import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock, SmallUpdateBlock
from .extractor import BasicEncoder, SmallEncoder
from .corr import CorrBlock, AlternateCorrBlock
from .myutil import bilinear_sampler, coords_grid, upflow8
import torchvision.models as models
from timm.models.vision_transformer import vit_base_patch16_224


#from torchvision.models import vit_b_16  # Example for ViT
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

    
class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        #self.encoder = resnets[num_layers](weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = resnets[num_layers](weights=None)
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        # Additional layers for FPN
        self.conv_l4_1x1 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv_l3_1x1 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv_l2_1x1 = nn.Conv2d(128, 256, kernel_size=1)

        #self.conv_out = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, input_image):
        self.features = []

        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.maxpool(self.features[-1]))
        self.features.append(self.encoder.layer1(self.features[-1]))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        # FPN
        l4 = self.features[-1]
        l3 = self.features[-2]
        l2 = self.features[-3]

        l4_1x1 = self.conv_l4_1x1(l4)
        l3_1x1 = self.conv_l3_1x1(l3)
        l2_1x1 = self.conv_l2_1x1(l2)

        l3_up = self.upsample(l4_1x1) + l3_1x1
        l2_up = self.upsample(l3_up) + l2_1x1

        #fused_features = self.conv_out(l2_up)
        #print(l2_up.shape)
        return l2_up


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class RAFT(nn.Module):
    def __init__(self):
        super(RAFT, self).__init__()

        # Mixed precision inference
        self.mixed_precision = False

        # Use smaller network variants if needed
        self.small = False

        # RAFT core parameters
        self.res_num = 18
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        self.corr_levels = 4
        self.corr_radius = 4

        self.leakyRELU = nn.LeakyReLU(0.1)
        self.dropout = 0

        # Fully connected layers for pose regression
        self.fc1 = nn.Linear(4096, 512)
        self.fc1_trasl = nn.Linear(512, 256)
        self.fc1_rot = nn.Linear(512, 256)
        self.fc2_trasl = nn.Linear(256, 3)
        self.fc2_rot = nn.Linear(256, 4)

        # Whether to use an alternate correlation volume
        self.alternate_corr = False

        # Set up 2D CNN encoder blocks (ResNet-based for both RGB and LiDAR)
        block = BasicBlock
        input_lidar = 1
        layers = [2, 2, 2, 2]
        self.Action_Func = 'leakyrelu'

        if self.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=self.dropout)
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=self.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
        else:
            self.fnet1 = ResnetEncoder(num_layers=self.res_num, pretrained=True, num_input_images=1)
            self.cnet = ResnetEncoder(num_layers=self.res_num, pretrained=True, num_input_images=1)
            self.update_block = BasicUpdateBlock(hidden_dim=hdim)

        # LiDAR encoder setup (ResNet-like)
        self.inplanes = 64
        self.conv1_lidar = nn.Conv2d(input_lidar, 64, kernel_size=7, stride=2, padding=3)
        self.elu_lidar = nn.ELU()
        self.leakyRELU_lidar = nn.LeakyReLU(0.1)
        self.maxpool_lidar = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_lidar = self._make_layer(block, 64, layers[0])
        self.layer2_lidar = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_lidar = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_lidar = self._make_layer(block, 512, layers[3], stride=2)

        # Feature reduction for correlation
        self.conv_l4_1x1 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv_l3_1x1 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv_l2_1x1 = nn.Conv2d(128, 256, kernel_size=1)

        self.conv_out = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def _make_layer(self, block, planes, blocks, stride=1):
        """Construct a sequence of residual blocks"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        """Set all BatchNorm layers to evaluation mode"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """Initialize coordinate grids for optical flow computation"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Learned upsampling using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """
        Forward pass of the RAFT model.
        Args:
            image1: RGB image tensor [B, C, H, W]
            image2: LiDAR projection tensor [B, 1, H, W]
            iters: Number of RAFT refinement iterations
            flow_init: Optional initial flow
            upsample: Whether to upsample flow
            test_mode: If True, return only final prediction
        """

        hdim = self.hidden_dim
        cdim = self.context_dim

        # Extract RGB features
        fmap1 = self.fnet1(image1)
        c16 = fmap1  # feature map from RGB (stride 8)

        # Extract LiDAR features through modified ResNet
        x2 = self.conv1_lidar(image2)
        if self.Action_Func == 'leakyrelu':
            c22 = self.leakyRELU_lidar(x2)
        elif self.Action_Func == 'elu':
            c22 = self.elu_lidar(x2)
        c23 = self.layer1_lidar(self.maxpool_lidar(c22))
        c24 = self.layer2_lidar(c23)
        c25 = self.layer3_lidar(c24)
        c26 = self.layer4_lidar(c25)

        # Multi-scale 1x1 convolutions
        l4_1x1 = self.conv_l4_1x1(c26)
        l3_1x1 = self.conv_l3_1x1(c25)
        l2_1x1 = self.conv_l2_1x1(c24)

        # Feature fusion through upsampling and addition
        l3_up = self.upsample(l4_1x1) + l3_1x1
        l2_up = self.upsample(l3_up) + l2_1x1

        # Correlation volume
        if self.alternate_corr:
            corr_fn = AlternateCorrBlock(c16, l2_up, radius=self.corr_radius)
        else:
            corr_fn = CorrBlock(c16, l2_up, radius=self.corr_radius)

        # Context network split into hidden state and input
        with autocast(enabled=self.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        # Initialize flow coordinates
        coords0, coords1 = self.initialize_flow(image1)
        if flow_init is not None:
            coords1 = coords1 + flow_init

        rot_predictions = []

        # RAFT iterative refinement
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # correlation volume
            flow = coords1 - coords0

            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            coords1 = coords1 + delta_flow
            x = coords1 - coords0
            x = x.view(x.shape[0], -1)  # flatten flow

            # Pose regression head
            x = self.leakyRELU(self.fc1(x))
            rot = self.leakyRELU(self.fc1_rot(x))
            rot = self.fc2_rot(rot)
            rot = F.normalize(rot, dim=1)  # normalize quaternion
            rot_predictions.append(rot)

        if test_mode:
            return [rot_predictions[-1]]  # return final prediction only

        return rot_predictions  # return all predictions across iterations