import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock, SmallUpdateBlock
from .extractor import BasicEncoder, SmallEncoder
from .corr import CorrBlock, AlternateCorrBlock
from .myutil import bilinear_sampler, coords_grid, upflow8
import torchvision.models as models

from timesformer_pytorch import TimeSformer

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

        self.encoder = resnets[num_layers](weights=models.ResNet18_Weights.DEFAULT)

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


class transraft(nn.Module):
    def __init__(self):
        super(transraft, self).__init__()

        # === Basic settings ===
        self.mixed_precision = False
        self.small = False
        self.res_num = 18
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        self.corr_levels = 4
        self.corr_radius = 4
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.dropout = 0

        # === Final pose estimation layers ===
        self.f1 = nn.Linear(2048, 512)
        self.f1_rot = nn.Linear(512, 256)
        self.f2_rot = nn.Linear(256, 4)

        self.alternate_corr = False
        block = BasicBlock
        input_lidar = 1
        layers = [2, 2, 2, 2]
        self.Action_Func = 'leakyrelu'

        # === Encoder & Update Block ===
        if self.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=self.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=self.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
        else:
            self.fnet1 = ResnetEncoder(num_layers=self.res_num, pretrained=True, num_input_images=1)
            self.cnet = ResnetEncoder(num_layers=self.res_num, pretrained=True, num_input_images=1)
            self.update_block = BasicUpdateBlock(hidden_dim=hdim)

        # === LiDAR encoder layers ===
        self.inplanes = 64
        self.conv1_lidar = nn.Conv2d(input_lidar, 64, kernel_size=7, stride=2, padding=3)
        self.elu_lidar = nn.ELU()
        self.leakyRELU_lidar = nn.LeakyReLU(0.1)
        self.maxpool_lidar = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_lidar = self._make_layer(block, 64, layers[0])
        self.layer2_lidar = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_lidar = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_lidar = self._make_layer(block, 512, layers[3], stride=2)

        # === 1x1 conv to reduce LiDAR features ===
        self.conv_l4_1x1 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv_l3_1x1 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv_l2_1x1 = nn.Conv2d(128, 256, kernel_size=1)

        self.conv_out = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # === Transformer-based temporal encoder ===
        self.timeSformer = TimeSformer(
            dim=192,
            patch_size=16,
            num_frames=5,
            num_classes=2048,
            depth=12,
            heads=8,
            dim_head=64,
            channels=2
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        # Build a residual block layer (e.g., ResNet)
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        # Freeze all BatchNorm layers
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """Initialize forward and backward coordinate grids"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Upsample coarse flow to full resolution using learned mask"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=10, flow_init=None, upsample=True, test_mode=False):
        """Main forward pass for flow + transformer-based pose estimation"""

        # RGB features from image1
        fmap1 = self.fnet1(image1)
        c16 = fmap1  # RGB features for correlation

        # LiDAR features from image2
        x2 = self.conv1_lidar(image2)
        c22 = self.leakyRELU_lidar(x2) if self.Action_Func == 'leakyrelu' else self.elu_lidar(x2)
        c23 = self.layer1_lidar(self.maxpool_lidar(c22))
        c24 = self.layer2_lidar(c23)
        c25 = self.layer3_lidar(c24)
        c26 = self.layer4_lidar(c25)

        # Fuse multi-scale LiDAR features
        l4_1x1 = self.conv_l4_1x1(c26)
        l3_1x1 = self.conv_l3_1x1(c25)
        l2_1x1 = self.conv_l2_1x1(c24)
        l3_up = self.upsample(l4_1x1) + l3_1x1
        l2_up = self.upsample(l3_up) + l2_1x1

        # Correlation layer
        corr_fn = AlternateCorrBlock(c16, l2_up, radius=self.corr_radius) if self.alternate_corr else CorrBlock(c16, l2_up, radius=self.corr_radius)

        # Context network to initialize hidden states
        with autocast(enabled=self.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)
        if flow_init is not None:
            coords1 = coords1 + flow_init

        # Iteratively refine flow
        x_list = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            coords1 = coords1 + delta_flow
            x = coords1 - coords0
            x_list.append(x)

        # Stack flow results and reshape to (B, T, C, H, W)
        x_tensor = torch.cat(x_list, dim=3)
        B, C, H, W = x_tensor.shape
        T = 5
        batch = B // T
        assert B % T == 0, "Batch size must be divisible by T"
        x_tensor = x_tensor.view(batch, T, C, H, W)

        # Transformer processing across time
        x_tensor = self.timeSformer(x_tensor)
        x_tensor = self.leakyRELU(self.f1(x_tensor))
        rot = self.leakyRELU(self.f1_rot(x_tensor))
        rot = self.f2_rot(rot)
        rot = F.normalize(rot, dim=1)
        rot = rot.repeat_interleave(T, dim=0)

        return rot

    def unfreeze_selected_layers(self):
        """Unfreeze only transformer + pose head layers for finetuning"""
        for param in self.parameters():
            param.requires_grad = False
        for param in self.timeSformer.parameters():
            param.requires_grad = True
        for param in self.f1.parameters():
            param.requires_grad = True
        for param in self.f1_rot.parameters():
            param.requires_grad = True
        for param in self.f2_rot.parameters():
            param.requires_grad = True