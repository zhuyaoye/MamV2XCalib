import math
import os
import random
import time
import sys


# import apex
import mathutils
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn as nn

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from DatasetLidarCamera import DatasetI2V_DAIR_V2X_Camera_Sequence, DatasetI2V_tumtraf_Sequence
from losses import listlossmam
from models.mamraft import mamraft
from quaternion_distances import quaternion_distance
from tensorboardX import SummaryWriter
from utils import (mat2xyzrpy, merge_inputs, overlay_imgs, quat2mat,
                   quaternion_from_matrix, rotate_back, rotate_forward,
                   tvector2mat)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
from torchvision import transforms
ex = Experiment("MamV2XCalib")
ex.captured_out_filter = apply_backspaces_and_linefeeds

try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# noinspection PyUnusedLocal
@ex.config
def config():
    checkpoints = './checkpoints/'
    use_reflectance = False
    epochs = 80
    BASE_LEARNING_RATE = 0.0001   #0.00002
    loss = 'listloss'
    max_t = 0
    max_r = 1.0# 20.0, 10.0, 5.0,  2.0,  1.0
    batch_size = 30  # 10
    num_worker = 6
    optimizer = 'adam'
    resume = False
    #weights = None
    weights = 'path_of_half_train.tar'   #replace with your own path
    rescale_rot = 1.0
    rescale_transl = 0
    max_depth = 80.
    weight_point_cloud = 0.1
    log_frequency = 10
    starting_epoch = -1
    dataset = 'v2x-seq'
    
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'


EPOCH = 1
def _init_fn(worker_id, seed):
    seed = seed + worker_id + EPOCH*100
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_2D_lidar_projection(pcl, cam_intrinsic):
    pcl_xyz = cam_intrinsic @ pcl.T
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
    pcl_uv = pcl_xyz[:, :2]

    return pcl_uv, pcl_z


def lidar_project_depth(pc_rotated, cam_calib, img_shape):
    pc_rotated = pc_rotated[:3, :].detach().cpu().numpy()
    cam_intrinsic = cam_calib.numpy()
    pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated.T, cam_intrinsic)
    mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (
            pcl_uv[:, 1] < img_shape[0]) & (pcl_z > 0)
    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]
    pcl_uv = pcl_uv.astype(np.uint32)
    pcl_z = pcl_z.reshape(-1, 1)
    depth_img = np.zeros((img_shape[0], img_shape[1], 1))
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z
    depth_img = torch.from_numpy(depth_img.astype(np.float32))
    depth_img = depth_img.cuda()
    depth_img = depth_img.permute(2, 0, 1)

    return depth_img, pcl_uv


# CCN training
@ex.capture
def train(model, optimizer, rgb_img, refl_img, target_transl, target_rot, loss_fn, point_clouds, loss):
    model.train()

    optimizer.zero_grad()

    # Run model
    rot_err = model(rgb_img, refl_img)

    if loss == 'points_distance' or loss == 'combined':
        losses = loss_fn(point_clouds, target_transl, target_rot, rot_err)
    elif loss=='listloss':
        losses = loss_fn(point_clouds, target_transl, target_rot, rot_err)
    else:
        losses = loss_fn(target_transl, target_rot,  rot_err)

    losses['total_loss'].backward()
    optimizer.step()

    return losses, rot_err


# CNN test
@ex.capture
def val(model, rgb_img, refl_img, target_transl, target_rot, loss_fn, point_clouds, loss):
    model.eval()

    # Run model
    with torch.no_grad():
        rot_err = model(rgb_img, refl_img)

    if loss == 'points_distance' or loss == 'combined':
        losses = loss_fn(point_clouds, target_transl, target_rot, rot_err)
    elif loss=='listloss':
        losses = loss_fn(point_clouds, target_transl, target_rot, rot_err)
    else:
        losses = loss_fn(target_transl, target_rot, rot_err)

    total_trasl_error = torch.tensor(0.0)
    total_trasl_error=total_trasl_error.cuda()
    total_rot_error = quaternion_distance(target_rot, rot_err, target_rot.device)
    total_rot_error = total_rot_error * 180. / math.pi

    return losses, total_rot_error.sum().item(), rot_err


@ex.automain
def main(_config, _run, seed):
    global EPOCH
    print('Loss Function Choice: {}'.format(_config['loss']))
    
    img_shape = (1080, 1920)
    input_size = (256, 512)
    
    max_t=_config['max_t']
    max_r=_config['max_r']
    if _config['dataset']=='v2x-seq':
        dataset_train = DatasetI2V_DAIR_V2X_Camera_Sequence(path='/home/JJ_Group/zhuyy2402/FFNet-VIC3D/data/v2x-seq/V2X-Seq/cooperative-vehicle-infrastructure', 
                                    split_path='/home/JJ_Group/zhuyy2402/DAIR-V2X/data/split_datas/cooperative-split-data-spd.json',
                                    max_t=max_t, max_r=max_r, split='train',
                                    sensortype="lidar")
        dataset_val = DatasetI2V_DAIR_V2X_Camera_Sequence(path='/home/JJ_Group/zhuyy2402/FFNet-VIC3D/data/v2x-seq/V2X-Seq/cooperative-vehicle-infrastructure', 
                                    split_path='/home/JJ_Group/zhuyy2402/DAIR-V2X/data/split_datas/cooperative-split-data-spd.json',
                                    max_t=max_t, max_r=max_r, split='val',
                                    sensortype="lidar")
    elif _config['dataset']=='tumtraf':
        dataset_train = DatasetI2V_tumtraf_Sequence(path='/home/JJ_Group/zhuyy2402/tumtraf_v2x_cooperative_perception_dataset/train',
                                  max_t=max_t, max_r=max_r, split='train')
        dataset_val = DatasetI2V_tumtraf_Sequence(path='/home/JJ_Group/zhuyy2402/tumtraf_v2x_cooperative_perception_dataset/val',
                                  max_t=max_t, max_r=max_r, split='val')
    else:
        raise TypeError('no such dataset!')
    model_savepath = os.path.join(_config['checkpoints'], 'val_seq_' , 'models')
    if not os.path.exists(model_savepath):
        os.makedirs(model_savepath)
    log_savepath = os.path.join(_config['checkpoints'], 'val_seq_' , 'log')
    if not os.path.exists(log_savepath):
        os.makedirs(log_savepath)

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    def init_fn(x): return _init_fn(x, seed)

    train_dataset_size = len(dataset_train)
    val_dataset_size = len(dataset_val)
    print('Number of the train dataset: {}'.format(train_dataset_size))
    print('Number of the val dataset: {}'.format(val_dataset_size))

    # Training and validation set creation
    num_worker = _config['num_worker']
    batch_size = _config['batch_size']
    TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                 shuffle=True,
                                                 batch_size=batch_size,
                                                 num_workers=num_worker,
                                                 worker_init_fn=init_fn,
                                                 collate_fn=merge_inputs,
                                                 drop_last=False,
                                                 pin_memory=True)

    ValImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                shuffle=False,
                                                batch_size=batch_size,
                                                num_workers=num_worker,
                                                worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=True)

    print(len(TrainImgLoader))
    print(len(ValImgLoader))

    if _config['loss'] == 'listloss':
        loss_fn = listlossmam(_config['rescale_transl'], _config['rescale_rot'], _config['weight_point_cloud'])
    else:
        raise ValueError("Unknown Loss Function")

    model=mamraft()
    model.unfreeze_selected_layers()

    if _config['weights'] is not None:
        print(f"Loading weights from {_config['weights']}")
        checkpoint = torch.load(_config['weights'], map_location='cpu')
        saved_state_dict = checkpoint['state_dict']

        # Get the state dict of the reduced model
        reduced_state_dict = model.state_dict()

        # Update the reduced model's state dict with the parameters from the original model
        for name, param in saved_state_dict.items():
            if name in reduced_state_dict and reduced_state_dict[name].size() == param.size():
                reduced_state_dict[name].copy_(param)

        # Load the updated state dict into the reduced model
        model.load_state_dict(reduced_state_dict)
        
    model = nn.DataParallel(model)
    model = model.cuda()

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    if _config['loss'] == 'geometric':
        parameters += list(loss_fn.parameters())
    if _config['optimizer'] == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=_config['BASE_LEARNING_RATE'], weight_decay=5e-6)
        # Probably this scheduler is not used
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,40,50], gamma=0.5)
        #adjust according to the dataset
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,100], gamma=0.5)
    else:
        optimizer = optim.SGD(parameters, lr=_config['BASE_LEARNING_RATE'], momentum=0.9,
                              weight_decay=5e-6, nesterov=True)

    starting_epoch = _config['starting_epoch']
    if _config['weights'] is not None and _config['resume']:
        checkpoint = torch.load(_config['weights'], map_location='cpu')
        opt_state_dict = checkpoint['optimizer']
        optimizer.load_state_dict(opt_state_dict)
        if starting_epoch != 0:
            starting_epoch = checkpoint['epoch']


    # Allow mixed-precision if needed
    # model, optimizer = apex.amp.initialize(model, optimizer, opt_level=_config["precision"])

    start_full_time = time.time()
    BEST_VAL_LOSS = 10000.
    old_save_filename = None

    train_iter = 0
    val_iter = 0
    for epoch in range(starting_epoch, _config['epochs'] + 1):
        if epoch==30:
            model.module.unfreeze_selected_layer_all()
            current_lr = optimizer.param_groups[0]['lr']
            print("current_lr:", current_lr)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=current_lr, weight_decay=5e-6)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,40,50], gamma=0.5)
            #adjust according to the dataset
        EPOCH = epoch
        print('This is %d-th epoch' % epoch)
        epoch_start_time = time.time()
        total_train_loss = 0
        local_loss = 0.
        if _config['optimizer'] != 'adam':
            _run.log_scalar("LR", _config['BASE_LEARNING_RATE'] *
                            math.exp((1 - epoch) * 4e-2), epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = _config['BASE_LEARNING_RATE'] * \
                                    math.exp((1 - epoch) * 4e-2)
        else:
            _run.log_scalar("LR", scheduler.get_last_lr()[0])


        ## Training ##
        time_for_50ep = time.time()
        
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            lidar_input = []
            rgb_input = []
            lidar_gt = []
            shape_pad_input = []
            real_shape_input = []
            pc_rotated_input = []

            # gt pose
            sample['tr_error'] = sample['tr_error'].cuda()
            sample['rot_error'] = sample['rot_error'].cuda()

            #start_preprocess = time.time()
            for idx in range(len(sample['rgb'])):
                # ProjectPointCloud in RT-pose
                real_shape = [sample['rgb'][idx].shape[0], sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2]]

                sample['point_cloud'][idx] = sample['point_cloud'][idx].cuda() 
                pc_lidar = sample['point_cloud'][idx].clone()

                if _config['max_depth'] < 80.:
                    pc_lidar = pc_lidar[:, pc_lidar[0, :] < _config['max_depth']].clone()
                
                depth_gt, uv = lidar_project_depth(pc_lidar, sample['calib'][idx], real_shape) # image_shape
                depth_gt /= _config['max_depth']

                R = mathutils.Quaternion(sample['rot_error'][idx]).to_matrix()
                R.resize_4x4()
                T = mathutils.Matrix.Translation(sample['tr_error'][idx])
                RT = T @ R
                
                pc_rotated = rotate_back(sample['point_cloud'][idx], RT) # Pc` = RT * Pc

                if _config['max_depth'] < 80.:
                    pc_rotated = pc_rotated[:, pc_rotated[0, :] < _config['max_depth']].clone()

                depth_img, uv = lidar_project_depth(pc_rotated, sample['calib'][idx], real_shape) # image_shape
                depth_img /= _config['max_depth']

                # PAD ONLY ON RIGHT AND BOTTOM SIDE
                normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                
                rgb = sample['rgb'][idx].cuda().permute(2,0,1).float()    #h w c----c h w
                rgb = normalization(rgb/255.0)
                shape_pad = [0, 0, 0, 0]

                shape_pad[3] = (img_shape[0] - rgb.shape[1])  # // 2
                shape_pad[1] = (img_shape[1] - rgb.shape[2])  # // 2 + 1

                rgb = F.pad(rgb, shape_pad)
                depth_img = F.pad(depth_img, shape_pad)
                depth_gt = F.pad(depth_gt, shape_pad)
 
  
                rgb_input.append(rgb)
                lidar_input.append(depth_img)
                lidar_gt.append(depth_gt)
                real_shape_input.append(real_shape)
                shape_pad_input.append(shape_pad)
                pc_rotated_input.append(pc_rotated)
                

            lidar_input = torch.stack(lidar_input)
            rgb_input = torch.stack(rgb_input)
            rgb_show = rgb_input.clone()
            lidar_show = lidar_input.clone()
            rgb_input = rgb_input.float() 
            lidar_input=lidar_input.float() 
            rgb_input = F.interpolate(rgb_input, size=[256, 512], mode="bilinear")
            lidar_input = F.interpolate(lidar_input, size=[256, 512], mode="bilinear")
        
            loss, R_predicted= train(model, optimizer, rgb_input, lidar_input,
                                                   sample['tr_error'], sample['rot_error'],
                                                   loss_fn, sample['point_cloud'], _config['loss'])
            T_predicted=sample['tr_error']
            
            for key in loss.keys():
                if loss[key].item() != loss[key].item():
                    raise ValueError("Loss {} is NaN".format(key))
           
            
            local_loss += loss['total_loss'].item()

            if batch_idx % 50 == 0 and batch_idx != 0:

                print(f'Iter {batch_idx}/{len(TrainImgLoader)} training loss = {local_loss/50:.3f}, '
                      f'time = {(time.time() - start_time)/lidar_input.shape[0]:.4f}, '
                      f'time for 50 iter: {time.time()-time_for_50ep:.4f}')
                time_for_50ep = time.time()
                _run.log_scalar("Loss", local_loss/50, train_iter)
                local_loss = 0.
            total_train_loss += loss['total_loss'].item() * len(sample['rgb'])
            train_iter += 1

        print("------------------------------------")
        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(dataset_train)))
        print('Total epoch time = %.2f' % (time.time() - epoch_start_time))
        print("------------------------------------")
        _run.log_scalar("Total training loss", total_train_loss / len(dataset_train), epoch)
            
        ## Validation ##
        total_val_loss = 0.
        total_val_t = 0.
        total_val_r = 0.

        local_loss = 0.0
        for batch_idx, sample in enumerate(ValImgLoader):
            #print(f'batch {batch_idx+1}/{len(TrainImgLoader)}', end='\r')
            start_time = time.time()
            lidar_input = []
            rgb_input = []
            lidar_gt = []
            shape_pad_input = []
            real_shape_input = []
            pc_rotated_input = []

            # gt pose
            sample['tr_error'] = sample['tr_error'].cuda()
            sample['rot_error'] = sample['rot_error'].cuda()

            for idx in range(len(sample['rgb'])):
                # ProjectPointCloud in RT-pose
                real_shape = [sample['rgb'][idx].shape[0], sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2]]
                sample['point_cloud'][idx] = sample['point_cloud'][idx].cuda() 
                pc_lidar = sample['point_cloud'][idx].clone()

                if _config['max_depth'] < 80.:
                    pc_lidar = pc_lidar[:, pc_lidar[0, :] < _config['max_depth']].clone()

                depth_gt, uv = lidar_project_depth(pc_lidar, sample['calib'][idx], real_shape) # image_shape
                depth_gt /= _config['max_depth']
            

                R = mathutils.Quaternion(sample['rot_error'][idx]).to_matrix()
                R.resize_4x4()
                T = mathutils.Matrix.Translation(sample['tr_error'][idx])
                RT = T @ R
                
                pc_rotated = rotate_back(sample['point_cloud'][idx], RT) # Pc` = RT * Pc

                if _config['max_depth'] < 80.:
                    pc_rotated = pc_rotated[:, pc_rotated[0, :] < _config['max_depth']].clone()

                depth_img, uv = lidar_project_depth(pc_rotated, sample['calib'][idx], real_shape) # image_shape
                depth_img /= _config['max_depth']


                normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                
                rgb = sample['rgb'][idx].cuda().permute(2,0,1).float()    #h w c----c h w
                
                rgb = normalization(rgb/255.0)
                shape_pad = [0, 0, 0, 0]

                shape_pad[3] = (img_shape[0] - rgb.shape[1]) 
                shape_pad[1] = (img_shape[1] - rgb.shape[2])  

                rgb = F.pad(rgb, shape_pad)
                depth_img = F.pad(depth_img, shape_pad)
                depth_gt = F.pad(depth_gt, shape_pad)

                rgb_input.append(rgb)
                lidar_input.append(depth_img)
                lidar_gt.append(depth_gt)
                real_shape_input.append(real_shape)
                shape_pad_input.append(shape_pad)
                pc_rotated_input.append(pc_rotated)

            lidar_input = torch.stack(lidar_input)
            rgb_input = torch.stack(rgb_input)
            rgb_show = rgb_input.clone()
            lidar_show = lidar_input.clone()
            rgb_input = rgb_input.float() 
            lidar_input=lidar_input.float() 
            rgb_input = F.interpolate(rgb_input, size=[256, 512], mode="bilinear")
            lidar_input = F.interpolate(lidar_input, size=[256, 512], mode="bilinear")

            loss, rot_e, R_predicted = val(model, rgb_input, lidar_input,
                                                                    sample['tr_error'], sample['rot_error'],
                                                                    loss_fn, sample['point_cloud'], _config['loss'])
            T_predicted=sample['tr_error']
            for key in loss.keys():
                if loss[key].item() != loss[key].item():
                    raise ValueError("Loss {} is NaN".format(key))

        
            total_val_r += rot_e
            local_loss += loss['total_loss'].item()

            if batch_idx % 50 == 0 and batch_idx != 0:
                print('Iter %d val loss = %.3f , time = %.2f' % (batch_idx, local_loss/50.,
                                                                    (time.time() - start_time)/lidar_input.shape[0]))
                local_loss = 0.0
            total_val_loss += loss['total_loss'].item() * len(sample['rgb'])
            val_iter += 1

        print("------------------------------------")
        print('total val loss = %.3f' % (total_val_loss / len(dataset_val)))
        print(f'total traslation error: {total_val_t / len(dataset_val)} cm')
        print(f'total rotation error: {total_val_r / len(dataset_val)} °')
        print("------------------------------------")

        _run.log_scalar("Val_Loss", total_val_loss / len(dataset_val), epoch)
        _run.log_scalar("Val_t_error", total_val_t / len(dataset_val), epoch)
        _run.log_scalar("Val_r_error", total_val_r / len(dataset_val), epoch)

        # SAVE
        val_loss = total_val_loss / len(dataset_val)
        if val_loss < BEST_VAL_LOSS:
            BEST_VAL_LOSS = val_loss
            if _config['rescale_transl'] > 0:
                _run.result = total_val_t / len(dataset_val)
            else:
                _run.result = total_val_r / len(dataset_val)
            savefilename = f'{model_savepath}/checkpoint_r{_config["max_r"]:.2f}_t{_config["max_t"]:.2f}_e{epoch}_{val_loss:.3f}.tar'
            torch.save({
                'config': _config,
                'epoch': epoch,
                'state_dict': model.module.state_dict(), # multi gpu
                'optimizer': optimizer.state_dict(),
                'train_loss': total_train_loss / len(dataset_train),
                'val_loss': total_val_loss / len(dataset_val),
            }, savefilename)
            print(f'Model saved as {savefilename}')
            if old_save_filename is not None:
                if os.path.exists(old_save_filename):
                    os.remove(old_save_filename)
            old_save_filename = savefilename
        scheduler.step()
    print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))
    return _run.result
