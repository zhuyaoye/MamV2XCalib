import csv
import random
import open3d as o3
import torch.nn as nn
import cv2
import mathutils
import sys

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from skimage import io
from tqdm import tqdm
import time
from models.mamraft import mamraft

from DatasetLidarCamera import DatasetI2V_DAIR_V2X_Camera_Sequence, DatasetI2V_tumtraf_Sequence

from quaternion_distances import quaternion_distance
from utils import (mat2xyzrpy, merge_inputs, overlay_imgs, quat2mat,
                   quaternion_from_matrix, rotate_back, rotate_forward,
                   tvector2mat)

from torchvision import transforms


plt.rcParams['axes.unicode_minus'] = False

font_EN = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
font_CN = {'family': 'AR PL UMing CN', 'weight': 'normal', 'size': 16}
plt_size = 10.5

ex = Experiment("MamV2XCalib-evaluate-iterative")
ex.captured_out_filter = apply_backspaces_and_linefeeds

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
# noinspection PyUnusedLocal
@ex.config
def config():
    max_t = 0
    max_r = 20
    dropout = 0.0
    max_depth = 80.
    iterative_method = 'multi_range' 
    weight = None
    #dataset = 'tumtraf'
    dataset = 'v2x-seq'
    

#replace with your own path
weights=[
    'checkpoint_r_20.tar',
    'checkpoint_r_10.tar',
    'checkpoint_r_5.tar',
    'checkpoint_r_2.tar',
    'checkpoint_r_1.tar'
    ]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 1


def _init_fn(worker_id, seed):
    seed = seed + worker_id + EPOCH * 100
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
    pc_valid = pc_rotated.T[mask]

    return depth_img, pcl_uv, pc_valid


@ex.automain
def main(_config, seed):  
    seed=12332412
    #seed=561740738
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    
    def init_fn(x):
        return _init_fn(x, seed)
    global EPOCH, weights
    if _config['weight'] is not None:
        weights = _config['weight']

    if _config['iterative_method'] == 'single':
        weights = [weights[0]]

    img_shape = (1080, 1920)
    input_size = (256, 512)
    max_t=_config['max_t']
    max_r=_config['max_r']

    if _config['dataset']=='v2x-seq':
        dataset_test = DatasetI2V_DAIR_V2X_Camera_Sequence(path='/home/JJ_Group/zhuyy2402/FFNet-VIC3D/data/v2x-seq/V2X-Seq/cooperative-vehicle-infrastructure', 
                                  split_path='/home/JJ_Group/zhuyy2402/DAIR-V2X/data/split_datas/cooperative-split-data-spd.json',
                                  max_t=max_t, max_r=max_r, split='test',
                                  sensortype="lidar")
    elif _config['dataset']=='tumtraf':
        dataset_test = DatasetI2V_tumtraf_Sequence(path='/home/JJ_Group/zhuyy2402/tumtraf_v2x_cooperative_perception_dataset/test',
                                  max_t=max_t, max_r=max_r, split='test')
    else:
        raise TypeError('no such dataset!')
        
    num_worker = 6
    batch_size = 5

    TestImgLoader = torch.utils.data.DataLoader(dataset=dataset_test,
                                                shuffle=False,
                                                batch_size=batch_size,
                                                num_workers=num_worker,
                                                worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=False)
    
   
    models = [] # iterative model
    
    for i in range(len(weights)):
        # network choice and settings      
        model = mamraft()
        checkpoint = torch.load(weights[i], map_location='cpu')
        saved_state_dict = checkpoint['state_dict']
       
        reduced_state_dict = model.state_dict()

        # Update the reduced model's state dict with the parameters from the original model
        for name, param in saved_state_dict.items():
            if name in reduced_state_dict and reduced_state_dict[name].size() == param.size():
                reduced_state_dict[name].copy_(param)

        # Load the updated state dict into the reduced model
        model.load_state_dict(reduced_state_dict)
        model = nn.DataParallel(model)
        model = model.cuda()
        model.eval()
        models.append(model)

        
   
    errors_r = []
    errors_t = []
    errors_t2 = []
    errors_xyz = []
    errors_rpy = []
    all_RTs = []
    mis_calib_list = []
    total_time = 0

    prev_tr_error = None
    prev_rot_error = None

    for i in range(len(weights) + 1):
        errors_r.append([])
        errors_t.append([])
        errors_t2.append([])
        errors_rpy.append([])

    # Initialize necessary variables
    for batch_idx, sample in enumerate(tqdm(TestImgLoader)):
        N = 100  # Adjust batch size as needed
        log_string = [str(batch_idx)]

        # Initialize lists to store input data
        lidar_input = []
        rgb_input = []
        lidar_gt = []
        shape_pad_input = []
        real_shape_input = []
        pc_rotated_input = []
        RTs = []
        shape_pad = [0, 0, 0, 0]
        outlier_filter = False

        
        sample['tr_error'] = sample['tr_error'].cuda()
        sample['rot_error'] = sample['rot_error'].cuda()
        

        for idx in range(len(sample['rgb'])):
            real_shape = [sample['rgb'][idx].shape[0], sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2]]

            sample['point_cloud'][idx] = sample['point_cloud'][idx].cuda()
            pc_lidar = sample['point_cloud'][idx].clone()

            if _config['max_depth'] < 80.:
                pc_lidar = pc_lidar[:, pc_lidar[0, :] < _config['max_depth']].clone()

            depth_gt, uv_gt, pc_gt_valid = lidar_project_depth(pc_lidar, sample['calib'][idx], real_shape)
            depth_gt /= _config['max_depth']

            
            R = quat2mat(sample['rot_error'][idx])
            T = tvector2mat(sample['tr_error'][idx])
            RT_inv = torch.mm(T, R)
            RT = RT_inv.clone().inverse()
            pc_rotated = rotate_back(sample['point_cloud'][idx], RT_inv)

            if _config['max_depth'] < 80.:
                pc_rotated = pc_rotated[:, pc_rotated[0, :] < _config['max_depth']].clone()

            depth_img, uv_input, pc_input_valid = lidar_project_depth(pc_rotated, sample['calib'][idx], real_shape)
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
            RTs.append([RT])
            
        if outlier_filter:
            continue

        lidar_input = torch.stack(lidar_input)
        rgb_input = torch.stack(rgb_input)
        rgb_resize = F.interpolate(rgb_input, size=[256, 512], mode="bilinear")
        lidar_resize = F.interpolate(lidar_input, size=[256, 512], mode="bilinear")
        rgb_show = rgb_input.clone()
        lidar_show = lidar_input.clone()
        

        target_transl = sample['tr_error'].to(device)
        target_rot = sample['rot_error'].to(device)

        RT1 = RTs[0][0]
        mis_calib = torch.stack(sample['initial_RT'])[1:]
        mis_calib_list.append(mis_calib)

        T_composed = RT1[:3, 3]
        R_composed = quaternion_from_matrix(RT1)
        errors_t[0].append(T_composed.norm().item())
        errors_t2[0].append(T_composed)
        errors_r[0].append(quaternion_distance(R_composed.unsqueeze(0),
                                            torch.tensor([1., 0., 0., 0.], device=R_composed.device).unsqueeze(0),
                                            R_composed.device))
        rpy_error = mat2xyzrpy(RT1)[3:]
        rpy_error *= (180.0 / 3.141592)
        errors_rpy[0].append(rpy_error)
        log_string += [str(errors_t[0][-1]), str(errors_r[0][-1]), str(errors_t2[0][-1][0].item()),
                    str(errors_t2[0][-1][1].item()), str(errors_t2[0][-1][2].item()),
                    str(errors_rpy[0][-1][0].item()), str(errors_rpy[0][-1][1].item()),
                    str(errors_rpy[0][-1][2].item())]

        start = 0

        with torch.no_grad():
            rotated_point_cloud_batch = []
            for iteration in range(start, len(weights)):
                t1 = time.time()

                if _config['iterative_method'] in ['single_range', 'single']:
                    T_predicted, R_predicted = models[0](rgb_resize, lidar_resize)
                elif _config['iterative_method'] == 'multi_range':
                    R_predicted = models[iteration](rgb_resize, lidar_resize)
                    
                    T_predicted=torch.zeros((5, 3), device='cuda:0')

                run_time = time.time() - t1
        

                RT_predicted_batch = []
                
                for i in range(T_predicted.size(0)):
                    R_predicted_mat = quat2mat(R_predicted[i])
                    T_predicted_mat = tvector2mat(T_predicted[i])
                    RT_predicted = torch.mm(T_predicted_mat, R_predicted_mat)
                    RT_predicted_batch.append(RT_predicted)

                    RTs[i].append(torch.mm(RTs[i][iteration], RT_predicted))
                    if iteration == 0:
                        rotated_point_cloud = pc_rotated_input[i]
                        rotated_point_cloud_batch.append(rotate_forward(rotated_point_cloud, RT_predicted))
                    else:
                        
                        rotated_point_cloud = rotated_point_cloud_batch[i]
                        rotated_point_cloud_batch[i]=rotate_forward(rotated_point_cloud, RT_predicted)

                depth_img_pred_batch, uv_pred_batch, pc_pred_valid_batch = [], [], []
                for i in range(len(rotated_point_cloud_batch)):
                    depth_img_pred, uv_pred, pc_pred_valid = lidar_project_depth(rotated_point_cloud_batch[i], sample['calib'][i], real_shape_input[i])
                    depth_img_pred /= _config['max_depth']
                    depth_img_pred_batch.append(F.pad(depth_img_pred, shape_pad_input[i]))
                    uv_pred_batch.append(uv_pred)
                    pc_pred_valid_batch.append(pc_pred_valid)
                    lidar_batch = [depth_img_pred.unsqueeze(0) for depth_img_pred in depth_img_pred_batch]
                    
                lidar_input = torch.stack(depth_img_pred_batch)
                lidar_resize = F.interpolate(lidar_input, size=[256, 512], mode="bilinear")
            

                for i in range(len(RTs)):
                    T_composed = RTs[i][iteration + 1][:3, 3]
                    
                    
                    R_composed = quaternion_from_matrix(RTs[i][iteration + 1])
                    
                    errors_t[iteration + 1].append(T_composed.norm().item())
                    errors_t2[iteration + 1].append(T_composed)
                    errors_r[iteration + 1].append(quaternion_distance(R_composed.unsqueeze(0), torch.tensor([1., 0., 0., 0.], device=R_composed.device).unsqueeze(0), R_composed.device))
                    rpy_error = mat2xyzrpy(RTs[i][iteration + 1])[3:] * (180.0 / 3.141592)
                    errors_rpy[iteration + 1].append(rpy_error)

                    log_string += [str(errors_t[iteration + 1][-1]), str(errors_r[iteration + 1][-1]),
                                str(errors_t2[iteration + 1][-1][0].item()), str(errors_t2[iteration + 1][-1][1].item()),
                                str(errors_t2[iteration + 1][-1][2].item()), str(errors_rpy[iteration + 1][-1][0].item()),
                                str(errors_rpy[iteration + 1][-1][1].item()), str(errors_rpy[iteration + 1][-1][2].item())]
          

        total_time += run_time

        for i in range(len(all_RTs)):
            all_RTs.append(RTs[i][-1])
            prev_RT = RTs[i][-1].inverse()
            prev_tr_error = prev_RT[:3, 3].unsqueeze(0)
            prev_rot_error = quaternion_from_matrix(prev_RT).unsqueeze(0)

        
    mis_calib_input = torch.stack(mis_calib_list)[:, :, 0]
    avg_time = total_time / 4540
    print("average runing time on {} iteration: {} s".format(len(weights), avg_time))
    print("End!")
   
    print("Iterative refinement: ")
    for i in range(len(weights) + 1):
        errors_r[i] = torch.tensor(errors_r[i]).abs() * (180.0 / 3.141592)
        errors_t[i] = torch.tensor(errors_t[i]).abs() * 100

        for k in range(len(errors_rpy[i])):
            errors_rpy[i][k] = errors_rpy[i][k].clone().detach().abs()
            errors_t2[i][k] = errors_t2[i][k].clone().detach().abs() * 100
        
        allx = [tensor[0].item() for tensor in errors_t2[i]]
        ally = [tensor[1].item() for tensor in errors_t2[i]]
        allz = [tensor[2].item() for tensor in errors_t2[i]]
        allr=[tensor[0].item() for tensor in errors_rpy[i]]
        allp=[tensor[1].item() for tensor in errors_rpy[i]]
        allyaw=[tensor[2].item() for tensor in errors_rpy[i]]



        mean_r = sum(allr) / len(allr)
        mean_p=sum(allp) / len(allp)
        mean_yaw=sum(allyaw) / len(allyaw)
        median_r = np.median(allr)
        median_p = np.median(allp)
        median_yaw = np.median(allyaw)
        var_r = np.std(allr)
        var_p=np.std(allp)
        var_yaw=np.std(allyaw)
        
        print(f"Iteration {i}: \tMean Rotation Error: {errors_r[i].mean():.4f} ° \n")
        print(f"Iteration {i}: \tMedian Rotation Error: {errors_r[i].median():.4f} ° \n")
        print(f"Iteration {i}: \tStd. Rotation Error: {errors_r[i].std():.4f} °\n ")
        print(f"Iteration {i}: \tMax. Rotation Error: {errors_r[i].max():.4f} °\n ")
             

       
        # rotation rpy
        print(f"Iteration {i}: \tMean Rotation Roll Error: {mean_r: .4f} °"
              f"     Median Rotation Roll Error: {median_r:.4f} °"
              f"     Std. Rotation Roll Error: {var_r:.4f} °")
        print(f"Iteration {i}: \tMean Rotation Pitch Error: {mean_p: .4f} °"
              f"     Median Rotation Pitch Error: {median_p:.4f} °"
              f"     Std. Rotation Pitch Error: {var_p:.4f} °")
        print(f"Iteration {i}: \tMean Rotation Yaw Error: {mean_yaw: .4f} °"
              f"     Median Rotation Yaw Error: {median_yaw:.4f} °"
              f"     Std. Rotation Yaw Error: {var_yaw:.4f} °\n")

    print("End!")

