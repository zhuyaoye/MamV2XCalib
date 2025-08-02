import torch
from torch import nn as nn

from quaternion_distances import quaternion_distance
from utils import quat2mat, rotate_back, rotate_forward, tvector2mat, quaternion_from_matrix
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d

class listLoss(nn.Module):
    def __init__(self, rescale_trans, rescale_rot, weight_point_cloud):
        super(listLoss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')
        self.weight_point_cloud = weight_point_cloud
        self.gamma = 0.8
    
    
    def forward(self, point_clouds, target_transl, target_rot, rot_list):
        n_predictions = len(rot_list)
        all_loss = {'total_loss': 0, 'rot_loss': 0, 'point_clouds_loss': 0}

        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)

            # Compute translation and rotation errors
            rot_err = rot_list[i]
            target_transl=target_transl.cuda()
            target_rot=target_rot.cuda()
            loss_transl = 0.

            loss_rot = 0.
            if self.rescale_rot != 0.:
                loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()

            pose_loss =  self.rescale_rot * loss_rot

            # Compute point clouds loss
            point_clouds_loss = torch.tensor([0.0]).to(rot_err.device)
            
            for j in range(len(point_clouds)):
                point_cloud_gt = point_clouds[j].to(rot_err.device)
                point_cloud_out = point_clouds[j].clone()
                point_cloud_out=point_cloud_out.cuda()
                
                R_target = quat2mat(target_rot[j])
                T_target = tvector2mat(target_transl[j])
                RT_target = torch.mm(T_target, R_target)

                R_predicted = quat2mat(rot_err[j])
                T_predicted = tvector2mat(target_transl[j])
                RT_predicted = torch.mm(T_predicted, R_predicted)

                RT_total = torch.mm(RT_target.inverse(), RT_predicted)
                
                point_cloud_out = rotate_forward(point_cloud_out, RT_total)

                error = (point_cloud_out - point_cloud_gt).norm(dim=0) 
                error.clamp(100.)
                point_clouds_loss += error.mean()
                
    

            total_loss = (1 - self.weight_point_cloud) * pose_loss + \
                         self.weight_point_cloud * (point_clouds_loss / len(point_clouds))
            all_loss['total_loss'] += i_weight * total_loss
            all_loss['rot_loss'] += i_weight * loss_rot
            all_loss['point_clouds_loss'] += i_weight * (point_clouds_loss / len(point_clouds))
            
        return all_loss
    
    
    
class listlossmam(nn.Module):
    def __init__(self, rescale_trans, rescale_rot, weight_point_cloud):
        super(listlossmam, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')
        self.weight_point_cloud = weight_point_cloud
        self.loss = {}

    def forward(self, point_clouds, target_transl, target_rot,  rot_err):
        """
        The Combination of Pose Error and Points Distance Error
        Args:
            point_cloud: list of B Point Clouds, each in the relative GT frame
            target_transl: groundtruth of the translations
            target_rot: groundtruth of the rotations
            transl_err: network estimate of the translations
            rot_err: network estimate of the rotations

        Returns:
            The combination loss of Pose error and the mean distance between 3D points
        """
        target_transl=target_transl.cuda()
        target_rot=target_rot.cuda()
        loss_transl = 0.
       
        loss_rot = 0.
        if self.rescale_rot != 0.:
            loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        pose_loss = self.rescale_rot*loss_rot

        point_clouds_loss = torch.tensor([0.0]).to(rot_err.device)
        for i in range(len(point_clouds)):
            point_cloud_gt = point_clouds[i].to(rot_err.device)
            point_cloud_out = point_clouds[i].clone()

            point_cloud_out=point_cloud_out.cuda()
            R_target = quat2mat(target_rot[i])
            T_target = tvector2mat(target_transl[i])
            RT_target = torch.mm(T_target, R_target)

            R_predicted = quat2mat(rot_err[i])
            T_predicted = tvector2mat(target_transl[i])
            RT_predicted = torch.mm(T_predicted, R_predicted)

            RT_total = torch.mm(RT_target.inverse(), RT_predicted)

            point_cloud_out = rotate_forward(point_cloud_out, RT_total)

            error = (point_cloud_out - point_cloud_gt).norm(dim=0)
            error.clamp(100.)
            point_clouds_loss += error.mean()

        
        total_loss = (1 - self.weight_point_cloud) * pose_loss +\
                     self.weight_point_cloud * (point_clouds_loss/ len(point_clouds))
        self.loss['total_loss'] = total_loss
        self.loss['rot_loss'] = loss_rot
        self.loss['point_clouds_loss'] = point_clouds_loss/ len(point_clouds)

        return self.loss