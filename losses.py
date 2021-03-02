import torch
import torch.nn as nn
import numpy as np
import glob
from utils.learning_helpers import disp_to_depth, save_obj
from utils.geometry_helpers import euler2mat
from models.stn import *
from models.plane_net import scale_recovery



class SSIM_Loss(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM_Loss, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y, valid_points = None):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        l = torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        return l

def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    
    mean_disp = disp.mean(2, True).mean(3, True)
    norm_disp = disp / (mean_disp + 1e-7)
    disp = norm_disp
    
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

class Plane_Height_loss(torch.nn.Module):
    def __init__(self, config):
        super(Plane_Height_loss, self).__init__()
        self.config = config
    
    def forward(self, plane_est, depth, intrinsics):
        scale_factor = scale_recovery(plane_est, depth, intrinsics, h_gt=self.config['camera_height']/30)
        target_depth = (depth.clone()*( scale_factor.reshape((-1,1,1,1)).expand_as(depth)) ).detach()
        depth_loss = (torch.abs(target_depth - depth)/target_depth).mean() 
        return scale_factor, depth_loss 

class Left_Right_Consist_Loss(torch.nn.Module):
    def __init__(self, config):
        super(Left_Right_Consist_Loss, self).__init__()
        self.config = config
        self.ssim =  SSIM_Loss()
    
    def forward(self, target_img_left, target_img_right, tgt_depth, intrinsics):
        baseline = torch.FloatTensor([self.config['stereo_baseline']/30, 0, 0,0,0,0]).unsqueeze(0).expand((tgt_depth.size(0),6)).type_as(tgt_depth)
        ref_img_warped, valid_mask, _, _ = inverse_warp2(target_img_right, tgt_depth, tgt_depth, -baseline, intrinsics, 'zeros')
        diff_img = (target_img_left - ref_img_warped).abs().clamp(0, 1)

        if self.config['with_auto_mask']:
            auto_mask = (diff_img.mean(dim=1, keepdim=True) < (target_img_left - target_img_right).abs().mean(dim=1, keepdim=True)).float() * valid_mask
            valid_mask = auto_mask
            
        ssim_map = self.ssim(target_img_left, ref_img_warped)
        diff_img = (0.5* diff_img + 0.5* ssim_map).mean(1,True)

        loss = (diff_img * valid_mask.expand_as(diff_img)).sum() / valid_mask.sum()
        return loss      

class Compute_Loss(nn.modules.Module):
    def __init__(self, config, plane_model=None):
        super(Compute_Loss, self).__init__()
        self.config = config
        self.ssim =  SSIM_Loss()
        self.l_lr_consist = Left_Right_Consist_Loss(config)
        self.plane_model = plane_model

            ## Load ground plane estimator network for extracting ground plane
        if self.config['l_scale_recovery']:
            if plane_model is not None:
                self.plane_loss = Plane_Height_loss(config)
                print('loaded Plane Estimator')
            else:
                print('warning - must load a plane estimator network')
            
        self.l1_weight = config['l1_weight']
        self.l_ssim_weight = config['l_ssim_weight']
        self.l_smooth_weight = config['l_smooth_weight']
        self.num_scales = config['num_scales']
        self.l_depth_consist_weight = config['l_depth_consist_weight']
        self.l_scale_depth_weight = config['l_scale_depth_weight']
        self.l_scale_pose_weight = config['l_scale_pose_weight']
        # self.l_left_right_consist_weight = config['l_left_right_consist_weight']
        self.scale_factor_list = {}
        for i in range(0, self.config['num_epochs']+1):
            self.scale_factor_list[i] = []
  
    def forward(self, source_imgs, target_img, poses, disparity, intrinsics, pose_vec_weight=None, validate=False,epoch=5, target_img_right=None):
        ''' Adopting from https://github.com/JiawangBian/SC-SfMLearner-Release/blob/master/loss_functions.py '''
        zero = torch.zeros(1).type_as(intrinsics)
        losses = {'l_reconstruct_inverse': zero.clone(), 'l_reconstruct_forward': zero.clone(), 'l_depth': zero.clone(), 'l_smooth': zero.clone(), \
            'l_scale_depth': zero.clone(), 'l_scale_pose': zero.clone(), 'l_left_right_consist': zero.clone(), 'l_brightness': zero.clone()  }
        disparity, source_disparities = disparity[0], disparity[1:] #separate disparity list into source and target disps
        poses, poses_inv = poses[0], poses[1] #separate pose change predictions
        B,_,h,w = target_img.size()     
        
                
        if self.config['l_scale_recovery'] and epoch > 0: #keep out of loop, only need to compute once
            plane_est = self.plane_model(target_img, epoch=epoch)[0].detach()

        for scale, disp in enumerate(disparity): 
            #upsample and convert to depth
            if scale!=0: 
                disp = nn.functional.interpolate(disp, (h, w), mode='nearest') 
            _,d = disp_to_depth(disp, self.config['min_depth'], self.config['max_depth'])

            # if self.config['l_left_right_consist']:
            #     if target_img_right is not None:
            #         losses['l_left_right_consist'] += self.l_left_right_consist_weight*self.l_lr_consist(target_img, target_img_right.to(self.config['device']), d, intrinsics)
            #     else:
            #         "warning - need to pass target_img_right for left-right constraints (make sure to use load_stereo=True in config)"
            
            ## Disparity Smoothness Loss
            if self.config['l_smooth']:
                losses['l_smooth'] += (self.l_smooth_weight*get_smooth_loss(disp, target_img) )/( 2**scale)  

            '''Ground Plane Loss (experimental)'''  
            if self.config['l_scale_recovery'] and epoch > 0:
                scale_factor, plane_loss = self.plane_loss(plane_est, d, intrinsics) 
                self.scale_factor_list[epoch].append(scale_factor.mean().item())
                losses['l_scale_depth'] += self.l_scale_depth_weight*plane_loss

                for pose in poses:
                    target_pose = (pose[:,0:3].clone()*( scale_factor.reshape((-1,1)).expand_as(pose[:,0:3])) ).detach()
                    losses['l_scale_pose'] += (1+min(2*epoch,9))*self.l_scale_pose_weight*(pose[:,0:3] - target_pose).abs().mean()

            reconstruction_errors = []
            masks = []
            proj_depths = []
            if self.config['l_reconstruction']:
                for j, source_img in enumerate(source_imgs): 
                    pose, pose_inv = poses[j], poses_inv[j]
                    source_disparity = source_disparities[j][scale]
                    if scale!=0: 
                        source_disparity = nn.functional.interpolate(source_disparity, (h,w), mode='nearest')
                    _, source_d = disp_to_depth(source_disparity, self.config['min_depth'], self.config['max_depth'])

                    ## Disparity Smoothness Loss
                    if self.config['l_smooth']:
                        losses['l_smooth'] += (self.l_smooth_weight*get_smooth_loss(source_disparity, source_img) )/( 2**scale)  

                    '''inverse reconstruction - reproject target frame to source frames'''
                    if self.config['l_inverse']:
                        l_reprojection, l_depth, _, _ = self.compute_pairwise_loss(source_img, target_img, source_d, d, -pose_inv.clone(), intrinsics, epoch)

                        if self.config['l_depth_consist']:
                            losses['l_depth'] += self.l_depth_consist_weight*l_depth 
                        losses['l_reconstruct_inverse'] += 0.3*l_reprojection #*** 0.3

                    '''forward reconstruction - reproject source frames to target frame'''
                    l_reprojection, l_depth, diff_img, valid_mask  = self.compute_pairwise_loss(target_img, source_img, d, source_d, -pose.clone(), intrinsics, epoch)

                    if self.config['l_depth_consist']:
                        losses['l_depth'] += self.l_depth_consist_weight*l_depth 

                    reconstruction_errors.append(diff_img)
                    masks.append(valid_mask)
                reconstruction_errors = torch.cat(reconstruction_errors, 1)
                reconstruction_errors, idx = torch.min(reconstruction_errors,1)
                losses['l_reconstruct_forward'] += reconstruction_errors.mean()

        losses['total'] = 0  
        for key, value in losses.items():
            if key is not 'total':
                losses[key] = value/(self.num_scales)  
                losses['total'] += losses[key]

        return losses
    
    def mean_on_mask(self, diff, valid_mask):
        mask = valid_mask.expand_as(diff)
        if mask.sum() > 10000:
            mean_value = (diff * mask).sum() / mask.sum()
        else:
            print('warning - most pixels are masked.')
            mean_value = torch.tensor(0).float().type_as(mask)
        return mean_value
    
    def compute_pairwise_loss(self, tgt_img, ref_img, tgt_depth, ref_depth, pose, intrinsic, epoch, padding_mode='zeros'): 
        ref_img_warped, valid_mask, projected_depth, computed_depth = inverse_warp2(ref_img, tgt_depth, ref_depth, pose, intrinsic, padding_mode)

        diff_img = (tgt_img - ref_img_warped).abs().clamp(0, 1)

        if self.config['with_auto_mask']==True:
            auto_mask = (diff_img.mean(dim=1, keepdim=True) < (tgt_img - ref_img).abs().mean(dim=1, keepdim=True)).float() * valid_mask
            valid_mask = auto_mask
            
        if self.config['l_ssim']==True:
            ssim_map = self.ssim(tgt_img, ref_img_warped)
            diff_img = (self.config['l1_weight'] * diff_img + self.config['l_ssim_weight'] * ssim_map).mean(1,True)

            ## Depth Reprojection
        l_depth = 0
        if self.config['l_depth_consist']==True:
            diff_depth = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth)).clamp(0, 1)
            if self.config['with_depth_mask']:
                weight_mask = (1 - diff_depth.clone()) 
                diff_img = diff_img * weight_mask
            l_depth = self.mean_on_mask(diff_depth, valid_mask)
            

        l_reprojection = self.mean_on_mask(diff_img, valid_mask)
        
        return l_reprojection, l_depth, diff_img, valid_mask
   
