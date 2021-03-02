## modified from https://github.com/nianticlabs/monodepth2
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from utils.learning_helpers import save_obj, load_obj, disp_to_depth, data_and_model_loader
from data.kitti_loader_eigen import KittiLoaderPytorch
from utils.learning_helpers import *
from utils.custom_transforms import *
import os
import glob
import cv2

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


if __name__=='__main__':
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    path_to_ws = '/home/brandonwagstaff/learned_scale_recovery/' ##update this
    path_to_dset_downsized = '/media/datasets/KITTI-eigen-split/'

    pretrained_plane_dir = 'results/plane-model-eigen-202101201842'
    dir = path_to_ws + 'results/final_models/eigen-depth-eval-scaled-202102141219'
    cam_height=1.70
    median_scaling=False #align scale of predicted depth with ground truth using median depth
    plane_rescaling=False #align scale using ground plane detection and known camera height
    post_process = True #use the standard post-processing that flips images, recomputes depth, and merges with unflipped depth
    benchmark = 'eigen' ### eigen_benchmark for improved gt, 'eigen' for standard benchmark


    splits_dir = '{}/data/splits/{}'.format(path_to_ws, benchmark)
    config = load_obj('{}/config'.format(dir))
    pretrained_depth_path = glob.glob('{}/**depth**best-loss-val_seq-**-test_seq-{}**.pth'.format(dir, ''))[0]
    pretrained_pose_path = glob.glob('{}/**pose**best-loss-val_seq-**-test_seq-{}**.pth'.format(dir, ''))[0]


    config['data_dir'] = path_to_dset_downsized+config['img_resolution'] + '_res/' #
    config['minibatch'] = 6
    config['load_pretrained'] = True

    _, models, device = data_and_model_loader(config, pretrained_depth_path, pretrained_pose_path, seq=None)
    depth_model, pose_model = models[0], models[1]

    if plane_rescaling == True:
        ### Plane Model
        from models.plane_net import PlaneModel, scale_recovery
        from losses import Plane_Height_loss
        plane_loss = Plane_Height_loss(config)
        plane_model = PlaneModel(config).to(config['device'])
        pretrained_plane_path = glob.glob('../{}/**plane**.pth'.format(pretrained_plane_dir))[0]
        plane_model.load_state_dict(torch.load(pretrained_plane_path))
        plane_model.train(False).eval()

    test_dset = KittiLoaderPytorch(config, None, mode=benchmark, transform_img=get_data_transforms(config)['test'])
    test_dset_loaders = torch.utils.data.DataLoader(test_dset, batch_size=config['minibatch'], shuffle=False, num_workers=8)


    depth_model, pose_model = depth_model.train(False).eval(), pose_model.train(False).eval()  # Set model to evaluate mode
    depth_list = []
    pred_disps = []
    scale_factor_list = []

    with torch.no_grad():
        for k, data in enumerate(test_dset_loaders):
            target_img, source_imgs, lie_alg, intrinsics, flow_imgs  = data
            target_img, source_imgs, intrinsics = target_img['color_left'], source_imgs['color_left'], intrinsics['color_left']
            target_img = target_img.to(device)
            B = target_img.shape[0]
            
            if post_process == True:
                # Post-processed results require each image to have two forward passes
                target_img = torch.cat((target_img, torch.flip(target_img, [3])), 0)        
            
            disparities = depth_model(target_img, epoch=50)
            
            disps, depths = disp_to_depth(disparities[0], config['min_depth'], config['max_depth'])
            
            if plane_rescaling==True:
                plane_est = plane_model(target_img[0:B], epoch=50)[0].detach()
                intrinsics = intrinsics[:,0].type(torch.FloatTensor).to(device).clone()
                scale_factor = scale_recovery(plane_est, depths[0:B], intrinsics, h_gt=cam_height/30.)
                scale_factor_list.append(scale_factor.cpu().numpy())    
            
            pred_disp = disps.cpu()[:, 0].numpy()
            
            if post_process == True:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])        

            depth = 30*depths
            depth = depth.cpu()[:, 0].numpy()
            depth_list.append(depth)
            pred_disps.append(pred_disp)
            

        depth_list = np.concatenate(depth_list)
        pred_disps = np.concatenate(pred_disps)
        if plane_rescaling==True:
            scale_factor_list = np.concatenate(scale_factor_list)

    gt_path = os.path.join(splits_dir, "gt_depths.npz")
    gt_depths = np.load(gt_path, allow_pickle=True, fix_imports=True, encoding='latin1')["data"]

    errors = []
    ratios = []
    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 30 / pred_disp
        
        
        if benchmark == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array(
                [0.40810811 * gt_height, 0.99189189 * gt_height,
                    0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
        else:
            mask = gt_depth > 0
        
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        
        ### median scaling
        if median_scaling == True:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio
        
        if plane_rescaling == True:
            pred_depth *= scale_factor_list[i]
            
        pred_depth = np.clip(pred_depth, 0, 80)
        # print(np.max(gt_depth), np.max(pred_depth))
        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        
        errors.append(compute_errors(gt_depth, pred_depth))
        
    ratios = np.array(ratios)
    med = np.median(ratios)
    print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 5.3f} " * 7).format(*mean_errors.tolist()) )# + "\\\\")
    print("\n-> Done!")