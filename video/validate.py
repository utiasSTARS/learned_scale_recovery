import time
import torch
import sys
sys.path.insert(0,'..')
from utils.learning_helpers import *
from data.kitti_loader import process_sample_batch, process_sample
from models.stn import *
from vis import UnNormalize_img_array
from losses import SSIM_Loss

def solve_pose(pose_model, depths, target_img, source_img_list, flow_imgs, intrinsics):
    depth, source_depths = depths[0], depths[1:] #separate disparity list into source and target disps
    poses, poses_inv = [], []
    outputs = {'source1':{}, 'source2':{}}
    source_names = ['source'+str(i) for i in range(1,len(source_img_list)+1)]
    flow_imgs_fwd, flow_imgs_back = flow_imgs
    
    for source_img, source_depth, flow_img_fwd, flow_img_back, source_name in zip(source_img_list, source_depths, flow_imgs_fwd, flow_imgs_back, source_names):
        pose = pose_model([target_img, source_img, flow_img_fwd])
        pose_inv = pose_model([source_img, target_img, flow_img_back])
       
        poses.append(pose)
        poses_inv.append(pose_inv)
        
        img_reconstructed, valid_mask, projected_depth, computed_depth = inverse_warp2(source_img, depth, source_depth, -pose.clone(), intrinsics, 'zeros') #forwards
        img_reconstructed_inv, valid_mask_inv, projected_depth_inv, computed_depth_inv = inverse_warp2(target_img, source_depth, depth, -pose_inv.clone(), intrinsics, 'zeros') #forwards

        mean_errors = []
        diff_depth = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth)).clamp(0, 1)
        weight_mask = [(1 - diff_depth)]
        diff_imgs = [ ((target_img - img_reconstructed).abs().mean(1,True) )*weight_mask[-1]*valid_mask ]
        mean_error = (diff_imgs[-1]).sum(3).sum(2) / (valid_mask).sum(3).sum(2)
        mean_errors.append(mean_error)
        
        mean_errors_inv = []
        diff_depth_inv = ((computed_depth_inv - projected_depth_inv).abs() / (computed_depth_inv + projected_depth_inv)).clamp(0, 1)
        weight_mask_inv = [(1 - diff_depth_inv)]
        diff_imgs_inv = [ ((source_img - img_reconstructed_inv).abs().mean(1,True) )*weight_mask_inv[-1]*valid_mask_inv ]
        mean_error_inv = (diff_imgs_inv[-1]).sum(3).sum(2) / (valid_mask_inv).sum(3).sum(2)
        mean_errors_inv.append(mean_error_inv)
        
        img_reconstructions = [img_reconstructed]
        img_reconstructions_inv = [img_reconstructed_inv]
        valid_masks = [valid_mask]
        valid_masks_inv = [valid_mask_inv]

        outputs[source_name]['mean_errors'] = torch.cat(mean_errors,1)
        outputs[source_name]['mean_errors_inv'] = torch.cat(mean_errors_inv,1)
        outputs[source_name]['diff_imgs'] = torch.stack(diff_imgs,dim=1)
        outputs[source_name]['diff_imgs_inv'] = torch.stack(diff_imgs_inv,dim=1)
        outputs[source_name]['img_reconstructions'] = torch.stack(img_reconstructions,dim=1)
        outputs[source_name]['img_reconstructions_inv'] = torch.stack(img_reconstructions_inv,dim=1)
        outputs[source_name]['weight_masks'] = torch.stack(weight_mask,dim=1)
        outputs[source_name]['weight_masks_inv'] = torch.stack(weight_mask_inv,dim=1)
        outputs[source_name]['valid_masks'] = torch.stack(valid_masks,dim=1)
        outputs[source_name]['valid_masks_inv'] = torch.stack(valid_masks_inv,dim=1)   
            
    return poses, poses_inv, outputs

def test_depth_and_reconstruction(device, models, data, config):
    source_img_idx = 0
    dpc = config['dpc']
    mode = config['pose_output_type']
    depth_model, pose_model = models[0], models[1]
    
    target_img, source_img_list, gt_lie_alg_list, vo_lie_alg_list, flow_imgs, intrinsics, target_img_aug, \
                    source_img_aug_list, gt_lie_alg_aug_list, vo_lie_alg_aug_list, intrinsics_aug = process_sample_batch(data, config)
    
    source_disp_array = torch.zeros(0)  
    depth_masks = torch.zeros(0)  
    valid_masks = torch.zeros(0)
    diff_imgs = torch.zeros(0)


    disparity = depth_model(target_img, epoch=50)
    disparities = [disparity]
    
    for i, im in enumerate(source_img_list):
        source_disparity = depth_model(im, epoch=50)
        disparities.append(source_disparity)

    depths = [disp_to_depth(disp[0], config['min_depth'], config['max_depth'])[1].detach() for disp in disparities]
    

    poses, poses_inv, outputs = solve_pose(pose_model, \
        depths, target_img, source_img_list, flow_imgs, intrinsics)
    
    
    # print('forward', stacked_poses[0][:,-1,2].mean().item(), stacked_poses_inv[0][:,-1,2].mean().item())
    # print('backward', stacked_poses[1][:,-1,2].mean().item(), stacked_poses_inv[1][:,-1,2].mean().item())
    
    outputs = outputs['source1']

    imgs = torch.cat([source_img_list[0].unsqueeze(1), outputs['img_reconstructions'], target_img.unsqueeze(1)],1).cpu().detach()
    diff_imgs = outputs['diff_imgs'].cpu().detach()

    depth_masks = torch.cat((depth_masks, outputs['weight_masks'].cpu().detach()))
    valid_masks = torch.cat((valid_masks, outputs['valid_masks'].cpu().detach()))
    # diff_img = torch.stack((diff_img1.mean(1,True)*weight_mask1, (diff_img2.mean(1,True)*weight_mask2)), dim=1).cpu().detach()
    # diff_imgs = torch.cat((diff_imgs, diff_img))
    # imgs = torch.stack((source_img_list[0],img_reconstructed1, img_reconstructed2, target_img),dim=1).cpu().detach()

    return imgs, disparities[0][0][:,0].cpu().detach(), depth_masks, valid_masks, diff_imgs

    # poses, poses_inv = solve_pose(pose_model, target_img, source_img_list, vo_lie_alg_list, dpc, mode, 50, intrinsics, flow_imgs)
    # imgs = torch.cat([source_img_list[0].unsqueeze(1), target_img.unsqueeze(1)],1).cpu().detach()
    # return imgs, disparities[0][0][:,0].cpu().detach()

def get_plane_masks(device, plane_model, data, config):
    
    plane_model = plane_model.train(False).eval()
    plane_array = torch.zeros(0)
    
    target_img, source_imgs, lie_alg, intrinsics, flow_imgs = data

    lie_alg = lie_alg['color']
    target_img, source_imgs, intrinsics = target_img['color_left'], source_imgs['color_left'], intrinsics['color_left']
    target_img = target_img.to(device)

    source_disp_array = torch.zeros(0)  
    depth_masks = torch.zeros(0)  
    valid_masks = torch.zeros(0)
    source_img_list = []
    gt_lie_alg_list = []
    vo_lie_alg_list = []

    for i, im, in enumerate(source_imgs):
        source_img_list.append(im.to(device))
        gt_lie_alg_list.append(lie_alg[i][0].type(torch.FloatTensor).to(device))
        vo_lie_alg_list.append(lie_alg[i][1].type(torch.FloatTensor).to(device))

    intrinsics = intrinsics.type(torch.FloatTensor).to(device)[:,0,:,:]
    plane = plane_model(target_img)

    plane = plane[0]
    plane = plane**3
    plane_array = torch.cat((plane_array, plane.cpu().detach()))

    return plane_array