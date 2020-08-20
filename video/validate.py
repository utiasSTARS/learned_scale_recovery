import time
import torch
import sys
sys.path.insert(0,'..')
from utils.learning_helpers import *
from train_mono import apply_dpc, solve_pose
from models.stn import *
from vis import UnNormalize_img_array

def test_depth_and_reconstruction(device, models, data, config):
    source_img_idx = 0
    dpc = config['dpc']
    mode = config['pose_output_type']
    depth_model, pose_model = models[0], models[1]
    
    target_img, source_imgs, lie_alg, intrinsics, flow_imgs = data
    # flow_imgs_fwd, flow_imgs_back = flow_imgs[0], flow_imgs[1]

    lie_alg = lie_alg['color']
    target_img, source_imgs, intrinsics = target_img['color_left'], source_imgs['color_left'], intrinsics['color_left']
    target_img = target_img.to(device)

    source_disp_array = torch.zeros(0)  
    depth_masks = torch.zeros(0)  
    valid_masks = torch.zeros(0)
    diff_imgs = torch.zeros(0)
    source_img_list = []
    gt_lie_alg_list = []
    vo_lie_alg_list = []

    for i, im, in enumerate(source_imgs):
        source_img_list.append(im.to(device))
        gt_lie_alg_list.append(lie_alg[i][0].type(torch.FloatTensor).to(device))
        vo_lie_alg_list.append(lie_alg[i][1].type(torch.FloatTensor).to(device))
        
        if config['flow_type'] == 'classical':
            flow_imgs_fwd, flow_imgs_back = flow_imgs
            flow_imgs_fwd_list, flow_imgs_back_list = [], []
            for i in range(0, len(flow_imgs_fwd)):
                flow_imgs_fwd_list.append(flow_imgs_fwd[i].to(device))
                flow_imgs_back_list.append(flow_imgs_back[i].to(device))
            flow_imgs = [flow_imgs_fwd_list, flow_imgs_back_list]
        else:
            flow_imgs = [[None for i in range(0,len(source_img_list))] for i in range(0,2)] #annoying but necessary

    intrinsics = intrinsics.type(torch.FloatTensor).to(device)[:,0,:,:]
    disparity = depth_model(target_img, epoch=50)
    disparities = [disparity]
    
    for i, im in enumerate(source_img_list):
        source_disparity = depth_model(im, epoch=50)
        disparities.append(source_disparity)

    poses, poses_inv = solve_pose(pose_model, target_img, source_img_list, vo_lie_alg_list, dpc, mode, 50, intrinsics, flow_imgs)
    

    imgs = torch.cat([source_img_list[0].unsqueeze(1), target_img.unsqueeze(1)],1).cpu().detach()
    return imgs, disparities[0][0][:,0].cpu().detach()

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