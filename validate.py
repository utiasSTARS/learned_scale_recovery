import torch
from utils.learning_helpers import *
import numpy as np
from liegroups import SE3
from models.stn import *
from pyslam.metrics import TrajectoryMetrics
import csv
from train_mono import solve_pose
from data.kitti_loader import process_sample, process_sample_batch

def test_depth_and_reconstruction(device, models,  dset, config, epoch=0, source_img_idx=0):

    depth_model, pose_model = models[0].train(False).eval(), models[1].train(False).eval()
    exp_mask_array = torch.zeros(0)
    img_array = torch.zeros(0)
    disp_array = torch.zeros(0)
    source_disp_array = torch.zeros(0)
    reconstructed_disp_array = torch.zeros(0)
    d_masks = torch.zeros(0)
    dset_length = dset.dataset.__len__()
    img_idx=np.arange(0,dset_length,int(dset_length/5 -1))

    for i in img_idx:
        data = dset.dataset.__getitem__(i)
        target_img, source_img_list, gt_lie_alg_list, vo_lie_alg_list, flow_imgs, intrinsics, \
            target_img_aug, source_img_aug_list, gt_lie_alg_aug_list, vo_lie_alg_aug_list, intrinsics_aug = process_sample(data,config)
               
        disparity = depth_model(target_img_aug, epoch=epoch)
        disparities = [disparity]
        for i, im in enumerate(source_img_aug_list):
            source_disparity = depth_model(im, epoch=epoch)
            disparities.append(source_disparity)       
               
        disp_array = torch.cat((disp_array, disparity[0].cpu().detach()))
        
        depths = [disp_to_depth(disp[0], config['min_depth'], config['max_depth'])[1].detach() for disp in disparities]

        poses, poses_inv = solve_pose(pose_model, target_img_aug, source_img_aug_list, flow_imgs)
        
        ### Visualize depth consistency 
        source_disp = depth_model(source_img_aug_list[source_img_idx])
        source_disp = source_disp[0]
        source_disp = source_disp.unsqueeze(1)
        source_disp_array = torch.cat((source_disp_array, source_disp[0].cpu().detach()))
        _, source_depth = disp_to_depth(source_disp[:,0].clone(), config['min_depth'], config['max_depth'])
        img_reconstructed, valid_mask, projected_depth, computed_depth = inverse_warp2(source_img_aug_list[source_img_idx], depths[0], source_depth, -poses[source_img_idx].clone(), intrinsics, 'zeros') #forwards
        diff_img = (source_img_aug_list[source_img_idx] - img_reconstructed).abs().clamp(0, 1)
        valid_mask = (diff_img.mean(dim=1, keepdim=True) < (target_img_aug - source_img_aug_list[source_img_idx]).abs().mean(dim=1, keepdim=True)).float() * valid_mask

 
        valid_mask = valid_mask * (img_reconstructed.mean(1,True) != 0).float()
        reconstructed_disp_array = torch.cat((reconstructed_disp_array, depth_to_disp(projected_depth, config['min_depth'], config['max_depth']).clamp(0,1).cpu().detach()))

        d_loss =  ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth)).clamp(0, 1)
        d_mask = (1-d_loss)
        d_masks = torch.cat((d_masks, d_mask.cpu().detach()))
        
        imgs = torch.stack((source_img_aug_list[source_img_idx],img_reconstructed,target_img_aug),dim=1)[0].cpu().detach()
        img_array = torch.cat((img_array, imgs))
        exp_mask_array = torch.cat((exp_mask_array, valid_mask.cpu().detach()))

    return img_array, disp_array.numpy().squeeze(), exp_mask_array, (source_disp_array.numpy().squeeze(), \
                reconstructed_disp_array.numpy().squeeze(), d_masks )

def compute_trajectory(pose_vec, gt_traj, method='odom'):
    est_traj = [gt_traj[0]]
    cum_dist = [0]
    for i in range(0,pose_vec.shape[0]):
        #classically estimated traj
        dT = SE3.exp(pose_vec[i])
        new_est = SE3.as_matrix((dT.dot(SE3.from_matrix(est_traj[i],normalize=True).inv())).inv())
        est_traj.append(new_est)
        cum_dist.append(cum_dist[i]+np.linalg.norm(dT.trans))
       
    gt_traj_se3 = [SE3.from_matrix(T,normalize=True) for T in gt_traj]
    est_traj_se3 = [SE3.from_matrix(T,normalize=True) for T in est_traj]
    
    tm_est = TrajectoryMetrics(gt_traj_se3, est_traj_se3, convention = 'Twv')
    est_mean_trans, est_mean_rot = tm_est.mean_err()
    est_mean_rot = ( est_mean_rot*180/np.pi ).round(3)
    est_mean_trans = est_mean_trans.round(3)
    
    seg_lengths = list(range(100,801,100))
    _, seg_errs_est = tm_est.segment_errors(seg_lengths, rot_unit='rad')
    
    print('trans. rel. err: {}, rot. rel. err: {}'.format(np.mean(tm_est.rel_errors()[0]), np.mean(tm_est.rel_errors()[1])))

    rot_seg_err = ( 100*np.mean(seg_errs_est[:,2])*180/np.pi).round(3)
    trans_seg_err = ( np.mean(seg_errs_est[:,1])*100).round(3)
    
    if np.isnan(trans_seg_err):
        max_dist = cum_dist[-1] - cum_dist[-1]%100 + 1 - 100
        print('max dist', max_dist)
        seg_lengths = list(range(100,int(max_dist),100))
        _, seg_errs_est = tm_est.segment_errors(seg_lengths, rot_unit='rad')

        rot_seg_err = ( 100*np.mean(seg_errs_est[:,2])*180/np.pi).round(3)
        trans_seg_err = ( np.mean(seg_errs_est[:,1])*100).round(3)        
    
    print("{} mean trans. error: {} | mean rot. error: {}".format(method, est_mean_trans, est_mean_rot))
    print("{} mean Segment Errors: {} (trans, %) | {} (rot, deg/100m)".format(method, trans_seg_err, rot_seg_err))
        
    errors = (est_mean_trans, est_mean_rot, trans_seg_err, rot_seg_err)

    return np.array(est_traj), np.array(gt_traj), errors, np.array(cum_dist)

def test_trajectory(config, device, models, dset, epoch):
    depth_model, pose_model = models[0].train(False).eval(), models[1].train(False).eval()

    #initialize the relevant outputs
    poses_stacked, gt_lie_alg_stacked = np.empty((0,6)), np.empty((0,6))
    
    for data in dset:
        target_img, source_img_list, gt_lie_alg_list, vo_lie_alg_list, flow_imgs, intrinsics, \
            target_img_aug, source_img_aug_list, gt_lie_alg_aug_list, vo_lie_alg_aug_list, intrinsics_aug = process_sample_batch(data, config)
        
        with torch.set_grad_enabled(False):
            poses, poses_inv = solve_pose(pose_model, target_img, source_img_list, flow_imgs)
            pose = poses[0].clone()
            
        pose[:,0:3] = 30*pose[:,0:3]

        poses_stacked = np.vstack((poses_stacked, pose.cpu().detach().numpy()))
        gt_lie_alg_stacked = np.vstack((gt_lie_alg_stacked, gt_lie_alg_list[0].cpu().detach().numpy()))
     
    gt_traj = dset.dataset.raw_gt_trials[0]
    est_traj, _, errors, _  = compute_trajectory(poses_stacked, gt_traj, method='est')

    return poses_stacked, gt_lie_alg_stacked, est_traj, gt_traj, errors
        
def get_plane_masks(device, models, dset, config, epoch=0, source_img_idx=0): #[100,250,400,500,600],epoch=0):

    depth_model, plane_model = models[0].train(False).eval(), models[1].train(False).eval()
    exp_mask_array = torch.zeros(0)
    img_array = torch.zeros(0)
    
    dset_length = dset.dataset.__len__()
    img_idx=np.arange(0,dset_length,int(dset_length/5 -1))

    for i in img_idx:
        data = dset.dataset.__getitem__(i)
        target_img, source_img_list, gt_lie_alg_list, vo_lie_alg_list, flow_imgs, intrinsics, \
            target_img_aug, source_img_aug_list, gt_lie_alg_aug_list, vo_lie_alg_aug_list, intrinsics_aug = process_sample(data, config)
        
        exp_mask = plane_model(target_img_aug)
        exp_mask = exp_mask[0]

        img_array = torch.cat((img_array, target_img_aug.cpu().detach()))
        exp_mask_array = torch.cat((exp_mask_array, exp_mask.cpu().detach()))

    return img_array, exp_mask_array

class ResultsLogger():
    def __init__(self, filename):
        self.filename = filename
        csv_header1 = ['', '', 'm-ATE', '', 'Mean Segment Errors', '']
        csv_header2 = ['Sequence (Length)', 'Name', 'Trans. (m)', 'Rot. (deg)', 'Trans. (%)', 'Rot. (deg/100m)']
        with open(filename, "w") as f:
            self.writer = csv.writer(f)
            self.writer.writerow(csv_header1)
            self.writer.writerow(csv_header2)
    
    def log(self, seq, name, t_ate, r_ate, t_mse, r_mse):
        stats_list = [seq, name, t_ate, r_ate, t_mse, r_mse]
        with open(self.filename, "a") as f:
            self.writer = csv.writer(f)
            self.writer.writerow(stats_list)
        