import numpy as np
import torch
import sys
sys.path.append('../')
import validate
from pyslam.metrics import TrajectoryMetrics
from run_inference import data_and_model_loader
from utils.learning_helpers import save_obj, load_obj
import os
from validate import compute_trajectory as tt
import glob
import visualizers
from liegroups import SE3
import matplotlib.pyplot as plt

path_to_ws = '/home/brandon-wagstaff/learned_scale_recovery/'
path_to_dset_downsized = '/media/m2-drive/datasets/KITTI-odometry-downsized/'
seq_list = ['00', '02', '05', '06', '07', '08', '09', '10'] 
method_list = ['scaled', 'unscaled']

dir_list = [path_to_ws+'results/202007111233-kitti-scaled-good', \
    path_to_ws+'results/202007100900-kitti-unscaled'
    ]



test_seq = '05'
val_seq = '00'
cam_height = 1.70 #1.52
plot_range =  slice(0,-1)
os.makedirs('figures', exist_ok=True)

for seq in seq_list:
    tm_dict = {}
    print('sequence: {}'.format(seq))
    for method, dir in zip(method_list, dir_list):
        tm_dict[method] = {}
        results_dir = dir + '/results/scale/'
        config = load_obj('{}/config'.format(dir))
        config['test_seq'] = [seq]
        config['data_dir'] = path_to_dset_downsized+config['img_resolution'] + '_res/' 
        dpc = config['dpc']
        mode = config['pose_output_type']
        if dpc:
            prefix = 'dpc'
        else:
            prefix=''
        test_dset_loaders, _, _ = data_and_model_loader(config, None, None, seq=seq)

        data = load_obj('{}/{}_plane_fit'.format(results_dir, config['test_seq'][0]))
        
        dist_to_plane = data['dist_to_plane']
        fwd_pose_vec1 = data['fwd_pose_vec1']
        fwd_pose_vec2 = data['fwd_pose_vec2']
        inv_pose_vec1 = data['inv_pose_vec1']
        inv_pose_vec2 = data['inv_pose_vec2']
        gt_pose_vec = data['gt_pose_vec']
        vo_pose_vec = data['vo_pose_vec']
        num_inliers = data['num_inliers']
        normals = data['normal']
        if config['dpc'] == False:
            prefix = ''
        if config['dpc'] == True:
            prefix = prefix = 'dpc-'

        d = [np.median(np.abs(i)) for i in dist_to_plane] 
        d  = np.array(d)
        average_d = np.average(d) 

        
        unscaled_pose_vec = fwd_pose_vec1
        # unscaled_pose_vec = (fwd_pose_vec1 - inv_pose_vec1)/2 
        unscaled_pose_vec[:,3:6] = gt_pose_vec[:,3:6]
        
        ## align with global scale
        gt_norm = np.linalg.norm(gt_pose_vec[:,0:3],axis=1)
        vo_norm = np.linalg.norm(vo_pose_vec[:,0:3],axis=1)
        vo_scale_factor = np.average(gt_norm/vo_norm)
        
        print('ground plane scale factor: {}'.format(cam_height/np.average(d)))
        unscaled_pose_vec = unscaled_pose_vec[:,0:6]


        scaled_pose_vec = np.array(unscaled_pose_vec)
        scaled_pose_vec[:,0:3] = scaled_pose_vec[:,0:3]*np.repeat(cam_height/d.reshape((-1,1)),3,axis=1)
        vo_pose_vec[:,0:3] = vo_pose_vec[:,0:3]*vo_scale_factor
        
        
        ## Compute Trajectories
        gt_traj = test_dset_loaders.dataset.raw_gt_trials[0]
        est, gt, errors, cum_dist = tt(unscaled_pose_vec,gt_traj,method='unscaled')
        scaled_est, gt, errors, cum_dist = tt(scaled_pose_vec,gt_traj, method='scaled')
        vo_est, _, _, _ = tt(vo_pose_vec, gt_traj, method='orbslam')


        gt_traj_se3 = [SE3.from_matrix(T,normalize=True) for T in gt_traj]
        est_se3 = [SE3.from_matrix(T, normalize=True) for T in est]
        scaled_se3 = [SE3.from_matrix(T, normalize=True) for T in scaled_est]
        vo_se3 = [SE3.from_matrix(T, normalize=True) for T in vo_est]
        
        
        est_tm = TrajectoryMetrics(gt_traj_se3, est_se3, convention = 'Twv')
        scaled_tm = TrajectoryMetrics(gt_traj_se3, scaled_se3, convention = 'Twv')
        vo_tm = TrajectoryMetrics(gt_traj_se3, vo_se3, convention = 'Twv')      
        
        tm_dict[method] = {'unscaled': est_tm,
                           'scaled': scaled_tm,
                           'orbslam': vo_tm,
               }
        
    # order_of_keys = ["Libviso2-m", "Ours (Corrected)"]
    # list_of_tuples = [(key, tm_dict[key]) for key in order_of_keys]
    # tm_dict = OrderedDict(list_of_tuples)
    
    
    ### CHOOSE DESIRED PLOTS HERE
    plotting_dict = {'Orbslam': tm_dict['scaled']['orbslam'],
                     'Ours': tm_dict['scaled']['unscaled'],
                     'DNet': tm_dict['unscaled']['scaled']
                     
                     }
    
    est_vis = visualizers.TrajectoryVisualizer(plotting_dict)
    plt.figure()
    fig, ax = est_vis.plot_topdown(which_plane='xy', outfile = 'figures/test-seq-{}.pdf'.format(seq), title=r'{}'.format(seq))
    
