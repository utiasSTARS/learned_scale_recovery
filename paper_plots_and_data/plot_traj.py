import numpy as np
import torch
import sys
sys.path.append('../')
import validate
from pyslam.metrics import TrajectoryMetrics
from utils.learning_helpers import save_obj, load_obj, data_and_model_loader
import os
from validate import compute_trajectory as tt
import glob
import visualizers
from liegroups import SE3
import matplotlib.pyplot as plt

if __name__=='__main__':
    path_to_ws = '/home/brandonwagstaff/learned_scale_recovery/'
    path_to_dset_downsized = '/media/datasets/KITTI-odometry-downsized-stereo/'
    seq_list = ['05_02','09_02', '10_02']
    method_list = ['scaled', 'unscaled']
    use_gt_rot = False
    dir_list = [path_to_ws+'results/final_models/vo-kitti-scaled-202102182020', \
        path_to_ws+'results/final_models/vo-kitti-unscaled-202102201302'
        ]

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
            fwd_pose_vec1 = data['fwd_pose_vec1']
            inv_pose_vec1 = data['inv_pose_vec1']
            gt_pose_vec = data['gt_pose_vec']

            if config['dpc'] == False:
                prefix = ''
            if config['dpc'] == True:
                prefix = prefix = 'dpc-'
            
            unscaled_pose_vec = fwd_pose_vec1

            if use_gt_rot == True:
                unscaled_pose_vec[:,3:6] = gt_pose_vec[:,3:6]
            

            scaled_pose_vec = np.array(unscaled_pose_vec)
            scaled_pose_vec[:,0:3] = scaled_pose_vec[:,0:3]*np.repeat(data['dnet_scale_factor'],3,axis=1)

            ## Compute Trajectories
            gt_traj = test_dset_loaders.dataset.raw_gt_trials[0]
            est, gt, errors, cum_dist = tt(unscaled_pose_vec,gt_traj,method='unscaled')
            scaled_est, gt, errors, cum_dist = tt(scaled_pose_vec,gt_traj, method='scaled')


            gt_traj_se3 = [SE3.from_matrix(T,normalize=True) for T in gt_traj]
            est_se3 = [SE3.from_matrix(T, normalize=True) for T in est]
            scaled_se3 = [SE3.from_matrix(T, normalize=True) for T in scaled_est]

            est_tm = TrajectoryMetrics(gt_traj_se3, est_se3, convention = 'Twv')
            scaled_tm = TrajectoryMetrics(gt_traj_se3, scaled_se3, convention = 'Twv')

            tm_dict[method] = {'unscaled': est_tm,
                            'scaled': scaled_tm,
                }
            
        plotting_dict = {
                        'Ours': tm_dict['scaled']['unscaled'],
                        'DNet': tm_dict['unscaled']['scaled']

                        }
        
        est_vis = visualizers.TrajectoryVisualizer(plotting_dict)
        plt.figure()
        if use_gt_rot==True:
            fig, ax = est_vis.plot_topdown(which_plane='xz', outfile = 'figures/test-seq-{}_gt_rot.pdf'.format(seq), title=r'{}'.format(seq))
        else:
            fig, ax = est_vis.plot_topdown(which_plane='xz', outfile = 'figures/test-seq-{}.pdf'.format(seq), title=r'{}'.format(seq))
