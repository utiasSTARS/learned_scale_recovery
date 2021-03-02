import numpy as np
import torch
import sys
sys.path.append('../')
import validate
from utils.learning_helpers import save_obj, load_obj, data_and_model_loader
import os
from validate import compute_trajectory as tt
import glob
import csv

path_to_ws = '/home/brandonwagstaff/learned_scale_recovery/'
path_to_dset_downsized = '/media/datasets/KITTI-odometry-downsized-stereo/'
# seq_list = ['00_02', '02_02', '06_02', '07_02', '08_02', '05_02', '09_02', '10_02']
seq_list = ['09_02', '10_02']
# method_list = ['Orbslam2', 'Unscaled', 'Scaled (online)', 'Scaled (ours)']
method_list = ['Unscaled', 'Scaled online (DNet)', 'Scaled (ours)']
use_gt_rot = True

dir_list =    [path_to_ws+'results/final_models/vo-kitti-unscaled-202102201302', \
    path_to_ws+'results/final_models/vo-kitti-unscaled-202102201302', \
    path_to_ws+'results/final_models/vo-kitti-scaled-202102182020',
     ]


csv_header1 = ['Method', 'Sequence']
csv_header2 = ['', 'Train', 'Val', 'Test']

seq_header = [' ']
stats_header = [' ']
stats = ['mse-trans (%)', 'mse-rot (deg/100m)']
for s in seq_list:
    seq_header.append(s)
    seq_header.append(' ')
    stats_header.append(stats[0])
    stats_header.append(stats[1])

if use_gt_rot==True:
    table_filename = 'table4_gt_rot.csv'
else:
    table_filename = 'table4.csv'

with open(table_filename, "w") as f:
    writer = csv.writer(f)
    writer.writerow(csv_header1)
    # writer.writerow(csv_header2)
    writer.writerow(seq_header)
    writer.writerow(stats_header)

    for method, dir in zip(method_list, dir_list):
        seq_results = []
        for seq in seq_list:
            print('sequence: {}'.format(seq))
            results_dir = dir + '/results/scale/'
            config = load_obj('{}/config'.format(dir))
            config['test_seq'] = [seq]
            config['data_dir'] = path_to_dset_downsized+config['img_resolution'] + '_res/'
            config['data_format'] = 'odometry'
            dpc = config['dpc']
            mode = config['pose_output_type']

            test_dset_loaders, _, _ = data_and_model_loader(config, None, None, seq=seq)

            data = load_obj('{}/{}_plane_fit'.format(results_dir, config['test_seq'][0]))
            
            dist_to_plane = data['dist_to_plane']
            fwd_pose_vec1 = data['fwd_pose_vec1']
            inv_pose_vec1 = data['inv_pose_vec1']
            gt_pose_vec = data['gt_pose_vec']

            unscaled_pose_vec = fwd_pose_vec1
            if use_gt_rot:
                unscaled_pose_vec[:,3:6] = gt_pose_vec[:,3:6]

            scaled_pose_vec = np.array(unscaled_pose_vec)
            scaled_pose_vec[:,0:3] = scaled_pose_vec[:,0:3]*np.repeat(data['dnet_scale_factor'],3,axis=1)

            ## Compute Trajectories
            gt_traj = test_dset_loaders.dataset.raw_gt_trials[0]

            if method == method_list[0] or method == method_list[2]:
                est, gt, errors, cum_dist = tt(unscaled_pose_vec,gt_traj,method=method)

            if method == method_list[1]:
                scaled_est, gt, errors, cum_dist = tt(scaled_pose_vec,gt_traj, method=method)

            
            seq_results.append(errors[2])
            seq_results.append(errors[3])
        errors = np.array(seq_results)
        # mean = np.mean(errors)
        # seq_results.append(mean)
        seq_results = ["%.2f" % e for e in seq_results]
        writer.writerow([method] + seq_results)
