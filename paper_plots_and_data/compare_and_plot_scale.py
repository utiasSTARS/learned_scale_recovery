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
import matplotlib.pyplot as plt
import matplotlib
# Removes the XWindows backend (useful for producing plots via tmux without -X)
matplotlib.use('Agg', warn=False)

path_to_ws = '/home/brandonwagstaff/learned_scale_recovery/'
path_to_dset_downsized = '/media/datasets/KITTI-odometry-downsized-stereo/'
seq_list = ['00_02', '02_02', '06_02', '07_02', '08_02', '05_02', '09_02', '10_02'] 
method_list = ['scaled', 'unscaled']

dir_list = [path_to_ws+'results/final_models/vo-kitti-scaled-202102182020', \
    path_to_ws+'results/final_models/vo-kitti-unscaled-202102201302'
    ]


csv_header1 = ['Method', 'Sequence']
csv_header2 = ['', 'Train', '', '', '','', 'Val', 'Test']
csv_header3 = [''] + seq_list + ['Mean']

with open('scale_variance_full.csv', "w") as f:
    writer = csv.writer(f)
    writer.writerow(csv_header1)
    writer.writerow(csv_header2)
    writer.writerow(csv_header3)
    
    scale_factors = {}
    for method, dir in zip(method_list, dir_list):
        scale_factors[method] = {}
        scale_factor_std_dev_list = []
        for seq in seq_list:
            results_dir = dir + '/results/scale/'
            config = load_obj('{}/config'.format(dir))
            config['test_seq'] = [seq]
            config['data_dir'] = path_to_dset_downsized+config['img_resolution'] + '_res/' #if grabbed from obelisk
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
            inv_pose_vec1 = data['inv_pose_vec1']
            gt_pose_vec = data['gt_pose_vec']

            if config['dpc'] == False:
                prefix = ''
            if config['dpc'] == True:
                prefix = prefix = 'dpc-'

            scale_factor = data['learned_scale_factor']
            scale_factor_mean = np.average(scale_factor)
            scale_factor_std = np.std(scale_factor)
            scale_factor_std_dev_list.append(scale_factor_std)
            scale_factors[method][seq] = scale_factor
            print('seq {} {} scale factor: {}'.format(seq, method, scale_factor_mean))
            print('seq {} {} scale factor std. dev.: {}'.format(seq, method, scale_factor_std))


        mean_std = np.mean(scale_factor_std_dev_list)
        scale_factor_std_dev_list.append(mean_std)
        scale_factor_std_dev_list = ["%.4f" % e for e in scale_factor_std_dev_list]
        writer.writerow([method] + scale_factor_std_dev_list)
        
for seq in seq_list:
    plt.figure()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.tick_params(labelsize=22)
    plt.grid()
    plt.plot(scale_factors['scaled'][seq], linewidth=2, label='Scaled', rasterized=True)
    plt.plot(scale_factors['unscaled'][seq], linewidth=2, label='Unscaled', rasterized=True)
    plt.legend(fontsize=15)
    plt.ylim([0.6,2.4])
    plt.ylabel('Scale Factor', fontsize=22)
    plt.xlabel('Timestep', fontsize=22)
    plt.savefig('figures/seq-{}-scale.pdf'.format(seq))