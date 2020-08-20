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
csv_header1 = ['Method', 'Sequence']
csv_header2 = ['', 'Train', 'Val', 'Test']
csv_header3 = [''] + seq_list + ['Mean']

with open('table3_scale_variance.csv', "w") as f:
    writer = csv.writer(f)
    writer.writerow(csv_header1)
    writer.writerow(csv_header2)
    writer.writerow(csv_header3)
    
    scale_factors = {}
    for method, dir in zip(method_list, dir_list):
        scale_factors[method] = {}
        scale_variance = []
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
            unscaled_pose_vec[:,3:6] = gt_pose_vec[:,3:6]
            
            print('seq {} {} scale factor: {}'.format(seq, method, cam_height/average_d))
            print('seq {} {} scale factor variance: {}'.format(seq, method, np.var(cam_height/d)))            
            
            scale_variance.append(np.var(cam_height/d))
            
            scale_factors[method][seq] = cam_height/d
            
        mean_var = np.mean(scale_variance)
        scale_variance.append(mean_var)
        scale_variance = ["%.4f" % e for e in scale_variance]
        writer.writerow([method] + scale_variance)
        
for seq in seq_list:
    plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.tick_params(labelsize=22)
    plt.grid()
    plt.plot(scale_factors['scaled'][seq][plot_range], linewidth=2, label='Scaled', rasterized=True)
    plt.plot(scale_factors['unscaled'][seq][plot_range], linewidth=2, label='Unscaled', rasterized=True)
    plt.legend(fontsize=15)
    plt.ylim([0.6,2.4])
    plt.ylabel('Scale Factor', fontsize=22)
    plt.xlabel('Timestep', fontsize=22)
    # plt.title('Seq. {} Scale Factor Comparison'.format(seq), fontsize=20)
    plt.savefig('figures/seq-{}-scale.pdf'.format(seq))