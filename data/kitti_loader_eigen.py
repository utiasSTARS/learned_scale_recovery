'''
Loads images from the Eigen split. There are three modes: train, val, and test. 

Must run 'create_kitti_eigen_data.py' before using this loader to preprocess the data.
'''

import cv2
import pykitti
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import scipy.io as sio
from liegroups import SE3, SO3
import os
import glob
import pickle

class KittiLoaderPytorch(torch.utils.data.Dataset):
    """Loads the KITTI Odometry Benchmark Dataset"""
    def __init__(self, config, seq, mode='train', transform_img=None, augment=False, skip=None, stereo_imgs=False):
        """
        Args:
            config file 
            seq isn't used but is here to be consistent with odometry loader
            mode is specified as 'train', 'val', 'test'
        """
        seq_names= {'train': 'train', 'val': 'val', 'test': 'test'}

        self.config = config
        basedir = config['data_dir']
        self.seq_len = config['img_per_sample']
        self.transform_img = transform_img
        self.num_frames = config['num_frames']
        self.augment = augment
        self.skip = skip
        self.stereo_imgs = stereo_imgs
        # self.load_stereo = config['load_stereo']
        self.raw_gt_trials = [np.zeros(10)]

            ###Iterate through all specified KITTI sequences and extract raw data, and trajectories

        data_filename = '{}/eigen_info_{}'.format(basedir, mode)
            
        print(data_filename)
        with open(data_filename + '.pkl', 'rb') as f:
            self.data = pickle.load(f)
            

    def __len__(self):
        return int(self.data['gt_poses'].shape[0])

    def __getitem__(self, idx):
        img_names = self.data['filenames'][idx]
        imgs = []
        for img_name in img_names:
            imgs.append(self.load_image(img_name))        
            
        intrinsics = self.data['intrinsics'][idx].reshape(1,3,3).repeat(3,axis=0) 
        
            
        target_idx = 1   
        source_idx = [0,2]
        poses = self.data['gt_poses'][idx]
        lie_alg = []
        transformed_lie_alg = []
        for i in range(0,self.seq_len-1):
            lie_alg.append(list(self.compute_target(poses, target_idx, source_idx[i])))    
            transformed_lie_alg.append(list(self.compute_target(poses, target_idx, source_idx[i])))   
        
        # if self.load_stereo:
        #     lie_alg = lie_alg+lie_alg
        #     transformed_lie_alg = transformed_lie_alg + transformed_lie_alg #make 2 copies for transforms    
        
        if self.transform_img != None:
            orig, transformed = self.transform_img((imgs, intrinsics, lie_alg), (imgs, intrinsics, transformed_lie_alg)) 
        orig_imgs, orig_intrinsics, orig_lie_alg = orig
        transformed_imgs, transformed_intrinsics, transformed_lie_alg = transformed  
      
        flow_imgs_fwd, flow_imgs_fwd_list = [], []
        flow_imgs_back, flow_imgs_back_list = [], []
        

        if self.config['flow_type'] == 'classical': # and self.config['preprocess_flow'] == False: ## compute flow online
            for i in range(0,len(imgs)-1):  
                flow_img_t = np.array(imgs[target_idx].convert('L'))
                flow_img_s = np.array(imgs[source_idx[i]].convert('L'))
                flow_img_fwd = cv2.calcOpticalFlowFarneback(flow_img_t,flow_img_s, None, 0.5, 3, 15, 3, 5, 1.2, 0) #fwd is target to source
                flow_img_fwd = torch.from_numpy(np.transpose(flow_img_fwd, (2,0,1))).float()
                flow_img_back = cv2.calcOpticalFlowFarneback(flow_img_s,flow_img_t, None, 0.5, 3, 15, 3, 5, 1.2, 0) #back is src to target
                flow_img_back = torch.from_numpy(np.transpose(flow_img_back, (2,0,1))).float()
                flow_imgs_fwd.append(flow_img_back) 
                flow_imgs_back.append(flow_img_fwd)     
            
        target_im = {'color_left': orig_imgs[0:self.seq_len][target_idx], 'color_aug_left': transformed_imgs[0:self.seq_len][target_idx] }
        source_imgs = {'color_left': [orig_imgs[0:self.seq_len][i] for i in source_idx], 'color_aug_left': [transformed_imgs[0:self.seq_len][i] for i in source_idx] }
        intrinsics = {'color_left': orig_intrinsics[0:self.seq_len], 'color_aug_left': transformed_intrinsics[0:self.seq_len]}
        lie_alg = {'color': orig_lie_alg[0:self.seq_len], 'color_aug': transformed_lie_alg[0:self.seq_len]}        

        
        return target_im, source_imgs, lie_alg, intrinsics, (flow_imgs_fwd, flow_imgs_back)


    def load_image(self, img_file):
        img = Image.open(img_file)
        return img
    
    def compute_target(self, poses, target_idx, source_idx):
        T_c2_w = SE3.from_matrix(poses[target_idx], normalize=True)
        T_c1_w = SE3.from_matrix(poses[source_idx], normalize=True)
        

        T_c2_c1 = T_c2_w.dot(T_c1_w.inv()) 
        gt_lie_alg = T_c2_c1.log()
        vo_lie_alg = np.copy(gt_lie_alg)
        gt_correction = np.zeros(gt_lie_alg.shape)
        dt = 0

        return gt_lie_alg, vo_lie_alg, gt_correction, dt


