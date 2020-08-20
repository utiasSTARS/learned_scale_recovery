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

class KittiLoaderPytorch(torch.utils.data.Dataset):
    """Loads the KITTI Odometry Benchmark Dataset"""
    def __init__(self, config, seq, mode='train', transform_img=None, augment=False, skip=None, stereo_imgs=False):
        """
        Args:
            config file 
            desired sequences
            this works for KITTI and robotcar
        """
        seq_names= {'00': '2011_10_03_drive_0027_sync',
                '01': '2011_10_03_drive_0042_sync',
                '02': '2011_10_03_drive_0034_sync',
                '04': '2011_09_30_drive_0016_sync',
                '05': '2011_09_30_drive_0018_sync',
                '06': '2011_09_30_drive_0020_sync',
                '07': '2011_09_30_drive_0027_sync',
                '08': '2011_09_30_drive_0028_sync',
                '09': '2011_09_30_drive_0033_sync',
                '10': '2011_09_30_drive_0034_sync',
                '11': '11',
                '12': '12',
                '13': '13',
                '14': '14',
                '15': '15',
                '16': '16',
                '17': '17',
                '18': '18',
                '19': '19',
                '20': '20',
                '21': '21',
                '2014-11-18-13-20-12_0': '2014-11-18-13-20-12_0', '2014-11-18-13-20-12_1': '2014-11-18-13-20-12_1', '2014-11-18-13-20-12_2': '2014-11-18-13-20-12_2', '2014-11-18-13-20-12_3': '2014-11-18-13-20-12_3',
                '2015-07-08-13-37-17_0': '2015-07-08-13-37-17_0', '2015-07-08-13-37-17_1': '2015-07-08-13-37-17_1', '2015-07-08-13-37-17_2': '2015-07-08-13-37-17_2', '2015-07-08-13-37-17_3': '2015-07-08-13-37-17_3', '2015-07-08-13-37-17_4': '2015-07-08-13-37-17_4', '2015-07-08-13-37-17_5': '2015-07-08-13-37-17_5',
                '2015-07-10-10-01-59_0': '2015-07-10-10-01-59_0', '2015-07-10-10-01-59_1': '2015-07-10-10-01-59_1', '2015-07-10-10-01-59_2': '2015-07-10-10-01-59_2', '2015-07-10-10-01-59_3': '2015-07-10-10-01-59_3', '2015-07-10-10-01-59_4': '2015-07-10-10-01-59_4', 
                '2015-08-12-15-04-18_0': '2015-08-12-15-04-18_0', '2015-08-12-15-04-18_1': '2015-08-12-15-04-18_1', '2015-08-12-15-04-18_2': '2015-08-12-15-04-18_2', '2015-08-12-15-04-18_3': '2015-08-12-15-04-18_3', '2015-08-12-15-04-18_4': '2015-08-12-15-04-18_4', 
                }

        self.config = config
        basedir = config['data_dir']
        self.seq_len = config['img_per_sample']
        self.transform_img = transform_img
        self.num_frames = config['num_frames']
        self.augment = augment
        self.skip = skip
        self.stereo_imgs = stereo_imgs
        self.load_stereo = config['load_stereo']

            ###Iterate through all specified KITTI sequences and extract raw data, and trajectories
        self.left_cam_filenames = []
        self.right_cam_filenames = []
        self.raw_intrinsic_trials_left = []
        self.raw_intrinsic_trials_right = []
        self.raw_gt_trials = []
        self.raw_vo_traj = []
        self.raw_ts = []
        train_seq, val_seq, test_seq = seq
        if train_seq == ['all'] and mode == 'train':
            seq = []
            seq_name={}
            
            for d in glob.glob('{}/**'.format(basedir), recursive=False):
                name = d.replace(basedir, '').replace('/','')
                i=0
                for s in val_seq:
                    if name == seq_names[s]:
                        i=1
                        print("excluding {}".format(name))
                for s in test_seq:
                    if name == seq_names[s]:
                        i=1
                        print("excluding {}".format(name))                    
                if i == 0:
                    seq.append(name)
                    seq_name[name] = name

        if train_seq != ['all'] and mode == 'train':
            seq = train_seq
            seq_name = seq_names
            
        if mode == 'val':
            seq_name = seq_names
            seq = val_seq
        if mode == 'test':
            seq_name = seq_names
            seq = test_seq

        for s,i in zip(seq,range(0,len(seq))):
            data = sio.loadmat(os.path.join(basedir, seq_name[s],'{}_data_{}.mat'.format(config['estimator_type'], config['estimator'])))
            
            self.left_cam_filenames.append(np.copy(data['cam_02'].reshape((-1,1))))
            self.right_cam_filenames.append(np.copy(data['cam_03'].reshape((-1,1))))
            self.raw_intrinsic_trials_left.append(np.copy(data['intrinsics_left']))
            self.raw_intrinsic_trials_right.append(np.copy(data['intrinsics_right']))
            self.raw_gt_trials.append(np.copy(data['sparse_gt_pose']))
            self.raw_vo_traj.append(np.copy(data['sparse_vo']))
            self.raw_ts.append(np.copy(data['ts'].reshape((-1))))
            

            if self.config['correction_rate'] != 1:
                cr = self.config['correction_rate']
                self.left_cam_filenames[i] = np.copy(self.left_cam_filenames[i][::cr])
                self.right_cam_filenames[i] = np.copy(self.right_cam_filenames[i][::cr])
                self.raw_intrinsic_trials_left[i] = np.copy(self.raw_intrinsic_trials_left[i][::cr])
                self.raw_intrinsic_trials_right[i] = np.copy(self.raw_intrinsic_trials_right[i][::cr])
                self.raw_gt_trials[i] = np.copy(self.raw_gt_trials[i][::cr])
                self.raw_vo_traj[i] = np.copy(self.raw_vo_traj[i][::cr])
                self.raw_ts[i] = np.copy(self.raw_ts[i][::cr])
            else:
                cr=1
            
            if self.num_frames:
                self.left_cam_filenames[i] = np.copy(self.left_cam_filenames[i][:self.num_frames])
                self.right_cam_filenames[i] = np.copy(self.right_cam_filenames[i][:self.num_frames])
                self.raw_intrinsic_trials_left[i] = np.copy(self.raw_intrinsic_trials_left[i][:self.num_frames])
                self.raw_intrinsic_trials_right[i] = np.copy(self.raw_intrinsic_trials_right[i][:self.num_frames])
                self.raw_gt_trials[i] = np.copy(self.raw_gt_trials[i][:self.num_frames])
                self.raw_vo_traj[i] = np.copy(self.raw_vo_traj[i][:self.num_frames])  
                self.raw_ts[i] = np.copy(self.raw_ts[i][:self.num_frames])      

            
            #also add a trial that skips every other frame, for augmented data that simulates faster motion.
        if self.augment == True:
            for skip_idx in range(1,2):
                for s, i in zip(seq, range(0,len(seq))):
                    data = sio.loadmat(os.path.join(basedir, seq_name[s],'{}_data_{}.mat'.format(config['estimator_type'], config['estimator'])))
                    self.left_cam_filenames.append(np.copy(data['cam_02'].reshape((-1,1)))[::(cr+skip_idx)])
                    self.right_cam_filenames.append(np.copy(data['cam_03'].reshape((-1,1)))[::(cr+skip_idx)])
                    self.raw_intrinsic_trials_left.append(np.copy(data['intrinsics_left'])[::(cr+skip_idx)])
                    self.raw_intrinsic_trials_right.append(np.copy(data['intrinsics_right'])[::(cr+skip_idx)])
                    self.raw_gt_trials.append(np.copy(data['sparse_gt_pose'])[::(cr+skip_idx)])
                    self.raw_vo_traj.append(np.copy(data['sparse_vo'])[::(cr+skip_idx)])
                    self.raw_ts.append(np.copy(data['ts'])[::(cr+skip_idx)])
         

###         Merge data from all trials   
        self.gt_samples, self.left_img_samples, self.right_img_samples, self.intrinsic_samples_left, self.intrinsic_samples_right, \
            self.vo_samples, self.ts_samples = self.reshape_data()

        if self.skip:
            skip = self.config['skip']
            self.gt_samples, self.left_img_samples, self.intrinsic_samples_left, self.intrinsic_samples_right = self.gt_samples[::skip], self.left_img_samples[::skip], self.intrinsic_samples_left[::skip], self.intrinsic_samples_right[::skip]
            self.vo_samples = self.vo_samples[::skip]
            self.right_img_samples = self.right_img_samples[::skip]
            self.ts_samples = self.ts_samples[::skip]
        
        print('length',len(self.gt_samples))


    def __len__(self):
        return int(self.gt_samples.shape[0])

    def __getitem__(self, idx):
        imgs_left = []
        for i in range(0,self.seq_len):
            imgs_left.append(self.load_image(self.left_img_samples[idx,i]))        
            
        imgs = list(imgs_left)
        intrinsics = self.intrinsic_samples_left[idx] 
        
        if self.load_stereo:
            imgs_left = imgs
            intrinsics_left = intrinsics
            imgs_right = []
            
            for i in range(0,self.seq_len):
                imgs_right.append(self.load_image(self.right_img_samples[idx,i]))     
        
            imgs_right = list(imgs_right)
            intrinsics_right = self.intrinsic_samples_right[idx]
            imgs = imgs_left + imgs_right
            intrinsics = np.vstack((intrinsics_left, intrinsics_right))
            
        target_idx = int(len(imgs_left)/2)   
        source_idx = list(range(0,self.seq_len))
        source_idx.pop(target_idx)             
        
        lie_alg = []
        transformed_lie_alg = []
        for i in range(0,self.seq_len-1):
            lie_alg.append(list(self.compute_target(idx,target_idx, source_idx[i])))    
            transformed_lie_alg.append(list(self.compute_target(idx,target_idx, source_idx[i])))   
        
        if self.load_stereo:
            lie_alg = lie_alg+lie_alg
            transformed_lie_alg = transformed_lie_alg + transformed_lie_alg #make 2 copies for transforms    
        
        if self.transform_img != None:
            orig, transformed = self.transform_img((imgs, intrinsics, lie_alg), (imgs, intrinsics, transformed_lie_alg)) 
        orig_imgs, orig_intrinsics, orig_lie_alg = orig
        transformed_imgs, transformed_intrinsics, transformed_lie_alg = transformed  
      
        flow_imgs_fwd, flow_imgs_fwd_list = [], []
        flow_imgs_back, flow_imgs_back_list = [], []
        

        if self.config['flow_type'] == 'classical': # and self.config['preprocess_flow'] == False: ## compute flow online
            for i in range(0,len(imgs_left)-1):  
                flow_img_t = np.array(imgs_left[target_idx].convert('L'))
                flow_img_s = np.array(imgs_left[source_idx[i]].convert('L'))
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
        if self.load_stereo:
            target_im['color_right'] = orig_imgs[self.seq_len:][target_idx]
            target_im['color_aug_right'] = transformed_imgs[self.seq_len:][target_idx]
            source_imgs['color_right'] = [orig_imgs[self.seq_len:][i] for i in source_idx]
            source_imgs['color_aug_right'] = [transformed_imgs[self.seq_len:][i] for i in source_idx]
            intrinsics['color_right'] = orig_intrinsics[self.seq_len:]
            intrinsics['color_aug_right'] = transformed_intrinsics[self.seq_len:]
        
        return target_im, source_imgs, lie_alg, intrinsics, (flow_imgs_fwd, flow_imgs_back)


    def load_image(self, img_file):
        img = Image.open(img_file[0])
        return img
    
    def compute_target(self, idx, target_idx, source_idx):
        ## Compute ts
        # dt = np.abs(self.ts_samples[idx,target_idx] - self.ts_samples[idx, source_idx])
        dt = self.ts_samples[idx,target_idx] - self.ts_samples[idx, source_idx]
        
        #compute Tau_gt
        T2 = SE3.from_matrix(self.gt_samples[idx,target_idx,:,:],normalize=True).inv()
        T1 = SE3.from_matrix(self.gt_samples[idx,source_idx,:,:],normalize=True)
        dT_gt = T2.dot(T1) #pose change from source to a target (for reconstructing source from target)
        gt_lie_alg = dT_gt.log()
         
        #compute Tau_vo
        T2 = SE3.from_matrix(self.vo_samples[idx,target_idx,:,:],normalize=True).inv()
        T1 = SE3.from_matrix(self.vo_samples[idx,source_idx,:,:],normalize=True)
        dT_vo = T2.dot(T1)
        vo_lie_alg = dT_vo.log()
        
        if self.config['estimator_type'] == 'mono':
            if np.linalg.norm(vo_lie_alg[0:3]) >= 1e-8:
                scale = np.linalg.norm(gt_lie_alg[0:3])/np.linalg.norm(vo_lie_alg[0:3])
                scale = 1
                vo_lie_alg[0:3] = scale*vo_lie_alg[0:3]
                dT_vo = SE3.exp(vo_lie_alg)

        gt_correction = (dT_gt.dot(dT_vo.inv())).log()
        return gt_lie_alg, vo_lie_alg, gt_correction, dt

    def reshape_data(self):
        self.samples_per_file=None
        gt_samples = self.raw_gt_trials[0][0:self.seq_len,:,:].reshape((1,self.seq_len,4,4))
        vo_samples = self.raw_vo_traj[0][0:self.seq_len,:,:].reshape((1,self.seq_len,4,4))
        intrinsic_samples_left = self.raw_intrinsic_trials_left[0][0:self.seq_len,:,:].reshape((1,self.seq_len,3,3))
        intrinsic_samples_right = self.raw_intrinsic_trials_right[0][0:self.seq_len,:,:].reshape((1,self.seq_len,3,3))
        left_img_samples = self.left_cam_filenames[0][0:self.seq_len].reshape((1,self.seq_len,1))
        right_img_samples = self.right_cam_filenames[0][0:self.seq_len].reshape((1,self.seq_len,1))
        ts_samples = self.raw_ts[0][0:self.seq_len].reshape((1,self.seq_len,1))
        pose_per_sample = []

        for gt in self.raw_gt_trials:
            gt = gt.reshape((-1,gt.shape[1]*gt.shape[2]))
            if self.samples_per_file == None:
                num_samples = int(gt.shape[0]/(self.seq_len))
            new_gt = self.split_data(gt, num_samples, self.seq_len)
            new_gt = new_gt.reshape((-1,self.seq_len,4,4))
            gt_samples = np.vstack((gt_samples, new_gt))
            pose_per_sample.append(new_gt.shape[0])
            
        for vo in self.raw_vo_traj:
            vo = vo.reshape((-1,vo.shape[1]*vo.shape[2]))
            if self.samples_per_file == None:
                num_samples = int(vo.shape[0]/(self.seq_len))
            new_vo = self.split_data(vo, num_samples, self.seq_len)
            new_vo = new_vo.reshape((-1,self.seq_len,4,4))
            vo_samples = np.vstack((vo_samples, new_vo))
            
        for intrins in self.raw_intrinsic_trials_left:
            intrins = intrins.reshape((-1,intrins.shape[1]*intrins.shape[2]))
            if self.samples_per_file == None:
                num_samples = int(intrins.shape[0]/(self.seq_len))
            new_intrins = self.split_data(intrins, num_samples, self.seq_len)
            new_intrins = new_intrins.reshape((-1,self.seq_len,3,3))
            intrinsic_samples_left = np.vstack((intrinsic_samples_left, new_intrins))
            
        for intrins_r in self.raw_intrinsic_trials_right:
            intrins_r = intrins_r.reshape((-1,intrins_r.shape[1]*intrins_r.shape[2]))
            if self.samples_per_file == None:
                num_samples = int(intrins_r.shape[0]/(self.seq_len))
            new_intrins = self.split_data(intrins_r, num_samples, self.seq_len)
            new_intrins = new_intrins.reshape((-1,self.seq_len,3,3))
            intrinsic_samples_right = np.vstack((intrinsic_samples_right, new_intrins))   
                 
        count = 0
        for im in self.left_cam_filenames:
            im = im.reshape((-1,1))
            if self.samples_per_file == None:
                num_samples = int(im.shape[0]/(self.seq_len))
            new_imgs = self.split_data(im, num_samples, self.seq_len)
            left_img_samples = np.vstack((left_img_samples, new_imgs[0:pose_per_sample[count]])) #get rid of extra imgs due to rounding of gt
            count +=1

        count = 0
        for im in self.right_cam_filenames:
            im = im.reshape((-1,1))
            if self.samples_per_file == None:
                num_samples = int(im.shape[0]/(self.seq_len))
            new_imgs = self.split_data(im, num_samples, self.seq_len)
            right_img_samples = np.vstack((right_img_samples, new_imgs[0:pose_per_sample[count]])) #get rid of extra imgs due to rounding of gt
            count +=1

        count = 0
        for t in self.raw_ts:
            t = t.reshape((-1,1))
            if self.samples_per_file == None:
                num_samples = int(t.shape[0]/(self.seq_len))
            new_ts = self.split_data(t, num_samples, self.config['img_per_sample'])
            ts_samples = np.vstack((ts_samples, new_ts[0:pose_per_sample[count]])) #get rid of extra imgs due to rounding of gt
            count +=1
           
        gt_samples = gt_samples[1:]
        intrinsic_samples_left = intrinsic_samples_left[1:]
        intrinsic_samples_right = intrinsic_samples_right[1:]
        left_img_samples = left_img_samples[1:]
        right_img_samples = right_img_samples[1:]
        vo_samples = vo_samples[1:]
        ts_samples = ts_samples[1:]

        return gt_samples, left_img_samples, right_img_samples, intrinsic_samples_left, intrinsic_samples_right, vo_samples, \
            ts_samples

        
    def split_data(self, data, num_samples,sample_size):
        Type = data.dtype
        samplesize=int(sample_size)
        output = np.zeros((1,samplesize,data.shape[1])).astype(Type)
        i=0
        while i <= (data.shape[0]-sample_size):
            output = np.vstack((output, data[i:i+samplesize].reshape(1,samplesize,data.shape[1])))
            # i+=(samplesize-1) 
            i+=1           
        return output[1:] 
