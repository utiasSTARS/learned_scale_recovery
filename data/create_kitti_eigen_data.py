import pykitti
import numpy as np
import scipy.io as sio
import imageio
import os
import concurrent.futures
from PIL import Image
import argparse
from liegroups import SE3
import pickle

parser = argparse.ArgumentParser(description='')
parser.add_argument("--source_dir", type=str, default='/media/HDD1/datasets/KITTI/raw/') #path to full raw dataset
parser.add_argument("--target_dir", type=str, default='/media/datasets/KITTI-eigen-split')
args = parser.parse_args()


resolutions = {'low_res': {'height':128, 'width': 448}, 'med_res': {'height':192, 'width': 640}, 'high_res': {'height':256,'width':832}}

for resolution in ['med_res']:
    target_dir = '{}/{}/'.format(args.target_dir,resolution)
    os.makedirs(target_dir, exist_ok=True)
    seq_info = {}

    args.height =  resolutions[resolution]['height']
    args.width =  resolutions[resolution]['width']

    def load_image(img_file):
        img_height = args.height 
        img_width = args.width 
        img = np.array(Image.open(img_file))
        orig_img_height = img.shape[0]
        orig_img_width = img.shape[1]
        zoom_y = img_height/orig_img_height
        zoom_x = img_width/orig_img_width
    #    img = np.array(Image.fromarray(img).crop([425, 65, 801, 305]))
        img = np.array(Image.fromarray(img).resize((img_width, img_height), resample = Image.ANTIALIAS))
        return img, zoom_x, zoom_y, orig_img_width, orig_img_height
    
        ###Iterate through all specified KITTI sequences and extract raw data, and trajectories
    for seq in ['train', 'val', 'eigen_benchmark', 'eigen']: 
        seq_info = {}
        filenames = []
        gt_poses = np.zeros((0,3,4,4)) #N triplets of SE3 poses (T_c_w)
        intrinsics = np.zeros((0,3,3))
        seq_target_dir = '{}/{}'.format(target_dir, seq)
        os.makedirs(seq_target_dir, exist_ok=True)
        img_list = 'splits/eigen_{}_split.txt'.format(seq)
        mult=1
        if seq == 'eigen_benchmark':
            img_list = 'splits/eigen_benchmark/test_files.txt'
            mult=3
        if seq == 'eigen':
            img_list = 'splits/eigen/test_files.txt'
            mult=3
        
        img_data = open(img_list, 'r')
        img_data = img_data.readlines()
        

        for line in img_data: 
            img_data = line.strip().split(' ')
            img_folder = img_data[0]
            img_num = img_data[1]
            cam_num = img_data[2]
            
            date = img_folder.split('/')[0]
            drive = img_folder.split('drive_')[1].replace('_sync','')
            
            print(date, drive, img_num)
            ### get 3 frames for training/val, and only need a target frame for testing
            if seq != 'eigen' and seq != 'eigen_benchmark':
                data = pykitti.raw(args.source_dir, date, drive, frames=range(int(img_num)-1,int(img_num)+2))
            else:
                data = pykitti.raw(args.source_dir, date, drive, frames=range(int(img_num),int(img_num)+1))
            # print(data.cam2_files)
            
            filenames_i = []
            if cam_num == 'l':
                img_files = mult*data.cam2_files
                intrinsics_i = data.calib.K_cam2
                num = '02'
                T_cam_imu = SE3.from_matrix(data.calib.T_cam2_imu, normalize=True)
            if cam_num == 'r':
                img_files = mult*data.cam3_files
                intrinsics_i = data.calib.K_cam3
                num = '03'
                T_cam_imu = SE3.from_matrix(data.calib.T_cam3_imu, normalize=True)
            
            for img_file in img_files:
                img, zoomx, zoomy, orig_img_width, orig_img_height = load_image(img_file)
                new_filename = img_file.split(args.source_dir)[1].replace('/image_03/data','').replace('.png','.jpg')
                new_filename = os.path.join(seq_target_dir, new_filename)
                new_filename = new_filename.replace('image_{}/data/'.format(num),'').replace('/'+date+'/','/').replace('sync/','')
                imageio.imwrite(new_filename, img)
                filenames_i.append(new_filename)

            intrinsics_i[0]*=zoomx
            intrinsics_i[1]*=zoomy
            T_c0_w = T_cam_imu.dot(SE3.from_matrix(data.oxts[0][-1],normalize=True).inv()).as_matrix().reshape((1,4,4))
            
            if seq != 'eigen' and seq != 'eigen_benchmark':
                T_c1_w = T_cam_imu.dot(SE3.from_matrix(data.oxts[1][-1],normalize=True).inv()).as_matrix().reshape((1,4,4))
                T_c2_w = T_cam_imu.dot(SE3.from_matrix(data.oxts[2][-1], normalize=True).inv()).as_matrix().reshape((1,4,4))
                gt_pose_i = np.vstack((T_c0_w, T_c1_w, T_c2_w)).reshape((1,3,4,4))
            else:
                gt_pose_i = np.vstack((T_c0_w, T_c0_w, T_c0_w)).reshape((1,3,4,4))

            gt_poses = np.vstack((gt_poses, gt_pose_i))
            intrinsics = np.vstack((intrinsics, intrinsics_i.reshape((1,3,3))))
            filenames.append(filenames_i)


        seq_info['intrinsics'] = intrinsics
        seq_info['gt_poses'] = gt_poses
        seq_info['filenames'] = filenames
        
        with open('{}/eigen_info_{}'.format(target_dir, seq) + '.pkl', 'wb') as f:
            pickle.dump(seq_info, f, pickle.HIGHEST_PROTOCOL)
