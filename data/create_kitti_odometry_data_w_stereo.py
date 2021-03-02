'''
Use this for preprocessing KITTI odometry data for the VO experiments
'''

import pykitti
import numpy as np
import scipy.io as sio
import imageio
import os
import concurrent.futures
from PIL import Image
import argparse
from liegroups import SE3

parser = argparse.ArgumentParser(description='')
parser.add_argument("--source_dir", type=str, default='/media/datasets/KITTI-odometry/')
parser.add_argument("--target_dir", type=str, default='/media/datasets/KITTI-odometry-downsized-stereo')
parser.add_argument("--remove_static", action='store_true', default=False)
args = parser.parse_args()


resolutions = {'low_res': {'height':128, 'width': 448}, 'med_res': {'height':192, 'width': 640}, 'high_res': {'height':256,'width':832}}

for resolution in ['med_res']:
    target_dir = '{}/{}/'.format(args.target_dir,resolution)
    os.makedirs(target_dir, exist_ok=True)
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
    }
    
    sequences = ['00', '01', '02', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19' , '20', '21']

    args.height =  resolutions[resolution]['height']
    args.width =  resolutions[resolution]['width']

        
    mono_orbslam_dir = 'orbslam-estimates/orbslam_mono_traj_odometry/'
    print(args.height, args.width)
          
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
    for seq in sequences:    
        data = pykitti.odometry(args.source_dir, seq)
        cam_ts = np.array([data.timestamps[i].total_seconds() for i in range(0,len(data.timestamps))])
        gt_poses = np.array(data.poses)
        if gt_poses.shape==(0,):
            gt_poses = np.zeros((len(cam_ts),4,4))
        
            ###make the new directories
        for cam_num in ['02','03']:
            seq_info = {}
            print(cam_num)
            seq_dir = os.path.join(target_dir, seq_names[seq]+'_{}'.format(cam_num))
            print(seq_dir)
            os.makedirs(seq_dir, exist_ok=True)
            os.makedirs(os.path.join(seq_dir, 'image_2'), exist_ok=True)
            # os.makedirs(os.path.join(seq_dir, 'image_3'), exist_ok=True)        
            
            # print(seq_info['intrinsics_left'], seq_info['intrinsics_right'])
            i = 0
            if cam_num == '02':
                seq_info['intrinsics_left'] = np.array(data.calib.K_cam2).reshape((-1,3,3)).repeat(len(data.cam2_files),0)
                with concurrent.futures.ProcessPoolExecutor() as executor: 
                    for filename, output in zip(data.cam2_files, executor.map(load_image, data.cam2_files)):
                        img, zoomx, zoomy, orig_img_width, orig_img_height = output
                        new_filename = os.path.join(target_dir, filename.split(args.source_dir)[1]).replace('sequences/','').replace('/'+seq+'/','/'+seq_names[seq]+'_'+cam_num+'/').replace('.png','.jpg')
                        imageio.imwrite(new_filename, img)
                        seq_info['intrinsics_left'][i,0] *= zoomx
                        seq_info['intrinsics_left'][i,1] *= zoomy
                        data.cam2_files[i] = np.array(new_filename)
                        i+=1
                    i = 0
                seq_info['cam_02'] = np.array(data.cam2_files)
            if cam_num == '03':
                seq_info['intrinsics_left'] = np.array(data.calib.K_cam3).reshape((-1,3,3)).repeat(len(data.cam3_files),0)
                with concurrent.futures.ProcessPoolExecutor() as executor: 
                    for filename, output in zip(data.cam3_files, executor.map(load_image, data.cam3_files)):
                        img, zoomx, zoomy, orig_img_width, orig_img_height = output
                        new_filename = os.path.join(target_dir, filename.split(args.source_dir)[1]).replace('sequences/','').replace('image_3', 'image_2').replace('/'+seq+'/','/'+seq_names[seq]+'_'+cam_num+'/').replace('.png','.jpg')
                        imageio.imwrite(new_filename, img)
                        seq_info['intrinsics_left'][i,0] *= zoomx
                        seq_info['intrinsics_left'][i,1] *= zoomy
                        data.cam3_files[i] = np.array(new_filename)
                        i+=1
                    i = 0
                seq_info['cam_02'] = np.array(data.cam3_files)                
        
                ### store the ground truth pose
            seq_info['sparse_gt_pose'] = gt_poses
        
            seq_info['sparse_vo'] = seq_info['sparse_gt_pose'] #use gt as a placeholder for now
            seq_info['ts'] = cam_ts         
            
            print(seq_info['ts'].shape, seq_info['cam_02'].shape, seq_info['sparse_gt_pose'].shape)
            
                        ###filter out frames with low rotational or translational velocities
            if args.remove_static:
                print("Removing Static frames from {}".format(seq))            
                deleting = True
                
                while deleting:
                    idx_list = []
                    sparse_traj = np.copy(seq_info['sparse_gt_pose']) ##using gt for now
                    for i in range(0,sparse_traj.shape[0]-1,2):
                        T2 = SE3.from_matrix(sparse_traj[i+1,:,:], normalize=True).inv()
                        T1 = SE3.from_matrix(sparse_traj[i,:,:], normalize=True)
                        dT = T2.dot(T1)
                        pose_vec = dT.log()
                        trans_norm = np.linalg.norm(pose_vec[0:3])
                        rot_norm = np.linalg.norm(pose_vec[3:6])
                        if trans_norm < 0.15 and rot_norm < 0.003: #0.007
                            idx_list.append(i)

                    if len(idx_list) == 0:
                        deleting = False
                    
                    print('deleting {} frames'.format(len(idx_list)))
                    print('original length: {}'.format(seq_info['cam_02'].shape))
                    
                    seq_info['intrinsics_left'] = np.delete(seq_info['intrinsics_left'],idx_list,axis=0)
                    # seq_info['intrinsics_right'] = np.delete(seq_info['intrinsics_right'],idx_list,axis=0)
                    seq_info['cam_02'] = np.delete(seq_info['cam_02'],idx_list,axis=0)
                    # seq_info['cam_03'] = np.delete(seq_info['cam_03'],idx_list,axis=0)
                    seq_info['sparse_gt_pose'] = np.delete(seq_info['sparse_gt_pose'],idx_list,axis=0)
                    seq_info['sparse_vo'] = np.delete(seq_info['sparse_vo'],idx_list,axis=0)
                    seq_info['ts'] = np.delete(seq_info['ts'],idx_list,axis=0)
                    print('final length: {}'.format(seq_info['cam_02'].shape))
                        
            sio.savemat(seq_dir + '/mono_data_orbslam.mat', seq_info)
        
        
        
