import pykitti
import numpy as np
import scipy.io as sio
import imageio
import os
import concurrent.futures
from PIL import Image
import argparse
from liegroups import SE3
from oxford.camera_model import CameraModel
import glob
import re
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic

parser = argparse.ArgumentParser(description='')
parser.add_argument("--source_dir", type=str, default='/media/HDD1/datasets/oxford-robotcar')
parser.add_argument("--target_dir", type=str, default='/media/m2-drive/datasets/oxford-robotcar-downsized')
parser.add_argument("--camera_type", type=str, default='stereo')
parser.add_argument("--remove_static", action='store_true', default=True)
args = parser.parse_args()

args.models_dir = '{}/camera_models'.format(args.source_dir)
args.source_dir = '{}/data'.format(args.source_dir)

sequences = ['2014-11-18-13-20-12', '2015-07-08-13-37-17', '2015-07-10-10-01-59', '2015-08-12-15-04-18']
crop = [200,-165, 0, 1280] #top, bottom, left, right
resolutions = {'low_res': {'height':128, 'width': 448}, 'med_res': {'height':192, 'width': 640}, 'high_res': {'height':256,'width':832}}

target_dir = args.target_dir
for resolution in ['med_res']:
    args.height =  resolutions[resolution]['height']
    args.width =  resolutions[resolution]['width']
    
    target_dir_res = '{}/{}/'.format(target_dir,resolution)
    os.makedirs(target_dir_res, exist_ok=True)
    sub_seq_info = {}
   
        
    vo_dir = 'oxford/stereo_vo_traj/'
  
    def load_image(img_file):
        img_height = args.height 
        img_width = args.width 
        
        img = Image.open(img_file)
        img = demosaic(img, 'gbrg')
        if 'left' in img_file:
            img = args.left_model.undistort(img)
        if 'right' in img_file:
            img = args.right_model.undistort(img)
        img = np.array(img).astype(np.uint8)
        img = img[crop[0]:crop[1],crop[2]:crop[3]]

        orig_img_height = img.shape[0]
        orig_img_width = img.shape[1]
        zoom_y = img_height/orig_img_height
        zoom_x = img_width/orig_img_width
        img = np.array(Image.fromarray(img).resize((img_width, img_height), resample = Image.ANTIALIAS))
        return img, zoom_x, zoom_y, orig_img_width, orig_img_height
    
        ###Iterate through all specified KITTI sequences and extract raw data, and trajectories
    for seq in sequences:    
        source_seq_dir = '{}/{}/{}'.format(args.source_dir, seq, args.camera_type)

        
        args.left_model = CameraModel(args.models_dir, source_seq_dir + '_left')
        args.right_model = CameraModel(args.models_dir, source_seq_dir + '_right')


        ###store filenames of camera data, and intrinsic matrix
        seq_info = {}
        seq_info['cam_02'] = []
        seq_info['cam_03'] = []
        for img_name in sorted(glob.glob('{}/left/**'.format(source_seq_dir))):
            seq_info['cam_02'].append(img_name)
        for img_name in sorted(glob.glob('{}/right/**'.format(source_seq_dir))):
            seq_info['cam_03'].append(img_name)            
        seq_info['cam_02'] = np.array(seq_info['cam_02'])
        seq_info['cam_03'] = np.array(seq_info['cam_03'])


        for cam, model, num_imgs in zip(['left', 'right'], [args.left_model, args.right_model], [len(seq_info['cam_02']), len(seq_info['cam_03'])]):
            K = np.zeros((3,3))
            K[0,0] = model.focal_length[0]
            K[1,1] = model.focal_length[1]
            K[0,2] = model.principal_point[0] - crop[2]
            K[1,2] = model.principal_point[1] - crop[0]
            K[2,2] = 1
            seq_info['intrinsics_{}'.format(cam)] = K.reshape((1,3,3)).repeat(num_imgs,0)


#             ###Import libviso2 estimate for correcting
        vo_data = sio.loadmat('{}/{}.mat'.format(vo_dir, seq))  #sparse VO
        est_traj = vo_data['poses_est'].transpose(2,0,1)
        keyframe_idx = vo_data['keyframe_idx'].reshape((-1))
        keyframe_ts = vo_data['keyframe_ts'].reshape((-1))
        keyframe_seq_idx = vo_data['keyframe_seq_idx'].reshape((-1))
        seq_changes = np.where((keyframe_seq_idx[1:] - keyframe_seq_idx[:-1]) == 1)[0] + 1
        seq_info['sparse_gt_pose'] = vo_data['poses_gt'].transpose(2,0,1) ### store the ground truth pose
        seq_info['sparse_vo'] = est_traj ### store the VO pose estimates to extract 

            ###Only keep keyframes        
        seq_info['intrinsics_left'] = seq_info['intrinsics_left'][keyframe_idx]
        seq_info['intrinsics_right'] = seq_info['intrinsics_right'][keyframe_idx]
        seq_info['cam_02'] = seq_info['cam_02'][keyframe_idx]
        seq_info['cam_03'] = seq_info['cam_03'][keyframe_idx]
        seq_info['sparse_gt_pose'] = seq_info['sparse_gt_pose']
        seq_info['sparse_vo'] = seq_info['sparse_vo']
        seq_info['ts'] = keyframe_ts 
        seq_info['keyframe_seq_idx'] = keyframe_seq_idx  

        for m in range(0,len(seq_changes)):
            sub_seq = '{}_{}'.format(seq, m)
            print(seq, sub_seq)
            target_seq_dir = '{}/{}'.format(target_dir_res, sub_seq)
            os.makedirs(target_seq_dir, exist_ok=True)
            os.makedirs(os.path.join(target_seq_dir, args.camera_type, 'left'), exist_ok=True)
            os.makedirs(os.path.join(target_seq_dir, args.camera_type, 'right'), exist_ok=True)
            
            sub_seq_info = {}
            for name in ['intrinsics_left', 'intrinsics_right', 'cam_02', 'cam_03', 'sparse_gt_pose', 'sparse_vo', 'ts']:
                sub_seq_info[name] = np.copy(np.split(seq_info[name],seq_changes)[m])
            

            ## Load images, preprocess (colour demosaicing, etc.) and place them in new location
            i = 0
            cam_02 = list(sub_seq_info['cam_02'])
            cam_03 = list(sub_seq_info['cam_03'])
            
            sub_seq_info['cam_02'] = []
            sub_seq_info['cam_03'] = []
            with concurrent.futures.ProcessPoolExecutor() as executor: 
                for filename, output in zip(cam_02, executor.map(load_image, cam_02)):
                    img, zoomx, zoomy, orig_img_width, orig_img_height = output
                    new_filename = np.copy(filename.replace(args.source_dir, args.target_dir).replace(seq,'{}/{}'.format(resolution,sub_seq)).replace('png','jpg'))
                    imageio.imwrite(str(new_filename), img)
                    
                    sub_seq_info['intrinsics_left'][i,0] *= zoomx
                    sub_seq_info['intrinsics_left'][i,1] *= zoomy
                    new_filename = np.array(new_filename)
                    sub_seq_info['cam_02'].append(new_filename)

                    i+=1
                i = 0
                for filename, output in zip(cam_03, executor.map(load_image, cam_03)):
                    img, zoomx, zoomy, orig_img_width, orig_img_height = output
                    new_filename = filename.replace(args.source_dir, args.target_dir).replace(seq,'{}/{}'.format(resolution,sub_seq)).replace('png','jpg')
                    imageio.imwrite(str(new_filename), img)
                    sub_seq_info['intrinsics_right'][i,0] *= zoomx
                    sub_seq_info['intrinsics_right'][i,1] *= zoomy
                    new_filename = np.array(new_filename)
                    sub_seq_info['cam_03'].append(new_filename)
                    i+=1

            sub_seq_info['cam_02'] = np.array(sub_seq_info['cam_02']).reshape((-1))
            sub_seq_info['cam_03'] = np.array(sub_seq_info['cam_03']).reshape((-1))

                        ###filter out frames with low rotational or translational velocities
            if args.remove_static:
                print("Removing Static frames from {}".format(sub_seq))            
                deleting = True
                
                while deleting:
                    idx_list = []
                    sparse_traj = np.copy(sub_seq_info['sparse_gt_pose']) ##using gt for now
                    for i in range(0,sparse_traj.shape[0]-1,2):
                        T2 = SE3.from_matrix(sparse_traj[i+1,:,:], normalize=True).inv()
                        T1 = SE3.from_matrix(sparse_traj[i,:,:], normalize=True)
                        dT = T2.dot(T1)
                        pose_vec = dT.log()
                        trans_norm = np.linalg.norm(pose_vec[0:3])
                        rot_norm = np.linalg.norm(pose_vec[3:6])
                        if trans_norm < 0.05 and rot_norm < 0.003: #0.007
                            idx_list.append(i)

                    if len(idx_list) == 0:
                        deleting = False
                    
                    print('deleting {} frames'.format(len(idx_list)))
                    print('original length: {}'.format(sub_seq_info['cam_02'].shape))
                    
                    for name in ['intrinsics_left', 'intrinsics_right', 'cam_02', 'cam_03', 'sparse_gt_pose', 'sparse_vo', 'ts']:
                        sub_seq_info[name] = np.delete(sub_seq_info[name],idx_list,axis=0)
                    print('final length: {}'.format(sub_seq_info['cam_02'].shape))

            sio.savemat(target_seq_dir + '/mono_data_stereo.mat'.format(sub_seq), sub_seq_info)

        
        

