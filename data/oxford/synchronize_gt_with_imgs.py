import numpy as np
from liegroups.numpy import SO3, SE3
from collections import namedtuple
from PIL import Image
import scipy.io as sio
import matplotlib.pyplot as plt
from pyslam.metrics import TrajectoryMetrics
import glob
import numpy as np
import pandas as pd
from interpolate_poses import interpolate_ins_poses, interpolate_vo_poses
from transform import se3_to_components

'''
This script processes ground truth poses for the oxford robotcar dataset

RTK ground truth must be synchronized with the image timestamps

'''
data_dir = '/media/HDD1/datasets/oxford-robotcar' ## enter path to robotcar folders here
seq_names = ['2014-11-18-13-20-12', '2015-07-08-13-37-17', '2015-07-10-10-01-59', '2015-08-12-15-04-18']
output_dir = 'stereo_vo_traj'

for seq in seq_names:
    seq_dir = '{}/data/{}'.format(data_dir, seq)
    data = {}
    stereo_img_ts_file = '{}/data/{}/stereo.timestamps'.format(data_dir, seq)
    stereo_img_ts_file = open(stereo_img_ts_file)
    stereo_img_ts = []
    img_seq_num = [] #split the data into the predefined segments
    for line in stereo_img_ts_file.readlines():
        ts, seq_num = line.split(' ')
        img_seq_num.append(int(seq_num))
        stereo_img_ts.append(int(line.split(' ')[0]))
    
    stereo_img_ts = np.array(stereo_img_ts)
    img_seq_num = np.array(img_seq_num)
    # stereo_img_ts = np.array(stereo_img_ts)/(10.**6)
    gt_dir = '{}/rtk/{}'.format(data_dir,seq)
    gt_data = pd.read_csv('{}/rtk.csv'.format(gt_dir), mangle_dupe_cols=True)
    gt_x = gt_data['easting'] - gt_data['easting'][0]
    gt_y = gt_data['northing'] - gt_data['northing'][0]
    gt_z = gt_data['down'] - gt_data['down'][0]
    gt_roll = gt_data['roll']
    gt_pitch = gt_data['pitch']
    gt_yaw = gt_data['yaw']
    gt_ts = np.array(gt_data['timestamp'])#/(10.**6)
    
    vo_data = pd.read_csv('{}/{}'.format(seq_dir, 'vo/vo.csv'), mangle_dupe_cols=True)
    vo_ts = vo_data['source_timestamp']

    rtk_filename = gt_dir+'/rtk.csv'
    
    first_idx = np.where(stereo_img_ts > gt_ts[0])[0][0] + 1
    last_idx = np.where(stereo_img_ts > gt_ts[-1])[0]
    if last_idx != []:
        last_idx = last_idx[0] - 1
    else:
        last_idx = len(gt_ts)-1
    

    stereo_img_idx = list(range(first_idx, last_idx))
    stereo_img_ts = list(stereo_img_ts[stereo_img_idx]) 
    gt_poses = interpolate_ins_poses(rtk_filename, stereo_img_ts, stereo_img_ts[0], use_rtk=True)
    gt_poses = [se3_to_components(p) for p in gt_poses]
    gt_poses = np.array(gt_poses)
    
    vo_filename = '{}/{}'.format(seq_dir, 'vo/vo.csv')
    vo_poses = interpolate_vo_poses(vo_filename, stereo_img_ts, stereo_img_ts[0])
    vo_poses = [se3_to_components(p) for p in vo_poses]
    vo_poses = np.array(vo_poses)
       
    r_w_imu_w = gt_poses[:,0:3]
    rpy_w_imu_w = gt_poses[:,3:6]

    plt.plot(gt_poses[:,0], gt_poses[:,1])
    plt.plot(vo_poses[:,0], -vo_poses[:,1])
    plt.savefig('{}_gt_traj'.format(seq))
    plt.figure()
    plt.plot(rpy_w_imu_w[:,2])
    plt.plot(vo_poses[:,5])
    plt.savefig('{}_gt_rpy'.format(seq))
    plt.figure()
    
    ''' GT '''   
    C_c_imu = SO3(np.array([[1,0,0],[0,0,1],[0,1,0]]))
    gt_traj=[np.array([[ 1, 0,  0,  0], \
                      [0, 0,  1,  0], \
                      [0, -1, 0,  0],  \
                      [ 0.,          0.,          0.,          1.        ]])]
    gt_T_w_c_list = []
    for i in range(0, r_w_imu_w.shape[0]):
        C_imu_w = SO3.from_rpy(rpy_w_imu_w[i,0], rpy_w_imu_w[i,1], rpy_w_imu_w[i,2]).inv()
        C_c_w = C_c_imu.dot(C_imu_w)
        C_w_c = SO3(np.copy(C_c_w.inv().as_matrix()))
        
        T_w_c = SE3(C_w_c, r_w_imu_w[i])
        gt_T_w_c_list.append(T_w_c)
    
    gt_pose_vec_list, gt_dT_list = [], []    
    for i in range(0,len(gt_T_w_c_list)-1):
        T_12 = (gt_T_w_c_list[i].inv().dot(gt_T_w_c_list[i+1]))
        gt_pose_vec = T_12.log()
        gt_pose_vec_list.append(gt_pose_vec)
        gt_dT_list.append(SE3.exp(gt_pose_vec).as_matrix())

        
    for i in range(0, len(gt_dT_list)):
        dT = SE3.from_matrix(gt_dT_list[i],normalize=True).inv()
        new_est = SE3.as_matrix(SE3.from_matrix(gt_traj[-1],normalize=True).dot(dT))
        gt_traj.append(new_est)
    

    est_traj = np.array(gt_traj) #use gt as placeholder because vo isn't set up correctly
    gt_traj = np.array(gt_traj)

    data['poses_est'] = est_traj.transpose(1,2,0)
    data['poses_gt'] = gt_traj.transpose(1,2,0)
    data['keyframe_ts'] = np.array(stereo_img_ts)
    data['keyframe_idx'] = np.array(stereo_img_idx)
    data['keyframe_seq_idx'] = img_seq_num[stereo_img_idx]
    sio.savemat('{}/{}.mat'.format(output_dir, seq),data)

   