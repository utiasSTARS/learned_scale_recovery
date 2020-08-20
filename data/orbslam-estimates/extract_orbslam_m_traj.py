import numpy as np
from liegroups.numpy import SO3, SE3
from collections import namedtuple
from PIL import Image
import scipy.io as sio
import matplotlib.pyplot as plt
from pyslam.metrics import TrajectoryMetrics
import glob
import numpy as np


seq_names= {'00': '2011_10_03_drive_0027',
            '01': '2011_10_03_drive_0042',
        '02': '2011_10_03_drive_0034',
        '04': '2011_09_30_drive_0016',
        '05': '2011_09_30_drive_0018',
        '06': '2011_09_30_drive_0020',
        '07': '2011_09_30_drive_0027',
        '08': '2011_09_30_drive_0028',
        '09': '2011_09_30_drive_0033',
        '10': '2011_09_30_drive_0034',
        }

gt_dir = 'ground_truth'
output_dir = 'orbslam_mono_traj'
data_dir = '/media/brandon/DATA/KITTI-odometry-gray/sequences'

for seq in seq_names:

    print(seq)
    drive = seq_names[seq]
    data = sio.loadmat('{}/{}.mat'.format(gt_dir,drive))
    kitti_ts = np.loadtxt('{}/{}/times.txt'.format(data_dir, seq))
    
    f = '/home/brandon/Desktop/Projects/VO-implementations/ORB_SLAM2/saved_trajectories/{}.txt'.format(seq)
    img_idx = []
    dT_list = []
    T = []
    l = np.loadtxt(f)
    ts = l[:,0]
    pos = l[:,1:4]
    quat = l[:,4:8]
    
    for i in range(0, pos.shape[0]):
        R = SO3.from_quaternion(quat[i],ordering='xyzw')
        T.append(SE3(R, pos[i]))
        img_idx.append(np.argmin(np.abs(ts[i]-kitti_ts)))

    for i in range(0, len(T)-1):
        dT = (T[i].inv()).dot(T[i+1])
        dT_list.append(dT.as_matrix())

    gt_traj = data['poses_gt'].transpose(2,0,1) #[0:4541]
    est_traj = []
    est_traj.append(gt_traj[0])
    
    for i in range(0,len(dT_list)):
        
        dT = SE3.from_matrix(dT_list[i],normalize=True)
        
        new_est = SE3.as_matrix(SE3.from_matrix(est_traj[i],normalize=True).dot(dT))
        est_traj.append(new_est)

    est_traj = np.array(est_traj)
    gt_traj = np.array(gt_traj)[img_idx]

    gt_traj_se3 = [SE3.from_matrix(T) for T in gt_traj]
    est_traj_se3 = [SE3.from_matrix(T,normalize=True) for T in est_traj]
    data['poses_est'] = est_traj.transpose(1,2,0)
    data['poses_gt'] = gt_traj.transpose(1,2,0)
    data['keyframe_idx'] = np.array(img_idx)
    sio.savemat('{}/{}.mat'.format(output_dir,drive), data)
    gt_pose_vec = []
    est_pose_vec = []
    
    ###create the pose vec
    for idx in range(0, est_traj.shape[0]-1):
        T2 = SE3.from_matrix(est_traj[idx+1,:,:], normalize=True).inv()
        T1 = SE3.from_matrix(est_traj[idx,:,:],normalize=True)
        dT = T2.dot(T1)
        
        gt2 = SE3.from_matrix(gt_traj[idx+1,:,:],normalize=True).inv()
        gt1 = SE3.from_matrix(gt_traj[idx,:,:],normalize=True)
        gt_dT = gt2.dot(gt1)
        gt_pose_vec = gt_dT.log()
        print(gt_pose_vec)
        
        svo_target = dT.log()
        if np.linalg.norm(svo_target[0:3]) >=1e-10:
            scale = np.linalg.norm(gt_pose_vec[0:3])/np.linalg.norm(svo_target[0:3])
            scale=1
            svo_target[0:3] = scale*svo_target[0:3]
            
        est_pose_vec.append(svo_target)
    
    scaled_traj = []
    scaled_traj.append(gt_traj[0]) 
    
    for i in range(0,est_traj.shape[0]-1):
        #classically estimated traj
        dT = SE3.exp(est_pose_vec[i])
        new_est = SE3.as_matrix((dT.dot(SE3.from_matrix(scaled_traj[i],normalize=True).inv())).inv())
        scaled_traj.append(new_est)
    
    scaled_traj_se3 = [SE3.from_matrix(T,normalize=True) for T in scaled_traj]
    scaled_traj = np.array(scaled_traj)
    

    plt.plot(scaled_traj[:,0,3], scaled_traj[:,1,3])    

    plt.plot(gt_traj[:,0,3], gt_traj[:,1,3])
    plt.legend(['Est', 'GT'])
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(seq)
    plt.grid()
    plt.savefig('{}.png'.format(seq))
    plt.figure()