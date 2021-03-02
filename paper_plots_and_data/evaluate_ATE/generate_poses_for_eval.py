import numpy as np
import torch
import sys
sys.path.append('../../')
from utils.learning_helpers import save_obj, load_obj
from data.kitti_loader_stereo import KittiLoaderPytorch
from utils.custom_transforms import get_data_transforms
from validate import compute_trajectory as tt
from pose_evaluation_utils import dump_pose_seq_TUM, rot2quat, pose_vec_to_mat, compute_ate
from glob import glob
import os
import matplotlib.pyplot as plt

seq = '10_02'
path_to_ws = '/home/brandonwagstaff/learned_scale_recovery/'
model_dir = 'results/202101281256-3-iter-unscaled-good'
path_to_dset_downsized = '/media/datasets/KITTI-odometry-downsized-stereo/'
out_folder = 'pose_data/our_predictions/{}'.format(seq[0:2])
# out_folder = 'pose_data/sfmlearner_results/{}'.format(seq[0:2])
os.makedirs(out_folder, exist_ok=True)
gt_folder = 'pose_data/ground_truth/{}/'.format(seq[0:2])
dir = path_to_ws + model_dir
config = load_obj('{}/config'.format(dir))
config['data_dir'] = path_to_dset_downsized+config['img_resolution'] + '_res/'
config['img_per_sample']=2

test_dset = KittiLoaderPytorch(config, [[seq], [seq], [seq]], mode='test', transform_img=get_data_transforms(config)['test'])
test_dset_loaders = torch.utils.data.DataLoader(test_dset, batch_size=config['minibatch'], shuffle=False, num_workers=6)

ts = test_dset.raw_ts[0]


data = load_obj('{}{}/results/depth_opt_bottleneck/{}_results'.format(path_to_ws, model_dir, seq))
# data = load_obj('{}{}/results/scale/{}_plane_fit'.format(path_to_ws, model_dir, seq))

poses = data['fwd_pose_vec_opt']
gt_pose_vec = data['gt_pose_vec']

# print(poses[1])
# print(gt_pose_vec[1])

gt_traj = test_dset_loaders.dataset.raw_gt_trials[0]
est_traj, gt_traj, _, _ = tt(poses,gt_traj,method='')


for i in range(0, poses.shape[0]-4):
    pose_window = np.copy(poses[i:i+4])
    init_pose = np.eye(4)
    this_pose = np.eye(4)
    ts_window = ts[i:i+5]
    filename = '%.6d.txt' % i
    # print(filename)
    out_file = '{}/{}'.format(out_folder,filename)
    
    with open(out_file, 'w') as f:
        f.write('%f %f %f %f %f %f %f %f\n' % (ts_window[0], 0, 0, 0, 0, 0, 0, 1))
        for j in range(0,pose_window.shape[0]):
            this_pose = np.dot(this_pose, pose_vec_to_mat(pose_window[j]))
            pose_out = np.dot(init_pose, np.linalg.inv(this_pose))
            tx = pose_out[0, 3]
            ty = pose_out[1, 3]
            tz = pose_out[2, 3]
            rot = pose_out[:3, :3]
            qw, qx, qy, qz = rot2quat(rot)
            f.write('%f %f %f %f %f %f %f %f\n' % (ts_window[j+1], tx, ty, tz, qx, qy, qz, qw))


pred_files = glob(out_folder + '/*.txt')

ate_all = []
for i in range(len(pred_files)):
    gtruth_file = gt_folder + os.path.basename(pred_files[i])

    if not os.path.exists(gtruth_file):
        continue

    ate = compute_ate(gtruth_file, pred_files[i])
    if ate == False:
        continue
    ate_all.append(ate)
ate_all = np.array(ate_all)

print("ATE mean: %.4f, std: %.4f" % (np.mean(ate_all), np.std(ate_all)))

plt.figure()
plt.grid()
plt.plot((ate_all))
plt.title('ATE')
plt.savefig('ATE_seq_{}.png'.format(seq))
