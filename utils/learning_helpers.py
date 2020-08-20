import math
import torch
import time
from torch.optim import Optimizer
import numpy as np
from liegroups import SE3, SO3
import pickle

import sys
sys.path.insert(0,'..')
from data.kitti_loader import KittiLoaderPytorch
import models.depth_and_egomotion as models
from utils.custom_transforms import *

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
def save_state(state, filename='test.pth.tar'):
    torch.save(state, filename)
    
def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=5):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch == 70 or epoch == 71 or epoch == 72 or epoch == 73 or epoch == 74 or epoch == 75:
        print('LR is reduced by {}'.format(0.5))
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
            param_group['lr'] = param_group['lr']*0.5
            
    if epoch !=0 and epoch%lr_decay_epoch==0:
        print('LR is reduced by {}'.format(0.5))
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
            param_group['lr'] = param_group['lr']*0.5

    return optimizer   

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def find_loop_closures(traj, cum_dist):
    num_loop_closures = 0
    filtered_loop_closures=0
    idx_list=[]
    for i in range(0,traj.shape[0],8): #check for loop closure points (compare current frame with all future points, and do this for every 20th frame)
        current_pose = traj[i]
        current_trans = current_pose[0:3,3]
        current_rot = SO3.from_matrix(current_pose[0:3,0:3], normalize=True).to_rpy()
        current_yaw = current_rot[2]
        
        current_cum_dist = cum_dist[i]
        loop_closure_idx = np.linalg.norm(np.abs(current_trans[0:3] - traj[i+1:,0:3,3]),axis=1) <= 7
        dist_idx = (cum_dist[i+1:]-current_cum_dist) >=10
        loop_closure_idx = loop_closure_idx & dist_idx
        
        idx = np.where(loop_closure_idx == 1)
        
        if idx != np.array([]):
            for pose_idx in idx[0]:
                T = traj[i+1:][pose_idx]
                yaw = SE3.from_matrix(T,normalize=True).rot.to_rpy()[2]
                yaw_diff = np.abs(np.abs(current_yaw) - np.abs(yaw))
                in_range = ((yaw_diff <= 0.15)) or (np.abs((np.pi - yaw_diff) <=0.15))
                filtered_loop_closures += in_range
                if in_range:
                    idx_list.append(pose_idx+i)
        
        num_loop_closures += np.sum(loop_closure_idx)

    return num_loop_closures, filtered_loop_closures, idx_list

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper. Taken from Monodepth2
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def depth_to_disp(depth, min_depth, max_depth):
    """Convert a depth prediction back into a disparity prediction
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    
    disp = 1/depth
    
    unscaled_disp = (disp- min_disp)/(max_disp - min_disp) 
    return unscaled_disp

###Moving average filter of size W
def moving_average(a, n) : #n must be odd)
    if n == 1:
        return a
    else:
        for i in range(a.shape[1]):
            if (n % 2) == 0:
                n -=1
            ret = np.cumsum(a[:,i], dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            a[:,i] = np.pad(ret[n - 1:-2] / n , int((n-1)/2+1), 'edge')
        return a
    
    import torch
    
def data_and_model_loader(config, pretrained_depth_path, pretrained_pose_path, seq=None, load_depth=True):
    if seq == None:
        seq = config['test_seq']
    else:
        seq = [seq]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    test_dset = KittiLoaderPytorch(config, [seq, seq, seq], mode='test', transform_img=get_data_transforms(config)['test'])
    test_dset_loaders = torch.utils.data.DataLoader(test_dset, batch_size=config['minibatch'], shuffle=False, num_workers=6)
    eval_dsets = {'test': test_dset_loaders}
    
    if load_depth:
        depth_model = models.depth_model(config).to(device)
    pose_model = models.pose_model(config).to(device)
    
    if pretrained_depth_path is not None and load_depth==True:
        depth_model.load_state_dict(torch.load(pretrained_depth_path))
    if pretrained_pose_path is not None:
        pose_model.load_state_dict(torch.load(pretrained_pose_path))
        
    
    pose_model.train(False)
    pose_model.eval()
    if load_depth:
        depth_model.train(False)
        depth_model.eval()
    else:
        depth_model = None
    
    mmodels = [depth_model, pose_model]
    return test_dset_loaders, mmodels, device