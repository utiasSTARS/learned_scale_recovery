import time
import torch
#import sys
#sys.path.insert(0,'..')
from utils.learning_helpers import *
from utils.lie_algebra import se3_log_exp
from models.stn import *

def apply_dpc(corr, vo_lie_alg, dpc, mode, epoch=25):
    vo = vo_lie_alg.clone()
    vo[:,0:3] = 0 #fully learn translation in all cases.
    if mode == 'translation':
        corr[:,3:6]=0 #fully use classical VO rotation
        # pose = se3_log_exp(corr, vo) 
        pose = corr + vo
        
    if mode == 'pose':
        if dpc == True: #learn full translation, but learn a correction to the classical VO estimate
            # pose = se3_log_exp(corr, vo_lie_alg)  
            pose = corr + vo 
        else: #learn full translation and full rotation
            pose = corr
    return pose

class Trainer():
    def __init__(self, config, models, loss, optimizer):
        self.config = config
        self.device = config['device']
        self.depth_model = models[0]
        self.pose_model = models[1]
        self.optimizer = optimizer
        self.loss = loss
        self.mode = self.config['pose_output_type']
        self.dpc = self.config['dpc']

    def forward(self, dset, epoch, phase):
        dev = self.device
        start = time.time()
        if phase == 'train' and self.config['freeze_depthnet'] is False: self.depth_model.train(True)
        else:  
            self.depth_model.train(False)  
            self.depth_model.eval()
        if phase == 'train' and self.config['freeze_depthnet'] is False: self.pose_model.train(True)
        else:
            self.pose_model.train(False)
            self.pose_model.eval()

        dset_size = len(dset)
        running_loss = None         
            # Iterate over data.
        for batch_num, data in enumerate(dset):
            target_img, source_imgs, lie_alg, intrinsics, flow_imgs = data
            target_img_aug = target_img['color_aug_left'].to(dev)
            lie_alg = lie_alg['color_aug']

            source_img_list = []
            source_img_aug_list = []
            gt_lie_alg_list = []
            vo_lie_alg_list = []
            for i, im, in enumerate(source_imgs['color_aug_left']):
                source_img_aug_list.append(im.to(dev))
                source_img_list.append(source_imgs['color_left'][i].to(dev))
                gt_lie_alg_list.append(lie_alg[i][0].type(torch.FloatTensor).to(dev))
                vo_lie_alg_list.append(lie_alg[i][1].type(torch.FloatTensor).to(dev))
    
            if self.config['flow_type'] == 'classical':
                flow_imgs_fwd, flow_imgs_back = flow_imgs
                flow_imgs_fwd_list, flow_imgs_back_list = [], []
                for i in range(0, len(flow_imgs_fwd)):
                    flow_imgs_fwd_list.append(flow_imgs_fwd[i].to(dev))
                    flow_imgs_back_list.append(flow_imgs_back[i].to(dev))
                flow_imgs = [flow_imgs_fwd_list, flow_imgs_back_list]
            else:
                flow_imgs = [[None for i in range(0,len(source_img_list))] for i in range(0,2)] #annoying but necessary

            intrinsics_aug = intrinsics['color_aug_left'].type(torch.FloatTensor).to(dev)[:,0,:,:] #only need one matrix since it's constant across the training sample
            intrinsics = intrinsics['color_left'].type(torch.FloatTensor).to(dev)[:,0,:,:]

            pose = []

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                disparity = self.depth_model(target_img_aug, epoch=epoch)
                if disparity[0].median() <=0.0000001 and disparity[0].mean() <=0.00000001:
                    print("warning - depth est has failed")
                disparities = [disparity]
                
                for i, im in enumerate(source_img_aug_list):
                    source_disparity = self.depth_model(im, epoch=epoch)
                    disparities.append(source_disparity)
                
                depths = [disp_to_depth(disp[0], self.config['min_depth'], self.config['max_depth'])[1] for disp in disparities] ####.detach()
                poses, poses_inv = solve_pose(self.pose_model, target_img_aug, source_img_aug_list, vo_lie_alg_list, self.dpc, self.mode, epoch, intrinsics_aug, flow_imgs)

                        ## Using GT orientation for paper results
                for i in range(0, len(poses)):
                    poses[i][:,3:6] = gt_lie_alg_list[i][:,3:6]
                    poses_inv[i][:,3:6] = -gt_lie_alg_list[i][:,3:6]
    
                # ## For testing
                # print('brightness fwd', 1+poses[0][:,6].mean().item(), poses[0][:,7].mean().item())
                # print('brightness back', poses_inv[0][:,6].mean().item(), poses_inv[0][:,7].mean().item())
                # print('forward', poses[0][:,2].mean().item(), poses_inv[0][:,2].mean().item())
                # print('backward', poses[1][:,2].mean().item(), poses_inv[1][:,2].mean().item())
                minibatch_loss=0
                if epoch ==0 and batch_num < 250 and (self.config['l_gt_supervised']!=True): ## Initial pretraining of posenet with VO estimates
                    for i in range(0, len(source_img_list)):
                            minibatch_loss += 10*torch.abs(poses[i][:,0:3] - 0.3*vo_lie_alg_list[i][:,0:3]).mean()
                            minibatch_loss += 10*torch.abs(poses_inv[i][:,0:3] + 0.3*vo_lie_alg_list[i][:,0:3]).mean()
                            minibatch_loss += 10*torch.pow(10*(poses[i][:,3:6] - vo_lie_alg_list[i][:,3:6]),2).mean()
                            minibatch_loss += 10*torch.pow(10*(poses_inv[i][:,3:6] + vo_lie_alg_list[i][:,3:6]),2).mean()
                            
                else:                   

                    losses = self.loss(source_img_list, target_img, [poses, poses_inv], disparities, intrinsics_aug, pose_vec_weight = vo_lie_alg_list[i],epoch=epoch)
                    
                    
                    ## pose losses (simpler to add here than in the main loss function)
                    if self.config['l_gt_supervised']:
                        for i in range(0, len(source_img_list)):
                            gt_lie_alg = gt_lie_alg_list[i].clone()
                            gt_lie_alg[:,0:3] = gt_lie_alg[:,0:3]/50
                            losses['l_gt_supervised'] = self.config['l_gt_supervised_weight']*torch.pow(10*(poses[i] - gt_lie_alg),2).mean()
                            losses['l_gt_supervised'] += self.config['l_gt_supervised_weight']*torch.pow(10*(poses_inv[i] + gt_lie_alg),2).mean() 
                            losses['total'] += losses['l_gt_supervised'] 
                                            
                    if self.config['l_pose_consist']:
                        losses['l_pose_consist'] = self.config['l_pose_consist_weight']*compute_pose_consistency_loss(poses, poses_inv)
                        losses['total'] += losses['l_pose_consist']
                        

                    minibatch_loss += losses['total']
                    
                    if running_loss is None:
                        running_loss = losses
                    else:
                        for key, val in losses.items():
                            if val.item() != 0:
                            #     print(key, val.item())
                                running_loss[key] += val.data
                                
                if phase == 'train':   
                    minibatch_loss.backward()
                    self.optimizer.step()
        
        print("{} epoch completed in {} seconds.".format(phase, timeSince(start)))  
        if epoch > 0:            
            for key, val in running_loss.items():
                running_loss[key] = val.item()/float(batch_num)
            print('{} Loss: {:.6f}'.format(phase, running_loss['total']))
            return running_loss
        else:
            return None



def compute_pose(pose_model, imgs, vo_lie_alg, dpc, mode, epoch):
    pose = pose_model(imgs, T21=vo_lie_alg, epoch=epoch)
    pose[:,0:6] = apply_dpc(pose[:,0:6], vo_lie_alg, dpc, mode, epoch=epoch)
    return pose

def compute_pose_consistency_loss(poses, poses_inv):
    pose_consistency_loss = 0

    for pose, pose_inv in zip(poses, poses_inv):
        t_s1 = pose[:,0:6]
        t_s1_inv = pose_inv[:,0:6]
        pose_consistency_loss += (t_s1 + t_s1_inv).abs()

    return pose_consistency_loss.mean()

def solve_pose(pose_model, target_img, source_img_list, vo_lie_alg_list, dpc, mode, epoch, intrinsics, flow_imgs):
    poses, poses_inv = [], []

    flow_imgs_fwd, flow_imgs_back = flow_imgs
    for source_img, vo_lie_alg, flow_img_fwd, flow_img_back in zip(source_img_list, vo_lie_alg_list, flow_imgs_fwd, flow_imgs_back):
        pose = compute_pose(pose_model, [target_img, source_img, flow_img_fwd], vo_lie_alg, dpc, mode, epoch)
        pose_inv = compute_pose(pose_model, [source_img, target_img, flow_img_back], -vo_lie_alg, dpc, mode, epoch)
   
                    
        poses.append(pose)
        poses_inv.append(pose_inv)

    return poses, poses_inv