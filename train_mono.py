import time
import torch
from utils.learning_helpers import *
from models.stn import *
from data.kitti_loader import process_sample_batch

def compute_pose_consistency_loss(poses, poses_inv):
    pose_consistency_loss = 0

    for pose, pose_inv in zip(poses, poses_inv):
        t_s1 = pose[:,0:6]
        t_s1_inv = pose_inv[:,0:6]
        pose_consistency_loss += (t_s1 + t_s1_inv).abs()

    return pose_consistency_loss.mean()

def solve_pose(pose_model, target_img, source_img_list, flow_imgs):
    poses, poses_inv = [], []

    flow_imgs_fwd, flow_imgs_back = flow_imgs
    
    for source_img, flow_img_fwd, flow_img_back in zip(source_img_list, flow_imgs_fwd, flow_imgs_back):
        pose = pose_model([target_img, source_img, flow_img_fwd])
        pose_inv = pose_model([source_img, target_img, flow_img_back])
       
        poses.append(pose)
        poses_inv.append(pose_inv)
        
    return poses, poses_inv


class Trainer():
    def __init__(self, config, models, loss, optimizer):
        self.config = config
        self.device = config['device']
        self.depth_model = models[0]
        self.pose_model = models[1]
        self.optimizer = optimizer
        self.loss = loss


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
            target_img, source_img_list, gt_lie_alg_list, vo_lie_alg_list, flow_imgs, intrinsics, target_img_aug, \
            source_img_aug_list, gt_lie_alg_aug_list, vo_lie_alg_aug_list, intrinsics_aug = process_sample_batch(data, self.config)
                
            pose = []
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                batch_size = target_img_aug.shape[0]
                
                ## compute disparities in same batch
                imgs = torch.cat([target_img_aug, source_img_aug_list[0], source_img_aug_list[1]],0)
                disparities = self.depth_model(imgs, epoch=epoch)
                target_disparities = [disp[0:batch_size] for disp in disparities]
                source_disp_1 = [disp[batch_size:(2*batch_size)] for disp in disparities]
                source_disp_2 = [disp[2*batch_size:(3*batch_size)] for disp in disparities]
                
                disparities = [target_disparities, source_disp_1, source_disp_2]

                if target_disparities[0].median() <=0.0000001 and target_disparities[0].mean() <=0.00000001:
                    print("warning - depth est has failed")


                poses, poses_inv = solve_pose(self.pose_model, target_img_aug, source_img_aug_list, flow_imgs)

                print('fwd',poses[0][0,2].item(), 'gt', gt_lie_alg_list[0][0,2].item())
                print('back',poses_inv[0][0,2].item(), 'gt inv',-gt_lie_alg_list[0][0,2].item())                    


                minibatch_loss=0
                # if epoch ==0 and batch_num < 400: ## Initial pretraining of posenet with orbslam2 estimates
                #     ###this isnt required and doesn't impact the final results
                #     #it simply speeds up training of posenet
                #     for i in range(0, len(source_img_list)):
                #             # minibatch_loss += 10*torch.abs(poses[i][:,0:3] - 0.01*vo_lie_alg_aug_list[i][:,0:3]).mean()
                #             # minibatch_loss += 10*torch.abs(poses_inv[i][:,0:3] + 0.01*vo_lie_alg_aug_list[i][:,0:3]).mean()
                #             minibatch_loss += 10*torch.abs(poses[i][:,0:3] - 0.3*vo_lie_alg_list[i][:,0:3]).mean()
                #             minibatch_loss += 10*torch.abs(poses_inv[i][:,0:3] + 0.3*vo_lie_alg_list[i][:,0:3]).mean()
                #             minibatch_loss += 10*torch.pow(10*(poses[i][:,3:6] - vo_lie_alg_aug_list[i][:,3:6]),2).mean()
                #             minibatch_loss += 10*torch.pow(10*(poses_inv[i][:,3:6] + vo_lie_alg_aug_list[i][:,3:6]),2).mean()
                            
                # else:                   
                if True:
                    losses  = self.loss(source_img_list, target_img, [poses, poses_inv], disparities, intrinsics_aug,epoch=epoch)
                    
                    
                    ## pose losses (simpler to add here than in the main loss function)
                    if self.config['l_gt_supervised']==True and epoch > 0:
                        for i in range(0, len(source_img_list)):
                            gt_lie_alg = gt_lie_alg_aug_list[i].clone()
                            gt_lie_alg[:,0:3] = gt_lie_alg[:,0:3]/30
                            losses['l_gt_supervised'] = self.config['l_gt_supervised_weight']*torch.pow(10*(poses[i] - gt_lie_alg),2).mean()
                            losses['l_gt_supervised'] += self.config['l_gt_supervised_weight']*torch.pow(10*(poses_inv[i] + gt_lie_alg),2).mean() 
                            losses['total'] += losses['l_gt_supervised'] 
                                            
                    if self.config['l_pose_consist']==True:
                        losses['l_pose_consist'] = self.config['l_pose_consist_weight']*compute_pose_consistency_loss(poses, poses_inv)
                        losses['total'] += losses['l_pose_consist']
                        

                    minibatch_loss += losses['total']
                    
                    if running_loss is None:
                        running_loss = losses
                    else:
                        for key, val in losses.items():
                            if val.item() != 0:
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
