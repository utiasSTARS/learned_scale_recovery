import torch
import sys
sys.path.insert(0,'..')
from train_mono import Trainer
from validate import test_depth_and_reconstruction, test_trajectory

import models.packetnet_depth_and_egomotion as models_packetnet
import models.depth_and_egomotion as models_monodepth   
from utils.learning_helpers import *
from utils.custom_transforms import *
import losses
from vis import *
import numpy as np
import datetime
import time
from tensorboardX import SummaryWriter
import argparse
import torch.backends.cudnn as cudnn
import os
import glob
 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
parser = argparse.ArgumentParser(description='training arguments.')

'''System Options'''
parser.add_argument('--estimator', type=str, default='libviso2') #libviso2 or orbslam
parser.add_argument('--estimator_type', type=str, default='mono') #mono or stereo
parser.add_argument('--flow_type', type=str, default='none', help='classical, or none')
#parser.add_argument('--load_stereo', action='store_true', default=False) #currently not implemented for left/right consist
parser.add_argument('--stereo_baseline', type=float, default=0.52)
parser.add_argument('--num_scales', type=int, default=1)
parser.add_argument('--img_resolution', type=str, default='med') # low (128x445) med (192 x640) or high (256 x 832) 
parser.add_argument('--img_per_sample', type=int, default=3) #1 target image, and rest are source images 

'''Training Arguments'''
parser.add_argument('--data_dir', type=str, default='/media/m2-drive/datasets/KITTI-downsized')
parser.add_argument('--data_format', type=str, default='odometry') #odmetry or eigen
parser.add_argument('--date', type=str, default='0000000')
parser.add_argument('--train_seq', nargs='+', type=str, default=['00_02', '02_02'])
parser.add_argument('--val_seq', nargs='+',type=str, default=['05_02'])
parser.add_argument('--test_seq', nargs='+', type=str, default=['09_02'])
parser.add_argument('--augment_motion', action='store_true', default=False)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--lr', type=float, default=9e-4)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--lr_decay_epoch', type=float, default=4)
parser.add_argument('--save_results', action='store_true', default=True)
parser.add_argument('--max_depth', type=float, default=80./30) #  80/30
parser.add_argument('--min_depth', type=float, default=0.06) #0.06

''' Losses'''
parser.add_argument('--l_reconstruction', action='store_true', default=True, help='use photometric reconstruction losses (l1, ssim)')
parser.add_argument('--l_ssim', action='store_true', default=True, help='without ssim, only use L1 error')
parser.add_argument('--l1_weight', type=float, default=0.15) #0.15
parser.add_argument('--l_ssim_weight', type=float, default=0.85) #0.85
parser.add_argument('--with_auto_mask', action='store_true', default=True, help='with the the mask for stationary points')

parser.add_argument('--l_pose_consist', action='store_true', default=True, help='ensure forward and backward pose predictions align')
parser.add_argument('--l_pose_consist_weight', type=float, default=5)
parser.add_argument('--l_inverse', action='store_true', default=True, help='reproject target image to source images as well')
parser.add_argument('--l_depth_consist', action='store_true', default=True, help='Depth consistency loss from https://arxiv.org/pdf/1908.10553.pdf')
parser.add_argument('--l_depth_consist_weight', type=float, default=0.14) #0.14
parser.add_argument('--with_depth_mask', action='store_true', default=True, help='with the depth consistency mask for moving objects and occlusions or not')
parser.add_argument('--l_scale_recovery', action='store_true', default=True, help='enforces metric scale consistency')
parser.add_argument('--l_scale_depth_weight', type=float, default=0.02) #0.02
parser.add_argument('--l_scale_pose_weight', type=float, default=0.6) # 0.6 for initial,  or 6
parser.add_argument('--camera_height', type=float, default=1.70) #1.52 for oxford, 1.70 for KITTI
parser.add_argument('--l_smooth', action='store_true', default=True)
parser.add_argument('--l_smooth_weight', type=float, default=0.05) #0.15


### Testing
parser.add_argument('--l_gt_supervised', action='store_true', default=False, help='gt pose used as supervision signal') 
parser.add_argument('--l_gt_supervised_weight', type=float, default=30) #60
# parser.add_argument('--l_left_right_consist', action='store_true', default=False, help='stereo loss (reproject left image to right) to resolve metric scale') 
# parser.add_argument('--l_left_right_consist_weight', type=float, default=0.4)
 
parser.add_argument('--load_pretrained_depth', action='store_true', default=False, help= 'Use an existing depth model')
parser.add_argument('--load_pretrained_pose', action='store_true', default=True, help= 'Use an existing pose model')
parser.add_argument('--pretrained_dir', type=str, default='results/202103011655')      
parser.add_argument('--pretrained_plane_dir', type=str, default='')   #'results/plane-model-med-res-oxford',    
        
args = parser.parse_args()
config={
    'num_frames': None,
    'skip':1,    ### if not one, we skip every 'skip' samples that are generated ({1,2}, {2,3}, {3,4} becomes {1,2}, {3,4})
    'correction_rate': 1, ### if not one, only perform corrections every 'correction_rate' frames (samples become {1,3},{3,5},{5,7} when 2)
    'minibatch':6,      ##minibatch size      
    'freeze_posenet': False,
    'freeze_depthnet': False,
    }
for k in args.__dict__:
    config[k] = args.__dict__[k]
print(config)

args.data_dir = '{}/{}_res'.format(args.data_dir, config['img_resolution'])
config['data_dir'] = '{}/{}_res'.format(config['data_dir'], config['img_resolution'])
if args.data_format == 'odometry':

    # if args.load_stereo == True:
    #     from data.kitti_loader import KittiLoaderPytorch # loads left and right images for each sample
    # else:
    from data.kitti_loader_stereo import KittiLoaderPytorch #loads left and right images as separate sequences

    dsets = {x: KittiLoaderPytorch(config, [args.train_seq, args.val_seq, args.test_seq], mode=x, transform_img=get_data_transforms(config)[x], \
                                augment=config['augment_motion']) for x in ['train', 'val']}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=config['minibatch'], shuffle=True, num_workers=8) for x in ['train', 'val']}

    val_dset = KittiLoaderPytorch(config, [args.train_seq, args.val_seq, args.test_seq], mode='val', transform_img=get_data_transforms(config)['val'])
    val_dset_loaders = torch.utils.data.DataLoader(val_dset, batch_size=config['minibatch'], shuffle=False, num_workers=8)

    test_dset = KittiLoaderPytorch(config, [args.train_seq, args.val_seq, args.test_seq], mode='test', transform_img=get_data_transforms(config)['test'])
    test_dset_loaders = torch.utils.data.DataLoader(test_dset, batch_size=config['minibatch'], shuffle=False, num_workers=8)

    eval_dsets = {'val': val_dset_loaders, 'test':test_dset_loaders}

if args.data_format == 'eigen':
    from data.kitti_loader_eigen import KittiLoaderPytorch
    dsets = {x: KittiLoaderPytorch(config, None, mode=x, transform_img=get_data_transforms(config)[x], \
                                augment=config['augment_motion'], skip=config['skip']) for x in ['train', 'val']}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=config['minibatch'], shuffle=True, num_workers=8) for x in ['train', 'val']}

    val_dset = KittiLoaderPytorch(config, None, mode='val', transform_img=get_data_transforms(config)['val'])
    val_dset_loaders = torch.utils.data.DataLoader(val_dset, batch_size=config['minibatch'], shuffle=False, num_workers=8)

    eval_dsets = {'val': val_dset_loaders}    

def main():
    results = {}
    config['device'] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    start = time.time()
    now= datetime.datetime.now()
    ts = '{}-{}-{}-{}-{}'.format(now.year, now.month, now.day, now.hour, now.minute)

    ''' Load Pretrained Models'''
    pretrained_depth_path, pretrained_pose_path = None, None
    if config['load_pretrained_depth']:
        pretrained_depth_path = glob.glob('{}/**depth**best-loss-val_seq-**-test_seq-**.pth'.format(config['pretrained_dir']))[0]
    
    if config['load_pretrained_pose']: ## skip epoch 0 if pose model is pretrained (epoch 0 initializes with VO)
        epochs = range(1,config['num_epochs'])
        pretrained_pose_path = glob.glob('{}/**pose**best-loss-val_seq-**-test_seq-**.pth'.format(config['pretrained_dir']))[0]
    else:
        epochs = range(0,config['num_epochs'])
    
    
    ### load the models
    depth_model = models_monodepth.depth_model(config).to(config['device'])
    pose_model = models_packetnet.pose_model(config).to(config['device'])
    
    if pretrained_depth_path is not None and config['load_pretrained_depth']==True:
        depth_model.load_state_dict(torch.load(pretrained_depth_path))
    if pretrained_pose_path is not None and config['load_pretrained_pose']==True:
        pose_model.load_state_dict(torch.load(pretrained_pose_path))    
    models = [depth_model, pose_model]
        ## Load the pretrained plane estimator if using the plane loss
    if config['l_scale_recovery']:
        from models.plane_net import PlaneModel
        plane_model = PlaneModel(config).to(config['device'])
        pretrained_plane_path = glob.glob('{}/**plane**.pth'.format(config['pretrained_plane_dir']))[0]
        plane_model.load_state_dict(torch.load(pretrained_plane_path))
        for param in plane_model.parameters():
            param.requires_grad = False
    else:
        plane_model = None
    
    if config['freeze_depthnet']: print('Freezing depth network weights.')
    if config['freeze_posenet']: print('Freezing pose network weights.')
    for param in depth_model.parameters():
        param.requires_grad = not config['freeze_depthnet']         
    for param in pose_model.parameters():
        param.requires_grad = not config['freeze_posenet']  

    params = [{'params': depth_model.parameters()},
                {'params': pose_model.parameters(), 'lr': 2*config['lr']}]
    loss = losses.Compute_Loss(config, plane_model=plane_model)
    optimizer = torch.optim.Adam(params, lr=config['lr'], weight_decay = config['wd']) #, amsgrad=True)
    trainer = Trainer(config, models, loss, optimizer)
    cudnn.benchmark = True



    best_val_loss = {}
    best_loss_epoch = {}
    for key, dset in eval_dsets.items():
        best_val_loss[key] = 1e5

    for epoch in epochs:
        optimizer = exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=config['lr_decay_epoch']) ## reduce learning rate as training progresses  
        print("Epoch {}".format(epoch))
        train_losses = trainer.forward(dset_loaders['train'], epoch, 'train')
        with torch.no_grad():
            val_losses = trainer.forward(dset_loaders['val'], epoch, 'val')    
#        
        if epoch == 0 or (epoch == 1 and (config['load_pretrained_pose'] == True) ):
            val_writer = SummaryWriter(comment="tw-val-{}-test_seq-{}_val".format(args.val_seq[0], args.test_seq[0]))
            train_writer = SummaryWriter(comment="tw-val-{}-test_seq-{}_train".format(args.val_seq[0], args.test_seq[0]))
            
        if train_losses is not None and val_losses is not None:
            for key, value in train_losses.items():
                train_writer.add_scalar('{}'.format(key), value, epoch+1)
                val_writer.add_scalar('{}'.format(key), val_losses[key], epoch+1)

        for key, dset in eval_dsets.items():
            print("{} Set, Epoch {}".format(key, epoch))
        
            if epoch > 0: 
                ###plot images, depth map, explainability mask
                img_array, disparity, exp_mask, d = test_depth_and_reconstruction(device, models, dset, config, epoch=epoch)
                
                source_disp, reconstructed_disp, d_masks = d[0], d[1], d[2]
                img_array = plot_img_array(img_array)
                train_writer.add_image(key+'/imgs',img_array,epoch+1) 
                for i,d in enumerate(disparity):
                    train_writer.add_image(key+'/depth-{}/target-depth'.format(i), plot_disp(d), epoch+1)

                    ### For depth consistency
                    train_writer.add_image(key+'/depth-{}/source-depth'.format(i), plot_disp(source_disp[i]), epoch+1)
                    train_writer.add_image(key+'/depth-{}/reconstructed-depth'.format(i), plot_disp(reconstructed_disp[i]), epoch+1)
                   
                d_masks = plot_img_array(d_masks)
                train_writer.add_image(key+'/depth/masks', d_masks, epoch+1)

                exp_mask = plot_img_array(exp_mask)
                train_writer.add_image(key+'/exp_mask', exp_mask, epoch+1)
                                    
                ###evaluate trajectories 
                if args.data_format == 'odometry':   
                    est_lie_alg, gt_lie_alg, est_traj, gt_traj, errors = test_trajectory(config, device, models, dset, epoch)
              
                    errors = plot_6_by_1(est_lie_alg - gt_lie_alg, title='6x1 Errors')  
                    pose_vec_est = plot_6_by_1(est_lie_alg, title='Est')   
                    pose_vec_gt = plot_6_by_1(gt_lie_alg, title='GT')    
                    est_traj_img = plot_multi_traj(est_traj, 'Est.', gt_traj, 'GT', key+' Set')

                    train_writer.add_image(key+'/est_traj', est_traj_img, epoch+1)
                    train_writer.add_image(key+'/errors', errors, epoch+1)
                    train_writer.add_image(key+'/gt_lie_alg', pose_vec_gt, epoch+1)
                    train_writer.add_image(key+'/est_lie_alg', pose_vec_est, epoch+1)
    

                    results[key] = {'val_seq': args.val_seq, 
                        'test_seq': args.test_seq,
                        'epochs': epoch+1,
                        'est_pose_vecs': est_lie_alg,
                        'gt_pose_vecs': gt_lie_alg,
                        'est_traj': est_traj,
                        'gt_traj': gt_traj, 
                    }
                    
                if args.save_results:   ##Save the best models
                    os.makedirs('results/{}'.format(config['date']), exist_ok=True)
                    
                    if (val_losses['l_reconstruct_forward'] + val_losses['l_reconstruct_inverse']) < best_val_loss[key] and epoch > 0: # and epoch > 2*(config['iterations']-1):
                        best_val_loss[key] = (val_losses['l_reconstruct_forward'] + val_losses['l_reconstruct_inverse'])
                        best_loss_epoch[key] = epoch
                        depth_dict_loss = depth_model.state_dict()
                        pose_dict_loss = pose_model.state_dict()
                        if key == 'val':
                            print("Lowest validation loss (saving model)")       
                            torch.save(depth_dict_loss, 'results/{}/{}-depth-best-loss-val_seq-{}-test_seq-{}.pth'.format(config['date'], ts, args.val_seq[0], args.test_seq[0]))
                            torch.save(pose_dict_loss, 'results/{}/{}-pose-best-loss-val_seq-{}-test_seq-{}.pth'.format(config['date'], ts, args.val_seq[0], args.test_seq[0]))

                        if args.data_format == 'odometry': 
                            results[key]['best_loss_epoch'] = best_loss_epoch[key]
                            save_obj(results, 'results/{}/{}-results-val_seq-{}-test_seq-{}'.format(config['date'], ts, args.val_seq[0], args.test_seq[0]))
                        save_obj(config, 'results/{}/config'.format(config['date']))
                        f = open("results/{}/config.txt".format(config['date']),"w")
                        f.write( str(config) )
                        f.close()
    save_obj(loss.scale_factor_list, 'results/{}/scale_factor'.format(config['date']))      
    duration = timeSince(start)    
    print("Training complete (duration: {})".format(duration))
 
main()
