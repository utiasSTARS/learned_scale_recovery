import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import validate
from train_mono import solve_pose
from data.kitti_loader import process_sample_batch, process_sample
from plane_fitting import img_to_3d_torch, fit_plane_torch
from models.dnet_layers import ScaleRecovery
from utils.learning_helpers import save_obj, load_obj, disp_to_depth, data_and_model_loader
from utils.custom_transforms import *
from vis import *
import os
from validate import compute_trajectory as tt
import glob

path_to_ws = '/home/brandonwagstaff/learned_scale_recovery/' ##update this
path_to_dset_downsized = '/media/datasets/KITTI-odometry-downsized-stereo/'

load_from_mat = False #Make True to load paper results rather than recomputing
plane_rescaling = True
dnet_rescaling = True
ransac_rescaling = False
# seq_list = ['00_02', '02_02', '06_02', '07_02', '08_02', '05_02', '09_02', '10_02']
seq_list = ['05_02', '09_02', '10_02']

dir = path_to_ws + 'results/final_models/vo-kitti-scaled-202102182020' 
plane_dir = 'results/plane-model-kitti-202101072240'
results_dir = dir + '/results/scale/'
os.makedirs(results_dir, exist_ok=True)
logger = validate.ResultsLogger('{}/metrics.csv'.format(results_dir))
for seq in seq_list:
    print('sequence: {}'.format(seq))
    ransac_iter = 250
    downscale_factor = 4
    depth_threshold = 15
    inlier_thresh = 0.02
    cam_height = 1.70
    plotting =  False
    plot_range =  slice(0,-1)
    
    config = load_obj('{}/config'.format(dir))
    print(config)
    config['data_dir'] = path_to_dset_downsized+config['img_resolution'] + '_res/' #
    
    config['load_stereo'] = False
    config['augment_motion'] = False
    config['augment_backwards'] = False
    config['img_per_sample']=2
    config['test_seq'] = [seq]
    config['minibatch'] = 1
    config['load_pretrained'] = True
    config['data_format'] = 'odometry'
    config['estimator'] = 'orbslam'

    device=config['device']
        
    ### dataset and model loading    
    from data.kitti_loader_stereo import KittiLoaderPytorch
    test_dset = KittiLoaderPytorch(config, [[seq], [seq], [seq]], mode='test', transform_img=get_data_transforms(config)['test'])
    test_dset_loaders = torch.utils.data.DataLoader(test_dset, batch_size=config['minibatch'], shuffle=False, num_workers=6)
    if load_from_mat == False:
        import models.packetnet_depth_and_egomotion as models_packetnet
        import models.depth_and_egomotion as models
        
        depth_model = models.depth_model(config).to(device)
        pose_model = models_packetnet.pose_model(config).to(device)
        pretrained_depth_path = glob.glob('{}/**depth**best-loss-val_seq-**-test_seq-{}**.pth'.format(dir, ''))[0]
        pretrained_pose_path = glob.glob('{}/**pose**best-loss-val_seq-**-test_seq-{}**.pth'.format(dir, ''))[0]
        depth_model.load_state_dict(torch.load(pretrained_depth_path))
        pose_model.load_state_dict(torch.load(pretrained_pose_path))
        pose_model.train(False).eval()
        depth_model.train(False).eval()    
    
    if plane_rescaling == True:
        ### Plane Model
        from models.plane_net import PlaneModel, scale_recovery
        from losses import Plane_Height_loss
        plane_loss = Plane_Height_loss(config)
        plane_model = PlaneModel(config).to(config['device'])
        pretrained_plane_path = glob.glob('../{}/**plane**.pth'.format(plane_dir))[0]
        plane_model.load_state_dict(torch.load(pretrained_plane_path))
        plane_model.train(False).eval()    
        
    if dnet_rescaling == True:
        dgc = ScaleRecovery(config['minibatch'], 192, 640).to(device) 
    
    os.makedirs('scale_results/plane_imgs', exist_ok=True)
    os.makedirs('scale_results/plane_imgs/{}'.format(config['test_seq'][0]),exist_ok=True)

    def pf(img_list, depth, intrinsics, img_for_plotting):
        img, d, K = img_list, depth, intrinsics
        img = img[int(v_crop_min_num*img_list.shape[0]/v_crop_min_den):int(v_crop_max_num*img_list.shape[0]/v_crop_max_den), int(u_crop_min_num*img_list.shape[1]/u_crop_min_den):int(u_crop_max_num*img_list.shape[1]/u_crop_max_den)]
        d = d[int(v_crop_min_num*img_list.shape[0]/v_crop_min_den):int(v_crop_max_num*img_list.shape[0]/v_crop_max_den), int(u_crop_min_num*img_list.shape[1]/u_crop_min_den):int(u_crop_max_num*img_list.shape[1]/u_crop_max_den)]
        K = K/downscale_factor

        p, pixel_grid = img_to_3d_torch(img, d, K, depth_threshold=depth_threshold, u_start =int(u_crop_min_num*img_list.shape[1]/u_crop_min_den), v_start = int(v_crop_min_num*img_list.shape[0]/v_crop_min_den))
        normal, inliers, inlier_pixel_grid = fit_plane_torch(p, pixel_grid, ransac_iter=ransac_iter, inlier_thresh=inlier_thresh)
        return p, normal, inliers, inlier_pixel_grid, img_for_plotting

    if load_from_mat == False:
        depth_model, pose_model = depth_model.train(False).eval(), pose_model.train(False).eval()  # Set model to evaluate mode
        fwd_pose_list1, inv_pose_list1 = [], []
        fwd_pose_list2, inv_pose_list2 = [], []
        vo_pose_list = []
        gt_list = []
        normal_list = []
        inlier_list = []
        dist_to_plane = []
        depth_list = []
        img_lists = []
        img_for_plotting = []
        intrinsics_list = []
        plane_pixel_weights = []
        learned_scale_factor_list = []
        dnet_scale_factor_list = []
        
        with torch.no_grad():
            for k, data in enumerate(test_dset_loaders):
                target_img, source_img_list, gt_lie_alg_list, vo_lie_alg_list, flow_imgs, intrinsics, target_img_aug, \
                    source_img_aug_list, gt_lie_alg_aug_list, vo_lie_alg_aug_list, intrinsics_aug = process_sample_batch(data, config)
                pose_results = {'source1': {}, 'source2': {} }

                batch_size = target_img.shape[0]
                imgs = torch.cat([target_img, source_img_list[0]],0)
                disparities = depth_model(imgs, epoch=50)

                target_disparities = [disp[0:batch_size] for disp in disparities]
                source_disp_1 = [disp[batch_size:(2*batch_size)] for disp in disparities]

                disparities = [target_disparities, source_disp_1]         
                depths = [disp_to_depth(disp[0], config['min_depth'], config['max_depth'])[1] for disp in disparities] ####.detach()

                flow_imgs_fwd_list, flow_imgs_back_list = flow_imgs
                poses, poses_inv = solve_pose(pose_model, target_img, source_img_list, flow_imgs)
                fwd_pose_vec1, inv_pose_vec1 = poses[0].clone(), poses_inv[0].clone()

                depth = 30*depths[0]
                fwd_pose_vec1[:,0:3] = 30*fwd_pose_vec1[:,0:3]
                inv_pose_vec1[:,0:3] = 30*inv_pose_vec1[:,0:3]

                if plane_rescaling == True:
                    plane_est = plane_model(target_img, epoch=50)[0].detach()
                    learned_scale_factor = scale_recovery(plane_est, depths[0], intrinsics, h_gt=cam_height/30.)
                    learned_scale_factor_list.append(learned_scale_factor.cpu().numpy())    
                    
                if dnet_rescaling == True:
                     dnet_scale_factor = dgc(depth, intrinsics, cam_height)
                     dnet_scale_factor_list.append(dnet_scale_factor.cpu().numpy())

                fwd_pose_list1.append(fwd_pose_vec1.cpu().detach().numpy())
                inv_pose_list1.append(inv_pose_vec1.cpu().detach().numpy())
                gt_list.append(gt_lie_alg_list[0].cpu().numpy())
                depth_list.append(depth[:,0].cpu().detach()[:, ::downscale_factor, ::downscale_factor])
                intrinsics_list.append(intrinsics.cpu().detach())
                img_lists.append(target_img.permute(0,2,3,1).cpu().detach()[:,::downscale_factor, ::downscale_factor,:])
                img_for_plotting.append(target_img.permute(0,2,3,1).cpu().detach()[:,::downscale_factor, ::downscale_factor,:])
                            

            fwd_pose_list1 = np.concatenate(fwd_pose_list1)
            inv_pose_list1 = np.concatenate(inv_pose_list1)
            gt_list = np.concatenate(gt_list)
            depth_list = torch.cat(depth_list,0)
            intrinsics_list = torch.cat(intrinsics_list,0)
            img_lists = torch.cat(img_lists,0)
            img_for_plotting = torch.cat(img_for_plotting,0)
            if plane_rescaling == True:
                learned_scale_factor_list = np.concatenate(learned_scale_factor_list).reshape((-1,1))
            if dnet_rescaling == True:
                dnet_scale_factor_list = np.concatenate(dnet_scale_factor_list).reshape((-1,1))

        i=0
        if ransac_rescaling==True:
            u_crop_min_num = 1
            u_crop_min_den = 6
            u_crop_max_num = 5
            u_crop_max_den = 6
            v_crop_min_num = 4
            v_crop_min_den = 7
            v_crop_max_num = 1
            v_crop_max_den = 1
            
            with torch.no_grad():  
                for i in range(0,len(img_lists)):  
                    p, normal, inliers, inlier_pixel_grid, plot_img = pf(img_lists[i], depth_list[i], intrinsics_list[i], img_for_plotting[i])
                    n = normal/torch.norm(normal)

                    inlier_points = p[inliers,:]
                    prod = inlier_points.mm(n)
                    dist_to_plane.append(prod.numpy())         
                    normal_list.append(n.numpy())
                    inlier_list.append(len(inliers))

                    ## Plotting
                    if plotting:
                        inlier_u_grid, inlier_v_grid = inlier_pixel_grid
                        inlier_u_grid, inlier_v_grid = inlier_u_grid.numpy(), inlier_v_grid.numpy()      
                        plt.figure()
                        plt.title('Normal {}'.format(n))
                        plt.imshow(plot_img.reshape(plot_img.shape[0], plot_img.shape[1],-1))
                        plt.scatter(inlier_u_grid, inlier_v_grid)
                        plt.savefig('scale_results/plane_imgs/{}/{}.png'.format(config['test_seq'][0], i))
                    i+=1

            normals = np.array(normal_list)[:,:,0]
            num_inliers = np.array(inlier_list)

        data = {'seq': config['test_seq'][0],
                'config': config,
                'ransac_iter': ransac_iter,
                'downscale_factor': downscale_factor,
                'inlier_thresh': inlier_thresh,
                'fwd_pose_vec1': fwd_pose_list1,
                'inv_pose_vec1': inv_pose_list1,
                'gt_pose_vec': gt_list,
                'dist_to_plane': dist_to_plane, 
                'learned_scale_factor': learned_scale_factor_list,    
                'dnet_scale_factor': dnet_scale_factor_list,    
        }
        save_obj(data, '{}/{}_plane_fit'.format(results_dir, config['test_seq'][0]))

    else:
        data = load_obj('{}/{}_plane_fit'.format(results_dir, config['test_seq'][0]))

                      
    dist_to_plane = data['dist_to_plane']
    gt_pose_vec = data['gt_pose_vec']
    unscaled_pose_vec = data['fwd_pose_vec1']
    scaled_pose_vec_ransac = np.array(unscaled_pose_vec)
    
    if ransac_rescaling == True:
        d = [np.median(np.abs(i)) for i in dist_to_plane]
        d  = np.array(d)
        average_d = np.average(d) 

        # print('Variance of ransac scale factor: {}'.format(np.var(cam_height/d)))
        print('ground plane mean scale factor (ransac): {}'.format(cam_height/np.average(d)))
        print('ground plane std. dev. scale factor (ransac): {}'.format(np.std(cam_height/d)))
        scaled_pose_vec_ransac[:,0:3] = scaled_pose_vec_ransac[:,0:3]*np.repeat(cam_height/d.reshape((-1,1)),3,axis=1)
        
    
    if plane_rescaling == True:
        print('ground plane mean scale factor (learned): {}'.format(np.mean(data['learned_scale_factor'])))
        print('ground plane std. dev. scale factor (learned): {}'.format(np.std(data['learned_scale_factor'])))
        scaled_pose_vec_learned = np.array(unscaled_pose_vec)
        scaled_pose_vec_learned[:,0:3] = scaled_pose_vec_learned[:,0:3]*np.repeat(data['learned_scale_factor'],3,axis=1)
    if dnet_rescaling == True:
        print('ground plane scale factor (dnet): {}'.format(np.mean(data['dnet_scale_factor'])))
        print('ground plane std. dev. scale factor (dnet): {}'.format(np.std(data['dnet_scale_factor'])))
        scaled_pose_vec_dnet = np.array(unscaled_pose_vec)
        scaled_pose_vec_dnet[:,0:3] = scaled_pose_vec_dnet[:,0:3]*np.repeat(data['dnet_scale_factor'],3,axis=1)
        


    ## Scale Factor
    gt_norm = np.linalg.norm(gt_pose_vec[:,0:3],axis=1)
    plt.figure()
    plt.grid()
    if ransac_rescaling == True:
        plt.plot((cam_height/d)[plot_range],label='ransac')
    if plane_rescaling == True:
        plt.plot(data['learned_scale_factor'], label='learned')
    plt.ylim([0.6,2.0])
    plt.legend()
    plt.title('Estimated Scale Factor Seq. {}'.format(seq.replace('_','-')))
    plt.savefig('{}/seq-{}-scale-vs-gt.png'.format(results_dir, seq))
    
       
    ## Compute Trajectories
    gt_traj = test_dset_loaders.dataset.raw_gt_trials[0]
    orig_est, gt, errors, cum_dist = tt(unscaled_pose_vec,gt_traj,method='unscaled')
    logger.log(seq, 'unscaled', errors[0], errors[1], errors[2], errors[3])
    if ransac_rescaling == True:
        scaled_est, gt, errors, cum_dist = tt(scaled_pose_vec_ransac,gt_traj, method='scaled (ransac)')
        logger.log(seq, 'ransac scaled', errors[0], errors[1], errors[2], errors[3])
    if plane_rescaling == True:
        scaled_est_learned, _, errors, _ = tt(scaled_pose_vec_learned,gt_traj, method='scaled (learned)')
        logger.log(seq, 'plane scaled', errors[0], errors[1], errors[2], errors[3])
    if dnet_rescaling == True:
        _, _, errors, _ = tt(scaled_pose_vec_dnet,gt_traj, method='scaled (dnet)')
        logger.log(seq, 'dnet scaled', errors[0], errors[1], errors[2], errors[3])
    logger.log('', '', '', '', '', '')
    
    
    ## Plot trajectories
    plt.figure()
    plt.grid()
    plt.plot(gt[:,0,3], gt[:,2,3], linewidth=1.5, color='black', label='gt')
    plt.plot(orig_est[:,0,3],orig_est[:,2,3], linewidth=1.5, linestyle='--', label='est')
    if ransac_rescaling == True:
        plt.plot(scaled_est[:,0,3],scaled_est[:,2,3], linewidth=1.5, linestyle='--', label='rescaled est')
    if plane_rescaling == True:
        plt.plot(scaled_est_learned[:,0,3],scaled_est_learned[:,2,3], linewidth=1.5, linestyle='--', label='rescaled est')
    plt.legend()
    plt.title('Topdown (XY) Trajectory Seq. {}'.format(seq.replace('_','-')))
    plt.savefig('{}/seq-{}-topdown-scaled.png'.format(results_dir, seq))

