import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import validate
from train_mono import compute_pose
from plane_fitting import img_to_3d_torch, fit_plane_torch
from utils.learning_helpers import save_obj, load_obj, disp_to_depth, data_and_model_loader
import os
from validate import compute_trajectory as tt
import glob

path_to_ws = '/home/brandon-wagstaff/learned_scale_recovery/' ##update this
path_to_dset_downsized = '/media/m2-drive/datasets/KITTI-odometry-downsized/'
# path_to_dset_downsized = '/media/m2-drive/datasets/oxford-robotcar-downsized/'

load_from_mat = True #Make True to load results rather than recomputing
seq_list = ['05', '09', '10'] #['00', '02', '06', '07', '08', '05', '09', '10']
test_seq = '05'
val_seq = '00'
dir = path_to_ws + 'results/202007111233-kitti-scaled-good'
results_dir = dir + '/results/scale/'
os.makedirs(results_dir, exist_ok=True)
logger = validate.ResultsLogger('{}/metrics.csv'.format(results_dir))
for seq in seq_list:
    print('sequence: {}'.format(seq))
    ransac_iter = 350
    downscale_factor = 4
    depth_threshold = 15
    inlier_thresh = 0.02
    cam_height = 1.70
    plotting =  False
    plot_range =  slice(0,-1)
    
    u_crop_min_num = 1
    u_crop_min_den = 6
    u_crop_max_num = 5
    u_crop_max_den = 6
    v_crop_min_num = 4
    v_crop_min_den = 7
    v_crop_max_num = 1
    v_crop_max_den = 1

    config = load_obj('{}/config'.format(dir))
    print(config)
    pretrained_depth_path = glob.glob('{}/**depth**best-loss-val_seq-**-test_seq-{}**.pth'.format(dir, ''))[0]
    pretrained_pose_path = glob.glob('{}/**pose**best-loss-val_seq-**-test_seq-{}**.pth'.format(dir, ''))[0]
    

    config['data_dir'] = path_to_dset_downsized+config['img_resolution'] + '_res/' #
    
    ## uncomment for KITTI to oxford
    # config['estimator'] = 'orbslam'
    
    
    config['load_stereo'] = False
    config['augment_motion'] = False
    config['augment_backwards'] = False
    config['test_seq'] = [seq]
    config['minibatch'] = 1
    config['load_pretrained'] = True
    dpc = config['dpc']
    mode = config['pose_output_type']
    if dpc:
        prefix = 'dpc'
    else:
        prefix=''
    test_dset_loaders, models, device = data_and_model_loader(config, pretrained_depth_path, pretrained_pose_path, seq=seq)
    depth_model, pose_model = models[0], models[1]
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
        
        with torch.no_grad():
            for k, data in enumerate(test_dset_loaders):
                target_img, source_imgs, lie_alg, intrinsics, flow_imgs  = data
                lie_alg = lie_alg['color']
                target_img, source_imgs, intrinsics = target_img['color_left'], source_imgs['color_left'], intrinsics['color_left']
                target_img = target_img.to(device)
                
                source_img_list = []
                gt_lie_alg_list = []
                vo_lie_alg_list = []
                pose_results = {'source1': {}, 'source2': {} }
                flow_imgs_fwd_list, flow_imgs_back_list = [], []
                for i, im, in enumerate(source_imgs):
                    source_img_list.append(im.to(device))
                    gt_lie_alg_list.append(lie_alg[i][0].type(torch.FloatTensor))
                    vo_lie_alg_list.append(lie_alg[i][1].type(torch.FloatTensor).to(device))

                if config['flow_type'] == 'classical':
                    flow_imgs_fwd, flow_imgs_back = flow_imgs
                    flow_imgs_fwd_list, flow_imgs_back_list = [], []
                    for i in range(0, len(flow_imgs_fwd)):
                        flow_imgs_fwd_list.append(flow_imgs_fwd[i].to(device))
                        flow_imgs_back_list.append(flow_imgs_back[i].to(device))
                    flow_imgs = [flow_imgs_fwd_list, flow_imgs_back_list]
                else:
                    flow_imgs = [[None for i in range(0,len(source_img_list))] for i in range(0,2)] #annoying but necessary
                    flow_imgs_fwd_list = [None for i in range(0,len(source_img_list))]
                    flow_imgs_back_list = [None for i in range(0,len(source_img_list))]

                intrinsics = intrinsics.type(torch.FloatTensor).to(device)[:,0,:,:] #only need one matrix since it's constant across the training sample
                disparity = depth_model(target_img)
                disp = disparity[0]
                _,depth = disp_to_depth(disp, config['min_depth'], config['max_depth'])
                
                poses = [compute_pose(pose_model, [target_img, source_img, flow_img_fwd], vo, dpc, mode, 50) for source_img, vo, flow_img_fwd in zip(source_img_list, vo_lie_alg_list, flow_imgs_fwd_list)]
                poses_inv = [compute_pose(pose_model, [source_img, target_img, flow_img_back], -vo, dpc, mode, 50) for source_img, vo, flow_img_back in zip(source_img_list, vo_lie_alg_list, flow_imgs_back_list)]

                if len(poses) == 2:
                    fwd_pose_vec1, inv_pose_vec1, fwd_pose_vec2, inv_pose_vec2 = poses[0].clone(), poses_inv[0].clone(), poses[1].clone(), poses_inv[1].clone()

                else:
                    fwd_pose_vec1, inv_pose_vec1, fwd_pose_vec2, inv_pose_vec2 = poses[0].clone(), -poses[0].clone(), -poses[0].clone(), poses[0].clone()

                depth = 30*depth
                fwd_pose_vec1[:,0:3] = 30*fwd_pose_vec1[:,0:3]
                inv_pose_vec1[:,0:3] = 30*inv_pose_vec1[:,0:3]
                fwd_pose_vec2[:,0:3] = 30*fwd_pose_vec2[:,0:3]
                inv_pose_vec2[:,0:3] = 30*inv_pose_vec2[:,0:3]

                for s in range(0, fwd_pose_vec1.size(0)):
                    fwd_pose_list1.append(fwd_pose_vec1[s].unsqueeze(0).cpu().detach().numpy())
                    inv_pose_list1.append(inv_pose_vec1[s].unsqueeze(0).cpu().detach().numpy())
                    fwd_pose_list2.append(fwd_pose_vec2[s].unsqueeze(0).cpu().detach().numpy())
                    inv_pose_list2.append(inv_pose_vec2[s].unsqueeze(0).cpu().detach().numpy())
                    vo_pose_list.append(vo_lie_alg_list[0][s].unsqueeze(0).cpu().detach().numpy())

                    gt_list.append(gt_lie_alg_list[0][s].unsqueeze(0).numpy())
                    depth_list.append(depth[s,0].cpu().detach()[::downscale_factor, ::downscale_factor])
                    intrinsics_list.append(intrinsics[0].cpu().detach())
                    img_lists.append(target_img[s].permute(1,2,0).cpu().detach()[::downscale_factor, ::downscale_factor,:])
                    img_for_plotting.append(target_img[s].permute(1,2,0).cpu().detach()[::downscale_factor, ::downscale_factor,:])

        i=0
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

        fwd_pose_vec1 = np.array(fwd_pose_list1)[:,0,:]
        inv_pose_vec1 = np.array(inv_pose_list1)[:,0,:]
        fwd_pose_vec2 = np.array(fwd_pose_list2)[:,0,:]
        inv_pose_vec2 = np.array(inv_pose_list2)[:,0,:]
        gt_pose_vec = np.array(gt_list)[:,0,:]
        vo_pose_vec = np.array(vo_pose_list)[:,0,:]
        normals = np.array(normal_list)[:,:,0]
        num_inliers = np.array(inlier_list)

        data = {'seq': config['test_seq'][0],
                'config': config,
                'ransac_iter': ransac_iter,
                'downscale_factor': downscale_factor,
                'inlier_thresh': inlier_thresh,
                'num_inliers': num_inliers,
                'normal': normals,
                'fwd_pose_vec1': fwd_pose_vec1,
                'fwd_pose_vec2': fwd_pose_vec2,
                'inv_pose_vec1': inv_pose_vec1,
                'inv_pose_vec2': inv_pose_vec2,
                'gt_pose_vec': gt_pose_vec,
                'vo_pose_vec': vo_pose_vec,
                'dist_to_plane': dist_to_plane,     
        }
        save_obj(data, '{}/{}_plane_fit'.format(results_dir, config['test_seq'][0]))

    else:
        data = load_obj('{}/{}_plane_fit'.format(results_dir, config['test_seq'][0]))
        dist_to_plane = data['dist_to_plane']
        fwd_pose_vec1 = data['fwd_pose_vec1']
        fwd_pose_vec2 = data['fwd_pose_vec2']
        inv_pose_vec1 = data['inv_pose_vec1']
        inv_pose_vec2 = data['inv_pose_vec2']
        gt_pose_vec = data['gt_pose_vec']
        vo_pose_vec = data['vo_pose_vec']
        num_inliers = data['num_inliers']
        normals = data['normal']
        if config['dpc'] == False:
            prefix = ''
        if config['dpc'] == True:
            prefix = prefix = 'dpc-'


    d = [np.median(np.abs(i)) for i in dist_to_plane]
    d  = np.array(d)
    average_d = np.average(d) 

    unscaled_pose_vec = fwd_pose_vec1
    # unscaled_pose_vec = -inv_pose_vec1
    # unscaled_pose_vec = (fwd_pose_vec1 - inv_pose_vec1)/2 
    unscaled_pose_vec[:,3:6] = gt_pose_vec[:,3:6]
    
    print('Variance of scale factor: {}'.format(np.var(cam_height/d)))
    print('ground plane scale factor: {}'.format(cam_height/np.average(d)))
    unscaled_pose_vec = unscaled_pose_vec[:,0:6]

    scaled_pose_vec = np.array(unscaled_pose_vec)
    scaled_pose_vec[:,0:3] = scaled_pose_vec[:,0:3]*np.repeat(cam_height/d.reshape((-1,1)),3,axis=1)

    
    ## Scale Factor
    gt_norm = np.linalg.norm(gt_pose_vec[:,0:3],axis=1)
    plt.figure()
    plt.grid()
    plt.plot((cam_height/d)[plot_range])
    plt.ylim([0.6,2.0])
    plt.title('Estimated Scale Factor Seq. {}'.format(config['test_seq'][0].replace('_','-')))
    plt.savefig('{}/{}seq-{}-scale-vs-gt.png'.format(results_dir, prefix, config['test_seq'][0]))
    
       
    ## Compute Trajectories
    gt_traj = test_dset_loaders.dataset.raw_gt_trials[0]
    orig_est, gt, errors, cum_dist = tt(unscaled_pose_vec,gt_traj,method='unscaled')
    logger.log(seq, 'unscaled', errors[0], errors[1], errors[2], errors[3])
    scaled_est, gt, errors, cum_dist = tt(scaled_pose_vec,gt_traj, method='scaled')
    logger.log(seq, 'ground plane scaled', errors[0], errors[1], errors[2], errors[3])
    logger.log('', '', '', '', '', '')
    
    vo_est, _, _, _ = tt(vo_pose_vec, gt_traj, method='vo')
    
    
    ## Plot trajectories
    plt.figure()
    plt.grid()
    plt.plot(gt[:,0,3], gt[:,1,3], linewidth=1.5, color='black', label='gt')
    plt.plot(orig_est[:,0,3],orig_est[:,1,3], linewidth=1.5, linestyle='--', label='est')
    plt.plot(scaled_est[:,0,3],scaled_est[:,1,3], linewidth=1.5, linestyle='--', label='rescaled est')
    # plt.plot(vo_est[:,0,3],vo_est[:,1,3], linewidth=1.5, linestyle='--', label='vo')
    plt.legend()
    plt.title('Topdown (XY) Trajectory Seq. {}'.format(config['test_seq'][0].replace('_','-')))
    plt.savefig('{}/{}seq-{}-topdown-scaled.png'.format(results_dir, prefix, config['test_seq'][0]))
