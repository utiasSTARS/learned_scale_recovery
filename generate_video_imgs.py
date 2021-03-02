import torch
from data.kitti_loader import KittiLoaderPytorch
from video.validate import test_depth_and_reconstruction, get_plane_masks
import models.stn as stn
from utils.learning_helpers import *
from utils.custom_transforms import *
import os 
import glob
from vis import *
import concurrent.futures
from torchvision.utils import save_image 

path_to_ws = '/home/brandonwagstaff/learned_scale_recovery/'
path_to_dset_downsized = '/media/datasets/KITTI-odometry-downsized-stereo/'
dir = 'results/final_models/vo-kitti-scaled-202102182020' #'results/202007111233-kitti-scaled-good'
plane_dir = 'results/plane-model-kitti-202101072240'

seq_list = ['10_02']

plot_imgs = True
plot_img_arrays = False
plot_depth = True
plot_plane  = True
plot_plane_overlay = True
plot_masks = True
plot_gradient = False
plot_reconstruction_errors = True

def save_imgs(j, img, disp, depth_mask, valid_mask, plane, img_array, gradient, diff_img, plane_overlay, seq):
    if img is not None:
        save_image(img, '{}{}img_seq_{}_img_{}.png'.format(figures_output_dir, '/imgs/', seq, j), nrow=1)

    if disp is not None:
        save_image(disp, '{}{}depth_seq_{}_img_{}.png'.format(figures_output_dir, '/depth/', seq, j), normalize=True)

    if depth_mask is not None:
        save_image(depth_mask, '{}{}depth_mask_seq_{}_img_{}.png'.format(figures_output_dir, '/depth_mask/', seq, j))

    if valid_mask is not None:
        save_image(valid_mask, '{}{}valid_mask_seq_{}_img_{}.png'.format(figures_output_dir, '/valid_mask/', seq, j))

    if plane is not None:
        save_image(plane,'{}{}plane_seq_{}_img_{}.png'.format(figures_output_dir, '/plane/', seq,j))

    if plane_overlay is not None:
        save_image(plane_overlay, '{}{}plane_overlay_seq_{}_img_{}.png'.format(figures_output_dir, '/plane_overlay/', seq,j))

    if diff_img is not None:
        save_image(diff_img, '{}{}diff_seq_{}_img_{}.png'.format(figures_output_dir, '/reconstruction_error/', seq,j))

    if img_array is not None:
       save_image(img_array, '{}{}img_array_seq_{}_img_{}.png'.format(figures_output_dir, '/combined/', seq,j))

    if gradient is not None:
       save_image(gradient, '{}{}gradient_seq_{}_img_{}.png'.format(figures_output_dir, '/gradient/', seq,j))

    return '{}{}{}.png'.format(figures_output_dir, '/imgs/', j)

for seq in seq_list:
    config = load_obj('{}/config'.format(dir))
    config['load_pretrained'] = True
    pretrained_depth_path = glob.glob('{}/**depth**best-loss-val_seq-**-test_seq-**.pth'.format(dir))[0]
    pretrained_pose_path = glob.glob('{}/**pose**best-loss-val_seq-**-test_seq-**.pth'.format(dir))[0]

    config['augment_motion'] = False
    config['augment_backwards'] = False
    config['test_seq'] = [seq]
    config['minibatch'] = 5
    device=config['device']
    config['data_dir'] = path_to_dset_downsized+config['img_resolution'] + '_res/'
    
    ### dataset and model loading    
    from data.kitti_loader_stereo import KittiLoaderPytorch
    test_dset = KittiLoaderPytorch(config, [[seq], [seq], [seq]], mode='test', transform_img=get_data_transforms(config)['test'])
    test_dset_loaders = torch.utils.data.DataLoader(test_dset, batch_size=config['minibatch'], shuffle=False, num_workers=6)

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
       

    ### Plane Model
    from models.plane_net import PlaneModel, scale_recovery
    from losses import Plane_Height_loss
    plane_loss = Plane_Height_loss(config)
    plane_model = PlaneModel(config).to(config['device'])
    pretrained_plane_path = glob.glob('{}/**plane**.pth'.format(plane_dir))[0]
    plane_model.load_state_dict(torch.load(pretrained_plane_path))
    plane_model.train(False).eval()    
    
    models = [depth_model, pose_model] 
    eval_dsets = {'test': test_dset_loaders}
    output_dir = 'video/{}/'.format(seq)

    figures_output_dir = '{}figs'.format(output_dir)
    # os.makedirs(figures_output_dir,exist_ok=True)
    os.makedirs(figures_output_dir+'/imgs', exist_ok=True)
    os.makedirs(figures_output_dir+'/depth', exist_ok=True)
    os.makedirs(figures_output_dir+'/exp_mask', exist_ok=True)
    os.makedirs(figures_output_dir+'/depth_mask', exist_ok=True)
    os.makedirs(figures_output_dir+'/valid_mask', exist_ok=True)
    os.makedirs(figures_output_dir+'/combined', exist_ok=True)
    os.makedirs(figures_output_dir+'/plane', exist_ok=True)
    os.makedirs(figures_output_dir+'/plane_overlay', exist_ok=True)
    os.makedirs(figures_output_dir+'/gradient', exist_ok=True)
    os.makedirs(figures_output_dir+'/reconstruction_error', exist_ok=True)
    
    j = 0
    with torch.set_grad_enabled(False):
        for key, dset in eval_dsets.items():
            ###plot images, depth map, explainability mask
            print("plotting images")

            for data in dset:
                img_array, disparity, depth_mask, valid_mask, diff_imgs = test_depth_and_reconstruction(device, models, data, config)
                if plot_plane or plot_plane_overlay:
                    plane_imgs = get_plane_masks(device, plane_model, data, config)

                if plot_gradient:
                    grads = compute_gradient_mask(img_array[:,:,0])

                plane_list = [] 
                plane_overlay = []   
                img_list = []
                img_arrays = []
                disp_list = []
                depth_mask_list = []
                valid_mask_list = []
                gradients = []
                diff_img_list = []
                seq_list = []
                for i in range(0, img_array.shape[0]):
                    
                    # else:
                    #     mask_list.append(None)
                    if plot_imgs:
                        img_list.append(img_array[i,-1])
                    else:
                        img_list.append(None)
                        
                    # save_image(img_list[0], '{}{}img_seq_{}_img_{}.png'.format(figures_output_dir, '/imgs/', seq, '0'), nrow=1)
                    
                    if plot_img_arrays:
                        img_arrays.append(img_array[i])
                    else:
                        img_arrays.append(None)
                        
                    if plot_depth:
                        disp_list.append(disparity[i].clamp(0,0.98))
                    else:
                        disp_list.append(None)
                        
                    if plot_plane_overlay:
                        overlay = torch.zeros(img_array[i,-1].size())
                        overlay[0] = (plane_imgs[i]**3).clone()
                        plane_idx = overlay > 0.8
                        # print('plane idx', plane_idx.size())
                        plane_img = img_array[i,-1].clone()
                        plane_img[plane_idx] = 0.6*overlay[plane_idx] + 0.4*img_array[i,-1][plane_idx]
                        plane_overlay.append(plane_img)
                    else:
                        plane_overlay.append(None)

                    if plot_plane:
                        plane_list.append(plane_imgs[i])
                    else:
                        plane_list.append(None)
                    
                    if plot_gradient:
                        gradients.append(grads[i])
                    else:
                        gradients.append(None)
                        
                    if plot_masks:
                        depth_mask_list.append(depth_mask[i])
                        valid_mask_list.append(valid_mask[i])
                    else:   
                        depth_mask_list.append(None)
                        valid_mask_list.append(None)
                        
                    if plot_reconstruction_errors:
                        diff_img_list.append(diff_imgs[i])
                    else:
                        diff_img_list.append(None)
                        
                    seq_list.append(seq)

                for idx, img, disp, depth_mask, valid_mask, plane, img_array, grad, diff_img, plane, s in zip(range(j, j+config['minibatch']), img_list, disp_list, depth_mask_list, valid_mask_list, plane_list, img_arrays, gradients, diff_img_list, plane_overlay, seq_list):
                    save_imgs(idx, img, disp, depth_mask, valid_mask, plane, img_array, grad, diff_img, plane, s)
                # with concurrent.futures.ProcessPoolExecutor() as executor: 
                #     for output in zip(executor.map(save_imgs, range(j, j+config['minibatch']), img_list, disp_list, depth_mask_list, valid_mask_list, plane_list, img_arrays, gradients, diff_img_list, plane_overlay, seq_list)):
                #         print(output)
                j = j+config['minibatch']
 
