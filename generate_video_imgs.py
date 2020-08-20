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

path_to_ws = '/home/brandon-wagstaff/learned_scale_recovery/'
path_to_dset_downsized = '/media/m2-drive/datasets/KITTI-odometry-downsized/'
dir = 'results/202007111233-kitti-scaled-good'


def save_img(j, img, disp, plane, plane_overlay, seq):
    if img is not None:
        # plot_img_array(img, nrow=1, save_file = '{}{}img_seq_{}_img_{}.png'.format(figures_output_dir, '/imgs/', seq, j))
        save_image(img, '{}{}img_seq_{}_img_{}.png'.format(figures_output_dir, '/imgs/', seq, j), nrow=1)
    if disp is not None:
        # plot_disp(disp, save_file = '{}{}depth_seq_{}_img_{}.png'.format(figures_output_dir, '/depth/', seq, j))
        save_image(disp, '{}{}depth_seq_{}_img_{}.png'.format(figures_output_dir, '/depth/', seq, j), normalize=True)
    if plane is not None:
        # plot_img_array(plane, nrow=1, save_file= '{}{}plane_seq_{}_img_{}.png'.format(figures_output_dir, '/plane/', seq,j))
        save_image(plane,'{}{}plane_seq_{}_img_{}.png'.format(figures_output_dir, '/plane/', seq,j))
    if plane_overlay is not None:
        # plot_img_array(plane_overlay, nrow=1, save_file= '{}{}plane_overlay_seq_{}_img_{}.png'.format(figures_output_dir, '/plane_overlay/', seq,j))
        save_image(plane_overlay, '{}{}plane_overlay_seq_{}_img_{}.png'.format(figures_output_dir, '/plane_overlay/', seq,j))
    return '{}{}{}.png'.format(figures_output_dir, '/imgs/', j)

seq_list = ['09']
test_seq = '09'
val_seq = '00'
plot_imgs = True
plot_depth = True
plot_plane  = True
plot_plane_overlay = True

for seq in seq_list:
    config = load_obj('{}/config'.format(dir))
    config['load_pretrained'] = True
    pretrained_depth_path = glob.glob('{}/**depth**best-loss-val_seq-**-test_seq-**.pth'.format(dir))[0]
    pretrained_pose_path = glob.glob('{}/**pose**best-loss-val_seq-**-test_seq-**.pth'.format(dir))[0]

    config['augment_motion'] = False
    config['augment_backwards'] = False
    config['test_seq'] = [seq]
    config['minibatch'] = 5
    
    config['data_dir'] = path_to_dset_downsized+config['img_resolution'] + '_res/'
    
    ## uncomment for evaluating KITTI model on oxford
    # config['estimator'] = 'stereo'

    ## Uncomment for evaluating oxford model on KITTI
    # config['estimator'] = 'orbslam'
    
    
    test_dset_loaders, models, device = data_and_model_loader(config, pretrained_depth_path, pretrained_pose_path, seq=seq)
    if plot_plane or plot_plane_overlay:
        from models.plane_net import PlaneModel
        plane_model = PlaneModel(config).to(device)
        pretrained_plane_path = glob.glob('{}/**plane**.pth'.format(config['pretrained_plane_dir']))[0]
        plane_model.load_state_dict(torch.load(pretrained_plane_path))   
    
    
    eval_dsets = {'test': test_dset_loaders}
    output_dir = 'video/{}/'.format(seq)

    figures_output_dir = '{}figs'.format(output_dir)
    os.makedirs(figures_output_dir+'/imgs', exist_ok=True)
    os.makedirs(figures_output_dir+'/depth', exist_ok=True)
    os.makedirs(figures_output_dir+'/plane', exist_ok=True)
    os.makedirs(figures_output_dir+'/plane_overlay', exist_ok=True)
    
    j = 0
    with torch.set_grad_enabled(False):
        for key, dset in eval_dsets.items():
            ###plot images, depth map, explainability mask
            print("plotting images")

            for data in dset:
                img_array, disparity = test_depth_and_reconstruction(device, models, data, config)
                if plot_plane or plot_plane_overlay:
                    plane_imgs = get_plane_masks(device, plane_model, data, config)

                plane_list = [] 
                plane_overlay = []   
                img_list = []
                img_arrays = []
                disp_list = []
                seq_list = []
                for i in range(0, img_array.shape[0]):
                    if plot_imgs:
                        img_list.append(img_array[i,-1])
                    else:
                        img_list.append(None)
                    
                    if plot_depth:
                        disp_list.append(disparity[i].clamp(0,0.98))
                    else:
                        disp_list.append(None)
                        
                    if plot_plane_overlay:
                        overlay = torch.zeros(img_array[i,-1].size())
                        overlay[0] = (plane_imgs[i]**3).clone()
                        plane_idx = overlay > 0.8
                        print('plane idx', plane_idx.size())
                        plane_img = img_array[i,-1].clone()
                        plane_img[plane_idx] = 0.6*overlay[plane_idx] + 0.4*img_array[i,-1][plane_idx]
                        plane_overlay.append(plane_img)
                    else:
                        plane_overlay.append(None)

                    if plot_plane:
                        plane_list.append(plane_imgs[i])
                    else:
                        plane_list.append(None)
                        
                    seq_list.append(seq)
                
                with concurrent.futures.ProcessPoolExecutor() as executor: 
                    for output in zip(executor.map(save_img, range(j, j+config['minibatch']), img_list, disp_list, plane_list, plane_overlay, seq_list)):
                        print(output)
                j = j+config['minibatch']
            
                # if j > 50:
                #     break
                # for i in range(0,img_array.shape[0]):
 
