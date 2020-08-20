import time
import torch
import sys
sys.path.insert(0,'..')
from utils.learning_helpers import *
from utils.learning_helpers import disp_to_depth

class Plain_Trainer():
    def __init__(self, config, device, models, optimizer):
        self.config = config
        self.device = device
        self.depth_model = models[0]
        self.plane_model = models[1]
        self.optimizer = optimizer
        
    def forward(self, dset, epoch, phase):
        dev = self.device
        start = time.time()
        self.depth_model.train(False)
        self.depth_model.eval()
        
        if phase == 'train':
            self.plane_model.train(True)
        else:
            self.plane_model.train(False)
            self.plane_model.eval()

        dset_size = dset.dataset.__len__()
        running_loss = 0.0           
            # Iterate over data.
        for data in dset:
            target_img, source_imgs, lie_alg, intrinsics, _ = data
            target_img_aug = target_img['color_aug_left'].to(dev)
            lie_alg = lie_alg['color_aug']

            source_img_list = []
            source_img_aug_list = []
            gt_lie_alg_list = []
            vo_lie_alg_list = []
            dt_list = []
            for i, im, in enumerate(source_imgs['color_aug_left']):
                source_img_aug_list.append(im.to(dev))
                source_img_list.append(source_imgs['color_left'][i].to(dev))
                gt_lie_alg_list.append(lie_alg[i][0].type(torch.FloatTensor).to(dev))
                vo_lie_alg_list.append(lie_alg[i][1].type(torch.FloatTensor).to(dev))
                dt_list.append(lie_alg[i][3].type(torch.FloatTensor).to(dev).expand_as(vo_lie_alg_list[-1][:,0:3]))

            intrinsics_aug = intrinsics['color_aug_left'].type(torch.FloatTensor).to(dev)[:,0,:,:] #only need one matrix since it's constant across the training sample
            intrinsics = intrinsics['color_left'].type(torch.FloatTensor).to(dev)[:,0,:,:]

            
            disparity = self.depth_model(target_img_aug, epoch=epoch)
            _,depth = disp_to_depth(disparity[0], self.config['min_depth'], self.config['max_depth'])
            self.optimizer.zero_grad()
            minibatch_loss = 0
            with torch.set_grad_enabled(phase == 'train'):
                plane_est = self.plane_model(target_img_aug, epoch=epoch)
                plane_est = plane_est[0]

                ones_var = torch.ones(1).expand_as(plane_est).type_as(plane_est)
                reg_loss = torch.nn.functional.binary_cross_entropy(plane_est, ones_var,reduction='none')  

                minibatch_loss = 0.05*reg_loss.mean() + 25*self.plane_loss(plane_est, depth, intrinsics_aug.inverse())
                #higher weight on plane loss makes plane estimate more conservative
                if phase == 'train':
                    minibatch_loss.backward()        
                    self.optimizer.step()
            running_loss += minibatch_loss.item()
            

        epoch_loss = running_loss / float(dset_size)

        print('{} Loss: {:.6f}'.format(phase, epoch_loss))
        print('{} epoch completed in {} seconds.'.format(phase, timeSince(start)))
        return epoch_loss

    def plane_loss(self, plane_est, depth, intrinsics_inv):
        b, _, h, w = depth.size()
        i_range = torch.arange(0, h).view(1, h, 1).expand(1,h,w).type_as(depth)  # [1, H, W]
        j_range = torch.arange(0, w).view(1, 1, w).expand(1,h,w).type_as(depth)  # [1, H, W]
        ones = torch.ones(1,h,w).type_as(depth)
        pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]
            ###pixel_coords is an array of camera pixel coordinates (x,y,1) where x,y origin is the upper left corner of the image.
        current_pixel_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w).view(b,3,-1) #.contiguous().view(b, 3, -1)  # [B, 3, H*W]

        cam_coords = intrinsics_inv.bmm(current_pixel_coords).view(b,3,h,w)
        cam_coords = cam_coords*depth
        gp_coords = cam_coords[:,:,int(6.*h/7.):,int(4.5*w/10.):int(5.5*w/10.)].clone()
        cam_coords = cam_coords.reshape(b,3,-1).permute(0,2,1)
        gp_coords = gp_coords.reshape(b,3,-1).permute(0,2,1)
        
        plane_est = plane_est.reshape(b,-1,1) #.expand_as(cam_coords)
        
        ones = torch.ones((b,gp_coords.size(1),1)).type_as(gp_coords)
        computed_normal = torch.pinverse(gp_coords).bmm(ones)
        normal = computed_normal/( torch.norm(computed_normal,dim=1).view(b,1,1).expand_as(computed_normal) )

        heights = (cam_coords.bmm(normal)) #.abs() #no abs to get rid of 'upper' pixels
        gp_height = gp_coords.bmm(normal).mean(1) #.abs()
        height_loss = plane_est*( (heights-gp_height.unsqueeze(1).expand_as(heights)).abs() )

        return height_loss.mean() 
