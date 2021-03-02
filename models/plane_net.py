import torch
import torch.nn.functional as F
import torch.nn as nn
from models.depth_and_egomotion import *

def scale_recovery(plane_est, depth, intrinsics, h_gt=1.70):
    plane_est = plane_est**3      
    b, _, h, w = depth.size()
    plane_est_down = torch.nn.functional.interpolate(plane_est.clone(),(int(h/4),int(w/4)),mode='bilinear') 
    depth_down = torch.nn.functional.interpolate(depth, (int(h/4),int(w/4)),mode='bilinear')
    int_inv = intrinsics.clone()
    int_inv[:,0:2,:] = int_inv[:,0:2,:]/4
    int_inv = int_inv.inverse()
    b, _, h, w = depth_down.size()
        
    i_range = torch.arange(0, h).view(1, h, 1).expand(1,h,w).type_as(depth_down)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1,h,w).type_as(depth_down)  # [1, H, W]
    ones = torch.ones(1,h,w).type_as(depth_down)
    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]
        ###pixel_coords is an array of camera pixel coordinates (x,y,1) where x,y origin is the upper left corner of the image.
    current_pixel_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w).view(b,3,-1) #.contiguous().view(b, 3, -1)  # [B, 3, H*W]

    cam_coords = int_inv.bmm(current_pixel_coords).view(b,3,h,w)
    cam_coords = cam_coords*depth_down
    cam_coords = cam_coords.reshape((b,3,-1)).permute(0,2,1) ## b, N, 3
    plane_est_down = plane_est_down.view(b,-1) ##b, N

    ## Weighted Least Squares
    W = torch.diag_embed(plane_est_down).type_as(plane_est_down)
    h = torch.ones(b,h*w,1).type_as(plane_est_down)

    left = (h.permute(0,2,1)).bmm(W).bmm(cam_coords)
    right = torch.pinverse((cam_coords.permute(0,2,1)).bmm(W).bmm(cam_coords))
    normal = (left.bmm(right)).permute(0,2,1)

    n = normal/( torch.norm(normal,dim=1).reshape(b,1,1).expand_as(normal) ) ## b,3,1 

    heights = cam_coords.bmm(n) ## b, N, 1
    height = ( (plane_est_down * heights[:,:,0]).sum(dim=1) )/(plane_est_down.sum(dim=1))

    scale_factor = ((h_gt)/height ).detach() ## scale factor is 1 if height is proper, smaller than 1 if height is too short
    # print(scale_factor)
    return scale_factor

class PlaneModel(nn.Module):
    def __init__(self, config): 
        super(PlaneModel, self).__init__()
        num_img_channels = 3
        self.num_scales = config['num_scales']
        self.nb_ref_imgs=1

        ## Encoder Layers
        self.encoder = ResnetEncoder(18,True)
        ## Upsampling
        upconv_planes = [256, 128, 64, 64,32]
        upconv_planes2 = [512]+ upconv_planes
        self.depth_upconvs = nn.ModuleList([self.upconv(upconv_planes2[i],upconv_planes2[i+1]) for i in range(0,len(upconv_planes))]) 
        self.iconvs = nn.ModuleList([self.conv(upconv_planes[i], upconv_planes[i]) for i in range(0,len(upconv_planes))])   
        self.predict_masks = nn.ModuleList([self.predict_mask(upconv_planes[i]) for i in range(len(upconv_planes)-4, len(upconv_planes))] ) 
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, epoch=0):
        x = (x - 0.45) / 0.22
        skips = self.encoder(x)

        ''' Depth UpSampling (depth upconv, and then out_iconv)'''
        out_iconvs = [skips[-1]]
        plane_est = []

        ## No disparity prediction for first 4 layers
        for i in range(0,len(self.iconvs)-1):
            if i > ( len(self.iconvs)-self.num_scales ):
                plane_est.append(self.predict_masks[i-2](out_iconvs[-1]))    
            upconv = self.depth_upconvs[i](out_iconvs[-1])
            upconv = upconv + skips[-(i+2)]

            out_iconvs.append( self.iconvs[i](upconv) )

        ## Final Block
        upconv = self.depth_upconvs[-1](out_iconvs[-1]) #final layer is different, so is out of loop
        plane_est.append(self.predict_masks[-2](out_iconvs[-1]))
        out_iconv = self.iconvs[-1](upconv)
        plane_est.append(self.predict_masks[-1](out_iconv))
        plane_est.reverse()

        return list(plane_est[0:self.num_scales])
   
        
    def upconv(self, in_planes, out_planes, kernel_size=3):
        return nn.Sequential(
            Interpolate(scale_factor=2, mode ='nearest'),
            Conv3x3(in_planes,  out_planes,  use_refl=True, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.ELU(inplace=True)

        )        
        
    def predict_mask(self, in_planes, kernel_size=3):
        return nn.Sequential(
            Conv3x3(in_planes, 1, use_refl=True, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.Sigmoid(),
        )

    def conv(self, in_planes, out_planes, kernel_size=3):
        return nn.Sequential(
            Conv3x3(in_planes, out_planes, use_refl=True, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.ELU(inplace=True),
        )