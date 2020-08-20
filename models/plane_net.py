import torch
import torch.nn.functional as F
import torch.nn as nn
from models.depth_and_egomotion import *

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