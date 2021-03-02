
from __future__ import absolute_import, division, print_function
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
from torch.nn import init
import numpy as np
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

'''
General Functions

Many of these have been adapted from https://github.com/nianticlabs/monodepth2/tree/master/networks
'''


class Conv3x3(nn.Module):
    """Layer to pad and convolve input from Monodepth2
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_refl=False, padding=1):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(padding)
            # self.pad = torch.nn.ReplicationPad1d(padding)
        else:
            self.pad = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=kernel_size, stride=stride,padding=0)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, x):
        # B,C,H,W = ref.size()
        x = self.interp(x, scale_factor=2, mode=self.mode)
        return x


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1, img_channels=3):
        super(ResNetMultiImageInput, self).__init__(block, layers, )
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * img_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1, img_channels=3):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images, img_channels=img_channels)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight'][:,0:img_channels]] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1, img_channels=3):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images, img_channels)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = input_image
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features

class depth_model(nn.Module):
    def __init__(self, config, nb_ref_imgs=1): 
        super(depth_model, self).__init__()
        num_img_channels = 3
        self.num_scales = config['num_scales']
        self.nb_ref_imgs=nb_ref_imgs

        ## Encoder Layers
        self.encoder = ResnetEncoder(18,True)
        ## Upsampling
        upconv_planes = [256, 128, 64, 64,32]
        upconv_planes2 = [512]+ upconv_planes
        self.depth_upconvs = nn.ModuleList([self.upconv(upconv_planes2[i],upconv_planes2[i+1]) for i in range(0,len(upconv_planes))]) 
        self.iconvs = nn.ModuleList([self.conv(upconv_planes[i], upconv_planes[i]) for i in range(0,len(upconv_planes))])   
        
        disp_feature_sizes = list(np.cumsum(self.num_scales*[8]))
        self.feature_convs = nn.ModuleList([self.conv(s,8) for s in upconv_planes[-self.num_scales:]])
        self.predict_disps = nn.ModuleList([self.predict_disp(disp_feature_sizes[i]) for i in range(0,self.num_scales)] ) 


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, x, epoch=0):

        x = (x - 0.45) / 0.22
        skips = self.encoder(x)
        
        ''' Depth UpSampling (depth upconv, and then out_iconv)'''
        out_iconvs = [skips[-1]]
        disps = []

        ## test
        depth_features = []
        for i in range(0,len(self.iconvs)-1):
            depth_features.append(out_iconvs[-1])             
            upconv = self.depth_upconvs[i](out_iconvs[-1])
            upconv = upconv + skips[-(i+2)]
            out_iconvs.append( self.iconvs[i](upconv) )

        depth_features.append(out_iconvs[-1])
        upconv = self.depth_upconvs[-1](out_iconvs[-1]) #final layer is different, so is out of loop
        out_iconv = self.iconvs[-1](upconv)
        depth_features.append(out_iconv)

        depth_features = depth_features[-self.num_scales:]
        
        #reduce # channels to 8 before merging
        for i in range(0, self.num_scales):
            depth_features[i] = self.feature_convs[i](depth_features[i])

        

        #merge features from all scales for depth prediction (upsize smaller scales before merging)
        concat_depth_features = []
        concat_depth_features.append(depth_features[-self.num_scales])

        for i in np.arange(self.num_scales-1, 0, -1):
            upsized = []
            _, _, h, w = depth_features[-i].size()
            
            for j in range(0, self.num_scales - i):
                upsized.append(nn.functional.interpolate(depth_features[j], (h, w), mode='nearest') )
            upsized.append(depth_features[-i])
            concat_depth_features.append(torch.cat(upsized,1))

        for i in np.arange(self.num_scales,0,-1):  
            disps.append(self.predict_disps[-i](concat_depth_features[-i]))
            
        disps.reverse()
        return list(disps[0:self.num_scales])

    
    def predict_disp(self, in_planes, kernel_size=3):
        return nn.Sequential(
            Conv3x3(in_planes, self.nb_ref_imgs, use_refl=True, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            # nn.Conv2d(in_planes, self.nb_ref_imgs, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            # nn.ReLU()
            nn.Sigmoid()
        )
        
    def upconv(self, in_planes, out_planes, kernel_size=3):
        return nn.Sequential(
            Interpolate(scale_factor=2, mode ='nearest'),
            Conv3x3(in_planes,  out_planes,  use_refl=True, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.ELU(inplace=True)
        )        


    def conv(self, in_planes, out_planes, kernel_size=3):
        return nn.Sequential(
            Conv3x3(in_planes, out_planes, use_refl=True, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.ELU(inplace=True),
        )

    
def conv(in_planes, out_planes, kernel_size=3, stride=2, padding=1, dilation=1, bn_layer=False, bias=True):
    if bn_layer:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    else: 
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
            nn.ReLU(inplace=True)
        )

def linear(in_planes, out_planes):
    return nn.Sequential(
        nn.Linear(in_planes, out_planes), 
        nn.ReLU(inplace=True)
        )

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = conv(inplanes, planes, 3, stride, pad, dilation, bn_layer=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        out += x

        return F.relu(out, inplace=True)

class ImgEncoder(nn.Module):
    def __init__(self, config):
        super(ImgEncoder, self).__init__()
        self.config = config
        if self.config['flow_type'] == 'classical':
            inputnum = 8
        else:
            inputnum = 6
        blocknums = [2,2,3,4,6,7,3]
        self.outputnums = [32,64,64,128,128,256,256]

        self.firstconv = nn.Sequential(conv(inputnum, 32, 3, 2, 1, 1, False),
                                       conv(32, 32, 3, 1, 1, 1),
                                       conv(32, 32, 3, 1, 1, 1))

        self.inplanes = 32

        self.layer1 = self._make_layer(BasicBlock, self.outputnums[2], blocknums[2], 2, 1, 1) # 40 x 28
        self.layer2 = self._make_layer(BasicBlock, self.outputnums[3], blocknums[3], 2, 1, 1) # 20 x 14
        self.layer3 = self._make_layer(BasicBlock, self.outputnums[4], blocknums[4], 2, 1, 1) # 10 x 7
        self.layer4 = self._make_layer(BasicBlock, self.outputnums[5], blocknums[5], 2, 1, 1) # 5 x 4
        self.layer5 = self._make_layer(BasicBlock, self.outputnums[6], blocknums[6], 2, 1, 1) # 3 x 2



    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.shape[0], -1)

        return x

class pose_model(nn.Module):
    def __init__(self, config):
        super(pose_model, self).__init__()
        self.config = config
        self.img_encoder = ImgEncoder(config)
        
        if config['img_resolution'] == 'med':
            fcnum = self.img_encoder.outputnums[6] * 30
        if config['img_resolution'] == 'low':
            fcnum = self.img_encoder.outputnums[6] * 14

        fc1_trans = linear(fcnum, 128)
        fc2_trans = linear(128,32)
        fc3_trans = nn.Linear(32,3)

        fc1_rot = linear(fcnum, 128)
        fc2_rot = linear(128,32)
        fc3_rot = nn.Linear(32,3)
        self.trans = nn.Sequential(fc1_trans, fc2_trans, fc3_trans)
        self.rot = nn.Sequential(fc1_rot, fc2_rot, fc3_rot)


        # self.voflow_trans = nn.ModuleList([nn.Sequential(fc1_trans, fc2_trans, fc3_trans) for i in range(0,num_heads)])
        # self.voflow_rot = nn.ModuleList([nn.Sequential(fc1_rot, fc2_rot, fc3_rot) for i in range(0, num_heads)])        
        

    def forward(self, imgs, flow=None, T21=None, epoch=None):
        if self.config['flow_type'] == 'none':
            imgs = imgs[0:2] # get rid third img (the optical flow image) if not wanted
        imgs = torch.cat(imgs,1)
        imgs = (imgs - 0.45)/0.22

        features = self.img_encoder( imgs )  #

        x_trans = self.trans(features)
        x_rot = self.rot(features)
        pose = torch.cat((0.01*x_trans, x_rot),dim=1)   

        return pose