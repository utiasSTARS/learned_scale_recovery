'''
From https://github.com/TRI-ML/packnet-sfm, modified slightly by adding weightnorm alonside groupnorm
'''

import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
'''
DepthNet
'''

class conv2d_wn(nn.Conv2d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(conv2d_wn, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Conv2D(nn.Module):
    """
    2D convolution with GroupNorm and ELU
    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int
        Kernel size
    stride : int
        Stride
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        # self.conv_base = nn.Conv2d(
        #     in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.conv_base = conv2d_wn(
            in_channels, out_channels, kernel_size, stride=stride)
        self.pad = nn.ConstantPad2d([kernel_size // 2] * 4, value=0)
        self.normalize = torch.nn.GroupNorm(16, out_channels)
        self.activ = nn.ELU(inplace=True)

    def forward(self, x):
        """Runs the Conv2D layer."""
        x = self.conv_base(self.pad(x))
        return self.activ(self.normalize(x))
    
class ResidualConv(nn.Module):
    """2D Convolutional residual block with GroupNorm and ELU"""
    def __init__(self, in_channels, out_channels, stride, dropout=None):
        """
        Initializes a ResidualConv object.
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        stride : int
            Stride
        dropout : float
            Dropout value
        """
        super().__init__()
        self.conv1 = Conv2D(in_channels, out_channels, 3, stride)
        self.conv2 = Conv2D(out_channels, out_channels, 3, 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.normalize = torch.nn.GroupNorm(16, out_channels)
        self.activ = nn.ELU(inplace=True)

        if dropout:
            self.conv3 = nn.Sequential(self.conv3, nn.Dropout2d(dropout))

    def forward(self, x):
        """Runs the ResidualConv layer."""
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        shortcut = self.conv3(x)
        return self.activ(self.normalize(x_out + shortcut))


def ResidualBlock(in_channels, out_channels, num_blocks, stride, dropout=None):
    """
    Returns a ResidualBlock with various ResidualConv layers.
    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    num_blocks : int
        Number of residual blocks
    stride : int
        Stride
    dropout : float
        Dropout value
    """
    layers = [ResidualConv(in_channels, out_channels, stride, dropout=dropout)]
    for i in range(1, num_blocks):
        layers.append(ResidualConv(out_channels, out_channels, 1, dropout=dropout))
    return nn.Sequential(*layers)


class InvDepth(nn.Module):
    """Inverse depth layer"""
    def __init__(self, in_channels, out_channels=1, min_depth=0.5):
        """
        Initializes an InvDepth object.
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        min_depth : float
            Minimum depth value to calculate
        """
        super().__init__()
        self.min_depth = min_depth
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)
        self.pad = nn.ConstantPad2d([1] * 4, value=0)
        self.activ = nn.Sigmoid()

    def forward(self, x):
        """Runs the InvDepth layer."""
        x = self.conv1(self.pad(x))
        return self.activ(x) / self.min_depth

########################################################################################################################

def packing(x, r=2):
    """
    Takes a [B,C,H,W] tensor and returns a [B,(r^2)C,H/r,W/r] tensor, by concatenating
    neighbor spatial pixels as extra channels. It is the inverse of nn.PixelShuffle
    (if you apply both sequentially you should get the same tensor)
    Parameters
    ----------
    x : torch.Tensor [B,C,H,W]
        Input tensor
    r : int
        Packing ratio
    Returns
    -------
    out : torch.Tensor [B,(r^2)C,H/r,W/r]
        Packed tensor
    """
    b, c, h, w = x.shape
    out_channel = c * (r ** 2)
    out_h, out_w = h // r, w // r
    x = x.contiguous().view(b, c, out_h, r, out_w, r)
    return x.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w)

########################################################################################################################

class PackLayerConv2d(nn.Module):
    """
    Packing layer with 2d convolutions. Takes a [B,C,H,W] tensor, packs it
    into [B,(r^2)C,H/r,W/r] and then convolves it to produce [B,C,H/r,W/r].
    """
    def __init__(self, in_channels, kernel_size, r=2):
        """
        Initializes a PackLayerConv2d object.
        Parameters
        ----------
        in_channels : int
            Number of input channels
        kernel_size : int
            Kernel size
        r : int
            Packing ratio
        """
        super().__init__()
        self.conv = Conv2D(in_channels * (r ** 2), in_channels, kernel_size, 1)
        self.pack = partial(packing, r=r)

    def forward(self, x):
        """Runs the PackLayerConv2d layer."""
        x = self.pack(x)
        x = self.conv(x)
        return x


class UnpackLayerConv2d(nn.Module):
    """
    Unpacking layer with 2d convolutions. Takes a [B,C,H,W] tensor, convolves it
    to produce [B,(r^2)C,H,W] and then unpacks it to produce [B,C,rH,rW].
    """
    def __init__(self, in_channels, out_channels, kernel_size, r=2):
        """
        Initializes a UnpackLayerConv2d object.
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            Kernel size
        r : int
            Packing ratio
        """
        super().__init__()
        self.conv = Conv2D(in_channels, out_channels * (r ** 2), kernel_size, 1)
        self.unpack = nn.PixelShuffle(r)

    def forward(self, x):
        """Runs the UnpackLayerConv2d layer."""
        x = self.conv(x)
        x = self.unpack(x)
        return x

########################################################################################################################

class PackLayerConv3d(nn.Module):
    """
    Packing layer with 3d convolutions. Takes a [B,C,H,W] tensor, packs it
    into [B,(r^2)C,H/r,W/r] and then convolves it to produce [B,C,H/r,W/r].
    """
    def __init__(self, in_channels, kernel_size, r=2, d=8):
        """
        Initializes a PackLayerConv3d object.
        Parameters
        ----------
        in_channels : int
            Number of input channels
        kernel_size : int
            Kernel size
        r : int
            Packing ratio
        d : int
            Number of 3D features
        """
        super().__init__()
        self.conv = Conv2D(in_channels * (r ** 2) * d, in_channels, kernel_size, 1)
        self.pack = partial(packing, r=r)
        self.conv3d = nn.Conv3d(1, d, kernel_size=(3, 3, 3),
                                stride=(1, 1, 1), padding=(1, 1, 1))

    def forward(self, x):
        """Runs the PackLayerConv3d layer."""
        x = self.pack(x)
        x = x.unsqueeze(1)
        x = self.conv3d(x)
        b, c, d, h, w = x.shape
        x = x.view(b, c * d, h, w)
        x = self.conv(x)
        return x


class UnpackLayerConv3d(nn.Module):
    """
    Unpacking layer with 3d convolutions. Takes a [B,C,H,W] tensor, convolves it
    to produce [B,(r^2)C,H,W] and then unpacks it to produce [B,C,rH,rW].
    """
    def __init__(self, in_channels, out_channels, kernel_size, r=2, d=8):
        """
        Initializes a UnpackLayerConv3d object.
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            Kernel size
        r : int
            Packing ratio
        d : int
            Number of 3D features
        """
        super().__init__()
        self.conv = Conv2D(in_channels, out_channels * (r ** 2) // d, kernel_size, 1)
        self.unpack = nn.PixelShuffle(r)
        self.conv3d = nn.Conv3d(1, d, kernel_size=(3, 3, 3),
                                stride=(1, 1, 1), padding=(1, 1, 1))

    def forward(self, x):
        """Runs the UnpackLayerConv3d layer."""
        x = self.conv(x)
        x = x.unsqueeze(1)
        x = self.conv3d(x)
        b, c, d, h, w = x.shape
        x = x.view(b, c * d, h, w)
        x = self.unpack(x)
        return x

########################################################################################################################
    
class depth_model(nn.Module):
    """
    PackNet network with 3d convolutions (version 01, from the CVPR paper).
    Slimmer version, with fewer feature channels
    https://arxiv.org/abs/1905.02693
    Parameters
    ----------
    dropout : float
        Dropout value to use
    version : str
        Has a XY format, where:
        X controls upsampling variations (not used at the moment).
        Y controls feature stacking (A for concatenation and B for addition)
    kwargs : dict
        Extra parameters
    """
    def __init__(self, config, dropout=None, version='AA', **kwargs):
        super().__init__()
        self.config = config
        self.num_scales = config['num_scales']
        self.version = version[1:]
        # Input/output channels
        in_channels = 3
        out_channels = 1
        # Hyper-parameters
        ni, no = 32, out_channels
        n1, n2, n3, n4, n5 = 32, 64, 128, 256, 512
        num_blocks = [2, 2, 3, 3]
        pack_kernel = [5, 3, 3, 3, 3]
        unpack_kernel = [3, 3, 3, 3, 3]
        iconv_kernel = [3, 3, 3, 3, 3]
        num_3d_feat = 4
        # Initial convolutional layer
        self.pre_calc = Conv2D(in_channels, ni, 5, 1)
        # Support for different versions
        if self.version == 'A':  # Channel concatenation
            n1o, n1i = n1, n1 + ni + no
            n2o, n2i = n2, n2 + n1 + no
            n3o, n3i = n3, n3 + n2 + no
            n4o, n4i = n4, n4 + n3
            n5o, n5i = n5, n5 + n4
        elif self.version == 'B':  # Channel addition
            n1o, n1i = n1, n1 + no
            n2o, n2i = n2, n2 + no
            n3o, n3i = n3//2, n3//2 + no
            n4o, n4i = n4//2, n4//2
            n5o, n5i = n5//2, n5//2
        else:
            raise ValueError('Unknown PackNet version {}'.format(version))

        # Encoder

        self.pack1 = PackLayerConv3d(n1, pack_kernel[0], d=num_3d_feat)
        self.pack2 = PackLayerConv3d(n2, pack_kernel[1], d=num_3d_feat)
        self.pack3 = PackLayerConv3d(n3, pack_kernel[2], d=num_3d_feat)
        self.pack4 = PackLayerConv3d(n4, pack_kernel[3], d=num_3d_feat)
        self.pack5 = PackLayerConv3d(n5, pack_kernel[4], d=num_3d_feat)

        self.conv1 = Conv2D(ni, n1, 7, 1)
        self.conv2 = ResidualBlock(n1, n2, num_blocks[0], 1, dropout=dropout)
        self.conv3 = ResidualBlock(n2, n3, num_blocks[1], 1, dropout=dropout)
        self.conv4 = ResidualBlock(n3, n4, num_blocks[2], 1, dropout=dropout)
        self.conv5 = ResidualBlock(n4, n5, num_blocks[3], 1, dropout=dropout)

        # Decoder

        self.unpack5 = UnpackLayerConv3d(n5, n5o, unpack_kernel[0], d=num_3d_feat)
        self.unpack4 = UnpackLayerConv3d(n5, n4o, unpack_kernel[1], d=num_3d_feat)
        self.unpack3 = UnpackLayerConv3d(n4, n3o, unpack_kernel[2], d=num_3d_feat)
        self.unpack2 = UnpackLayerConv3d(n3, n2o, unpack_kernel[3], d=num_3d_feat)
        self.unpack1 = UnpackLayerConv3d(n2, n1o, unpack_kernel[4], d=num_3d_feat)

        self.iconv5 = Conv2D(n5i, n5, iconv_kernel[0], 1)
        self.iconv4 = Conv2D(n4i, n4, iconv_kernel[1], 1)
        self.iconv3 = Conv2D(n3i, n3, iconv_kernel[2], 1)
        self.iconv2 = Conv2D(n2i, n2, iconv_kernel[3], 1)
        self.iconv1 = Conv2D(n1i, n1, iconv_kernel[4], 1)

        # Depth Layers

        self.unpack_disps = nn.PixelShuffle(2)
        self.unpack_disp4 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp3 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp2 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)

        self.disp4_layer = InvDepth(n4, out_channels=out_channels)
        self.disp3_layer = InvDepth(n3, out_channels=out_channels)
        self.disp2_layer = InvDepth(n2, out_channels=out_channels)
        self.disp1_layer = InvDepth(n1, out_channels=out_channels)

        self.init_weights()

    def init_weights(self):
        """Initializes network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x,epoch=0):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        x = self.pre_calc(x)

        # Encoder

        x1 = self.conv1(x)
        x1p = self.pack1(x1)
        # print(x1p.shape)
        x2 = self.conv2(x1p)
        x2p = self.pack2(x2)
        x3 = self.conv3(x2p)
        x3p = self.pack3(x3)
        x4 = self.conv4(x3p)
        x4p = self.pack4(x4)
        x5 = self.conv5(x4p)
        x5p = self.pack5(x5)

        # Skips

        skip1 = x
        skip2 = x1p
        skip3 = x2p
        skip4 = x3p
        skip5 = x4p

        # Decoder

        unpack5 = self.unpack5(x5p)
        if self.version == 'A':
            concat5 = torch.cat((unpack5, skip5), 1)
        else:
            concat5 = unpack5 + skip5
        iconv5 = self.iconv5(concat5)

        unpack4 = self.unpack4(iconv5)
        if self.version == 'A':
            concat4 = torch.cat((unpack4, skip4), 1)
        else:
            concat4 = unpack4 + skip4
        iconv4 = self.iconv4(concat4)
        disp4 = self.disp4_layer(iconv4)
        udisp4 = self.unpack_disp4(disp4)

        unpack3 = self.unpack3(iconv4)
        if self.version == 'A':
            concat3 = torch.cat((unpack3, skip3, udisp4), 1)
        else:
            concat3 = torch.cat((unpack3 + skip3, udisp4), 1)
        iconv3 = self.iconv3(concat3)
        disp3 = self.disp3_layer(iconv3)
        udisp3 = self.unpack_disp3(disp3)

        unpack2 = self.unpack2(iconv3)
        if self.version == 'A':
            concat2 = torch.cat((unpack2, skip2, udisp3), 1)
        else:
            concat2 = torch.cat((unpack2 + skip2, udisp3), 1)
        iconv2 = self.iconv2(concat2)
        disp2 = self.disp2_layer(iconv2)
        udisp2 = self.unpack_disp2(disp2)

        unpack1 = self.unpack1(iconv2)
        if self.version == 'A':
            concat1 = torch.cat((unpack1, skip1, udisp2), 1)
        else:
            concat1 = torch.cat((unpack1 +  skip1, udisp2), 1)
        iconv1 = self.iconv1(concat1)
        disp1 = self.disp1_layer(iconv1)

        # if self.training:
        #     return [disp1, disp2, disp3, disp4]
        # else:
        #     return disp1
        disps = [disp1, disp2, disp3, disp4]
        return list(disps[0:self.num_scales])

'''
PoseNet
'''
########################################################################################################################

def conv_gn(in_planes, out_planes, kernel_size=3):
    """
    Convolutional block with GroupNorm
    Parameters
    ----------
    in_planes : int
        Number of input channels
    out_planes : int
        Number of output channels
    kernel_size : int
        Convolutional kernel size
    Returns
    -------
    layers : nn.Sequential
        Sequence of Conv2D + GroupNorm + ReLU
    """
    return nn.Sequential(
        # nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
        #           padding=(kernel_size - 1) // 2, stride=2),
        conv2d_wn(in_planes, out_planes, kernel_size, padding=(kernel_size-1) // 2, stride=2),
        nn.GroupNorm(16, out_planes),
        nn.ReLU(inplace=True)
    )

########################################################################################################################

class pose_model(nn.Module):
    """Pose network """

    def __init__(self, config, epoch=0, nb_ref_imgs=2, rotation_mode='euler', **kwargs):
        super().__init__()
        self.nb_ref_imgs = 1
        self.rotation_mode = rotation_mode

        conv_channels = [16, 32, 64, 128, 256, 256, 256]
        
        self.config = config
        if self.config['flow_type'] == 'classical':
            inputnum = 8
        else:
            inputnum = 6
        
        
        self.conv1 = conv_gn(inputnum, conv_channels[0], kernel_size=7)
        self.conv2 = conv_gn(conv_channels[0], conv_channels[1], kernel_size=5)
        self.conv3 = conv_gn(conv_channels[1], conv_channels[2])
        self.conv4 = conv_gn(conv_channels[2], conv_channels[3])
        self.conv5 = conv_gn(conv_channels[3], conv_channels[4])
        self.conv6 = conv_gn(conv_channels[4], conv_channels[5])
        self.conv7 = conv_gn(conv_channels[5], conv_channels[6])

        self.pose_pred = nn.Conv2d(conv_channels[6], 6 * self.nb_ref_imgs,
                                   kernel_size=1, padding=0)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, imgs, epoch=0, T21=None):
        if self.config['flow_type'] == 'none':
            imgs = imgs[0:2] # get rid third img (the optical flow image) if not wanted
        imgs = torch.cat(imgs,1)
        imgs = (imgs - 0.45)/0.22
        out_conv1 = self.conv1(imgs)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7)
        pose = pose.mean(3).mean(2)
        pose = 0.01 * pose.view(pose.size(0), 6)

        return pose