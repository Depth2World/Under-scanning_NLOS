from turtle import forward
from pickle import NONE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

import numpy as np
import scipy.signal as ssi
import cv2 
from torch.autograd import Variable
import math

from .utils_pytorch import fk_1_10
from .utils_pytorch import lct_1_10
from models.utils_pytorch import phasor_1_10
from models.utils_pytorch import RSDEfficient

from .modules import *


def rodrigues(ctr, rot_vec, world=True):
    """
    Construct world-to-camera transformation using Rodrigues' algorithm.

    Args:
        ctr (float 3-tuple): camera center.
            (NOTE: this is the image center for an orthographic camera.)
        rot_vec (float 3-tuple): rotation vector.
        world (bool): if True, the given camera center is in world frame.
            Otherwise, it is in the frame after rotation.

    Returns:
        Rt (float array, [4, 4]): world-to-camera transformation.
    """
    ctr = np.array(ctr, dtype=float)
    rot_vec = np.array(rot_vec, dtype=float)

    Rt = np.eye(4)
    Rt[:3, :3] = cv2.Rodrigues(rot_vec)[0]
    if world:
        Rt[:3, 3] = np.matmul(Rt[:3, :3], -ctr)
    else:
        Rt[:3, 3] = -ctr
    return Rt

def get_all_views(r=2, n_views=26, ratio=0.25):
    """
    Obtain camera views for multi-view supervision.

    NOTE: The camera principal axes are drawn using the Fibonacci spiral 
    algorithm. The implementation follows Chen et al., SIGGRAPH Asia 20.

    Args:
        r (float): radius of circular camera trajectory.
        n_views (int): number of views including the canonical view.
        ratio (float): ratio that controls the span of rotations.

    Returns:
        rot_vecs (float array, (v, 3, 3)): world-to-camera transformations.
    """
    z_axis = np.array([0., 0., 1.])
    n_max = (n_views - 1) / ratio
    n_min = n_max - (n_views - 1)
    
    idx = np.arange(n_min + 1, n_max + 1)
    sinp = idx / (n_max + 1)
    cosp = np.sqrt(1 - sinp ** 2)

    igr = (5 ** 0.5 - 1) / 2                            # inverse golden ratio
    ga = 2 * np.pi * igr                                # golden angle
    t = ga * idx                                        # longitude

    x = cosp * np.cos(t)
    y = cosp * np.sin(t)
    z = sinp

    p_axis = np.stack([x, y, z], -1)                    # camera principal axis
    rv = np.cross(z_axis, p_axis)                       # rotation vector
    rv /= np.linalg.norm(rv, axis=-1, keepdims=True)    # normalized rotation vector
    ra = np.arccos(z)                                   # rotation angle
    
    rot_vecs = ra[:, None] * rv
    rot_vecs = np.concatenate([np.zeros((1, 3)), rot_vecs], 0)

    Rt = []
    for i in range(n_views):
        Rt.append(rodrigues([0, 0, r], rot_vecs[i], world=False))
    Rt = np.stack(Rt, 0)                                # (v, 3, 3)
    return Rt


Rt_all = get_all_views()


def sample_views(n_views=1, include_orthogonal=True):
    """
    Sample camera views.

    Args:
        n_views (int): number of camera views.
        include_orthogonal (bool): if True, always include the orthogonal view.

    Returns:
        idx (int list): sampled view indices.
        Rt (float array, (v, 3)): sampled camera views.
    """
    if include_orthogonal:
        idx = np.random.choice(
            np.arange(1, len(Rt_all)), n_views - 1, replace=False
        )
        idx = [0] + list(idx)
    else:
        idx = np.random.choice(len(Rt_all), n_views, replace=False)
    Rt = Rt_all[idx]
    return idx, Rt



class Interpsacle2d(nn.Module):
    
    def __init__(self, factor=2, gain=1, align_corners=False):
        """
            the first upsample method in G_synthesis.
        :param factor:
        :param gain:
        """
        super(Interpsacle2d, self).__init__()
        self.gain = gain
        self.factor = factor
        self.align_corners = align_corners

    def forward(self, x):
        if self.gain != 1:
            x = x * self.gain
        
        x = nn.functional.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=self.align_corners)
        
        return x


class ResConv2D(nn.Module):

    def __init__(self, nf0, inplace=False):
        super(ResConv2D, self).__init__()
        
        self.tmp = nn.Sequential(
                
                nn.ReplicationPad2d(1),
                nn.Conv2d(nf0 * 1,
                          nf0 * 1,
                          kernel_size=[3, 3],
                          padding=0,
                          stride=[1, 1],
                          bias=True),
                
                nn.LeakyReLU(negative_slope=0.2, inplace=inplace),
                # nn.Dropout3d(0.1, inplace),
                
                nn.ReplicationPad2d(1),
                nn.Conv2d(nf0 * 1,
                          nf0 * 1,
                          kernel_size=[3, 3],
                          padding=0,
                          stride=[1, 1],
                          bias=True),
        )
        
        self.inplace = inplace

    def forward(self, x):
        re = F.leaky_relu(self.tmp(x) + x, negative_slope=0.2, inplace=self.inplace)
        return re

class Projector(nn.Module):
    """
    A projection module that projects 3D feature volume into 2D feature maps.
    """
    def __init__(self):
        super(Projector, self).__init__()

    def forward(self, x):
        # input_size  # (bs*v, c, d, hi, wi)
        x, idx = x.max(2)
        d = x.size(2)
        # larger value for closer planes
        depth = (d - 1 - idx.float()) / (d - 1)
        out = torch.cat([x, depth], dim=1)
        return out

######################################################################
class ResConv3D(nn.Module):

    def __init__(self, nf0, inplace=False):
        super(ResConv3D, self).__init__()
        self.tmp = nn.Sequential(
                
                nn.ReplicationPad3d(1),
                nn.Conv3d(nf0 * 1,
                          nf0 * 1,
                          kernel_size=[3, 3, 3],
                          padding=0,
                          stride=[1, 1, 1],
                          bias=True),
                
                nn.LeakyReLU(negative_slope=0.2, inplace=inplace),
                # nn.Dropout3d(0.1, inplace),
                
                nn.ReplicationPad3d(1),
                nn.Conv3d(nf0 * 1,
                          nf0 * 1,
                          kernel_size=[3, 3, 3],
                          padding=0,
                          stride=[1, 1, 1],
                          bias=True),
        )
        
        self.inplace = inplace

    def forward(self, x):
        re = F.leaky_relu(self.tmp(x) + x, negative_slope=0.2, inplace=self.inplace)
        return re
    

class Transient2volumn_gray(nn.Module):
    def __init__(self, nf0, in_channels, \
                 norm=nn.InstanceNorm3d):
        super(Transient2volumn_gray, self).__init__()
        ###############################################
        # assert in_channels == 1
        weights = np.zeros((in_channels, in_channels, 3, 3, 3), dtype=np.float32)
        weights[:, :, 1:, 1:, 1:] = 1.0
        tfweights = torch.from_numpy(weights / np.sum(weights))
        tfweights.requires_grad = True
        self.weights = nn.Parameter(tfweights)
        
        ##############################################
        self.conv1 = nn.Sequential(
            # begin, no norm
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels,
                      nf0 * 1,
                      kernel_size=[3, 3, 3],
                      padding=0,
                      stride=[2, 2, 2],
                      bias=True),
            ResConv3D(nf0 * 1, inplace=False),
            ResConv3D(nf0 * 1, inplace=False)
        )
    def forward(self, x0):
        # x0 is from 0 to 1
        x0_conv = F.conv3d(x0, self.weights, \
                           bias=None, stride=2, padding=1, dilation=1, groups=1)
        x1 = self.conv1(x0)
        re = torch.cat([x0_conv, x1], dim=1)
        return re
    


class volumefusion(nn.Module):
    def __init__(self,dim, spatial,blocks):
        super().__init__()

        self.dim = dim
        self.spatial = spatial
        self.blocks = blocks
        self.conv = nn.Sequential(
            # begin, no norm
            nn.ReplicationPad3d(1),
            nn.Conv3d(self.dim,
                      self.dim,
                      kernel_size=[3, 3, 3],
                      padding=0,
                      stride=1,
                      bias=True),
            ResConv3D(self.dim, inplace=False),
            ResConv3D(self.dim, inplace=False)
        )
        self.conv1 = nn.Sequential(
            nn.ReplicationPad3d(1),
            nn.Conv3d(self.dim,
                      self.dim,
                      kernel_size=[3, 3, 3],
                      padding=0,
                      stride=1,
                      bias=True),
            ResConv3D(self.dim, inplace=False),
            ResConv3D(self.dim, inplace=False),
            nn.ConvTranspose3d(self.dim, self.dim, (3,6,6), stride=(1,2,2),padding=(1,2,2),bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
        )
        self.conv2 = nn.Sequential(
            nn.ReplicationPad3d(1),
            nn.Conv3d(self.dim,
                      self.dim,
                      kernel_size=[3, 3, 3],
                      padding=0,
                      stride=1,
                      bias=True),
            ResConv3D(self.dim, inplace=False),
            ResConv3D(self.dim, inplace=False),
            nn.ConvTranspose3d(self.dim, self.dim, (3,6,6), stride=(1,2,2),padding=(1,2,2),bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.ConvTranspose3d(self.dim, self.dim, (3,6,6), stride=(1,2,2),padding=(1,2,2),bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
        )
        self.fusion = nn.Sequential(
            nn.Conv3d(self.dim*3,
                      self.dim*2,
                      kernel_size=[3, 3, 3],
                      padding=1,
                      stride=1,
                      bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv3d(self.dim*2,
                      self.dim,
                      kernel_size=[3, 3, 3],
                      padding=1,
                      stride=1,
                      bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            )

    def forward(self, vlo, vlo_1, vlo_2):
        vlo = self.conv(vlo)
        vlo_1 = self.conv1(vlo_1)
        # vlo_1 = F.interpolate(vlo_1, scale_factor=[1,2,2])
        vlo_2 = self.conv2(vlo_2)
        # vlo_2 = F.interpolate(vlo_2, scale_factor=[1,4,4])
        # print(vlo.shape,vlo_1.shape,vlo_2.shape)
        return self.fusion(torch.cat([vlo,vlo_1,vlo_2],dim=1)) + vlo

    
class Rendering(nn.Module):
    
    def __init__(self, nf0, out_channels, \
                 norm=nn.InstanceNorm2d, isdep=False):
        super(Rendering, self).__init__()
        
        ######################################
        # assert out_channels == 1
        self.out_channels = out_channels
        weights = np.zeros((out_channels, out_channels*2, 1, 1), dtype=np.float32)
        if isdep:
            weights[:, out_channels:, :, :] = 1.0
        else:
            weights[:, :out_channels, :, :] = 1.0
        tfweights = torch.from_numpy(weights)
        tfweights.requires_grad = True
        self.weights = nn.Parameter(tfweights)
        
        self.resize = Interpsacle2d(factor=2, gain=1, align_corners=False)
        
        #######################################
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 1,
                      nf0 * 1,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
            ResConv2D(nf0 * 1, inplace=False),
            ResConv2D(nf0 * 1, inplace=False),
        )
        
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 1 + out_channels,
                      nf0 * 2,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
            ResConv2D(nf0 * 2, inplace=False),
            ResConv2D(nf0 * 2, inplace=False),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 2,
                      out_channels,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
        )
    
    def forward(self, x0):
        
        dim = x0.shape[1] // 2
        x0_im = x0[:, 0:self.out_channels, :, :]
        x0_dep = x0[:, dim:dim + self.out_channels, :, :]
        x0_raw_128 = torch.cat([x0_im, x0_dep], dim=1)
        x0_raw_256 = self.resize(x0_raw_128)
        x0_conv_256 = F.conv2d(x0_raw_256, self.weights, \
                               bias=None, stride=1, padding=0, dilation=1, groups=1)
        
        ###################################
        x1 = self.conv1(x0)
        x1_up = self.resize(x1)
        
        x2 = torch.cat([x0_conv_256, x1_up], dim=1)
        x2 = self.conv2(x2)
        
        re = x0_conv_256 + 1 * x2
        
        return re
    

    
class VisibleNet(nn.Module):
    
    def __init__(self, nf0, layernum=0, norm=nn.InstanceNorm3d):
        super(VisibleNet, self).__init__()
        
        self.layernum = layernum
    
    def forward(self, x):
        
        x5 = x
        ###############################################
        depdim = x5.shape[2]
        raw_pred_bxcxhxw, raw_dep_bxcxhxw = x5.max(dim=2)
        
        # -1 to 1
        # the nearer, the bigger
        raw_dep_bxcxhxw = depdim - 1 - raw_dep_bxcxhxw.float()
        raw_dep_bxcxhxw = raw_dep_bxcxhxw / (depdim - 1)
        
        xflatdep = torch.cat([raw_pred_bxcxhxw, raw_dep_bxcxhxw], dim=1)
        
        return xflatdep


class Transformer(nn.Module):
    """
    A transformation module that re-samples 3D feature volume given 
    arbitrary camera pose.
    """
    def __init__(
        self, 
        d=256,              # depth dimension of feature volume
        h=128,              # height dimension of feature volume
        w=128,              # width dimension of feature volume
        wall_size=2,    # wall size (unit: m)
        zmin=0,        # near plane w.r.t. world frame (unit: m)
        zmax=2,         # far plane w.r.t. world frame (unit: m)
    ):
        super(Transformer, self).__init__()
        assert zmax > zmin, \
            "far plane depth must be larger than near plane depth"

        self.d = d
        self.h = h
        self.w = w

        # sampling grid
        width = height = wall_size / 2
        zdim = np.linspace(zmin, zmax, d + 1)
        ydim = np.linspace(height, -height, h + 1)
        xdim = np.linspace(-width, width, w + 1)
        zdim = (zdim[:-1] + zdim[1:]) / 2
        ydim = (ydim[:-1] + ydim[1:]) / 2
        xdim = (xdim[:-1] + xdim[1:]) / 2
        [zgrid, ygrid, xgrid] = np.meshgrid(zdim, ydim, xdim, indexing="ij")
        grid = np.stack([xgrid, ygrid, zgrid], -1)         # (d, h, w, 3)
        grid = grid.reshape(-1, 3).T                       # (3, d*h*w)
        self.register_buffer("grid", torch.from_numpy(grid.astype(np.float32)))
        
        bb_ctr = [0, 0, (zmax + zmin) / 2]
        bb_radius = [width, height, (zmax - zmin) / 2]
        self.register_buffer("bb_ctr", torch.Tensor(bb_ctr))
        self.register_buffer("bb_radius", torch.Tensor(bb_radius))

    def forward(self, x, rot):
        """
        Args:
            x (float tensor, (bs, c, d, h, w)): feature volume aligned with the 
                orthogonal view.
            rot (float tensor, (3, 3)): rotation matrix.

        Returns:
            x (float tensor, (bs, c, d, h, w)): re-sampled feature volume.
        """
        bs, _, d, h, w = x.shape
        assert [d, h, w] == [self.d, self.h, self.w]

        # rotate grid to align with the camera view
        p = torch.matmul(rot, self.grid).T                 # (d*h*w, 3)

        # normalize grid points to [-1, 1]
        ## NOTE: F.grid_sample assumes that y-axis points DOWNWARD
        p = p.sub_(self.bb_ctr).div_(self.bb_radius)
        p[:, 1].mul_(-1)     # flip y-axis
        p = p.reshape(d, h, w, 3).repeat(bs, 1, 1, 1, 1)   # (bs, d, h, w, 3)

        # trilinear interpolation
        x = F.grid_sample(x, p, align_corners=False)       # (bs, c, d, h, w)
        return x


class SPUP(nn.Module):
    """The SPUP block
    Upsample spatial dim and keep the temporal dimension
    Input: B, C, D, H, W
    Output: B, C, D, H*up, W*up (default)
    """

    def __init__(self, ch_in, up=2):
        super().__init__()

        self.in_chans = ch_in
        self.up = int(log2(up))
        upblk = []
        for n in range(self.up):
            sp_upblk = nn.Sequential(
            nn.ConvTranspose3d(self.in_chans, self.in_chans, (3,6,6), stride=(1,2,2),padding=(1,2,2),bias=False),
            nn.ReLU(inplace=True)
            )
            init.kaiming_normal_(sp_upblk[0].weight, 0, 'fan_in', 'relu')
            upblk.append(sp_upblk)
        
        self.upop = nn.Sequential(*upblk)

    def forward(self, inputs):

        out = self.upop(inputs)

        return out
    

class mask_conv_fusion(nn.Module):
    def __init__(self, in_chans=1,temp_chans=32,sp_ds_scale=8,targrt_size=128,inplace=False):
        super(mask_conv_fusion, self).__init__()
        self.in_chans = in_chans
        self.temp_chans = temp_chans
        self.targrt_size =targrt_size
        self.inplace = inplace
        self.sp_ds_scale = sp_ds_scale
        self.conv1 = nn.Sequential(nn.Conv3d(self.in_chans, self.temp_chans, (3,17,17), stride=(1,1,1), padding=(1,8,8), dilation=1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
                                   ResConv3D(self.temp_chans,self.inplace),)
        
        self.conv2 = nn.Sequential(nn.Conv3d(self.in_chans, self.temp_chans, (3,9,9), stride=(1,1,1), padding=(1,4,4), dilation=1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
                                   ResConv3D(self.temp_chans,self.inplace),)
        
        self.conv3 = nn.Sequential(nn.Conv3d(self.in_chans, self.temp_chans, (3,5,5), stride=(1,1,1), padding=(1,2,2), dilation=1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
                                   ResConv3D(self.temp_chans,self.inplace),)
        
        self.conv4 = nn.Sequential(nn.Conv3d(self.in_chans, self.temp_chans, (3,3,3), stride=(1,1,1), padding=(1,1,1), dilation=1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
                                   ResConv3D(self.temp_chans,self.inplace),)
        
        self.fusion = nn.Sequential(nn.Conv3d(self.temp_chans*4 +1, self.temp_chans*2, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
                                    ResConv3D(self.temp_chans*2,self.inplace),
                                    nn.Conv3d(self.temp_chans*2, self.temp_chans*1, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
                                    ResConv3D(self.temp_chans*1,self.inplace),
                                    nn.Conv3d(self.temp_chans*1, self.in_chans, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
                                    ResConv3D(self.in_chans,self.inplace),)
        
    def forward(self, mask_mea):
        mea_1 = self.conv1(mask_mea)
        mea_2 = self.conv2(mask_mea)
        mea_3 = self.conv3(mask_mea)
        mea_4 = self.conv4(mask_mea)
        
        if self.sp_ds_scale >=1:
            b, c, t, h, w = mask_mea.shape
            ds_mea_reshape = mask_mea.reshape(-1,t,h,w)
            ds_mea_reshape = ds_mea_reshape[:,:,self.sp_ds_scale//2::self.sp_ds_scale,self.sp_ds_scale//2::self.sp_ds_scale]
            simple_up = F.interpolate(ds_mea_reshape, size=(self.targrt_size,) * 2) # , mode='nearest'
            simple_up = simple_up.reshape(b,c,t,self.targrt_size,self.targrt_size)
            # final+=simple_up
        
        mea_fusion = torch.cat([simple_up, mea_4,mea_3,mea_2,mea_1],dim=1)
        # mea_fusion = torch.cat([simple_up,mea_1],dim=1)
        final = self.fusion(mea_fusion)  
        return final
    
    
class EmbedFeatureModel_MUL_gray_former_mask_refine(nn.Module):
    """ Encoder-decoder model for feed-forward inference """
    def __init__(self, basedim = 3, in_ch=1,out_ch=1,spatial=64,tlen=256,bin_len=0.02,views=1,wall_size=2, sp_ds_scale=1):
        super(EmbedFeatureModel_MUL_gray_former_mask_refine, self).__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.spatial = spatial
        self.tlen = tlen
        self.bin_len = bin_len
        self.views = views
        self.wall_size = wall_size
        self.up_num = int(np.log2(sp_ds_scale))
        self.cs_ch = 4
        self.cs_blk = mask_conv_fusion(in_chans=1,temp_chans=8,sp_ds_scale=sp_ds_scale,targrt_size=128,inplace=False)
        
        self.downnet = Transient2volumn_gray(nf0=basedim, in_channels=self.in_channels)
        
        # self.tra2vlo = phasor_1_10.phasor(spatial=self.spatial, crop=self.tlen, bin_len=self.bin_len, wall_size=self.wall_size, sampling_coeff=2.0, cycles=5,dnum=basedim + self.in_channels)
        # self.tra2vlo = fk_1_10.lct_fk(spatial=self.spatial, crop=self.tlen, bin_len=self.bin_len,dnum=basedim + self.in_channels)
        self.tra2vlo =  lct_1_10.lct(spatial=self.spatial, crop=self.tlen, wall_size=self.wall_size, bin_len=self.bin_len,method='lct',dnum=basedim + self.in_channels)
        # self.tra2vlo = RSDEfficient.RSD(t=512,d=48,h=128,w=128,in_plane=basedim + self.in_channels,wall_size=2,bin_len=0.0096,scale_coef=2,n_cycles=5,ratio=0.0001)
        
        refine_chann = basedim + self.in_channels
        self.refine = nn.Sequential(nn.Conv3d(refine_chann, refine_chann*2, 7, stride=(1,1,1), padding=3, dilation=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=False),
                                    ResConv3D(refine_chann*2,False),
                                    nn.Conv3d(refine_chann*2, refine_chann*2, 5, stride=(1,1,1), padding=2, dilation=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=False),
                                    ResConv3D(refine_chann*2,False),
                                    nn.Conv3d(refine_chann*2, refine_chann, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=False),
                                    ResConv3D(refine_chann,False),)
        # self.transform = Transformer(d=self.tlen,h=self.spatial,w=self.spatial,wall_size=self.wall_size)  
        self.transform = Transformer(d=200,h=self.spatial,w=self.spatial,wall_size=self.wall_size)  
        self.visnet = Projector()     
        layernum = 0
        self.rendernet = Rendering(nf0=(basedim * 1 + self.in_channels) * 2, out_channels=self.out_channels)
        self.depnet = Rendering(nf0=(basedim * 1 +self.in_channels) * 2, out_channels=self.out_channels, isdep=True)
    
    def normalize(self, data_bxcxdxhxw):   #  min max scaling
        b, c, d, h, w = data_bxcxdxhxw.shape
        data_bxcxk = data_bxcxdxhxw.reshape(b, c, -1)
        data_min = data_bxcxk.min(2, keepdim=True)[0]
        data_zmean = data_bxcxk - data_min
        # most are 0
        data_max = data_zmean.max(2, keepdim=True)[0]
        data_norm = data_zmean / (data_max + 1e-15)

        return data_norm.view(b, c, d, h, w)  

    def forward(self, inputs, target=None,targetd=None):
        if not self.training:
            self.views = 1
        # sample target views
        view_idx, Rt = sample_views(
            n_views=self.views,
            include_orthogonal=True)
        rot = None
        if view_idx != [0]:
            rot = Rt[:, :3, :3]
            rot = torch.from_numpy(rot.astype(np.float32))
            rot = rot.cuda(non_blocking=True)
        
        # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # starter.record()
        sig_exp = self.cs_blk(inputs)
        # ender.record()
        # torch.cuda.synchronize()
        # curr_time = starter.elapsed_time(ender) 
        # print('curr_time',curr_time)
        inputs = self.normalize(sig_exp)      #  b,1,512,128,128
        fea = self.downnet(inputs)      #  b,4,512,128,128
        vlo = self.tra2vlo(fea, [0,0,0], [self.tlen,self.tlen,self.tlen]) #  b,4,512,128,128
        
        zdim = vlo.shape[2]
        zdimnew = zdim * 100 // 128
        vlo = vlo[:, :, :zdimnew]
        vlo = nn.ReLU()(vlo)
        vlo = self.normalize(vlo)   
        
        if rot is not None:
            views = []
            for r in rot:
                views.append(self.transform(vlo, r)) # (bs, c, d, hi, wi)
            results = torch.stack(views, 1).flatten(0, 1)  # (bs*v, c, d, hi, wi)
        else:
            results = vlo
        results = self.refine(results)
        raw = self.visnet(results)  # (bs*v, 2c, hi, wi)   former intensity later depth   
        rendered_img = self.rendernet(raw)  # (bs*v, 1/3, hi, wi) 
        rendered_img = rendered_img.reshape(inputs.shape[0], -1, *rendered_img.shape[-3:])        
        rendered_depth = self.depnet(raw)
        rendered_depth = rendered_depth.reshape(inputs.shape[0], -1, *rendered_img.shape[-3:])         
        rendered_depth = torch.mean(rendered_depth,dim=2,keepdim=True)
        
        if target is not None:
            target = target[:, view_idx]                    
            target = target.cuda(non_blocking=True)
            targetd = targetd[:, view_idx]                    
            targetd = targetd.cuda(non_blocking=True)
            return sig_exp, results, rendered_img, target, rendered_depth, targetd
        else:
            # for validate on realworld dataset
            return results, torch.squeeze(rendered_img,1)   

    
  


