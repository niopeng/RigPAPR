# credit: https://github.com/princeton-vl/SNP
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from einops import rearrange
from .utils import activation_func


class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, norm='none', act='relu', groups=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if norm == 'instance':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=groups),
                nn.InstanceNorm2d(mid_channels),
                activation_func(act, inplace=True),
            )
        elif norm == 'batch':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=groups),
                nn.BatchNorm2d(mid_channels),
                activation_func(act, inplace=True),
            )
        elif norm == 'group':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=groups),
                nn.GroupNorm(1, mid_channels),
                activation_func(act, inplace=True),
            )
        elif norm == 'none':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=groups),
                activation_func(act, inplace=True),
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, norm='none', act='relu', groups=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if norm == 'instance':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=groups),
                nn.InstanceNorm2d(mid_channels),
                activation_func(act, inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, groups=groups),
                nn.InstanceNorm2d(out_channels),
                activation_func(act, inplace=True),
            )
        elif norm == 'batch':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=groups),
                nn.BatchNorm2d(mid_channels),
                activation_func(act, inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, groups=groups),
                nn.BatchNorm2d(out_channels),
                activation_func(act, inplace=True),
            )
        elif norm == 'group':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=groups),
                nn.GroupNorm(1, mid_channels),
                activation_func(act, inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, groups=groups),
                nn.GroupNorm(1, out_channels),
                activation_func(act, inplace=True),
            )
        elif norm == 'none':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=groups),
                activation_func(act, inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, groups=groups),
                activation_func(act, inplace=True),
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, single=False, norm='none', act='relu', groups=1):
        super().__init__()
        # print('down norm:', norm)

        if single:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                SingleConv(in_channels, out_channels, norm=norm, act=act, groups=groups)
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels, norm=norm, act=act, groups=groups)
            )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, single=False, norm='none', act='relu', groups=1):
        super().__init__()
        self.groups = groups

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if single:
                self.conv = SingleConv(in_channels, out_channels, in_channels // 2, norm=norm, act=act, groups=groups)
            else:
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm=norm, act=act, groups=groups)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, groups=groups)
            if single:
                self.conv = SingleConv(in_channels, out_channels, norm=norm, act=act, groups=groups)
            else:
                self.conv = DoubleConv(in_channels, out_channels, norm=norm, act=act, groups=groups)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is NCHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        
        # Handle grouped convolutions: interleave groups from x1 and x2
        if self.groups > 1:
            # Get group information
            groups = self.groups
            channels_per_group_x1 = x1.size(1) // groups
            channels_per_group_x2 = x2.size(1) // groups
            
            # Reshape to separate groups: (N, groups, channels_per_group, H, W)
            x1_grouped = x1.view(x1.size(0), groups, channels_per_group_x1, x1.size(2), x1.size(3))
            x2_grouped = x2.view(x2.size(0), groups, channels_per_group_x2, x2.size(2), x2.size(3))
            
            # Interleave groups using einops: alternate between x2 and x1 groups
            # Stack along group dimension and then rearrange to interleave
            stacked = torch.stack([x2_grouped, x1_grouped], dim=1)  # (b, 2, g, c, h, w)
            x = rearrange(stacked, 'b two g c h w -> b (g two c) h w')
        else:
            # Standard concatenation for non-grouped convolutions
            x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


# class UpAdd(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True, single=False, norm='none', act='relu', groups=1):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             if single:
#                 self.conv = SingleConv(in_channels, out_channels, in_channels // 2, norm=norm, act=act, groups=groups)
#             else:
#                 self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm=norm, act=act, groups=groups)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, groups=groups)
#             if single:
#                 self.conv = SingleConv(in_channels, out_channels, norm=norm, act=act, groups=groups)
#             else:
#                 self.conv = DoubleConv(in_channels, out_channels, norm=norm, act=act, groups=groups)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         # if you have padding issues, see
#         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)


# class UpSample(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True, single=False, norm='none', act='relu', groups=1):
#         super().__init__()
#         # print('up norm:', norm)

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             if single:
#                 self.conv = SingleConv(in_channels, out_channels, in_channels // 2, norm=norm, act=act, groups=groups)
#             else:
#                 self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm=norm, act=act, groups=groups)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, groups=groups)
#             if single:
#                 self.conv = SingleConv(in_channels, out_channels, norm=norm, act=act, groups=groups)
#             else:
#                 self.conv = DoubleConv(in_channels, out_channels, norm=norm, act=act, groups=groups)

#     def forward(self, x1):
#         x1 = self.up(x1)
#         return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, single=True, norm='none', last_act='none', act='relu',
                 inp_scale=1.0, use_amp=False, amp_dtype=torch.float16, affine_layer=-1, use_outc=True, groups=1, 
                 group_last=False, channel_factor=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.affine_layer = affine_layer
        self.inp_scale = inp_scale
        self.use_outc = use_outc
        self.groups = groups
        self.channel_factor = channel_factor
        self.group_last = group_last

        self.inc = SingleConv(n_channels, 128 * channel_factor, norm=norm, act=act, groups=groups)
        self.down1 = Down(128 * channel_factor, 256 * channel_factor, single=single, norm=norm, act=act, groups=groups)
        self.down2 = Down(256 * channel_factor, 512 * channel_factor, single=single, norm=norm, act=act, groups=groups)
        self.up1 = Up(512 * channel_factor, 256 * channel_factor, bilinear, single=single, norm=norm, act=act, groups=groups)

        if self.use_outc:
            self.up2 = Up(256 * channel_factor, 128 * channel_factor, bilinear, single=single, norm=norm, act=act, groups=groups)
            self.outc = OutConv(128 * channel_factor, n_classes, groups=groups if self.group_last else 1)
        else:
            self.up2 = Up(256 * channel_factor, n_classes, bilinear, single=single, norm=norm, act=act, groups=groups)

        self.last_act = activation_func(last_act)

    def forward(self, x, log=False, gamma=None, beta=None):

        with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            x = x * self.inp_scale

            x1 = self.inc(x)    # (N, 128 * channel_factor, H, W)
            x2 = self.down1(x1)    # (N, 256 * channel_factor, H/2, W/2)
            x3 = self.down2(x2)    # (N, 512 * channel_factor, H/4, W/4)
            x = self.up1(x3, x2)   # (N, 256 * channel_factor, H/2, W/2)
            x = self.up2(x, x1)    # (N, 128 * channel_factor, H, W)
            if self.use_outc:
                x = self.outc(x)    # (N, n_classes, H, W)
                
            x = self.last_act(x)    # (N, n_classes, H, W)

            return x
