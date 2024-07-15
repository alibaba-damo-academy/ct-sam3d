import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm


# https://github.com/NVlabs/SPADE/blob/master/models/networks/normalization.py
class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc, kernel_size=3):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm3d(norm_nc, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        ks = kernel_size
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv3d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv3d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv3d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        # normalized = x

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='trilinear')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SPADE_concat(nn.Module):
    def __init__(self, norm_nc, label_nc, kernel_size=3):
        super().__init__()

        # self.param_free_norm = nn.InstanceNorm3d(norm_nc, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        ks = kernel_size
        pw = ks // 2
        # self.mlp_shared = nn.Sequential(
        #     nn.Conv3d(label_nc, nhidden, kernel_size=ks, padding=pw),
        #     nn.ReLU()
        # )
        # self.mlp_gamma = nn.Conv3d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        # self.mlp_beta = nn.Conv3d(nhidden, norm_nc, kernel_size=ks, padding=pw)

        self.channel_mapper = nn.Sequential(
            nn.Conv3d(label_nc, norm_nc, kernel_size=ks, padding=pw),
            nn.LeakyReLU()
        )

        self.reduce = nn.Sequential(
            nn.Conv3d(norm_nc*2, norm_nc, kernel_size=ks, padding=pw),
            nn.LeakyReLU()
        )

    def forward(self, x, segmap):

        # # Part 1. generate parameter-free normalized activations
        # # normalized = self.param_free_norm(x)
        # normalized = x

        # # Part 2. produce scaling and bias conditioned on semantic map
        # segmap = F.interpolate(segmap, size=x.size()[2:], mode='trilinear')
        # actv = self.mlp_shared(segmap)
        # gamma = self.mlp_gamma(actv)
        # beta = self.mlp_beta(actv)

        # # apply scale and bias
        # out = normalized * (1 + gamma) + beta

        segmap = F.interpolate(segmap, size=x.size()[2:], mode='trilinear')
        segmap_fea = self.channel_mapper(segmap)
        concat_fea = torch.concat([x, segmap_fea], dim=1)
        out = self.reduce(concat_fea)

        return out

# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, label_nc, nhidden=128):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv3d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv3d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv3d(fin, fout, kernel_size=1, bias=False)

        # # apply spectral norm if specified
        # if 'spectral' in opt.norm_G:
        #     self.conv_0 = spectral_norm(self.conv_0)
        #     self.conv_1 = spectral_norm(self.conv_1)
        #     if self.learned_shortcut:
        #         self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        self.norm_0 = SPADE(fin, label_nc=label_nc, kernel_size=3)
        self.norm_1 = SPADE(fmiddle, label_nc=label_nc, kernel_size=3)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc=label_nc, kernel_size=3)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
    
class SPADEResnetBlock_concat(nn.Module):
    def __init__(self, fin, fout, label_nc):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv3d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv3d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv3d(fin, fout, kernel_size=1, bias=False)

        # # apply spectral norm if specified
        # if 'spectral' in opt.norm_G:
        #     self.conv_0 = spectral_norm(self.conv_0)
        #     self.conv_1 = spectral_norm(self.conv_1)
        #     if self.learned_shortcut:
        #         self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        self.norm_0 = SPADE_concat(fin, label_nc=label_nc, kernel_size=3)
        self.norm_1 = SPADE_concat(fmiddle, label_nc=label_nc, kernel_size=3)
        if self.learned_shortcut:
            self.norm_s = SPADE_concat(fin, label_nc=label_nc, kernel_size=3)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)