import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4) 
    N, C = size[:2] 
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1) 
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    
    return feat_mean, feat_std
def affine_transformation(X, alpha, beta):
    x = X.clone()  
    mean, std = calc_mean_std(x) 
    mean = mean.expand_as(x)  
    std = std.expand_as(x)  
    return alpha * ((x-mean)/std) + beta

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out
    
class bFMFusion(nn.Module):
    def __init__(self, sketch_nc, texture_nc, ngf=160, norm_layer=nn.BatchNorm2d, bottleneck_depth=100, num_resblocks=4):
        super(bFMFusion, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.downconv1 = nn.Sequential(*[
            nn.Conv2d(sketch_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)
            ])
        self.downconv_channel_unchanged = nn.Sequential(*[
            nn.Conv2d(ngf, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)
            ])
        self.downconv2 = nn.Sequential(*[
            nn.LeakyReLU(0.2, True), 
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias)
            ])
        self.downconv3 = nn.Sequential(*[
            nn.LeakyReLU(0.2, True), 
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias)
            ])
        self.downconv4 = nn.Sequential(*[
            nn.LeakyReLU(0.2, True), 
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)
            ])
        self.downconv5 = nn.Sequential(*[
            nn.LeakyReLU(0.2, True), 
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)
            ])
        
        resblocks = []
        for i in range(num_resblocks):
            resblocks.append(ResBlock(ngf))
        self.resblocks = nn.Sequential(*resblocks)

        
        ### texture downsampling
        self.G_downconv1 = nn.Sequential(*[
            nn.Conv2d(texture_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)
            ])
        self.G_downconv_channel_unchanged = nn.Sequential(*[
            nn.Conv2d(ngf, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)
            ])
        self.G_downconv2 = nn.Sequential(*[
            nn.LeakyReLU(0.2, True), 
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias)
            ])

        self.G_downconv3 = nn.Sequential(*[
            nn.LeakyReLU(0.2, True), 
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias)
            ])

        self.G_downconv4 = nn.Sequential(*[
            nn.LeakyReLU(0.2, True), 
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)
            ])
  
        self.G_downconv5 = nn.Sequential(*[
            nn.LeakyReLU(0.2, True), 
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)
            ])
        G_resblocks = []
        for i in range(num_resblocks):
            G_resblocks.append(ResBlock(ngf))
        self.G_resblocks = nn.Sequential(*G_resblocks)

        ### bottlenecks for param generation
        self.bottleneck_alpha_2 = nn.Sequential(*self.bottleneck_layer(ngf * 2, bottleneck_depth))
        self.bottleneck_beta_2 = nn.Sequential(*self.bottleneck_layer(ngf * 2, bottleneck_depth))
        self.bottleneck_alpha_3 = nn.Sequential(*self.bottleneck_layer(ngf * 4, bottleneck_depth))
        self.bottleneck_beta_3 = nn.Sequential(*self.bottleneck_layer(ngf * 4, bottleneck_depth))
        self.bottleneck_alpha_4 = nn.Sequential(*self.bottleneck_layer(ngf * 8, bottleneck_depth))
        self.bottleneck_beta_4 = nn.Sequential(*self.bottleneck_layer(ngf * 8, bottleneck_depth))
        self.bottleneck_alpha_5 = nn.Sequential(*self.bottleneck_layer(ngf * 8, bottleneck_depth))
        self.bottleneck_beta_5 = nn.Sequential(*self.bottleneck_layer(ngf * 8, bottleneck_depth))

        ### for texture
        self.G_bottleneck_alpha_2 = nn.Sequential(*self.bottleneck_layer(ngf * 2, bottleneck_depth))
        self.G_bottleneck_beta_2 = nn.Sequential(*self.bottleneck_layer(ngf * 2, bottleneck_depth))
        self.G_bottleneck_alpha_3 = nn.Sequential(*self.bottleneck_layer(ngf * 4, bottleneck_depth))
        self.G_bottleneck_beta_3 = nn.Sequential(*self.bottleneck_layer(ngf * 4, bottleneck_depth))
        self.G_bottleneck_alpha_4 = nn.Sequential(*self.bottleneck_layer(ngf * 8, bottleneck_depth))
        self.G_bottleneck_beta_4 = nn.Sequential(*self.bottleneck_layer(ngf * 8, bottleneck_depth))
        self.G_bottleneck_alpha_5 = nn.Sequential(*self.bottleneck_layer(ngf * 8, bottleneck_depth))
        self.G_bottleneck_beta_5 = nn.Sequential(*self.bottleneck_layer(ngf * 8, bottleneck_depth))

    def bottleneck_layer(self, nc, bottleneck_depth):   
        return [
            nn.Conv2d(nc, bottleneck_depth, kernel_size=1), 
            nn.ReLU(True), 
            nn.Conv2d(bottleneck_depth, nc, kernel_size=1)
            ]

    # per pixel
    def get_FiLM_param_(self, X, i, texture=False):
        x = X.clone()
        # bottleneck
        if texture:
            if (i=='2'):
                alpha_layer = self.G_bottleneck_alpha_2
                beta_layer = self.G_bottleneck_beta_2
            elif (i=='3'):
                alpha_layer = self.G_bottleneck_alpha_3
                beta_layer = self.G_bottleneck_beta_3
            elif (i=='4'): 
                alpha_layer = self.G_bottleneck_alpha_4
                beta_layer = self.G_bottleneck_beta_4
            elif (i=='5'):
                alpha_layer = self.G_bottleneck_alpha_5
                beta_layer = self.G_bottleneck_beta_5
        else:
            if (i=='2'):
                alpha_layer = self.bottleneck_alpha_2
                beta_layer = self.bottleneck_beta_2
            elif (i=='3'):
                alpha_layer = self.bottleneck_alpha_3
                beta_layer = self.bottleneck_beta_3
            elif (i=='4'):
                alpha_layer = self.bottleneck_alpha_4
                beta_layer = self.bottleneck_beta_4
            elif (i=='5'):
                alpha_layer = self.bottleneck_alpha_5
                beta_layer = self.bottleneck_beta_5
            
        alpha = alpha_layer(x)
        beta = beta_layer(x)
        return alpha, beta

    def forward (self, sketch, texture):
        ## downconv
        down1 = self.downconv1(sketch)
        G_down1 = self.G_downconv1(texture)
        

        #---------------------------------------------------------------------------------
        down1 = self.resblocks(down1)
        G_down1 = self.G_resblocks(G_down1)

        down1 = self.downconv_channel_unchanged(down1)
        G_down1 = self.G_downconv_channel_unchanged(G_down1)
        #---------------------------------------------------------------------------------
        down2 = self.downconv2(down1)
        G_down2 = self.G_downconv2(G_down1)

        g_alpha2, g_beta2 = self.get_FiLM_param_(G_down2, '2', texture=True)
        i_alpha2, i_beta2 = self.get_FiLM_param_(down2, '2')
        down2 = affine_transformation(down2, g_alpha2, g_beta2)
        G_down2 = affine_transformation(G_down2, i_alpha2, i_beta2)

        down3 = self.downconv3(down2)
        G_down3 = self.G_downconv3(G_down2)

        g_alpha3, g_beta3 = self.get_FiLM_param_(G_down3, '3', texture=True)
        i_alpha3, i_beta3 = self.get_FiLM_param_(down3, '3')
        down3 = affine_transformation(down3, g_alpha3, g_beta3)
        G_down3 = affine_transformation(G_down3, i_alpha3, i_beta3)

        down4 = self.downconv4(down3)
        G_down4 = self.G_downconv4(G_down3)

        g_alpha4, g_beta4 = self.get_FiLM_param_(G_down4, '4', texture=True)
        i_alpha4, i_beta4 = self.get_FiLM_param_(down4, '4')
        down4 = affine_transformation(down4, g_alpha4, g_beta4) 
        G_down4 = affine_transformation(G_down4, i_alpha4, i_beta4)

        down5 = self.downconv5(down4)
        G_down5 = self.G_downconv5(G_down4)


        g_alpha5, g_beta5 = self.get_FiLM_param_(G_down5, '5', texture=True)
        i_alpha5, i_beta5 = self.get_FiLM_param_(down5, '5')
        down5 = affine_transformation(down5, g_alpha5, g_beta5)
        G_down5 = affine_transformation(G_down5, i_alpha5, i_beta5)

        res = [down2, down3, down4, down5]
        return res