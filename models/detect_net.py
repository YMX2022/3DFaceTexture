import torch
import torch.nn as nn
import numpy as np
from torch.nn import init

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class Encoder(nn.Module):
    
    def __init__(self, cin, ngf=64, norm=nn.BatchNorm2d):
        super().__init__()
        network = [\
            nn.Conv2d(cin, ngf, kernel_size=3, stride=1, padding=1, bias=False),  # 224 -> 112, 3 => 64
            norm(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf, kernel_size=4, stride=2, padding=1, bias=False),  # 224 -> 112, 3 => 64
            norm(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 112 -> 56, 64 => 128
            norm(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 56 -> 28, 128 => 256
            norm(ngf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 28 -> 14, 256 => 512
            norm(ngf*8),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.network = nn.Sequential(*network)
    
    def forward(self, inputs):
        return self.network(inputs)
                
class Decoder(nn.Module):
        
    def _get_bilinear(self, in_channels, out_channels, bias=False):
        bilinear_layer = [\
                             nn.UpsamplingBilinear2d(scale_factor=2.0),\
                             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),\
                         ]
        return nn.Sequential(*bilinear_layer)
    
    def _get_conv2d_trans(self, in_channels, out_channels, bias=False):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=bias)
            
    def __init__(self, cout, ngf=64, norm=nn.BatchNorm2d, activation=nn.Tanh(), bilinear=False):
        super().__init__()
        if bilinear:
            upsample_func = self._get_bilinear
        else:
            upsample_func = self._get_conv2d_trans
            
        network = [            
            upsample_func(ngf*8, ngf*4),  # 28 -> 56, 512 => 256
            norm(ngf*4),
            nn.ReLU(inplace=True),
            upsample_func(ngf*4, ngf*2),  # 56 -> 112, 256 => 128
            norm(ngf*2),
            nn.ReLU(inplace=True),
            upsample_func(ngf*2, ngf),    # 112 -> 224, 128 => 64
            norm(ngf),
            nn.ReLU(inplace=True),
            upsample_func(ngf, ngf),    # 112 -> 224, 128 => 64
            norm(ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, cout, kernel_size=3, stride=1, padding=1, bias=True),
        ]
        if activation is not None:
            network += [activation]
        self.network = nn.Sequential(*network)
        
    def forward(self, inputs):
        return self.network(inputs)
    
class UVNet(nn.Module):
    
    def __init__(self, cin, cout, ngf=64, norm=nn.BatchNorm2d, last_activation=nn.Tanh(), init_type='kaiming', n_blocks=6):
        super().__init__()
        self.encoder =  Encoder(cin=cin, ngf=ngf, norm=norm)
        self.tex_decoder = Decoder(cout=cout, ngf=ngf, norm=norm, activation=last_activation)
        self.light_decoder = Decoder(cout=27, ngf=ngf, norm=norm, activation=None, bilinear=True)
        mid_res_list = []
        for i in range(n_blocks):
            mid_res_list.append(ResnetBlock(dim=ngf*8, padding_type='zero', norm_layer=norm, use_dropout=False, use_bias=False))
        self.mid_res_block = nn.Sequential(*mid_res_list)
        # Init params
        init_net(self.encoder, init_type)
        init_net(self.tex_decoder, init_type)
        init_net(self.light_decoder, init_type)
        init_net(self.mid_res_block, init_type)
        
    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.mid_res_block(x)
        x_tex = self.tex_decoder(x)
        x_light = self.light_decoder(x)
        return x_tex, x_light