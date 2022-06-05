# %%
import torch
import torch.nn as nn
from torch.optim import Adam

# %%
# from .detect_net import UVDetectNet
from .detect_net_test import UVDetectNet
from .decoder_tex import ReconstructionLayer
from .utils import UVSampleLayer
from .basic_loss import *
from .pix2pix import define_D
import os.path as osp
from PIL import Image
import numpy as np
from .debugging import GradRecord

# %%
import yaml

# %%
class BasicModel:
    
    def __init__(self, model_config, is_train=True, is_continue=False, continue_path=None, continue_gan=False):

        self.model_config = model_config
        self.device = self.model_config['device']
        self.is_train = is_train
        self.is_continue = is_continue
        # Initialize training
        self.init()
        if self.is_continue and continue_path is not None:
            self.continue_train(continue_path, continue_gan)
            
    def continue_train(self, continue_path, continue_gan):
        state = torch.load(continue_path)
        self.generator.load_state_dict(state['generator'])
    
    def init(self):
        # 1. Init model
        if self.model_config['activation'] == 'tanh':
            activation = nn.Tanh
        elif self.model_config['activation'] == 'sigmoid':
            activation = nn.Sigmoid
        else:
            activation = None        
        self.generator = UVDetectNet(cin=3,cout=3,\
                                     activation=activation).to(self.device)        
        self.sampler = UVSampleLayer('./models/model_config.json').to(self.device)
        self.renderer = ReconstructionLayer('./models/model_config.json').to(self.device) 
        if self.is_train:
            self.generator.train()
                    
        # 4. Init Params
        self.uv_mask = torch.as_tensor(np.load('./Data/uv_mask.npy').astype(np.float32)[None,...,None]).to(self.device)
                
    @staticmethod
    def toTensor2D(maps):
        # [B,H,W,C]
        if len(maps.shape) == 4:
            return torch.as_tensor(maps).permute(0,3,1,2).contiguous()
        # [B,H,W]
        elif len(maps.shape) == 3:
            return torch.as_tensor(maps)[:,None,...]
        
    def set_input(self, inputs):
        self.mask = self.toTensor2D(inputs['mask']).to(self.device)
        self.img = self.toTensor2D(inputs['image']).to(self.device)
        self.coarse_tex_map = self.toTensor2D(inputs['render_tex']).to(self.device)
        self.param = inputs['param'].to(self.device)
        self.render = self.toTensor2D(inputs['render']).to(self.device)
        
    def forward(self):
        self.detail_tex_map, self.light_maps, self.local_maps, self.local_probs= self.generator(self.img, self.coarse_tex_map)
        self.delta_local_map = (self.local_maps * self.local_probs[...,None,None,None]).sum(dim=1)
        self.detail_tex_map = (self.detail_tex_map + 1) / 2.0
        self.detail_tex_map_update = self.detail_tex_map + self.delta_local_map        
        if self.model_config['activation'] == 'tanh':
            # Reshape to [None,224,224,3]
            self.detail_tex_map, self.detail_tex_map_update = self.detail_tex_map.permute(0,2,3,1).contiguous(), self.detail_tex_map_update.permute(0,2,3,1).contiguous()
        self.detail_tex, self.detail_tex_update = self.sampler(self.detail_tex_map), self.sampler(self.detail_tex_map_update)
        # Generate results
        self.detail_tex_face_illu,\
        self.detail_tex_full_illu,\
        self.detail_face_image,\
        self.detail_full_image,\
        self.detail_face_image_rand,\
        self.coarse_tex = self.renderer(self.param,\
                                        self.detail_tex_update)
            
        self.detail_face_image = self.detail_face_image.permute(0,3,1,2).contiguous()
        if self.detail_face_image_rand is not None:
            self.detail_face_image_rand = self.detail_face_image_rand.permute(0,3,1,2).contiguous()        
    
    def validate(self, inputs):
        self.set_input(inputs)
        self.forward()
            
    def train(self):
        self.mask_reg_detail_loss.changeWeight(self.mask_weight)
        self.generator.train()
        
    def eval(self):
        self.generator.eval()
        
    '''This should only apply on discriminator, not generator!'''
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad