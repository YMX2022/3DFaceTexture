# %%
import torch
import torch.nn as nn
from torch.optim import Adam

# %%
# from .detect_net import UVDetectNet
from .detect_net import UVNet
from .decoder_tex_white import ReconstructionLayer
from .utils import UVSampleLayer, ImgGrad
from .basic_loss import *
from .pix2pix import define_D, define_G
import os.path as osp
from PIL import Image
import numpy as np

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
        self.generator = UVNet(cin=6, cout=3, ngf=64).to(self.device)
        self.sampler = UVSampleLayer('./models_new_test/model_config.json').to(self.device)
        self.renderer = ReconstructionLayer('./models_new_test/model_config.json').to(self.device) 
        if self.is_train:
            self.generator.train()

        # 4. Init Params
        self.uv_mask = torch.as_tensor(np.load('./Data/uv_mask_without_eye_256.npy').astype(np.float32)[None,None,...]).to(self.device)        
    
    @staticmethod
    def toTensor2D(maps):
        # [B,H,W,C]
        if len(maps.shape) == 4:
            return torch.as_tensor(maps).permute(0,3,1,2).contiguous()
        # [B,H,W]
        elif len(maps.shape) == 3:
            return torch.as_tensor(maps)[:,None,...]
        
    def set_input(self, inputs):
        self.img = self.toTensor2D(inputs['image']).to(self.device)
        self.coarse_tex_map = self.toTensor2D(inputs['coarse_tex']).to(self.device)
        self.im_tex = self.toTensor2D(inputs['im_tex']).to(self.device)
        self.param = inputs['param'].to(self.device)
        self.light_param = self.param[:,227:227+27]
        
    def forward(self):
        inputs = torch.cat([2 * self.im_tex - 1.0, 2 * self.coarse_tex_map - 1.0], dim=1)
        self.detail_tex_map, self.detail_light_map = self.generator(inputs)
        self.detail_tex_map = (self.detail_tex_map + 1.0) / 2.0
        self.detail_tex, self.detail_light = self.sampler(self.detail_tex_map.permute(0,2,3,1).contiguous()), self.sampler(self.detail_light_map.permute(0,2,3,1).contiguous())
        # Generate results

        # self.coarse_face_image_update,\
        # self.detail_tex_face_illu,\
        # self.detail_face_image,\
        # self.detail_face_image_update,\
        # self.light_image = self.renderer(self.param,\
        #                                               self.detail_tex,\
        #                                               self.detail_light)       
        self.detail_face_image, self.detail_face_image_illu = self.renderer(self.param,\
                                                      self.detail_tex,\
                                                      self.detail_light)
        self.detail_face_image = self.detail_face_image.permute(0,3,1,2).contiguous()                                                      
        
    def train(self):
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