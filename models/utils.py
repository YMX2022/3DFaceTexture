import torch
import torch.nn as nn
from .interface import readConfig, MatFile
import numpy as np

class UVSampleLayer(nn.Module):
    
    # Return uv-coordinate([0,1])
    @staticmethod
    def loadAndParameterization(uv_mat_path, select_id_path):
        uv_mat_reader = MatFile()
        select_id_reader = MatFile()
        if uv_mat_reader(uv_mat_path) == 0:
            uv = getattr(uv_mat_reader, 'UV')
        if select_id_reader(select_id_path) == 0:
            select_id = getattr(select_id_reader, 'select_id') - 1
        return uv[select_id].squeeze()
    
    # Turn uv-coordinates to uv-image coordinates
    @staticmethod
    def process_uv(uv_coords, uv_size=224):
        uv_coords[:,0] = uv_coords[:,0] * (uv_size-1)
        uv_coords[:,1] = uv_coords[:,1] * (uv_size-1)
        uv_coords[:,1] = (uv_size-1) - uv_coords[:,1]
        return uv_coords
    
    def sampleWeights(self):
        if self.uv is None:
            raise ValueError('uv should be initialized first!!!')
        if self.config['uv_inter_type'] == 'bilinear':
            u_low = np.floor(self.uv[:,0])
            u_high = np.ceil(self.uv[:,0])
            v_low = np.floor(self.uv[:,1])
            v_high = np.ceil(self.uv[:,1])
            v_high[np.where(v_low == v_high)] += 1
            u_high[np.where(u_low == u_high)] += 1
            u,v = self.uv[:,0], self.uv[:,1]
            w_u, w_v = (u - u_low) / (u_high - u_low), (v - v_low) / (v_high - v_low)
            self.register_buffer('weight_u', torch.as_tensor(w_u)[None,:,None].float())
            self.register_buffer('weight_v', torch.as_tensor(w_v)[None,:,None].float())
            # transform to torch
            self.register_buffer('u_low', torch.as_tensor(u_low).long())
            self.register_buffer('v_low', torch.as_tensor(v_low).long())
            self.register_buffer('u_high', torch.as_tensor(u_high).long())
            self.register_buffer('v_high', torch.as_tensor(v_high).long())
        elif self.config['uv_inter_type'] == 'nn':
            u_round = np.round(self.uv[:,0])
            v_round = np.round(self.uv[:,1])
            self.register_buffer('u_round', torch.as_tensor(u_round).long())
            self.register_buffer('v_round', torch.as_tensor(v_round).long())
                
    def __init__(self, config):
        super().__init__()
        self.config = readConfig(config)
        self.uv = self.loadAndParameterization(self.config['uv_mat_path'], self.config['select_id_path'])
        self.uv = self.process_uv(self.uv, uv_size=self.config['uv_size'])
        self.sampleWeights()        
        
    def forward(self, attr_map):
        if self.config['uv_inter_type'] == 'bilinear':
            u0v0 = attr_map[:, self.v_low, self.u_low, :]
            u1v0 = attr_map[:, self.v_high, self.u_high, :]
            u0v1 = attr_map[:, self.v_high, self.u_low, :]
            u1v1 = attr_map[:, self.v_high, self.u_high, :]

            tmp0 = self.weight_u * (u1v0 - u0v0) + u0v0
            tmp1 = self.weight_u * (u1v1 - u0v1) + u0v1
            return self.weight_v * (tmp1 - tmp0) + tmp0
        elif self.config['uv_inter_type'] == 'nn':
            return attr_map[:,self.v_round, self.u_round, :]

class ImgGrad(nn.Module):

    def __init__(self, cin, pattern='grad_loss'):
        super().__init__()
        if pattern == 'grad_loss':
            grad_filter = torch.as_tensor([[0, 0.5, 0], [-0.5, 0, 0.5], [0, -0.5, 0]]).repeat(1, cin, 1, 1)
            self.grad_conv = nn.Conv2d(in_channels=cin, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)
            self.grad_conv.weight.data = grad_filter
            self.grad_conv.weight.requires_grad = False
        elif 'smooth' in pattern:
            grad_filter_x = torch.as_tensor([[0, 0, 0], [0, -1.0, 1.0], [0, 0, 0]]).repeat(cin, cin, 1, 1)
            grad_filter_y = torch.as_tensor([[0, 1.0, 0], [0, -1.0, 0], [0, 0, 0]]).repeat(cin, cin, 1, 1)
            self.grad_conv_x = nn.Conv2d(in_channels=cin, out_channels=cin, kernel_size=3, stride=1, padding=0, bias=False)
            self.grad_conv_x.weight.data = grad_filter_x
            self.grad_conv_y = nn.Conv2d(in_channels=cin, out_channels=cin, kernel_size=3, stride=1, padding=0, bias=False)
            self.grad_conv_y.weight.data = grad_filter_y
            self.grad_conv_x.weight.requires_grad = False
            self.grad_conv_y.weight.requires_grad = False
        self.pattern = pattern
        self.pad = nn.ReplicationPad2d(1)        

    # image : [None,3,224,224]
    def forward(self, image):
        if self.pattern == 'grad_loss':
            maps = self.pad(image)
            return self.grad_conv(maps)
        elif self.pattern == 'smooth_loss':
            maps = self.pad(image)
            return torch.abs(self.grad_conv_x(maps).norm(dim=1)) + torch.abs(self.grad_conv_y(maps).norm(dim=1))
        elif self.pattern == 'smooth':
            maps = self.pad(image)
            return torch.abs(self.grad_conv_x(maps).norm(dim=1)), torch.abs(self.grad_conv_y(maps).norm(dim=1))
