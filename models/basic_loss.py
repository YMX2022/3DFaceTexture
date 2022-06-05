#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import numpy as np
from .inception_resnet_v1 import InceptionResnetV1
from .vggface import VGGFace
from .resnet50_task import resnet50_use
from torchvision.models import vgg16
from scipy.io import loadmat
from .utils import ImgGrad
from .arcface import Backbone

class LightRegLoss(nn.Module):
    
    def __init__(self, with_rgb_reg=True, loss_type='l1'):
        super().__init__()
        self.with_rgb_reg = with_rgb_reg
        if loss_type == 'l1':
            self.loss_func = torch.abs
        else:
            self.loss_func = torch.norm
     
    # light : [B,N,27]
    # light_param : [B,27]
    def forward(self, light, light_param, mask=None):
        loss = 0.0
        B, N = light.shape[:2]
        light_tmp = light.reshape(B, N, 3, 9)
        if self.with_rgb_reg:
            loss += self.loss_func(light_tmp - light_tmp.mean(dim=2, keepdim=True)).mean()        
        if mask is None:
            loss += self.loss_func(light-light_param[:,None,:].repeat(1,light.shape[1],1)).mean()
        else:
            loss += self.loss_func(mask * (light - light_param[:,None,:].repeat(1,light.shape[1],1)))
        return loss
                

class ChromaSmoothLoss(nn.Module):
    
    def __init__(self, uv_mask):
        super().__init__()
        self.uv_mask = uv_mask
        self.uv_size = torch.sum(uv_mask).float()
        self.network = ImgGrad(cin=3, pattern='smooth')
    
    def chromatic(self, maps):
        sim_x, sim_y = self.network(maps)   # sim_x, sim_y : [B,1,H,W]
        self.sim_x, self.sim_y = torch.exp(-80.0 * sim_x), torch.exp(-80.0 * sim_y)
        
    def forward(self, ref_map, inf_map):
        self.chromatic(ref_map)
        grad_x, grad_y = self.network(inf_map)
        return ((self.uv_mask * (self.sim_x.detach() * grad_x + self.sim_y.detach() * grad_y)).sum(dim=(-2,-1)) / self.uv_size).mean()
    
class MapSymmetryLoss(nn.Module):
    
    def __init__(self, uv_mask, thres=0.0):
        super().__init__()
        self.uv_mask = uv_mask
        self.uv_count = uv_mask.sum().item()
        self.thres = thres
        
    # uv_map : [None,3,224,224]
    def forward(self, uv_map):
        uv_map = self.uv_mask * uv_map
        uv_sym = (torch.flip(uv_map, dims=(-1,)).detach() - uv_map).norm(dim=-1)
        return (uv_sym.where(uv_sym>self.thres, torch.zeros(1).to(uv_map.device)).sum(dim=(1,2)) / self.uv_count).mean()
    
class MapSparsityLoss(nn.Module):
    
    def __init__(self, weight_mask):
        super().__init__()
        self.weight_mask = weight_mask[None,...,None]
        self.weight_sum = self.weight_mask.sum()
        
    # uv_map : [None,224,224,3]
    def forward(self, uv_map):
        return (torch.norm(self.weight_mask * uv_map, dim=-1).sum(dim=(1,2)) / self.weight_sum).mean()

# ### Photometric-loss
# - Using photometric loss between outputs and labels;
# - Return shape is $[nb,1]$

class MaskPhotoLoss(nn.Module):
    
    def __init__(self, loss_type='l2'):
        super().__init__()
        self.loss_type = loss_type
        
    # face_masks : [None, 1, H, W]
    # rendered_images : [None, 3, H, W]
    # face_images(GT) : [None, 3, H, W]
    def forward(self, rendered_images, face_images, face_masks):
        # [None,3,W,H]
        loss = ((rendered_images - face_images).norm(dim=1, keepdim=True) * face_masks).sum(dim=(1,2,3)) / face_masks.sum(dim=(1,2,3))
        return loss.mean()             
        
        
# ### Perceptual-loss
# - Using facenet as default;
# - Return cosine loss, shape is $[nb,1]$, whose value is in [0,2];

class GramMatLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    # maps : [B,C,H,W]    
    @staticmethod
    def gram_mat(maps):
        b,c = maps.shape[0:2]
        vecs = maps.permute(0,2,3,1).reshape(b,-1,c)
        return torch.matmul(vecs, vecs.permute(0,2,1))
    
    def forward(self, fm_a, fm_b):
        H, W = fm_a.shape[-2:]
        return (1 / (4 * H * H * W * W) * ((self.gram_mat(fm_a)-self.gram_mat(fm_b)) ** 2).sum(dim=(-2,-1))).mean()
    
    
class FeatureSpaceLoss(nn.Module):
    
    def __init__(self, perceptual_net='facenet', weights=[1.0,1.0], type='cos'):
        super().__init__()
        self.perceptual = perceptual_net
        if self.perceptual == 'facenet':
            self.network = InceptionResnetV1(pretrained='vggface2').eval()
        else:
            raise ValueError('No such perceptual net included!')
        self.type = type
        self.weights = weights
        if self.weights[1] > 0.0:
            self.style_loss = GramMatLoss()
            
    '''
        rendered_images : [0,1]
        face_images : [0,1]
    '''    
    def forward(self, rendered_images, face_images):
        face_images = face_images.detach()
        # Resize and renormalize
        rendered_images, face_images =\
        (2.0 * F.interpolate(rendered_images, mode='bilinear', size=160, align_corners=False) - 1.0),\
        (2.0 * F.interpolate(face_images, mode='bilinear', size=160, align_corners=False) - 1.0)
        rendered_fm, rendered_fv = self.network(rendered_images)
        face_fm, face_fv = self.network(face_images)
        loss = 0.0
        if self.type == 'cos':
            loss += (1.0 - torch.matmul(rendered_fv[:,None,:], face_fv[:,:,None]))[:,0].mean()
        if self.weights[1] > 0.0:
            loss += self.style_loss(rendered_fm, face_fm)
        return loss

class ArcFaceLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.network = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se').eval()
        
    '''
        rendered_images : [0,1]
        face_images : [0,1]
    '''    
    def forward(self, rendered_images, face_images):
        face_images = face_images.detach()
        # Resize and renormalize
        rendered_images, face_images =\
        (2.0 * F.interpolate(rendered_images, mode='bilinear', size=112, align_corners=False) - 0.5),\
        (2.0 * F.interpolate(face_images, mode='bilinear', size=112, align_corners=False) - 0.5)
        render_vec = self.network(rendered_images)
        face_vec = self.network(face_images)
        return (1.0 - torch.matmul(render_vec[:,None,:], face_vec[...,None])).mean()    
    
class FaceNetLoss(nn.Module):
    
    def __init__(self, perceptual_net='facenet'):
        super().__init__()
        self.perceptual = perceptual_net
        if self.perceptual == 'facenet':
            self.network = InceptionResnetV1(pretrained='vggface2').eval()
        else:
            raise ValueError('No such perceptual net included!')
            
    '''
        rendered_images : [0,1]
        face_images : [0,1]
    '''    
    def forward(self, rendered_images, face_images):
        face_images = face_images.detach()
        # Resize and renormalize
        rendered_images, face_images =\
        (2.0 * F.interpolate(rendered_images, mode='bilinear', size=160, align_corners=False) - 1.0),\
        (2.0 * F.interpolate(face_images, mode='bilinear', size=160, align_corners=False) - 1.0)
        rendered_fv = self.network(rendered_images)
        face_fv = self.network(face_images)
        loss = 0.0
        loss += (1.0 - torch.matmul(rendered_fv[:,None,:], face_fv[:,:,None]))[:,0].mean()
        return loss    

class RegDetailLoss(nn.Module):
    
    def __init__(self, type='l1'):
        super().__init__()
        self.type = type
        
    '''
        render : [None,35709,3],
        detail_render : [None,35709,3],
    '''
    def forward(self, render, detail_render):
        if self.type == 'l2':
            return torch.norm(render - detail_render, dim=-1).mean(dim=(0,1))
        elif self.type == 'l1':
            return torch.abs(render-detail_render).sum(dim=-1).mean(dim=(0,1))
        
class RegDetailMapLoss(nn.Module):
    
    def __init__(self, uv_mask, type='l2'):
        super().__init__()
        self.type = type
        self.uv_mask = uv_mask
        self.uv_pix_sum = torch.sum(self.uv_mask)
        
    '''
        coarse_tex_map : [None,C,H,W],
        detail_tex_map : [None,C,H,W],
    '''
    def forward(self, coarse_tex_map, detail_tex_map):
        if self.type == 'l2':
            return ((self.uv_mask * torch.norm(coarse_tex_map - detail_tex_map, dim=1, keepdim=True)).sum((1,2,3)) / self.uv_pix_sum).mean()
        elif self.type == 'l1':
            return ((self.uv_mask * torch.abs(coarse_tex_map - detail_tex_map, dim=1, keepdim=True)).sum((1,2,3)) / self.uv_pix_sum).mean()
            
class LSGANLoss(nn.Module):
    
    def __init__(self, target_real=1.0, target_fake=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.as_tensor(target_real))
        self.register_buffer('fake_label', torch.as_tensor(target_fake))
        self.gan_loss = nn.MSELoss()
        
    def forward(self, pred, is_real):
        if is_real:
            return self.gan_loss(pred, self.real_label.expand_as(pred))
        else:
            return self.gan_loss(pred, self.fake_label.expand_as(pred))
        
class OrigGANLoss(nn.Module):
    
    def __init__(self, target_real=1.0, target_fake=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.as_tensor(target_real))
        self.register_buffer('fake_label', torch.as_tensor(target_fake))
        self.gan_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, pred, is_real):
        if is_real:
            return self.gan_loss(pred, self.real_label.expand_as(pred))
        else:
            return self.gan_loss(pred, self.fake_label.expand_as(pred))
    