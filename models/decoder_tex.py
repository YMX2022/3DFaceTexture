#!/usr/bin/env python
# coding: utf-8

from .interface import MatFile, H5File, readConfig
import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pt_mesh_renderer.mesh_renderer as mesh_renderer
    
class FaceRenderer(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        # Config Params
        self.config = readConfig(config)
        if self.config['proxy'] == 'BFM09':
            self.register_buffer('triangles', BFM09(self.config['proxy_path']).triangles.long())
        self.im_size = int(self.config['image_size'])
        # Camera Params
        self.fov_y = 2 * np.arctan(112/1015.0)*180.0/np.pi
        self.register_buffer('camera_position',torch.as_tensor([0,0,10.0]).float())
        self.register_buffer('camera_lookat',torch.as_tensor([0.0,0.0,0.0]).float())
        self.register_buffer('camera_up',torch.as_tensor([0.0,1.0,0.0]).float())
        self.register_buffer('light_positions',torch.as_tensor([0,0,1e5]).float()[None,None,:])
        self.register_buffer('light_intensities',torch.zeros(1,3).float()[None,...])
        self.register_buffer('ambient_color',torch.as_tensor([1.0,1.0,1.0]).float()[None,:])
        self.near_clip = 0.01
        self.far_clip = 50.0
    
    # default color : [0,255]
    def forward(self, shape_rt, color):
        B = color.shape[0]
        renderer, _ = mesh_renderer.mesh_renderer(vertices=shape_rt,\
                                 triangles=self.triangles,\
                                 normals=torch.zeros_like(shape_rt).float().to(shape_rt.device),\
                                 diffuse_colors=color,\
                                 camera_position=self.camera_position,\
                                 camera_lookat=self.camera_lookat,\
                                 camera_up=self.camera_up,
                                 light_positions=self.light_positions.repeat(B,1,1),
                                 light_intensities=self.light_intensities.repeat(B,1,1),\
                                 image_width=self.im_size,
                                 image_height=self.im_size,
                                 fov_y=self.fov_y,
                                 ambient_color=self.ambient_color.repeat(B,1),\
                                 near_clip=self.near_clip,
                                 far_clip=self.far_clip
                                 )
        # Return : [None,224,224,3]
        return renderer[...,:-1]  
    
    
    
class FaceModelBase(nn.Module):
    def __init__(self, save_type='mat'):
        super().__init__()
        if save_type == 'mat':
            self.reader = MatFile()
        elif save_type == 'h5':
            self.reader = H5File()
    
    def parse_file(self):
        pass
        
# Todo: Extende the model to full BFM2009 and maybe full-head BFM2019
# Including necessary information of BFM2009
class BFM09(FaceModelBase):
    def __init__(self, mat_path):
        super().__init__('mat')
        if self.reader(mat_path) == 0:
            self.parse_file()
        else:
            print('[Error] Reading .mat file failed!!!')

    def parse_file(self):
        self.register_buffer('shape_mean', torch.from_numpy(getattr(self.reader, 'meanshape').reshape(-1)).float())
        self.register_buffer('tex_mean', torch.from_numpy(getattr(self.reader, 'meantex').reshape(-1)).float())
        self.register_buffer('id_base', torch.from_numpy(getattr(self.reader, 'idBase')).float())
        self.register_buffer('ex_base', torch.from_numpy(getattr(self.reader, 'exBase')).float())
        self.register_buffer('tex_base', torch.from_numpy(getattr(self.reader, 'texBase')).float())
        self.register_buffer('point_buf', torch.from_numpy(getattr(self.reader, 'point_buf') - 1).long())  # adjacent face index for each vertex, starts from 1(used for calculate vertex normal)
        self.register_buffer('kps', torch.from_numpy(np.squeeze(getattr(self.reader, 'keypoints')).astype(np.int32) - 1).long())    # vertex index for all the landmark points
        self.register_buffer('triangles', torch.from_numpy(getattr(self.reader, 'tri') - 1).long())    # vertex index for all the triangles, start from 1

# params_vec : [None, 257]
class SplitParams(nn.Module):
    def __init__(self, config):
        super().__init__()
        key_list = ['id','exp','tex','projection','illumination','XY','Z']
        self.n_params_vec = []
        for key in key_list:
            self.n_params_vec.append(config[key])
        self.n_params_vec[-2] = self.n_params_vec[-2] + self.n_params_vec[-1]
        self.n_params_vec.pop()

    def forward(self, params_vec):
        params = []
        lower = 0
        upper = self.n_params_vec[0]
        for i in range(len(self.n_params_vec) - 1):
            params.append(params_vec[:,lower:upper])
            lower += self.n_params_vec[i]
            upper += self.n_params_vec[i+1]
        params.append(params_vec[:,lower:].to(params_vec.device))
        return params

# Using the regressed params to reconstruct
# mesh, texture, illumination and transformation
class ReconstructionLayer(nn.Module):
    
    ## id_params : [None, 80]
    ## ex_params : [None, 64]
    ## return : [None, N, 3]
    def shapeFormation(self, id_params, ex_params):
        id_params = id_params.unsqueeze(-1)
        ex_params = ex_params.unsqueeze(-1)
        shape_mean = self.proxy_model.shape_mean.unsqueeze(0).unsqueeze(-1)
        id_base = self.proxy_model.id_base.unsqueeze(0)
        ex_base = self.proxy_model.ex_base.unsqueeze(0)
        face_shape = shape_mean
        id_shape = torch.matmul(id_base, id_params)
        face_shape = face_shape + id_shape
        deforming_shape = torch.matmul(ex_base, ex_params)
        face_shape += deforming_shape
        ## re-center faceshape
        face_shape = face_shape.reshape(face_shape.size()[0], -1, 3) - torch.mean(shape_mean.reshape(1, -1, 3), dim=1, keepdim=True)
        return face_shape

    ## tex_params : [None, 80]
    ## return : [None, N, 3]
    def texFormation(self, tex_params):
        tex_params = tex_params.unsqueeze(-1)
        tex_mean = self.proxy_model.tex_mean.unsqueeze(0).unsqueeze(-1)
        tex_base = self.proxy_model.tex_base.unsqueeze(0)
        face_tex = tex_mean + torch.matmul(tex_base, tex_params)
        return face_tex.reshape(face_tex.size()[0], -1, 3)

    ## Eular-angle_params : [None, 3]
    ## return column-multiply-matrix : [None, 3, 3]
    def rotationFormation(self, trans_params):
        angle_x = trans_params[:,0]
        angle_y = trans_params[:,1]
        angle_z = trans_params[:,2]
        device = trans_params.device
        ones = torch.ones_like(angle_x).to(device)
        zeros = torch.zeros_like(angle_x).to(device)
        rot_x = torch.stack([ones,zeros,zeros,zeros,torch.cos(angle_x),-torch.sin(angle_x),zeros,torch.sin(angle_x),torch.cos(angle_x)], dim=-1).reshape(-1,3,3)
        rot_y = torch.stack([torch.cos(angle_y),zeros,torch.sin(angle_y),zeros,ones,zeros,-torch.sin(angle_y),zeros,torch.cos(angle_y)], dim=-1).reshape(-1,3,3)
        rot_z = torch.stack([torch.cos(angle_z),-torch.sin(angle_z),zeros,torch.sin(angle_z),torch.cos(angle_z),zeros,zeros,zeros,ones], dim=-1).reshape(-1,3,3)
        rot = torch.matmul(torch.matmul(rot_z, rot_y), rot_x).permute(0,2,1)
        return rot

    # SH function illumination
    ## illu_params : [None, 27]
    ## face_norms : [None, N, 3]
    ## diffuse_albedo : [None, N, 3]
    def illuFormation(self, illu_params, face_norms, diffuse_albedo):
        # Compute Illu SH-coeff
        # illu_vecotr : [None, 9, 3]
        illu_vector = illu_params.reshape(illu_params.shape[0], 3, 9).permute(0,2,1).contiguous()
        illu_vector[:,0,:] += 0.8
        # Compute Transfer SH-coeff
        a0 = np.float(np.pi)
        a1 = np.float(2*np.pi/np.sqrt(3.0))
        a2 = np.float(2*np.pi/np.sqrt(8.0))
        c0 = np.float(1/np.sqrt(4*np.pi))
        c1 = np.float(np.sqrt(3.0)/np.sqrt(4*np.pi))
        c2 = np.float(3*np.sqrt(5.0)/np.sqrt(12*np.pi))
        
        norm = face_norms.view(-1, 3)
        nx, ny, nz = norm[:,0], norm[:,1], norm[:,2]
        
        ones = torch.ones(face_norms.shape[0]*face_norms.shape[1], dtype=torch.float32)
        ones = ones.to(illu_params.device)
        transfer_list = []
        transfer_list.append(a0*c0 * ones)
        transfer_list.append(-a1*c1 * ny)
        transfer_list.append(a1*c1 * nz)
        transfer_list.append(-a1*c1 * nx)
        transfer_list.append(a2*c2 * nx*ny)
        transfer_list.append(-a2*c2 * ny*nz)
        transfer_list.append(a2*c2*0.5/np.sqrt(3.0) * (3*torch.pow(nz,2)-1))
        transfer_list.append(-a2*c2* nx * nz)
        transfer_list.append(a2*c2*0.5*(torch.pow(nx,2)-torch.pow(ny,2)))
        # transfer_vector shape : [None, N, 9]
        transfer_vector = torch.stack(transfer_list, dim=-1).reshape(-1,face_norms.shape[1],9)    
        return (diffuse_albedo * torch.bmm(transfer_vector, illu_vector))    
    
    def illuFormationUpdate(self, illu, face_norms, diffuse_albedo):
        B, N = face_norms.shape[:2]        
        # Compute Illu SH-coeff
        # illu_vecotr : [B,N,9,3]
        illu_vector = illu.reshape(B,N,3,9).permute(0,1,3,2).contiguous()
        illu_vector[:,:,0,:] += 0.8
        # Compute Transfer SH-coeff
        a0 = np.float(np.pi)
        a1 = np.float(2*np.pi/np.sqrt(3.0))
        a2 = np.float(2*np.pi/np.sqrt(8.0))
        c0 = np.float(1/np.sqrt(4*np.pi))
        c1 = np.float(np.sqrt(3.0)/np.sqrt(4*np.pi))
        c2 = np.float(3*np.sqrt(5.0)/np.sqrt(12*np.pi))
        
        norm = face_norms.view(-1, 3)
        nx, ny, nz = norm[:,0], norm[:,1], norm[:,2]
        
        ones = torch.ones(face_norms.shape[0]*face_norms.shape[1], dtype=torch.float32)
        ones = ones.to(face_norms.device)
        transfer_list = []
        transfer_list.append(a0*c0 * ones)
        transfer_list.append(-a1*c1 * ny)
        transfer_list.append(a1*c1 * nz)
        transfer_list.append(-a1*c1 * nx)
        transfer_list.append(a2*c2 * nx*ny)
        transfer_list.append(-a2*c2 * ny*nz)
        transfer_list.append(a2*c2*0.5/np.sqrt(3.0) * (3*torch.pow(nz,2)-1))
        transfer_list.append(-a2*c2* nx * nz)
        transfer_list.append(a2*c2*0.5*(torch.pow(nx,2)-torch.pow(ny,2)))
        # transfer_vector shape : [None, N, 9]
        transfer_vector = torch.stack(transfer_list, dim=-1).reshape(-1,face_norms.shape[1],9)
        new_light = torch.matmul(transfer_vector[...,None,:], illu_vector)[:,:,0,:]
        return (diffuse_albedo * new_light)
    
        
    # Compute vertex-normal or face-normal
    ## face_shape : [None, N, 3]
    ## face_model : A facemodel class object
    ## return : 'vertex':[None, N, 3]
    def computeNorm(self, face_shape, face_model):
        # Calculate face norm first
        tri = face_model.triangles
        e1 = face_shape[:,tri[:,0],:] - face_shape[:,tri[:,1],:]
        e2 = face_shape[:,tri[:,1],:] - face_shape[:,tri[:,2],:]
        face_norm = torch.cross(e1, e2)
        face_norm = nn.functional.normalize(face_norm, dim=-1)
        # Calculate vertex norm then
        point_buf = face_model.point_buf
        device = face_shape.device
        face_norm = torch.cat([face_norm, torch.zeros(face_norm.shape[0],1,3, dtype=torch.float32, device=device)], dim=1)
        vertex_norm = torch.sum(face_norm[:,point_buf,:], dim=2)
        vertex_norm = nn.functional.normalize(vertex_norm, dim=2)
        return vertex_norm


    def rigidTrans(self, shape, rot_params, xyz_params):
        # rotation formation
        rot_mat = self.rotationFormation(rot_params)
        xyz = torch.unsqueeze(xyz_params, dim=1).float()
        shape_rt = torch.matmul(shape, rot_mat) + xyz
        return shape_rt        

    # Project Mesh to Image
    def projection(self, shape_rt):
        shape_rt_p = shape_rt + self.cam_trans
        shape_rt_p = torch.matmul(shape_rt_p, self.cam_mat)
        depth = shape_rt_p[...,-1]
        shape_rt_p = shape_rt_p / depth.unsqueeze(-1)
        shape_rt_p[...,-1] = depth
        return shape_rt_p

    # Compute landmark coordinates
    # face_shape : [None, N, 3]
    def computeLandmark(self, face_shape):
        kps_index = self.proxy_model.kps
        return face_shape[:,kps_index,:]
    
    # All the constants initialized as buffers
    def register_constants(self):
        # Used in projection, Camera Internal matrix
        half_w = np.float32(self.image_size / 2)
        ## camera external formation
        cam_trans = torch.from_numpy(np.array([0,0,self.cam_trans_z]))[None,None,:].float()
        ## camera internal formation(from Camera-Space to NDC to Image-Space)
        cam_mat = torch.from_numpy(np.array([self.focal,0,half_w,0,-self.focal,half_w,0,0,1]).reshape(3,3)).unsqueeze(0).float()
        cam_mat = cam_mat.permute(0,2,1)
        self.register_buffer('cam_trans', cam_trans)
        self.register_buffer('cam_mat', cam_mat)

    def __init__(self, config):
        super().__init__()
        ## Load config dict
        self.config = readConfig(config)
        if self.config['proxy'] == 'BFM09':
            self.proxy_model = BFM09(self.config['proxy_path'])
        self.splitParams = SplitParams(self.config)
        # Image Attrs
        self.image_size = float(self.config['image_size'])
        # Camera Attrs
        self.focal = float(self.config['focal'])
        self.cam_trans_z = float(self.config['cam_trans_z'])
        self.register_constants()
        self.face_renderer = FaceRenderer(config)

    # Todo : Check whether config-file matches the BFM-model
    # If not, cutting BFM-model
    def checkConfig(self):
        pass
    
    def forward(self, params, detail_tex, light=None):
        params_id, params_ex, params_tex, params_rot, params_illu, params_xyz = self.splitParams(params)
        # 3dmm decode
        shape = self.shapeFormation(params_id, params_ex)
        coarse_tex = self.texFormation(params_tex)
        # tex gen
        self.norm = self.computeNorm(shape, self.proxy_model)
        detail_tex_face_illu = self.illuFormation(params_illu, self.norm, detail_tex * 255.0) / 255.0
        if light is not None:
            detail_tex_face_illu_update = self.illuFormationUpdate(light, self.norm, detail_tex * 255.0) / 255.0
            coarse_tex_face_illu_update = self.illuFormationUpdate(light, self.norm, coarse_tex) / 255.0
            light_sample = self.illuFormationUpdate(light, self.norm, torch.ones_like(coarse_tex).to(coarse_tex.device) * 255.0) / 255.0
#             coarse_tex_face = self.illuFormationUpdate(light, self.norm, coarse_tex) / 255.0
        # shape gen
        self.shape_rt = self.rigidTrans(shape, params_rot, params_xyz)
        # Render
        coarse_face_image_update = self.face_renderer(self.shape_rt, coarse_tex_face_illu_update)
        detail_face_image = self.face_renderer(self.shape_rt, detail_tex_face_illu)
        detail_face_image_update = self.face_renderer(self.shape_rt, detail_tex_face_illu_update)
        light_sample_image = self.face_renderer(self.shape_rt, light_sample)
#         coarse_face_image = self.face_renderer(self.shape_rt, coarse_tex_face)    
#         return detail_tex_face, detail_face_image, (coarse_tex/255.0), coarse_face_image            
        return coarse_face_image_update, detail_tex_face_illu_update, detail_face_image, detail_face_image_update, light_sample_image