from torch.utils.data import Dataset
import os.path as osp
import numpy as np
from PIL import Image

class DataUniversal(Dataset):
    
    def __init__(self, root='../eval_data', name='Zhoukun'):
        super().__init__()
        data_root = osp.join(root, name)
        with open(osp.join(data_root, 'file_list.txt')) as fh:
            self.file_list = fh.readlines()
        self.length = len(self.file_list)
        self.im_folder = osp.join(data_root, 'outputs/img')
        self.coarse_tex_folder = osp.join(data_root, 'outputs/tex_map')
        self.im_tex_folder = osp.join(data_root, 'outputs/im_tex_uv')
        self.gen_mask_folder = osp.join(data_root, 'outputs/gen_mask')        
        self.params = np.load(osp.join(data_root, 'outputs/params.npy')).astype(np.float32)
        self.uv_mask = np.load('./Data/uv_mask_256.npy')[...,None]
        assert(len(self.params) == self.length)
        
    def __len__(self):
        return self.length
        
    @staticmethod
    def normImg(img):
        if img.shape[-1] > 3:
            img = img[...,:-1]
        if img.max() > 1:
            img = img / 255.0
        return img
        
    def __getitem__(self, index):
        param = self.params[index]
        fn = self.file_list[index].strip()        
        im = np.asarray(Image.open(osp.join(self.im_folder, fn))).astype(np.float32)
        coarse_tex = np.load(osp.join(self.coarse_tex_folder, fn.replace('.jpg', '.npy'))).astype(np.float32)
        im_tex = np.load(osp.join(self.im_tex_folder, fn.replace('.jpg', '.npy'))).astype(np.float32)
        zero_indices = (im_tex.sum(axis=-1) == 0)
        im_tex[zero_indices,:] = np.random.normal(0,1,size=[int(zero_indices.sum()),3])
        im_tex *= self.uv_mask
        gen_mask = np.load(osp.join(self.gen_mask_folder, fn.replace('.jpg', '.npy'))).astype(np.float32)
        # Normalize 0-1, if needed
        im = self.normImg(im)
        coarse_tex = self.normImg(coarse_tex)
        data_dict = {
                        'image' : im,
                        'coarse_tex' : coarse_tex,
                        'im_tex' : im_tex,
                        'fn' : fn.replace('.jpg', ''),
                        'param' : param,
                        'gen_mask' : gen_mask
                    }
        return data_dict
    