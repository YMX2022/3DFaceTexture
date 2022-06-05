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
        self.mask_tex_folder = osp.join(data_root, 'outputs/mask_tex')
        self.img_tex_folder = osp.join(data_root, 'outputs/im_tex')        
        self.params = np.load(osp.join(data_root, 'outputs/params.npy')).astype(np.float32)
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
        img_tex = np.load(osp.join(self.img_tex_folder, fn.replace('.jpg', '.npy'))).astype(np.float32).squeeze()
        mask_tex = np.load(osp.join(self.mask_tex_folder, fn.replace('.jpg', '.npy'))).astype(np.float32).squeeze()[...,0:1]
        im = self.normImg(im)
        data_dict = {
                        'image' : im,
                        'im_tex' : img_tex,
                        'mask_tex' : mask_tex,
                        'fn' : fn.replace('.jpg', ''),
                        'param' : param
                    }
        return data_dict
    