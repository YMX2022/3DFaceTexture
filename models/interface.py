# %%
import os
import json
import codecs
from collections import OrderedDict
from scipy.io import loadmat
import h5py
import struct
import numpy as np
import matplotlib.pyplot as plt

# %%
class BinFile:
    def __init__(self):
        pass

    def call(self, bin_file_path):
        if not os.path.isfile(bin_file_path):
            print('[Error] Wrong input .bin file!!!')
            return -1
        if mat_file_path[-3:] != 'bin':
            print('[Error] Not an .bin file!!!')
            return -1
        with open(bin_file_path, 'rb') as fh:
            size = os.path.getsize(bin_file_path)
            for i in range(size):
                data = fh.read(1)
                n
# %%
class MatFile:
    def __init__(self):
        pass

    def call(self, mat_file_path):
        if not os.path.isfile(mat_file_path):
            print('[Error] Wrong input .mat file!!!')
            return -1
        if mat_file_path[-3:] != 'mat':
            print('[Error] Not an .mat file!!!')
            return -1
        mat = loadmat(mat_file_path)
        # If has inputs, clear current and read next mat file
        if self.__dict__:
            self.__dict__.clear()
        for key in mat.keys():
            if '__' not in key:
                self.__dict__[key] = mat[key]
        self.key_list = list(self.__dict__.keys())
        return 0

    def __call__(self, mat_file_path):
        return self.call(mat_file_path)

    def __getitem__(self, index):
        if self.key_list is None:
            print('[Error] Should read-in a mat file first!!!')
        str_index = self.key_list[index]
        return getattr(self, str_index)
# %%
class ObjFile:
    def __init__(self):
        # According to baidu, .obj file should not contain info
        # of color, but Deep3D output obj maybe has color-info in it;
        self.key_ele_dict = {'v':3, 'f':3, 'vt':3, 'vn':3}
        self.v = []
        self.c = []
        self.f = []
        self.vt = []
        self.vn = []

    def clear(self):
        self.v = []
        self.c = []
        self.f = []
        self.vt = []
        self.vn = []

    def writeObj(self, obj_write_path):
        with open(obj_write_path, 'w') as fh:
            line_lists = []
            if len(self.v) != 0 and len(self.c) != 0:
                for v, c in zip(self.v, self.c):
                    line_lists.append('v {} {} {} {} {} {}\n'.format(v[0],v[1],v[2],c[0],c[1],c[2]))
            if len(self.f) != 0:
                for f in self.f:
                    line_lists.append('f {} {} {}\n'.format(f[0],f[1],f[2]))
            fh.writelines(line_lists)

    def _parse_obj(self, obj_file_path):
        if not os.path.isfile(obj_file_path):
            print('[Error] Input .obj file doesnot exit!!!')
            return -1
        if obj_file_path[-3:] != 'obj':
            print('[Error] Not an .obj file!!!')
            return -1
        with open(obj_file_path, 'r') as fh:
            obj_info = [l.strip() for l in fh.readlines()\
                        if len(l.split())>0 and not l.startswith('#')]
            for line in obj_info:
                lsp = line.split()
                if lsp[0] in self.key_ele_dict.keys():
                    if lsp[0] == 'v':
                        if len(lsp) - 1 == self.key_ele_dict[lsp[0]]:
                            self.__dict__[lsp[0]].append(np.array([lsp[1:]], dtype=np.float))
                        elif len(lsp) - 1 == 6:     # Including color info
                            self.__dict__[lsp[0]].append(np.array([lsp[1:4]], dtype=np.float))
                            self.__dict__['c'].append(np.array([lsp[4:]], dtype=np.float))
                    elif lsp[0] == 'f':
                        assert(len(lsp) - 1 == self.key_ele_dict[lsp[0]])
                        self.__dict__[lsp[0]].append(np.array([lsp[1:]], dtype=np.int))
                    else:
                        assert(len(lsp) - 1 == self.key_ele_dict[lsp[0]])
                        self.__dict__[lsp[0]].append(np.array([lsp[1:]], dtype=np.float))
        return 0

    def call(self, obj_file_path):
        # If has content, clear that and read next one
        if len(self.v) != 0 or isinstance(self.v, np.ndarray):
            self.clear()
        flag = self._parse_obj(obj_file_path)
        for key in self.__dict__:
            if isinstance(self.__dict__[key], list):
                self.__dict__[key] = np.squeeze(np.array(self.__dict__[key]))
        return flag

    def __call__(self, obj_file_path):
        return self.call(obj_file_path)
        
# %%
# unfinished
class H5File:
    def __init__(self):
        pass

# %%
class Deep3DCoeffParser:
    def __init__(self, obj_path, mat_path):
        mat = MatFile()
        obj = ObjFile()
        if mat(mat_path) == -1 or obj(obj_path) == -1:
            print('[Error] Wrong input file path!!!')
            return
        try:
            coeff = getattr(mat, 'coeff')
            camera_tuple = self.Split_coeff(coeff, 'camera')
        except AttributeError as e:
            print('[Error] Wrong mat file, no coeff key in it!!!')            
    
    # Deep3DCoeffParser provided
    @staticmethod
    def Split_coeff(coeff, key):
        assert isinstance(key, str)
        ref_key_list = ['identity', 'expression', 'texture', 'lighting', 'camera']
        if key not in ref_key_list:
            print('[Error] Wrong key, there is no {} in coeff!!!'.format(key))
            return
        id_coeff = coeff[:,:80] # identity(shape) coeff of dim 80
        ex_coeff = coeff[:,80:144] # expression coeff of dim 64
        tex_coeff = coeff[:,144:224] # texture(albedo) coeff of dim 80
        angles = coeff[:,224:227] # eular angles(x,y,z) for rotation of dim 3
        gamma = coeff[:,227:254] # lighting coeff for 3 channel SH function of dim 27
        translation = coeff[:,254:] # translation coeff of dim 3
        if key == 'identity':
            return id_coeff
        elif key == 'expression':
            return ex_coeff
        elif key == 'texture':
            return tex_coeff
        elif key == 'lighting':
            return gamma
        elif key == 'camera':
            return (angles, translation)
        
# %%
def saveConfig(f_path):
    content = {}
    content['id'] = 80
    content['exp'] = 64
    content['tex'] = 80
    content['projection'] = 3
    content['illumination'] = 27
    content['XY'] = 2
    content['Z'] = 1

    json_content = json.dumps(content, ensure_ascii=False, indent=0)
    # with open(f_path, 'w') as fh:
    #     fh.write(json_content)
    with codecs.open(f_path, 'w', 'utf-8') as file:
        file.write(json_content)

# %%
def readConfig(f_path):
    if not os.path.isfile(f_path):
        print('[Error] No configure file!!!')
    with open(f_path, 'r') as fh:
        return json.load(fh)
