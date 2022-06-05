import numpy as np 
from scipy.io import loadmat,savemat
from PIL import Image
import sys
sys.path.append('./utils')
from file_parser import MatFile

#calculating least sqaures problem
def POS(xp,x):
    npts = xp.shape[1]

    A = np.zeros([2*npts,8])

    A[0:2*npts-1:2,0:3] = x.transpose()
    A[0:2*npts-1:2,3] = 1

    A[1:2*npts:2,4:7] = x.transpose()
    A[1:2*npts:2,7] = 1;

    b = np.reshape(xp.transpose(),[2*npts,1])

    k,_,_,_ = np.linalg.lstsq(A,b,rcond=None)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
    t = np.stack([sTx,sTy],axis = 0)

    return t,s

def process_img(img,lm,t,s,mask=None,target_size = 224.):
    w0,h0 = img.size
    w = (w0/s*102).astype(np.int32)
    h = (h0/s*102).astype(np.int32)
    img = img.resize((w,h),resample = Image.BICUBIC)

        
    left = (w/2 - target_size/2 + float((t[0] - w0/2)*102/s)).astype(np.int32)
    right = left + target_size
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*102/s)).astype(np.int32)
    below = up + target_size

    img = img.crop((left,up,right,below))
    
    img = np.array(img)
    # img = img[:,:,::-1] #RGBtoBGR
    # img = np.expand_dims(img,0)
    lm = np.stack([lm[:,0] - t[0] + w0/2,lm[:,1] - t[1] + h0/2],axis = 1)/s*102
    lm = lm - np.reshape(np.array([(w/2 - target_size/2),(h/2-target_size/2)]),[1,2])
    if mask is not None:
        mask = mask.resize((w,h),resample = Image.BICUBIC)
        mask = mask.crop((left,up,right,below))
        mask = np.array(mask)
        return img,mask,lm
    else:
        return img, lm


# resize and crop input images before sending to the R-Net
def Preprocess(img,lm,mask=None):
    w0,h0 = img.size

    # change from image plane coordinates to 3D sapce coordinates(X-Y plane)
    lm = np.stack([lm[:,0],h0 - 1 - lm[:,1]], axis = 1)
    lm_idx = np.array([31,37,40,43,46,49,55]) - 1
    if len(lm) > 5:
        lm_aligned = np.stack([lm[lm_idx[0],:],np.mean(lm[lm_idx[[1,2]],:],0),np.mean(lm[lm_idx[[3,4]],:],0),lm[lm_idx[5],:],lm[lm_idx[6],:]], axis = 0)
        lm_aligned = lm_aligned[[1,2,0,3,4],:]
    elif len(lm) == 5:
        lm_aligned = lm
    else:
        raise ValueError('input lm length must be at least 5 !')
    
    # landmark3d
    m = MatFile()
    m('./BFM/similarity_Lm3D_all.mat')
    lm3D = m.lm
    # calculate 5 facial landmarks using 68 landmarks
    lm3D = np.stack([lm3D[lm_idx[0],:],np.mean(lm3D[lm_idx[[1,2]],:],0),np.mean(lm3D[lm_idx[[3,4]],:],0),lm3D[lm_idx[5],:],lm3D[lm_idx[6],:]], axis = 0)
    lm3D = lm3D[[1,2,0,3,4],:]
    # calculate translation and scale factors using 5 facial landmarks and standard landmarks
    t,s = POS(lm_aligned.transpose(),lm3D.transpose())
    # processing the image
    if mask is not None:
        img_new,mask_new,lm_new = process_img(img,lm,t,s,mask)
        lm_new = np.stack([lm_new[:,0],223 - lm_new[:,1]], axis = 1)
        trans_params = np.array([w0,h0,102.0/s,t[0],t[1]])
        return img_new,mask_new,lm_new,trans_params
    else:
        img_new,lm_new = process_img(img,lm,t,s)
        lm_new = np.stack([lm_new[:,0],223 - lm_new[:,1]], axis = 1)
        trans_params = np.array([w0,h0,102.0/s,t[0],t[1]])
        return img_new,lm_new,trans_params



