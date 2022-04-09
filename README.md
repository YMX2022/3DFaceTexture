# 3DFaceTexture

##  **Joint Specular Highlight Detection and Removal in Single Images via Unet-Transformer** 

	Reconstructing high-fidelity 3D facial texture from a single image is a quite challenging task due to the lack of complete face information and the domain gap between the 3D face and 2D image.  Further, obtaining re-renderable 3D faces has become a strongly desired property in many graphics applications, where the term 're-renderable' demands the facial texture to be spatially complete and disentangled with environmental illumination. In this paper, we propose a new self-supervised deep learning framework for reconstructing high-quality and re-renderable facial albedos from single-view images in-the-wild.
   Our main idea is to first utilize a \emph{prior generation module} based on the 3DMM proxy model to produce an unwrapped texture and a globally parameterized prior albedo. Then we apply a \emph{detail refinement module} to synthesize the final texture with both high-frequency details and completeness.  To further make facial textures disentangled with illumination, we propose a novel detailed illumination representation which is reconstructed with the detailed albedo together.We also design several novel regularization losses on both the albedo and illumination maps to facilitate the disentanglement of these two factors. Finally, by leveraging a differentiable renderer, each face attribute can be jointly trained in a self-supervised manner without requiring ground-truth facial reflectance.
    Extensive comparisons and ablation studies on challenging datasets demonstrate that our framework substantially outperforms state-of-the-art approaches.
## Requirements

- Python   3.6
- PyTorch 1.5.0
- Cuda 10.1

### Training

------

- Modify gpu id, dataset path, and checkpoint path. Adjusting some other parameters if you like.

-  Please run the following code: 

  ```
  python shiq.py --gpus 5\--id UNet --model UNet \
      --optim Adam --lr 1e-4\
      --epochs 80 --batchsize 4 --threads 16
  ```

  

### Testing

------

- Modify test dataset path and result path.

-  Please run the following code: 

  ```
  python shiq.py --gpus 0 --id UNet --model UNet --mode test --epochs 60 --saveimg true
  ```

  
