import os
import torch
import numpy as np

class GradRecord:

    def __init__(self, root, grad_name, batch_size=None):
        if not os.path.isdir(root):
            os.makedirs(root)
        self.grad_path = os.path.join(root, grad_name)
        if batch_size is not None:
            self.batch_size = batch_size

    def record_grad(self, grad):
        grad = torch.abs(grad)
        if not os.path.isfile(self.grad_path):
            with open(self.grad_path, 'w') as fh:
                fh.writelines(['grad_max : {}, grad_min : {}\n'.format(grad.max().item(), grad.min().item())])
        else:
            with open(self.grad_path, 'a') as fh:
                fh.writelines(['grad_max : {}, grad_min : {}\n'.format(grad.max().item(), grad.min().item())])
                
    def record_grad_all(self, grad):
        grad_list = grad.detach().cpu().numpy().reshape(-1)
        line = str(' ').join(str(i) for i in list(grad_list)) + '\n'
        if not os.path.isfile(self.grad_path):
            with open(self.grad_path, 'w') as fh:
                fh.write(line)
        else:
            with open(self.grad_path, 'a') as fh:
                fh.write(line)
                

    def record_grad_for_map(self, grad):
        grad = grad.detach().cpu().numpy().reshape(self.batch_size, -1)
        np.savetxt(self.grad_path, grad)
                

    def debug_gradient(self, grad):
        print('grad_max: {}, grad_min: {}'.format(grad.max().item(), grad.min().item()))
        
    def print_grad(self, grad):
#         print('grad_max: {}\tgrad_min: {}'.format(grad.max().item(), grad.min().item()))
#         print('grad_shapes: {}'.format(grad.shape))
        print('grad', grad)


def record_weight_grad(content, log_folder, log_fn):
    log_path = os.path.join(log_folder, log_fn)
    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)
    if not os.path.isfile(log_path):
        with open(log_path, 'w') as fh:
            fh.writelines(content)
    else:
        with open(log_path, 'a') as fh:
            fh.writelines(content)
