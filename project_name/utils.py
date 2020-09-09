import os
from time import time
import shutil
import functools

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.utils as vutils
from omegaconf import OmegaConf


def get_conf(name: str):
    cfg = OmegaConf.load(f"{name}.yaml")
    return cfg


def get_device():
    cfg = get_conf("conf/train")
    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and cfg.ngpu > 0) else "cpu"
    )
    return device
        
        
def check_grad_norm(net):
    """Compute and return the grad norm of all parameters of the network.
    To see gradients flowing in the network or not"""
    total_norm = 0                                               
    for p in list(filter(lambda p: p.grad is not None, net.parameters())): 
        param_norm = p.grad.data.norm(2)                         
        total_norm += param_norm.item() ** 2                     
    total_norm = total_norm ** (1. / 2)                          
    return total_norm
    
    
def timeit_cuda(fn):
    """A function decorator to calculate the time a funcion needed for completion on GPU.
    returns: the function result and the time taken
    """
    @functools.wraps(fn)
    def wrapper_fn(*args, **kwargs):
        torch.cuda.synchronize()
        t1 = time()
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()
        t2 = time()
        take = t2 - t1
        return result, take

    return wrapper_fn
    

def save_checkpoint(state, is_best):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
    """
    cfg = get_conf("conf/dirs")
    filepath = os.path.join(cfg.save, 'last.pth.tar')
    if not os.path.exists(cfg.save):
        print("Checkpoint Directory does not exist! Making directory {}".format(cfg.save))
        os.mkdir(cfg.save)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(cfg.save, 'best.pth.tar'))


def load_checkpoint(model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    cfg = get_conf("conf/dirs")
    if not os.path.exists(cfg.save):
        raise("File doesn't exist {}".format(cfg.save))
    checkpoint = torch.load(cfg.save)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def plot_images(batch):
	n_samples = batch.size(0)
    plt.figure(figsize=(n_samples // 2, n_samples //2))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(vutils.make_grid(batch, padding=2, normalize=True), (1, 2, 0))
    )


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
