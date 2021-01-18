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
    """Returns yaml config file in DictConfig format

    Args:
        name: (str) name of the yaml file without .yaml extension
    """
    cfg = OmegaConf.load(f"{name}.yaml")
    return cfg


def check_grad_norm(net: nn.Module):
    """Compute and return the grad norm of all parameters of the network.
    To see gradients flowing in the network or not
    """
    total_norm = 0
    for p in list(filter(lambda p: p.grad is not None, net.parameters())):
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def timeit(fn):
    """A function decorator to calculate the time a funcion needed for completion on GPU.
    returns: the function result and the time taken
    """
    # first, check if cuda is available
    cuda = True if torch.cuda.is_available() else False
    if cuda:

        @functools.wraps(fn)
        def wrapper_fn(*args, **kwargs):
            torch.cuda.synchronize()
            t1 = time()
            result = fn(*args, **kwargs)
            torch.cuda.synchronize()
            t2 = time()
            take = t2 - t1
            return result, take

    else:

        @functools.wraps(fn)
        def wrapper_fn(*args, **kwargs):
            t1 = time()
            result = fn(*args, **kwargs)
            t2 = time()
            take = t2 - t1
            return result, take

    return wrapper_fn


def save_checkpoint(state: dict, is_best: bool, save_dir: str, name: str):
    """Saves model and training parameters at checkpoint + 'epoch.pth'. If is_best==True, also saves
    checkpoint + 'best.pth'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        save_dir: (str) the location where to save the checkpoint
        name: (str) file name to be written
    """
    filepath = os.path.join(save_dir, f"{name}.pth")
    if not os.path.exists(save_dir):
        print(
            "Checkpoint Directory does not exist! Making directory {}".format(save_dir)
        )
        os.mkdir(save_dir)
    else:
        print(f"Checkpoint Directory exists! Saving in {save_dir}")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(save_dir, "best.pth"))


def load_checkpoint(save: str, device: str):
    """Loads model parameters (state_dict) from file_path.

    Args:
        save: (str) directory of the saved checkpoint
        device: (str) map location
    """
    if not os.path.exists(save):
        raise ("File doesn't exist {}".format(save))
    checkpoint = torch.load(save, map_location=device)

    return checkpoint


def plot_images(batch: torch.Tensor, title: str):
    """Plot a batch of images

    Args:
        batch: (torch.Tensor) a batch of images with dimensions (batch, channels, height, width)
        title: (str) title of the plot and saved file
    """
    n_samples = batch.size(0)
    plt.figure(figsize=(n_samples // 2, n_samples // 2))
    plt.axis("off")
    plt.title(title)
    plt.imshow(
        np.transpose(vutils.make_grid(batch, padding=2, normalize=True), (1, 2, 0))
    )
    plt.savefig(f"{title}.png")


def init_weights_normal(m: nn.Module, mean: float = 0.0, std: float = 0.5):
    """Initialize the network's weights based on normal distribution

    Args:
        m: (nn.Module) network itself
        mean: (float) mean of normal distribution
        std: (float) standard deviation for normal distribution
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, mean=mean, std=std)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, mean=mean, std=std)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, mean=mean, std=std)
        nn.init.constant_(m.bias.data, 0)


def init_weights_uniform(m: nn.Module, low: float = 0.0, high: float = 1.0):
    """Initialize the network's weights based on uniform distribution

    Args:
        low: (float) minimum threshold for uniform distribution
        high: (float) maximum threshold for uniform distribution
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.uniform_(m.weight.data, a=low, b=high)
    elif classname.find("BatchNorm") != -1:
        nn.init.uniform_(m.weight.data, a=low, b=high)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("Linear") != -1:
        nn.init.uniform_(m.weight.data, a=low, b=high)
        nn.init.constant_(m.bias.data, 0)


def init_weights_xavier_normal(m: nn.Module):
    """Initialize the network's weights based on xaviar normal distribution"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find("BatchNorm") != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)