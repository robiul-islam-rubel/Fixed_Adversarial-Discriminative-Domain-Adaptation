"""Utilities for ADDA."""

import os
import random

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import params
from datasets import get_mnist, get_usps


def make_variable(tensor, requires_grad=False):
    """Move tensor to GPU if available, set requires_grad."""
    tensor = tensor.to("cuda" if torch.cuda.is_available() else "cpu")
    return tensor.requires_grad_(requires_grad)


def make_cuda(tensor):
    """Use CUDA if it's available."""
    return tensor.to("cuda" if torch.cuda.is_available() else "cpu")


def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)


def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


def init_random_seed(manual_seed):
    """Init random seed."""
    seed = random.randint(1, 10000) if manual_seed is None else manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data_loader(name, train=True, batch_size=128):
    """Get data loader by name. Force consistent 3-channel input."""
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.Grayscale(num_output_channels=1),  # force 3 channels
        transforms.ToTensor()
    ])

    if name.upper() == "MNIST":
        return get_mnist(train, transform=transform, batch_size=batch_size)
    elif name.upper() == "USPS":
        return get_usps(train, transform=transform, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def init_model(net, restore=None):
    """Init models with cuda and optional restore."""
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from:", os.path.abspath(restore))
    else:
        net.apply(init_weights)
        net.restored = False

    if torch.cuda.is_available():
        cudnn.benchmark = True
        net = net.cuda()

    return net

def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    path = os.path.join(params.model_root, filename)
    torch.save(net.state_dict(), path)
    print("save pretrained model to:", path)
