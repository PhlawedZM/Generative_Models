import torch


def numel(x):
    batch_size = x.size(0)
    return x.numel() // batch_size
