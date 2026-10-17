import torch
import numpy as np 


def split_indices(n,n_val,seed):
    g = torch.Generator()
    g.manual_seed(seed)
    result = torch.randperm(n=n,generator=g)
    validation_idx = result[:n_val]
    training_idx = result[n_val:]
    return training_idx,validation_idx

def compute_pixel_mean(images,train_idx):
    training_images = images[train_idx.numpy()]
    floated_images = training_images.astype(np.float32)
    avg = floated_images.mean(axis=0)
    avg /= 255
    return avg 
