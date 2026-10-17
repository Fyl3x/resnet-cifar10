import torch
import torchvision


from cifar_resnet.data import split_indices,compute_pixel_mean

train_idx, val_idx = split_indices(50000,5000,0)
images = torchvision.datasets.CIFAR10(root="data",train=True,download=True)
computed_mean = compute_pixel_mean(images=images.data,train_idx=train_idx)

print(computed_mean.shape,computed_mean.dtype,computed_mean.mean(axis=(0,1)),computed_mean.min(),computed_mean.max())