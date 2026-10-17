import torch
import torchvision
import numpy as np 
from cifar_resnet.data import split_indices,compute_pixel_mean


data = torchvision.datasets.CIFAR10(root="data",train=True,download=True)

train_idx , val_idx = split_indices(50000,5000,0)

computed_mean_45k = compute_pixel_mean(images=data.data,train_idx=train_idx)


np.save("artifacts/pixel_mean_45k_seed0.npy",computed_mean_45k)

loaded_mean = np.load("artifacts/pixel_mean_45k_seed0.npy")

print(np.array_equal(loaded_mean,computed_mean_45k))