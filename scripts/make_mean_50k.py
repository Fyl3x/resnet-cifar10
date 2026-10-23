import torch
import torchvision
import numpy as np 
from cifar_resnet.data import split_indices,compute_pixel_mean


data = torchvision.datasets.CIFAR10(root="data",train=True,download=True)

train_idx = torch.arange(50000)

computed_mean_50k = compute_pixel_mean(images=data.data,train_idx=train_idx)


np.save("artifacts/pixel_mean_50k.npy",computed_mean_50k)

loaded_mean = np.load("artifacts/pixel_mean_50k.npy")

print(np.array_equal(loaded_mean,computed_mean_50k))