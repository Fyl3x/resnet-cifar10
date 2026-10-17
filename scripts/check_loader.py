import torch
import torchvision
import numpy as np 
from cifar_resnet.data import split_indices,CIFARDataset
from torch.utils.data import DataLoader
data = torchvision.datasets.CIFAR10(root="data",train=True,download=True)

mean_saved = np.load("C:/dev/Project3/resnet-cifar10/artifacts/pixel_mean_45k_seed0.npy")

train_idx, val_idx = split_indices(50000,5000,0)

def make_train_loader(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    cifset = CIFARDataset(images=data.data,labels=data.targets,indices=train_idx,mean=mean_saved,train=True,seed=seed)
    dataloader_instance = DataLoader(cifset,batch_size=128,shuffle=True,num_workers=0,generator=g)
    return dataloader_instance

data_current = make_train_loader(0)
images,labels = next(iter(data_current))
print(images.shape,labels.shape,images.dtype,labels.dtype)

data_current_copy = make_train_loader(0)
images_copy,labels_copy = next(iter(data_current_copy))
print(torch.equal(images,images_copy),torch.equal(labels,labels_copy))

data_current_diff = make_train_loader(1)
images_diff, labels_diff = next(iter(data_current_diff))
print(torch.equal(images,images_diff),torch.equal(labels,labels_diff))


def make_val_loader(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    cifset = CIFARDataset(images=data.data,labels=data.targets,indices=val_idx,mean=mean_saved,train=False,seed=seed)
    dataloader_instance = DataLoader(cifset,batch_size=5000,shuffle=False,num_workers=0,generator=g)
    return dataloader_instance

val_data_current = make_val_loader(0)
images_val, labels_val = next(iter(val_data_current))
print(images_val.mean(dim=(0, 2, 3)))