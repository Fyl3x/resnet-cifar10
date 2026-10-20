import torch
import numpy as np 
import torchvision
import time 
torch.manual_seed(0)

import torch 
import torchvision
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from cifar_resnet.model import ResNetCIFAR
from cifar_resnet.data import CIFARDataset,split_indices
from cifar_resnet.train import train_one_epoch,evaluate

device = "cuda"
model = ResNetCIFAR(3)

model = model.to(device=device)

############ Data Loading  ############

train_idx, val_idx =  split_indices(50000,5000,0)

mean = np.load("artifacts/pixel_mean_45k_seed0.npy")
data = torchvision.datasets.CIFAR10(root="data",train=True,download=True)

val_dataset = CIFARDataset(images=data.data,labels=data.targets,indices=val_idx,mean=mean,train=False,seed=0)
train_dataset = CIFARDataset(images=data.data,labels=data.targets,indices=train_idx,mean=mean,train=True,seed=0)

loader = DataLoader(dataset=train_dataset,batch_size=128,shuffle=True)
val_loader = DataLoader(dataset=val_dataset,batch_size=128,shuffle=False)

image,label = next(iter(loader))
val_image, val_label = next(iter(val_loader))

############### Optimizer ############

optim = torch.optim.SGD(params=model.parameters(),lr=0.1,momentum=0.9,weight_decay=1e-4)

############ Loss Function ############

loss_fct = nn.CrossEntropyLoss()

############ Epoch_training ############
torch.cuda.synchronize()
start_time = time.perf_counter()
avg_loss, accuracy , iteration_number = train_one_epoch(model=model,loader=loader,optimizer=optim,loss_fn=loss_fct,device=device,iteration=0)
torch.cuda.synchronize()
end_time = time.perf_counter()

epoch_time = end_time - start_time


val_loss, val_acc = evaluate(model=model,loader=val_loader,loss_fn=loss_fct,device=device)

print(f"Training Loss and Accuracy : {avg_loss}, {accuracy}")
print(f"Epoch time: {epoch_time:.2f} seconds")
print(f"validation Loss and Accuracy : {val_loss}, {val_acc}")
print(f"Final iteration count: {iteration_number}")


