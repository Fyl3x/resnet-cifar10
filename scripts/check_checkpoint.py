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
from cifar_resnet.train import train_one_epoch,evaluate,load_checkpoint,save_checkpoint

device = "cuda"

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

# optim = torch.optim.SGD(params=model.parameters(),lr=0.1,momentum=0.9,weight_decay=1e-4)

############ Loss Function ############

loss_fct = nn.CrossEntropyLoss()


# Model A
model_a = ResNetCIFAR(3)
model_a = model_a.to(device=device)
# Model A's Optimizer
optim_a = torch.optim.SGD(params=model_a.parameters(),lr=0.1,momentum=0.9,weight_decay=1e-4)
loss_cum_a,correct_pred_a,iterer_a = train_one_epoch(model=model_a,loader=loader,optimizer=optim_a,loss_fn=loss_fct,device=device,iteration=0,max_iterations=20)
# Saving the Model's Checkpoint
save_checkpoint("artifacts/checkpoints/check.pt", model_a, optim_a, iterer_a, 0)



# Model B
model_b = ResNetCIFAR(3)
model_b = model_b.to(device=device)
# Model B's Optimizer
optim_b = torch.optim.SGD(params=model_b.parameters(),lr=0.1,momentum=0.9,weight_decay=1e-4)
# Saving the Model's Checkpoint
iteration_b, seed_b = load_checkpoint("artifacts/checkpoints/check.pt", model_b, optim_b,device=device)

print("Loaded iteration:", iteration_b)

sd_a = model_a.state_dict()
sd_b = model_b.state_dict()

same_state = all(
    torch.equal(sd_a[k], sd_b[k])
    for k in sd_a
)

print("State dictionaries equal:", same_state)


val_loss_a, val_acc_a = evaluate(model_a, val_loader, loss_fct, device)
val_loss_b, val_acc_b = evaluate(model_b, val_loader, loss_fct, device)

print("Model A — loss:", val_loss_a, "accuracy:", val_acc_a)
print("Model B — loss:", val_loss_b, "accuracy:", val_acc_b)