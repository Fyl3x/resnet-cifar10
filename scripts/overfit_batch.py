import torch 
import torchvision
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from cifar_resnet.model import ResNetCIFAR
from cifar_resnet.data import CIFARDataset,split_indices

torch.manual_seed(42)
data = torchvision.datasets.CIFAR10(root="data",train=True,download=True)


device = "cuda"
model = ResNetCIFAR(3)
model = model.to(device=device)
model.train()
mean = np.load("artifacts/pixel_mean_45k_seed0.npy")

train_idx, val_idx = split_indices(50000,5000,0)
dataset = CIFARDataset(images=data.data,labels=data.targets,indices=train_idx,mean=mean,train=True,seed=42)
loader = DataLoader(dataset=dataset,batch_size=128,shuffle=False)
image,label = next(iter(loader))
image = image.to(device) 
label = label.to(device)
from cifar_resnet.train import train_step
optim = torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=1e-4)
loss_fct = nn.CrossEntropyLoss()
for i in range(200):
    logits, loss = train_step(model=model,x=image,y=label,optimizer=optim,loss_fn=loss_fct)
    if i == 0:
        print(f"loss at the first iteration IS : {loss}")
    if (i+1) % 20 == 0:
        print(i)
        print(loss)
        print((logits.argmax(dim=1) == label).float().mean().item())