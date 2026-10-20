import torch 
import torchvision
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from cifar_resnet.model import ResNetCIFAR
from cifar_resnet.data import CIFARDataset,split_indices
from cifar_resnet.train import evaluate
device = "cuda"

model = ResNetCIFAR(3)
model = model.to(device=device)

train_idx, val_idx =  split_indices(50000,5000,0)

mean = np.load("artifacts/pixel_mean_45k_seed0.npy")
data = torchvision.datasets.CIFAR10(root="data",train=True,download=True)
dataset = CIFARDataset(images=data.data,labels=data.targets,indices=val_idx,mean=mean,train=False,seed=42)

loader = DataLoader(dataset=dataset,batch_size=128,shuffle=False)
image,label = next(iter(loader))
loss_fct = nn.CrossEntropyLoss()
model.train()
for i in range(20):
    with torch.no_grad():
        image,label = next(iter(loader))
        image = image.to(device)
        logits = model(image)
loss, acc = evaluate(model=model,loader=loader,loss_fn=loss_fct,device=device)


print(loss,acc)