import torch
import numpy as np 
import argparse
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import os 

from cifar_resnet.train import evaluate,load_checkpoint
from cifar_resnet.model import ResNetCIFAR
from cifar_resnet.data import CIFARDataset
from cifar_resnet.evaluate import predict


parser = argparse.ArgumentParser()


parser.add_argument("--checkpoint", type=str,required=True)
parser.add_argument("--mean", type=str,required=True)
parser.add_argument("--n", type=int,default=3)
parser.add_argument("--plain", action="store_true")
parser.add_argument("--name", type=str,required=True)

args = parser.parse_args()


# creating the data

data = torchvision.datasets.CIFAR10(root="data",train=False,download=True)

mean = np.load(args.mean)

eval_idx = torch.arange(10000)

eval_loader_dataset = CIFARDataset(images=data.data,labels=data.targets,indices=eval_idx,mean=mean,train=False,seed=0)
eval_loader = DataLoader(dataset=eval_loader_dataset,shuffle=False,batch_size=128)

model = ResNetCIFAR(args.n,residual=not args.plain).to("cuda")

load_checkpoint(args.checkpoint,model=model,optimizer=None,device="cuda")


loss, acc = evaluate(model=model,loader=eval_loader,loss_fn=nn.CrossEntropyLoss(),device="cuda")

print(f"Number of test images = {len(eval_idx)} |"
      f"loss obtained = {loss}|"
      f"accuracy obtained is {acc} |"
      f"error is {100*(1-acc)} |")

probs, labels = predict(model=model,loader=eval_loader,device="cuda")
os.makedirs("artifacts/preds",exist_ok=True)

np.savez(f"artifacts/preds/{args.name}.npz", probs=probs, labels=labels)
