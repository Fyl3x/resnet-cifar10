import torch
import numpy as np 
import torchvision
import argparse
import torch.nn as nn

import csv, os, time
from torch.utils.data import DataLoader
from cifar_resnet.model import ResNetCIFAR
from cifar_resnet.data import CIFARDataset,split_indices
from cifar_resnet.train import train_one_epoch,evaluate,load_checkpoint,save_checkpoint

# this part serves as putting the arguments in the cmd line to run 
parser = argparse.ArgumentParser()


parser.add_argument("--n",type=int,default=3)
parser.add_argument("--plain",action="store_true")
parser.add_argument("--seed",type=int,default=0)
parser.add_argument("--full",action="store_true")
parser.add_argument("--name",type=str,required=True)
parser.add_argument("--max-iters",type=int,default=64000)


args = parser.parse_args()
print(args)

# this part is now the data,model and optimizer

torch.manual_seed(args.seed)
device = "cuda"
g = torch.Generator()
g.manual_seed(args.seed)

# loading the dataset CIFAR10

data = torchvision.datasets.CIFAR10(root="data",train=True,download=True)

# distinguishing between the full training/training with validation cases

if args.full:
    train_idx = torch.arange(50000)
    # mean = np.load("artifacts/pixel_mean_50k.npy")
    val_loader = None
    val_idx = None
else:

    train_idx, val_idx = split_indices(50000,5000,0) 
    mean = np.load("artifacts/pixel_mean_45k_seed0.npy")
    val_dataset = CIFARDataset(images=data.data,labels=data.targets,indices=val_idx,mean=mean,train=False,seed=args.seed)
    val_loader = DataLoader(dataset=val_dataset,batch_size=128,shuffle=False,generator=g)

# creating training dataset and loader
train_dataset= CIFARDataset(images=data.data,labels=data.targets,indices=train_idx,mean=mean,train=True,seed=args.seed)
train_loader = DataLoader(dataset=train_dataset,batch_size=128,shuffle=True,generator=g)

# creating the model using the args inputted 
model = ResNetCIFAR(n=args.n,residual= not args.plain).to(device=device)

# creating the optimizer 
optim = torch.optim.SGD(params=model.parameters(),lr=0.1,momentum=0.9,weight_decay=1e-4)

# creating the loss function
loss_fct = nn.CrossEntropyLoss()

print(
    f"Training Set Size = {len(train_idx)} | "
    f"Validation Set Size = {len(val_idx)} |"
    f"Parameters Count = {sum(p.numel() for p in model.parameters())} | "
)



# Making and Saving the Logs 


os.makedirs("artifacts/logs", exist_ok=True)
os.makedirs("artifacts/checkpoints", exist_ok=True)
with open(f"artifacts/logs/{args.name}.csv",mode="w",newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
    "epoch",
    "iteration",
    "lr",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "elapsed_s"
                    ])
    iteration = 0
    epoch = 0
    start_time = time.time()
    while iteration < args.max_iters:
        epoch += 1
        cum_loss,pred_acc,iteration = train_one_epoch(model=model,loader=train_loader,optimizer=optim,loss_fn=loss_fct,device=device,iteration=iteration,max_iterations=args.max_iters)
        lr = optim.param_groups[0]["lr"]
        elapsed_s = time.time() - start_time

        if val_loader is not None:
            val_loss, val_acc = evaluate(
        model=model,
        loader=val_loader,
        loss_fn=loss_fct,
        device=device
                                        )
        else:
            val_loss = ""
            val_acc = ""
        print(
    f"epoch={epoch} | iteration={iteration} | lr={lr:.6f} | "
    f"train_acc={pred_acc:.4f} | val_acc={val_acc} | "
    f"elapsed={elapsed_s:.1f}s"
                )

        
        writer.writerow([
    epoch,
    iteration,
    lr,
    cum_loss,
    pred_acc,
    val_loss,
    val_acc,
    elapsed_s
                        ])
        f.flush()
    save_checkpoint(f"artifacts/checkpoints/{args.name}.pt", model, optim,iteration,args.seed)