import torch
import numpy as np 
import torchvision
import time 
torch.manual_seed(0)
import torch.nn as nn


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

optim_a = torch.optim.SGD(params=model_a.parameters(),
                          lr=0.1,
                          momentum=0.9,
                          weight_decay=1e-4)
iteration = 0
for i in range(5):

    loss_cum_a,correct_pred_a,iteration = train_one_epoch(model=model_a,
                                                     loader=loader,
                                                     optimizer=optim_a,
                                                     loss_fn=loss_fct,
                                                     device=device,
                                                     iteration=iteration,
                                                     max_iterations=64000)
    val_loss_a, val_acc_a = evaluate(model_a, val_loader, loss_fct, device)
    print(
        f"Epoch {i + 1} | "
        f"Train loss: {loss_cum_a} | "
        f"Train accuracy: {correct_pred_a} | "
        f"Val loss: {val_loss_a} | "
        f"Val accuracy: {val_acc_a} | "
        f"Iteration: {iteration}"
    )


# Model B
model_b = ResNetCIFAR(9)
model_b = model_b.to(device=device)
# Model B's Optimizer
optim_b = torch.optim.SGD(params=model_b.parameters(),lr=0.1,momentum=0.9,weight_decay=1e-4)
torch.cuda.synchronize()
start = time.perf_counter()

loss_b, acc_b, iteration_b = train_one_epoch(
    model=model_b,
    loader=loader,
    optimizer=optim_b,
    loss_fn=loss_fct,
    device=device,
    iteration=0,
    max_iterations=64000
)

torch.cuda.synchronize()
end = time.perf_counter()

print(f"ResNet-56 one-epoch time: {end - start:.2f} seconds")
