import torch
import numpy as np 
import os 
import shutil

from cifar_resnet.model import ResNetCIFAR
from cifar_resnet.train import load_checkpoint




model = ResNetCIFAR(3)
load_checkpoint("artifacts/checkpoints/final_res20.pt",model=model,optimizer=None,device="cpu")
os.makedirs("model",exist_ok=True)
torch.save({"model": model.state_dict(), "iteration": 64000, "seed": 0}, "model/resnet20.pt")
shutil.copy("artifacts/pixel_mean_50k.npy", "model/pixel_mean_50k.npy")

mean_size = os.path.getsize("model/pixel_mean_50k.npy")
model_size = os.path.getsize("model/resnet20.pt")

print(mean_size,model_size)