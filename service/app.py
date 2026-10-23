from fastapi import FastAPI
import torch
import numpy as np
from cifar_resnet.model import ResNetCIFAR
from cifar_resnet.train import load_checkpoint
device = "cpu"

model = ResNetCIFAR(3)
load_checkpoint("artifacts/checkpoints/final_res20.pt", model, None, device)
model.eval()
print(sum(p.numel() for p in model.parameters()),model.training)
mean = np.load("artifacts/pixel_mean_50k.npy")
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}