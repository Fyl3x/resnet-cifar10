import time
import numpy as np
import torch
from PIL import Image
from cifar_resnet.data import preprocess
from cifar_resnet.model import ResNetCIFAR
from cifar_resnet.train import load_checkpoint

device = "cpu"

model = ResNetCIFAR(3)
load_checkpoint("artifacts/checkpoints/final_res20.pt", model, None, device)
model.eval()

mean = np.load("artifacts/pixel_mean_50k.npy")

img = Image.open("samples/test0.png").convert("RGB")
arr = np.asarray(img)
x = preprocess(arr, mean=mean).unsqueeze(0)

with torch.no_grad():
    for _ in range(10):
        model(x)

    times = []
    for _ in range(100):
        start = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - start) * 1000)

print(f"Median model latency: {np.percentile(times, 50):.2f} ms")