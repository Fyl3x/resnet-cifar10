import torch
import torchvision
import os 
from PIL import Image


import numpy as np 
from cifar_resnet.data import split_indices
data = torchvision.datasets.CIFAR10(root="data",train=False,download=True)




os.makedirs("samples", exist_ok=True)

Image.fromarray(data.data[0]).save("samples/test0.png")
print(data.classes[data.targets[0]])