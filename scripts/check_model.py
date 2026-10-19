import torch
import numpy as np

from cifar_resnet.model import ResNetCIFAR


resnet3 = ResNetCIFAR(3)
resenet3_false = ResNetCIFAR(3,residual=False)

print(sum(p.numel() for p in resnet3.parameters()) )
print(sum(p.numel() for p in resenet3_false.parameters()) )
input = torch.randn(128,3,32,32)
output = resnet3(input)

print(output.shape)

print(resnet3.stem.weight.std())
print(resnet3.stage3[-1].second_conv.weight.std().item())
resnet9 = ResNetCIFAR(9)
print(sum(p.numel() for p in resnet9.parameters()) )
