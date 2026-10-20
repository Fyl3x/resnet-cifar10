
from cifar_resnet.train import lr_at

iterations = [0, 31999, 32000, 47999, 48000, 63999]

for iteration in iterations:
    print(iteration, lr_at(iteration))
