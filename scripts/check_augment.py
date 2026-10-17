import torch
import torchvision
import numpy as np 
import matplotlib.pyplot as plt

from cifar_resnet.data import augment

data = torchvision.datasets.CIFAR10(root="data",train=True,download=True)
rng = np.random.default_rng(0)

fig, axes = plt.subplots(8, 4, figsize=(10, 16))

for i in range(8):
    original = data.data[i]
    axes[i, 0].imshow(original)
    axes[i, 0].axis("off")

    for j in range(1, 4):
        augmented = augment(original, rng)
        axes[i, j].imshow(augmented)
        axes[i, j].axis("off")

plt.tight_layout()
plt.savefig("artifacts/aug_grid.png")
plt.show()

print("Shape:", augmented.shape)
print("Dtype:", augmented.dtype)
