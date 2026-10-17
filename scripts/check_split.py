from cifar_resnet.data import split_indices
import torch
train_idx, val_idx = split_indices(n=50000,n_val=5000,seed=0)
print(len(train_idx),len(val_idx))


train_set = set(train_idx.tolist())
val_set = set(val_idx.tolist())

print("Overlapping indices:", len(train_set & val_set))
print("Unique indices across both:", len(train_set | val_set))


train_idx2, val_idx2 = split_indices(50000, 5000, seed=0)

print("Same seed gives identical training:", torch.equal(train_idx, train_idx2))
print("Same seed gives identical validation:", torch.equal(val_idx, val_idx2))

train_idx3, val_idx3 = split_indices(50000, 5000, seed=1)

print("Different seed changes training:", not torch.equal(train_idx, train_idx3))