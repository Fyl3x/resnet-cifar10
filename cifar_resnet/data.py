import torch
import numpy as np 


def split_indices(n,n_val,seed):
    g = torch.Generator()
    g.manual_seed(seed)
    result = torch.randperm(n=n,generator=g)
    validation_idx = result[:n_val]
    training_idx = result[n_val:]
    return training_idx,validation_idx

def compute_pixel_mean(images,train_idx):
    training_images = images[train_idx.numpy()]
    floated_images = training_images.astype(np.float32)
    avg = floated_images.mean(axis=0)
    avg /= 255
    return avg 

def augment(image,rng):
    padded_image = np.pad(image,pad_width=((4, 4), (4, 4), (0, 0)))
    top_offset = rng.integers(0,9)
    left_offset = rng.integers(0,9)
    cropped_image = padded_image[top_offset:top_offset+32, left_offset:left_offset+32, :]
    if rng.random() < 0.5 :
        cropped_image = np.ascontiguousarray(np.flip(cropped_image, axis=1))
    return cropped_image


class CIFARDataset(torch.utils.data.Dataset):
    def __init__(self,images,labels,indices,mean,train:bool,seed):
        super().__init__()
        self.images = images
        self.labels = labels
        self.indices = indices
        self.mean = mean
        self.train = train
        self.seed = seed
        if self.train:
            self.rng = np.random.default_rng(self.seed)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        idx = int(self.indices[i])
        image, label = self.images[idx], self.labels[idx]
        if self.train:
            image = augment(image=image,rng=self.rng)
        image = image.astype(np.float32) / 255 - self.mean
        image = torch.tensor(image)
        rearranged_image = image.permute(2, 0, 1).contiguous()
        return rearranged_image,label