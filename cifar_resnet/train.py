import numpy as np
import torch
import torchvision

def train_step(model,x,y,optimizer,loss_fn):
    optimizer.zero_grad()
    logits = model(x)
    loss = loss_fn(logits,y)
    loss.backward()
    optimizer.step()
    return logits.detach(), loss.item()
