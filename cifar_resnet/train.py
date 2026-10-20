import numpy as np
import torch
import torchvision
import os
def train_step(model,x,y,optimizer,loss_fn):
    optimizer.zero_grad()
    logits = model(x)
    loss = loss_fn(logits,y)
    loss.backward()
    optimizer.step()
    return logits.detach(), loss.item()



def evaluate(model,loader,loss_fn,device):
    model.eval()
    cum_loss = 0
    num_exmpl = 0
    correct_pred = 0
    with torch.no_grad():
        for image,label in loader:
            image = image.to(device)
            label = label.to(device)
            logits = model(image)
            compare = (torch.argmax(logits,dim=1) == label)
            compare = compare.sum().item()
            correct_pred+= compare
            loss = loss_fn(logits,label)
            num_exmpl += logits.shape[0]
            cum_loss += loss.item() * logits.shape[0]
        cum_loss /= num_exmpl
        correct_pred /= num_exmpl
    return cum_loss,correct_pred


def train_one_epoch(model,loader,optimizer,loss_fn,device,iteration,max_iterations):
    model.train()
    num_exmpl  = 0 
    correct_pred = 0
    loss_cum = 0
    iterer = iteration
    for image,label in loader:
        
        image = image.to(device)
        label = label.to(device)
        if iterer >= max_iterations:
             break
        set_lr(optimizer, lr_at(iterer))
        logits,loss = train_step(model=model,x=image,y=label,optimizer=optimizer,loss_fn=loss_fn)

        compare = (torch.argmax(logits,dim=1) == label)
        compare = compare.sum().item()
        correct_pred+= compare

        iterer+=1
        num_exmpl += logits.shape[0]
        loss_cum+=loss*logits.shape[0]
       

    
    loss_cum /= num_exmpl
    correct_pred /= num_exmpl
    return loss_cum,correct_pred,iterer



def lr_at(iteration):
    if iteration < 32000:
        return 0.1
    elif iteration < 48000:
        return 0.01
    else:
        return 0.001
def set_lr(optimizer,lr):
    for p in optimizer.param_groups:
        p["lr"] = lr

    
def save_checkpoint(path,model,optimizer,iteration,seed):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration, "seed": seed}
    torch.save(checkpoint,path)
def load_checkpoint(path,model,optimizer,device):
    checkpoint = torch.load(path,map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint["iteration"], checkpoint["seed"]