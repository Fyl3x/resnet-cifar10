import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self,in_ch,out_ch,stride,residual):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.residual = residual
        self.first_conv = nn.Conv2d(bias=False,in_channels=in_ch,out_channels=out_ch,stride=stride,padding=1,kernel_size=3)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.second_conv = nn.Conv2d(bias=False,in_channels=out_ch,out_channels=out_ch,stride=1,padding=1,kernel_size=3)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsampling_need = False
        if residual:
            if (stride != 1) or (in_ch != out_ch):
                self.downsampling_need = True
    def forward(self,x):
        
        f = self.first_conv(x)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.second_conv(f)
        f = self.bn2(f)
        if self.residual:
            if self.downsampling_need:
                x = x[:,:,::2,::2]
                pads = (self.out_ch - self.in_ch) // 2 
                x = F.pad(x, (0, 0, 0, 0, pads, pads))

             
            x = x + f 
        else:
            x = f 
        x = self.relu(x)
        return x 


                
        