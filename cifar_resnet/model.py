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


class ResNetCIFAR(nn.Module):
    def __init__(self, n,residual=True,num_classes=10,in_channels=3,widths=(16,32,64)):
        super().__init__()
        self.n = n
        self.residual = residual
        self.num_classes = num_classes
        self.widths = widths
        self.in_channels = in_channels
        self.stem = nn.Conv2d(bias=False,stride=1,padding=1,kernel_size=3,in_channels=in_channels,out_channels=self.widths[0])
        self.bn1 = nn.BatchNorm2d(self.widths[0])
        self.relu = nn.ReLU()
        self.linear = nn.Linear(self.widths[2],self.num_classes)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1,-1)
        self.stage1 = self._make_stage(in_ch=widths[0],out_ch=widths[0],stride=1)
        self.stage2 = self._make_stage(in_ch=widths[0],out_ch=widths[1],stride=2)
        self.stage3 = self._make_stage(in_ch=widths[1],out_ch=widths[2],stride=2)
        self._init_weights()
    def _make_stage(self,in_ch,out_ch,stride):
        blocks = []
        blocks.append(BasicBlock(in_ch=in_ch,out_ch=out_ch,stride=stride,residual=self.residual))
        for i in range(self.n - 1):
            blocks.append(BasicBlock(in_ch=out_ch,out_ch=out_ch,stride=1,residual=self.residual))
        return nn.Sequential(*blocks)
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                fan_in = m.weight[0].numel()
                std = (2/fan_in) ** 0.5
                with torch.no_grad():
                    m.weight.normal_(0,std)
    def forward(self,x):
        x = self.stem(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg_pool(x)
        x =self.flatten(x)
        x = self.linear(x)
        return x 

                
        