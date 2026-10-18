import torch
import numpy as np

from cifar_resnet.model import BasicBlock

test_blk1 = BasicBlock(in_ch=16,out_ch=16,stride=1,residual=True)
test_blk2 = BasicBlock(in_ch=16,out_ch=32,stride=2,residual=True)
test_blk3 = BasicBlock(in_ch=32,out_ch=64,stride=2,residual=True)
test_blk4 = BasicBlock(in_ch=16,out_ch=32,stride=2,residual=False)

test_input1 = torch.randn((2,16,32,32))
test_input2 = torch.randn((2,16,32,32))
test_input3 = torch.randn((2,32,16,16))
test_input4 = torch.randn((2,16,32,32))


print(test_blk1(test_input1).shape)

print(test_blk2(test_input2).shape)

print(test_blk3(test_input3).shape)

print(test_blk4(test_input4).shape)

blk1_params = sum(p.numel() for p in test_blk1.parameters())
blk2_params = sum(p.numel() for p in test_blk2.parameters())
blk3_params = sum(p.numel() for p in test_blk3.parameters())
blk4_params = sum(p.numel() for p in test_blk4.parameters())

print(blk2_params)