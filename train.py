import torch
import torch.nn as nn
from torch.nn import functional as F
from model.UNet import UNet

def train():
    x = torch.rand(4,4,256,256)
    in_channels = [4,64,64]
    out_channels = [64,64,128]
    blocks = len(in_channels)
    model = UNet(in_channels,out_channels,blocks=blocks,bn_blocks=2)
    out = model(x)
    print(x.shape,out.shape)

if __name__ == '__main__':
    train()
