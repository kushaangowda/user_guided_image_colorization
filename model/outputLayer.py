import torch
import torch.nn as nn
from torch.nn import functional as F

class OutputLayer(nn.Module):
    ''' output layer of the Unet '''
    def __init__(self,in_channels,out_channels):
        super(OutputLayer,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,1)
        self.activation = nn.Tanh()
    
    def forward(self,x):
        return self.activation(self.conv(x))