import torch
import torch.nn as nn
from torch.nn import functional as F
from .encoder import Encoder
from .decoder import Decoder
from layers import TransformerEncoderBlock
from .outputLayer import OutputLayer

class UNet(nn.Module):
    ''' Unet for image colorization '''
    def __init__(self,in_channels,out_channels,patch_dim=16,n_heads=4,blocks=1,bn_blocks=1,num_bins=40):
        super(UNet,self).__init__()
        assert len(in_channels) == blocks and len(out_channels) == blocks,\
        'Error: The len of in_channels and out_channels should be same as blocks'
        d_in_channels = [elem for elem in out_channels[::-1]]
        d_out_channels = in_channels[::-1]
        self.encoder = Encoder(in_channels,out_channels,patch_dim,n_heads,num_layers=blocks)
        self.botNeck = TransformerEncoderBlock(out_channels[-1],n_heads,bn_blocks)
        self.decoder = Decoder(d_in_channels,d_out_channels,
                                patch_dim//2**(blocks-1),n_heads,num_layers=blocks)
        self.out = OutputLayer(4,num_bins)

    def forward(self,x):
        x,skips = self.encoder(x)
        b,c,h,w = x.shape
        x = x.view(b,c,h*w).permute(0,2,1)
        x = self.botNeck(x)
        x = x.permute(0,2,1).reshape(b,c,h,w)
        x = self.decoder(x,skips)
        x = self.out(x)
        return x