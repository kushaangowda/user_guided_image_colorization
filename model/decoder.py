import torch
import torch.nn as nn
from torch.nn import functional as F
from layers import BasicBlock, TransformerDecoderBlock, Upsample

class DecoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,patch_dim,n_heads,stride,padding,resnet_bias
                    ,dim_ff,num_layers):
        super(DecoderBlock,self).__init__()
        self.p = patch_dim*patch_dim
        self.resnet = BasicBlock(in_channels,out_channels,stride,padding,resnet_bias)
        self.transformer = TransformerDecoderBlock(self.p*in_channels,n_heads,dim_ff,num_layers)
        self.upsample = Upsample(in_channels)
    
    def forward(self, x, context):
        print(f"Before: {x.shape}")
        x = self.upsample(x)
        b,c,h,w = x.shape
        x = x.reshape(b,c*self.p,h*w//self.p).permute(0,2,1)
        _,t_context = context
        # context = context.view(b,c*self.p,h*w//self.p).permute(0,2,1)
        x = self.transformer(x,t_context)
        x = x.permute(0,2,1).reshape(b,c,h,w)
        x = self.resnet(x)
        print(f"After: {x.shape}")
        return x

class Decoder(nn.Module):
    def __init__(self,in_channels,out_channels,patch_dim,n_heads,stride=1,padding=1,resnet_bias=False
                    ,dim_ff=2048,t_layers=1,num_layers=1):
        super(Decoder,self).__init__()
        assert len(in_channels) == num_layers and len(out_channels) == num_layers,\
        'Error: The len of in_channels and out_channels should be same as num_layers'
        self.layers = nn.ModuleList([
            DecoderBlock(in_channels[i],out_channels[i],patch_dim*2**i,n_heads,stride,padding,
                            resnet_bias,dim_ff,t_layers)
            for i in range(num_layers)
        ])
    def forward(self, x, skips):
        for i,layer in enumerate(self.layers):
            x = layer(x,skips.pop())
        return x