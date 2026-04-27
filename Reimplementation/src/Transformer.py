from turtle import forward

from Reimplementation.src.MLP import MLP
from Reimplementation.src.attention import Attention

import torch
import torch.nn as nn

class Transformer(nn.Module):
    """
    Pre-Norm Transformer block 
    with residual Attention and residual MLP.
    """

    def __init__(self, channels, head_channels, expansion):
        super().__init__()

        self.attention = Attention(channels, head_channels)
        self.mlp = MLP(channels, expansion)

    def forward(self, x, attn_mask=None, attn_temp=1.0, which_cache='cond'):

        x = x + self.attention(x, attn_mask, attn_temp, which_cache)
        x = x + self.mlp(x)
        
        return x
