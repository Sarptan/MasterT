import torch
import torch.nn as nn

class MLP(nn.Module):

    """
    Two layer feedforward network with GELU and LayerNorm
    """

    def __init__(self, channels, expansion):
        super().__init__()
        
        self.norm = nn.LayerNorm(channels)
        self.main = nn.Sequential(
                  nn.Linear(channels, channels*expansion),
                  nn.GELU(),
                  nn.Linear(channels*expansion, channels))
        
    def forward(self, x):
        return self.main(self.norm(x.float()).type(x.dtype))
    