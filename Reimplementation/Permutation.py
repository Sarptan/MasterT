from turtle import forward

import torch
import torch.nn as nn

class Permutation(nn.Module):

    def __init__(self, seq_length):
        super().__init__()
        
        self.seq_length = seq_length

    def forward(self, x, dim=1, inverse=False):
        raise NotImplementedError("Subclasses have the forward()")
    

class PermutationIdentity(Permutation):
    def forward(self, x, dim=1, inverse=False):
        return x
    
class PermuatationFlip(Permutation):
    def forward(self, x, dim=1, inverse=False):
        return x.flip(dims=[dim])
