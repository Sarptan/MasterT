import torch
import torch.nn as nn
import torch.nn.functional as F
from TARBlock import TARBlock
from Permutation import PermuatationFlip, PermutationIdentity


class TARFlow(nn.Module):
    """
    Full model architecture: 
    Images are divided into non overlapping patches
    Stack of TARBlocks
    Alternating permutations 

    Variance prior:
    (nvp=False), a learnable per patch variance buffer ('var')
    Updated via EMA: var <- (1-lr)*var + lr*mean(z^2)
    (nvp=True), the prior should be N(0,1) (var=1). 
    """

    VAR_LR = 0.1
    var: torch.Tensor

    def __init__(self, in_channels, img_size, patch_size,
                 channels, num_blocks, layers_per_block,
                 nvp=True, num_classes=0):
        
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size//patch_size)**2

        permutations = [
            PermutationIdentity(self.num_patches),
            PermuatationFlip(self.num_patches)
        ]

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                TARBlock(in_channels=in_channels*patch_size**2,
                         channels = channels,
                         num_patches = self.num_patches,
                         permutation=permutations[i%2],
                         num_layers=layers_per_block,
                         nvp=nvp,
                         num_classes=num_classes
                         )
            )
        self.blocks = nn.ModuleList(blocks)

        self.register_buffer('var',
                             torch.ones(self.num_patches, in_channels*patch_size**2))
        
    
    def patchify(self, x):
        u = F.unfold(x, self.patch_size, stride=self.patch_size)
        return u.transpose(1,2)
    
    def unpatchify(self, x):
        u = x.transpose(1,2)
        return F.fold(u, (self.img_size, self.img_size),
                      self.patch_size, stride=self.patch_size)
    
    def forward(self, x, y=None):
        x = self.patchify(x)
        outputs = []
        logdets = torch.zeros((), device=x.device)

        for block in self.blocks:
            x, logdet = block(x,y)
            logdets = logdets + logdet
            outputs.append(x)

        return x, outputs, logdets

    def update_prior(self, z):
        z2 = (z**2).mean(dim=0) # per patch per channel variance estimate
        self.var.lerp_(z2.detach(), weight=self.VAR_LR)

    def get_loss(self, z, logdets):
        return 0.5*z.pow(2).mean() - logdets.mean()
    
    def reverse(self, x, y, guidance=0, guide_what='ab',
            attn_temp=1.0, annealed_guidance=False,
            return_sequence=False):

        seq = [self.unpatchify(x)]
        x = x * self.var.sqrt()        # ← multiply, not add

        for block in reversed(self.blocks):
            x = block.reverse(x, y, guidance, guide_what,
                            attn_temp, annealed_guidance)
            seq.append(self.unpatchify(x))   # ← inside the loop

        x = self.unpatchify(x)               # ← unpatchify before returning

        if not return_sequence:
            return x
        else:
            return seq