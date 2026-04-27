from matplotlib.pylab import permutation

from Reimplementation.src.Transformer import Transformer
import torch
import torch.nn as nn

from Reimplementation.src.attention import Attention

class TARBlock(nn.Module):
    """
    One block of Transformer Autoregr4essive Flow model,
    that processes a sequence of omage patches

    NVP mode: both log-scale and shift 
    VP mode: only shift

    Forward:
    Permutation (either Identity or Flip)
    Map patches to Transformer dimension
    Add class embeddings
    Transformer with attention mask num_layers times
    Project back to in_channels dimension (if VP, else 2*in_channels)
    Causal shift
    split and return log-scale and shift

    Backward:
    Sampling token-by-token via `reverse_step`, 
    using a KV-cache. For each position i (0 to T-2):
    Run the Transformer on token i, reading from the cached keys/values.
    Recover (xa, xb) for position i+1.
    Apply the inverse affine: x_{i+1} = x_{i+1} * exp(xa) + xb.
    """

    attn_mask: torch.Tensor

    def __init__(self, in_channels, channels, num_patches,
                 permutation, num_layers = 1, head_dim = 64,
                 expansion = 4, nvp = True, num_classes=0):
        super().__init__()

        self.proj_in = nn.Linear(in_channels, channels)
        self.pos_embed = nn.Parameter(
            torch.randn(num_patches, channels)*1e-2)
        
        if num_classes:
            self.class_embed = nn.Parameter(
                torch.randn(num_classes, 1, channels)*1e-2)
        else:
            self.class_embed = None

        self.attn_blocks = nn.ModuleList(
            [Transformer(channels, head_dim, expansion)
             for _ in range(num_layers)]
        )

        self.nvp = nvp
        
        output_dim = in_channels * 2 if nvp else in_channels
        self.proj_out = nn.Linear(channels, output_dim)

        #This ensures the transformation starts as an exact identity:
        self.proj_out.weight.data.fill_(0.0) 

        self.permutation = permutation

        self.register_buffer(
            'attn_mask',
            torch.tril(torch.ones(num_patches, num_patches))
        )
    
    def forward(self, x, y):

        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)

        x_in = x

        x = self.proj_in(x) + pos_embed

        if self.class_embed is not None:
            if y is not None:

                if (y<0).any():
                    m = (y<0).float().view(-1, 1, 1)
                    class_embed = (1-m) * self.class_embed[y] + m * self.class_embed.mean(0)
                    
                else:
                    class_embed = self.class_embed[y]

                x = x + class_embed
            
            else:
                x = x + self.class_embed.mean(dim=0)

        for block in self.attn_blocks:
            x = block(x, self.attn_mask)

        x = self.proj_out(x)

        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)

        if self.nvp: 
            xa, xb = x.chunk(2, dim=-1)

        else: 
            xb = x
            xa = torch.zeros_like(x)

        scale = (-xa.float()).exp().type(xa.dtype)
        z = self.permutation((x_in - xb) * scale, inverse = True)                
        logdet = -xa.mean(dim=[1,2])
        return z, logdet

    def reverse_step(self, x, pos_embed, i, y, attn_temp, which_cache = 'cond'):
        
        x_in = x[:, i:i+1]
        x = self.proj_in(x_in) + pos_embed[i:i+1]

        if self.class_embed is not None:
            if y is not None:
                x = x + self.class_embed[y]
            else:
                x = x + self.class_embed.mean(dim=0)

        for block in self.attn_blocks:
            x = block(x, attn_temp=attn_temp, which_cache=which_cache)

        x = self.proj_out(x)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)

        return xa, xb
    
    def set_sample_mode(self, flag=True):

        for m in self.modules():
            if isinstance(m, Attention):
                m.sample = flag
                m.k_cache = {'cond': [], 'uncond': []}
                m.v_cache = {'cond': [], 'uncond': []}

    def reverse(self, x, y, guidance=0, guide_what='ab',
                attn_temp=1.0,
                annealed_guidance = False):
        
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)

        self.set_sample_mode(True)
        T = x.size(1)

        for i in range(T-1):
            za, zb = self.reverse_step(
            x, pos_embed, i, y, attn_temp=attn_temp, which_cache='cond'
                )

            if guidance > 0 and guide_what:
                za_u, zb_u = self.reverse_step(
                    x, pos_embed, i, None, attn_temp, which_cache='uncond'
                )

                g = ((i + 1) / (T - 1) * guidance) if annealed_guidance else guidance

                if 'a' in guide_what:
                    za = za + g * (za - za_u)
                if 'b' in guide_what:
                    zb = zb + g * (zb - zb_u)

            scale = za[:,0].float().exp().type(za.dtype)

            x[:, i+1] = x[:, i+1] * scale + zb[:,0]

        self.set_sample_mode(False)
        return self.permutation(x, inverse=True)
        