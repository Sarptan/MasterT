import torch
import torch.nn as nn
import torch.nn.functional as F 


class Attention(nn.Module):
    """
    Multihead self attention 

    - Pre LayerNorm applied before qkv projection

    - Causal Masking: low-triangular mask for autoregressiveness

    - KV cache: keys and values are accumulated, sepereate caches for cond and uncond for guidance
    """

    USE_SPDA: bool = True

    def __init__(self, in_channels, head_channels):
        assert in_channels % head_channels == 0 , (f"{in_channels} mmust be divisible by {head_channels}")
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)

        self.qkv = nn.Linear(in_channels, in_channels*3)

        self.proj = nn.Linear(in_channels, in_channels)

        self.num_heads = in_channels // head_channels

        self.sqrt_scale = head_channels ** (-0.25)

        self.sample = False

        self.k_cache = {'cond': [], 'uncond': []}
        self.v_cache = {'cond': [], 'uncond': []}

    def forward_spda(self, x, mask, temp = 1.0, which_cache = 'cond'):
        
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)

        q, k, v = (self.qkv(x)
                   .reshape(B, T, 3*self.num_heads, -1)
                   .transpose(1,2)
                   .chunk(3, dim=1))

        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)

            k = torch.cat(self.k_cache[which_cache], dim=2)
            v = torch.cat(self.v_cache[which_cache], dim=2)

        scale = self.sqrt_scale ** 2 / temp

        if mask is not None:
            mask = mask.bool()

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        x = x.transpose(1,2).reshape(B, T, C)

        return self.proj(x)
    
    def forward_base(self, x, mask, temp = 1.0, which_cache = 'cond'):
        
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)

        q, k, v = (self.qkv(x)
                   .reshape(B, T, 3*self.num_heads, -1)
                   .transpose(1,2)
                   .chunk(3, dim=1))

        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)

            k = torch.cat(self.k_cache[which_cache], dim=2)
            v = torch.cat(self.v_cache[which_cache], dim=2)

            attn = torch.einsum('bmnh,bnhd->bmnh', q, self.sqrt_scale, k*self.sqrt_scale) / temp
            
            if mask is not None:
                attn = attn.masked_fill(mask.unsqueeze(-1)==0, float("-inf"))

        attn = attn.float().softmax(dim=2).type(attn.dtype)
            
        x = torch.einsum('bmnh,bnhd->bmhd', attn, v).reshape(B, T, C)
        return self.proj(x)
        
    def forward(self, x, mask, temp=1.0, which_cache ='cond'):

        if self.USE_SPDA:
            return self.forward_spda(x, mask, temp, which_cache)
        return self.forward_base(x, mask, temp, which_cache)            
    
