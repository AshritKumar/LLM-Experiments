import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, device=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.key = nn.Linear(embed_dim, embed_dim, bias=False, device=device)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False, device=device)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False, device=device)

        self.W0 = nn.Linear(embed_dim, embed_dim, device=device)

    def forward(self, x, track_sizes=False): # x is [batch, context_size, embed_dim]
        B, CL,  ED = x.shape
        if track_sizes: print(f"Input shape: {x.shape}")

        qx = self.query(x) # qx [B, CL, ED]
        kx = self.key(x) # kx [B, CL, ED]
        vx = self.value(x) # vx [B, CL, ED]

        if track_sizes: print(f"kx,qx,vx pre split {qx.shape}")

        # Now split up the qx,kx and vx based on num_heads and head dim
        qx = qx.view(B, CL, self.num_heads, self.head_dim) # qx [B, CL, num_heads, head_dim]
        kx = kx.view(B, CL, self.num_heads, self.head_dim) # kx [B, CL, num_heads, head_dim]
        vx = vx.view(B, CL, self.num_heads, self.head_dim) # vx [B, CL, num_heads, head_dim]

        if track_sizes: print(f"kx,qx,vx post split {qx.shape}")

        # Now we will have to transpose qx,kx and vx, so that each head is treated as a separate batch and each head gets all the context_len tokens
        qx = qx.transpose(1, 2) # qx [B, num_heads, CL, head_dim]
        kx = kx.transpose(1, 2) # kx [B, num_heads, CL, head_dim]
        vx = vx.transpose(1, 2) # vx [B, num_heads, CL, head_dim]

        if track_sizes: print(f"kx,qx,vx post split and transpose {qx.shape}")

        out = F.scaled_dot_product_attention(qx, kx, vx, is_causal=True) # out [B, num_heads, CL, head_dim]
        
        if track_sizes: print(f"out shape after scaled_dot_product_attention: {out.shape}")

        # we sill have to swap back the contxt_len and num_heads dims so that we can concatinate all the attention results
        out = out.transpose(1, 2) # out [B, CL, num_heads, head_dim]
        
        if track_sizes: print(f"out shape after transpose: {out.shape}")

        # Now concatinate all the attention results
        # This will concatinate all the attention results into a single tensor
        out = out.reshape(B, CL, ED)

        if track_sizes: print(f"out shape after concatination {out.shape}")

        # finally linearly mix with W0
        out = self.W0(out)
        
        if track_sizes: print(f"out shape after W0 {out.shape}")

        return out


        
       

