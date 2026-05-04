# Adds multi head attention
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, device=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # We will club the K,Q and V into one matrix for perfrmance
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, device=device)

        self.W0 = nn.Linear(embed_dim, embed_dim, device=device)

    def forward(self, x): # x is [batch, context_size, embed_dim]
        B, CL,  ED = x.shape

        qkvX = self.qkv(x) # qkvX [B, CL, 3 * ED]

        # Now split the qkvX across dim 3 (emb_dim)
        qx, kx, vx = qkvX.split(ED, dim=2)

        # Now split up the qx,kx and vx based on num_heads and head dim
        qx = qx.view(B, CL, self.num_heads, self.head_dim) # qx [B, CL, num_heads, head_dim]
        kx = kx.view(B, CL, self.num_heads, self.head_dim) # kx [B, CL, num_heads, head_dim]
        vx = vx.view(B, CL, self.num_heads, self.head_dim) # vx [B, CL, num_heads, head_dim]

        # Now we will have to transpose qx,kx and vx, so that each head is treated as a separate batch and each head gets all the context_len tokens
        qx = qx.transpose(1, 2) # qx [B, num_heads, CL, head_dim]
        kx = kx.transpose(1, 2) # kx [B, num_heads, CL, head_dim]
        vx = vx.transpose(1, 2) # vx [B, num_heads, CL, head_dim]

        out = F.scaled_dot_product_attention(qx, kx, vx, is_causal=True) # out [B, num_heads, CL, head_dim]
        
        # we sill have to swap back the contxt_len and num_heads dims so that we can concatinate all the attention results
        out = out.transpose(1, 2) # out [B, CL, num_heads, head_dim]
        

        # Now concatinate all the attention results
        # This will concatinate all the attention results into a single tensor
        out = out.reshape(B, CL, ED)

        # finally linearly mix with W0
        out = self.W0(out)

        return out

# Define transformer block
# Attention and MLP

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, device):
        super().__init__()
        # Attention block
        self.layerNormAttn = nn.LayerNorm(embed_dim, device=device)
        self.attn = MultiHeadAttention(embed_dim, num_heads, device)

        # MLP
        self.layerNormMLP = nn.LayerNorm(embed_dim, device=device)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim, device=device), # 4x expansion
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim, device=device) # 4x contraction
        )

    def forward(self, x): # x [B, CL, ED]
        x = x + self.attn(self.layerNormAttn(x))
        x = x + self.mlp(self.layerNormMLP(x))
        return x

class GPTModel5(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_len, num_transformerBlocks, num_attn_heads, device):
        super().__init__()

        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device

        self.embed_dim = embed_dim
        self.context_len = context_len
        self.wte = nn.Embedding(vocab_size, embed_dim, device=device)
        self.wpe = nn.Embedding(context_len, embed_dim, device=device)

        # trasformer blocks
        self.transformerBlocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_attn_heads, device) for _ in range(num_transformerBlocks)
        ])

        #Final layer norm
        self.finalLayerNorm = nn.LayerNorm(embed_dim, device=device)

        #Final unembeddings
        self.finalLayer = nn.Linear(embed_dim, vocab_size, device=device)
        self.finalLayer.weight = nn.Parameter(self.wte.weight)
    
    def forward(self, inpTox): # inpTox [batch, context_size]
        x = self.wte(inpTox) # [B, CL, ED]
        pos = torch.arange(0, inpTox.shape[1], device=self.device) # context len size
        x = x + self.wpe(pos) # [B, CL, ED]

        # Pass through transformer blocks
        x = self.transformerBlocks(x) # [B, CL, ED]
        
        # Apply final layer norm
        x = self.finalLayerNorm(x) # [B, CL, ED]
        
        # Apply final linear layer
        logits = self.finalLayer(x) # [B, CL, vocab_size]
        
        return logits