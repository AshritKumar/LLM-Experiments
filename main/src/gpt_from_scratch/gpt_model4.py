# Add Multiple transfomer blocks
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SingleAttentionHead(nn.Module):
    def __init__(self, embed_dim, device=None):
        super().__init__()
        self.key = nn.Linear(embed_dim, embed_dim, bias=False, device=device)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False, device=device)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False, device=device)

        self.W0 = nn.Linear(embed_dim, embed_dim, device=device)
    
    def forward(self, x): # x is [batch, context_size, embed_dim]
        kx = self.key(x) # [batch, context_size, embed_dim]
        qx = self.query(x) # [batch, context_size, embed_dim]
        vx = self.value(x) # [batch, context_size, embed_dim]

        # qk = qx @ kx.mT # [batch, context_size, context_size]
        # qk = qk / np.sqrt(self.embed_dim)
        # mask = torch.tril(torch.ones(x.shape[0], x.shape[1], x.shape[1])) # [batch, context_size, context_size]
        # qk[mask==0] = -np.inf
        # y = F.softmax(qk, dim=-1) @ vx # [batch, context_size, embed_dim]


        # Instead of calculating attention manually like above, we can use the scaled_dot_product_attention
        y = F.scaled_dot_product_attention(qx, kx, vx, is_causal=True)

        y = self.W0(y)
        return y


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, device=None):
        super().__init__()

        # Attention sub layer
        self.layerNormAttn = nn.LayerNorm(embed_dim, device=device)
        self.attn = SingleAttentionHead(embed_dim, device=device)

        # MLP feed forward sub layer
        self.layerNormMLP = nn.LayerNorm(embed_dim, device=device)
        self.linear1 = nn.Linear(embed_dim, embed_dim*4, device=device) # 4x expansion
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(embed_dim*4, embed_dim, device=device) # 4x contraction

    def forward(self, x): # x is [batch, context_size, embed_dim]
        # Attention
        residual = x
        x = self.layerNormAttn(x)
        x = self.attn(x)
        x = residual + x
        # or we can do it in one line
        # x = x + self.attn(self.layerNormAttn(x))

        # MLP
        y = x + self.linear2(
                self.gelu(
                    self.linear1(self.layerNormMLP(x))
                )
            )

        return y


class GPTModel4(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_size, num_blocks=12, device=None):
        super().__init__()
        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.context_size = context_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, device=device)
        self.positions = nn.Embedding(context_size, embed_dim, device=device)

        self.transformerBlocks = nn.Sequential(
            *[TransformerBlock(embed_dim, device=device) for _ in range(num_blocks)]
        )

        # Final layer normalization
        self.layerNormFinal = nn.LayerNorm(embed_dim, device=device)

        self.finalLinear = nn.Linear(embed_dim, vocab_size, bias=False, device=device)
        # tying the weights of final linear and embedding. This is a common practice in GPT models
        self.finalLinear.weight = nn.Parameter(self.embedding.weight)

    
    def forward(self, x): # x is [batch, context_size]
        ### Embedding ###
        x = self.embedding(x) # [batch, context_size, embed_dim]
        pos = self.positions(torch.arange(x.shape[1], device=x.device)) # [context_size, embed_dim]
        x = x + pos # [batch, context_size, embed_dim]

        ### Transformer Blocks ###
        x = self.transformerBlocks(x) # [batch, context_size, embed_dim]

        ### Final Layer Normalization ###
        # Instead of dividing the final liner layer o/p by sqrt(embed_dim) [self.finalLinear(y) / np.sqrt(self.embed_dim)],
        # we do a layer norm here. We can avoide the scaling 
        x = self.layerNormFinal(x) # [batch, context_size, embed_dim]

        ### Final Linear ###
        logits = self.finalLinear(x) # [batch, context_size, vocab_size]

        return logits
    
    def generate(self, inpTok, T=1, gen_len=30): # inpTok is (bathsize, context_size)
        for _ in range(gen_len):
            # pass the last context_size tokens to the model
            x = self(inpTok[:, -self.context_size:]) # [batch, context_size, vocab_size]

            # get the final token of the sequence from each i/p of the batch and apply softmax
            logits = x[:, -1, :] # [batch, vocab_size]

            # apply softmax
            sf_probs = F.softmax(logits/T, dim=1)

            # sample from the softmax distribution
            nxt_token = torch.multinomial(sf_probs, num_samples=1) # [batch, 1]
            # append the next token to the input sequence
            inpTok = torch.cat([inpTok, nxt_token], dim=1) # [batch, context_size+1]
        
        return inpTok


        

        