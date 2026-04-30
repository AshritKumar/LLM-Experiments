# Adds the attention blocks.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GPTModel3(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, context_size=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positions = nn.Embedding(context_size, embed_dim)

        self.layerNorm = nn.LayerNorm(embed_dim)

        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

        # Optional in single head attention
        self.W0 = nn.Linear(embed_dim, embed_dim)

        self.finalLinear = nn.Linear(embed_dim, vocab_size, bias=False)
        self.finalLinear.weight = nn.Parameter(self.embedding.weight)

    
    def forward(self, tokx):
        x = self.embedding(tokx) # [batch, context_size, embed_dim]
        pos = self.positions(torch.arange(tokx.shape[1])) # [context_size, embed_dim]
        x = x + pos # [batch, context_size, embed_dim]

        ####### BEGIN ATTENTION BLOCK ########
        x = self.layerNorm(x) # normalize the embeddings

        kx = self.key(x) # [batch, context_size, embed_dim]
        qx = self.query(x) # [batch, context_size, embed_dim]
        vx = self.value(x) # [batch, context_size, embed_dim]

      
        # We could also use kx.transpose(-2,-1) instead of kx.mT
        qk = qx @ kx.mT # [batch, context_size, context_size]
        # scale the qk by sqrt(embed_dim)
        qk = qk / np.sqrt(self.embed_dim)

        # add the mask
        mask = torch.tril(torch.ones(x.shape[0], x.shape[1], x.shape[1])) # [batch, context_size, context_size]
        # this is boolean mask indexing in pytorch
        # mask == 0 will return a boolean tensor of the same shape as mask, with True where the mask is 0 and False elsewhere
        # when we do qk[mask==0], this acts as a boolean mask, pytorch will select elemets from qk when mask isTrue.
        # when we do qk[mask==0] = -inf changes the positions selected to -inf inpace.
        qk[mask==0] = -np.inf

        # apply softmax along the last dimension (i.e dim=2 also written as dim=-1)
        qk_softmaxed = F.softmax(qk, dim=-1) # [batch, context_size, context_size]
        y = qk_softmaxed @ vx # [batch, context_size, embed_dim]

        # Resedual connection
        y += self.W0(y)

        ########## END of attention block ########

        ##### WOULD ADD MLP LAYER NEXT #######

        # raw logits
        y = self.finalLinear(y) / np.sqrt(self.embed_dim) # [batch, context_size, vocab_size]
        # few xtra o/p's for debugging
        return y, (mask, qk_softmaxed)

    def generate(self, inpTok, T=1, gen_len=30): # inpTok is (bathsize, context_size)
        for _ in range(gen_len):
            # pass the last context_size tokens to the model
            x,_ = self(inpTok[:, -self.context_size:]) # [batch, context_size, vocab_size]

            # get the final token of the sequence from each i/p of the batch and apply softmax
            logits = x[:, -1, :] # [batch, vocab_size]

            # apply softmax
            sf_probs = F.softmax(logits/T, dim=1)

            # sample from the softmax distribution
            nxt_token = torch.multinomial(sf_probs, num_samples=1) # [batch, 1]
            # append the next token to the input sequence
            inpTok = torch.cat([inpTok, nxt_token], dim=1) # [batch, context_size+1]
        
        return inpTok





        
