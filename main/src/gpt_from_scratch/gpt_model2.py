# from gpt_model1 import TokenDataset
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# Added layer norm and positional embeddings

class GPTModel2(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, context_size=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positions = nn.Embedding(context_size, embed_dim)
        self.gelu = nn.GELU()
        self.layerNorm = nn.LayerNorm(embed_dim)
        self.finalLinear = nn.Linear(embed_dim, vocab_size, bias=False)
        self.finalLinear.weight = nn.Parameter(self.embedding.weight)
    
    def forward(self, tokx): # tokx is a tensor of shape (batch_size, context_size)
        # forward pass
        x = self.embedding(tokx) # [batch, token, embed_dim]
        # add positional embeddings
        pos = self.positions(torch.arange(tokx.shape[1])) # [token, embed_dim]. Use tokx.shape[1] to make it dynamic, if we use self.context_size, it will be static we can't accept varied sequence length.

        # add positional embeddings
        x = x + pos # the addition work here because broadcasting rules are satisfied
        
         # IMPORTANT: Here we are not not flatting the embedding result like how we did in mlp_char_level.py. 
         # If we flat the embedding result, we will lose the information about the position of the token in the sequence (i.e. we will not be able to tell the difference between the first token and the last token in the sequence).
         # This is because the positional embeddings are added to the embedding result before the linear transformation.
         # So, we need to keep the embedding result in the shape of (batch_size, context_size, embed_dim).
         # Keeping the embedding result in the shape of (batch_size, context_size, embed_dim) will make the model see all the tokens in the sequence. 
         # And we can compute the loss for each token in the sequence, not just the last token
         # ALSO THIS WILL MAKE THE MODEL MORE FLEXIBLE IN ACCEPTING VARIED SEQUENCE LENGTH, UP TO THE MAX CONTEXT SIZE

        x = self.layerNorm(x)
        x = self.gelu(x)
        # we divide by sqrt(embed_dim) to keep the scale of the output small. This is a common practice in transformers.
        x = self.finalLinear(x) / np.sqrt(self.embed_dim) # [batch, token, vocab_size]
        return x
    
    def generate(self, inpTok, T=1, gen_len=30): # inpTok is (bathsize, context_size)
        for _ in range(gen_len):
            # Always pass the last context_size tokens to the model
            x = self(inpTok[:, -self.context_size:]) # [batch, context_size, vocab_size]
            # print("G: logits shape = ", x.shape)
            # get the final token of the sequence from each i/p of the batch and apply softmax
            logits = x[:, -1, :] # [batch, vocab_size]
            sf_probs = F.softmax(logits / T, dim=-1) # dim=-1 means we apply softmax along the last dimension, i.e vocab_size (same as dim=1)
            # print("G: sf_probs shape = ", sf_probs.shape)
            # sample from the softmax distribution
            nxt_token = torch.multinomial(sf_probs, num_samples=1) # [batch, 1]
            # print("G: nxt_token shape = ", nxt_token.shape)

            # append the next token to the input sequence
            inpTok = torch.cat([inpTok, nxt_token], dim=1) # [batch, context_size+1]
        return inpTok

            
        