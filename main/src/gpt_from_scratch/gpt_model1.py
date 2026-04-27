# pytorch stuff
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.nn import functional as F
import re

class TokenDataset(Dataset):
    def __init__(self, tokenizer, text: str, context_len=8, stride=4):
        # character strings to replace with space
        strings2replace = [ '\r\n\r\nâ\x80\x9c','â\x80\x9c','â\x80\x9d','\r\n','â\x80\x94','â\x80\x99','â\x80\x98','_', ]

        # use regular expression (re) to replace those strings with space
        for str2match in strings2replace:
            text = re.compile(r'%s'%str2match).sub(' ',text)
        
        # remove non-ASCII characters and numbers, and make lower-case
        # text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        # self.words = re.split(f'[{string.punctuation}\s]+',text)
        self.context_len = context_len
        self.stride = stride
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.tokens = torch.tensor(tokenizer.encode(text))

        self._prep_dataset()
    
    def _prep_dataset(self):
        for i in range(0, len(self.tokens) - self.context_len, self.stride):
            inpToks = self.tokens[i: i+self.context_len]
            targetToks = self.tokens[i+1: i+1+self.context_len]
            self.inputs.append(inpToks)
            self.targets.append(targetToks)
            
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


class GPTModel1(nn.Module):
    def __init__(self, vocab_size, embed_dim=64):
        super().__init__()

        # embedding matrix
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        print("Embedding shape ", self.embedding.weight.shape)

        # unembedding (linear layer)
        self.gelu = nn.GELU()
        # self.gelu = nn.ReLU()
        self.finalLinear = nn.Linear(embed_dim,vocab_size,bias=False)

    def forward(self,tokx):
        print("Input shape ", tokx.shape)
        # forward pass
        x = self.embedding(tokx) # [batch, token, embed_dim]
        print("Afetr EMB shape ", x.shape)
        x = self.gelu(x)
        print("Afetr gelu shape ", x.shape)
        x = self.finalLinear(x)  # [batch, token, vocab_size]
        print("Fial gelu shape ", x.shape)

        # no softmax here
        return x # logits

