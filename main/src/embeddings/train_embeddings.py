import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from tokenization import SBTokenizer
from torch.utils.data import Dataset
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

class WordDataSet(Dataset):
    def __init__(self, tokenizer: SBTokenizer, text: str, context_len=8, stride=4):
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
        self.tokens = tokenizer.encode(text)

        self._prep_dataset()
    
    # Ashrit is a good boy updated the notebook cell to work with Jupyter
    def _prep_dataset(self):
        for i in range(0, len(self.tokens) - self.context_len - 1, self.stride):
            inpToks = self.tokens[i: i+self.context_len]
            targetToks = self.tokens[i+1: i+1+self.context_len]
            # print(f"Input =>  {self.tokenizer.decode(inpToks)}, target => {self.tokenizer.decode(targetToks)} ")
            # inpIds = self.tokenizer.encode(" ".join(inpTxt))
            # tarIds = self.tokenizer.encode(" ".join(tarTxt))
            self.inputs.append(torch.tensor(inpToks))
            self.targets.append(torch.tensor(targetToks))
            
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]
            

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, layer1_out=128, device = 'cpu'): # context_size or block_size or sequence_len
        print("CS = ", context_size)
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_dim)
        print("Embedding shape = ", self.embedding.weight.shape)
        
        self.device = device

        # init the layers
        self.linear1 = nn.Linear(context_size * embedding_dim, layer1_out)
        self.linear2 = nn.Linear(layer1_out, vocab_size)
        self.apply(EmbeddingModel.init_weights)
        self.to(self.device)

    def init_weights(layer):
        # scale the the weights, we generally don't do it for embedding layer
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Embedding):
            print("In init weights applying for ", layer)
            # Generally Kaiming for ReLU or Xavier for Tanh/Sigmoid, but are following the lecture here
            nn.init.xavier_normal_(layer.weight)
            if isinstance(layer, nn.Linear) and layer.bias is not None:
                nn.init.zeros_(layer.bias)
       
    def traiModel(self, data_loader, optimizer=None, loss_function=None, num_epochs=25):
        if data_loader is None:
            print("Empty data loader")
            raise ValueError("Empty data loader")
        
        loss_per_epoch = torch.zeros(num_epochs)
        for i in range(num_epochs):
            cur_epoch_loss = 0.0
            for inp,tg in data_loader:
                inp,tg = inp.to(self.device), tg.to(self.device)
                # we are getting log soft max probs here
                log_probs = self.forward(inp)
                # print("Log probs shape = ", log_probs.shape)
                if loss_function is not None:
                    loss = loss_function(log_probs, tg[:, -1])
                else:
                    loss = self.loss_function(log_probs, tg[:, -1])
                
                # clear prev grads
                # equal to 
                # for p in self.parameters:
                #   p.grad = None
                self.zero_grad()
                loss.backward()
            
                if optimizer is not None:
                    optimizer.step()
                else:
                    Optimizer(self, lr=0.01).step()

                cur_epoch_loss += loss.item()

            # scale by the number of examples in this dataloader
            loss_per_epoch[i] = cur_epoch_loss/len(data_loader.dataset)
            
        return loss_per_epoch
                

    # this is called when we do model(X)
    def forward(self, inputs): #(batch_size, context_size)
        # get embeddings for the inputs
        emb = self.embedding(inputs) # (batch_size, context_size, embedding_dim)
        # print("Emb sape = ", emb.shape)
        # flatten the embeddings
        emb = emb.view(inputs.shape[0], -1) #(batch_size, context_size * embedding_dim)
        # print("Emb sape after view = ", emb.shape)
        # pass through linear layers
        h = F.gelu(self.linear1(emb))
        # print("Layer 1 out shape= ", h.shape)
        
        out = self.linear2(h)
        # print("Layer 2 out shape= ", out.shape)

        # softmax log probs
        # soft_max probs = e^out/sum(e^out)
        # log_soft_max = log(soft_max)
        # n_log_soft_max = -log_soft_max
        # loss = n_log_soft_max[ipd_idx].mean()

        log_sf = F.log_softmax(out, dim=1)
        
        return log_sf

    def loss_function(self, log_probs, targes):
        print("Using default loss fn")
        loss_fn = torch.nn.NLLLoss().to(self.device)
        return loss_fn(log_probs, targes)


class Optimizer:
    def __init__(self, model, lr=0.001):
        self.model = model
        self.lr = lr
    
    def step(self):
        print("Using default optimizer")
        for param in self.model.parameters():
            param.data -= self.lr * param.grad



