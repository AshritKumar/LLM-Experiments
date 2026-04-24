from math import sqrt
from mlp_char_level import SimpleCharacterMLP
import torch
import torch.nn.functional as F
import csv

# Goal: We will do different excersies to deep dive into back prop by doing it manually

class ExplicitCharMLP(SimpleCharacterMLP):
    
    def _init_parms(self):
        # create embedding lookup C
        self.C = torch.randn(self.total_chars, self.embedding_dim_size)

        # create a layer 1 with layer1_neurons size neurons and  embedding_dim_size * block_size inputs 
        # Each column is is weights of one neuron
        # outputs (batch_size, layer1_neurons) sized vector
        # emb = 32 x 6 
        # W1 = 6 @ 100 => 
        # emb @ W1 => 32 @ 100
        kaiming_init_for_tanh = (5/3)/ sqrt(self.embedding_dim_size * self.block_size)
        self.W1 = torch.randn(self.embedding_dim_size * self.block_size, self.layer1_neurons) * kaiming_init_for_tanh #(inp_size, num_neurons) #(6, 100)
        self.b1 = torch.randn(self.layer1_neurons) * 0.1

        # create layer 2 with total_chars neurons and output size of layer 1 (as inputs)
        layer2_neurons = self.total_chars # we want one o/p for  each char
        self.W2 = torch.randn(self.layer1_neurons, layer2_neurons) * 0.01 #(inp_size, num_neurons) #(100, 27)
        self.b2 = torch.randn(self.total_chars) * 0.1

        # batch norm after layer 1 (befor tanh)
        self.bn_gain = torch.randn(1, self.layer1_neurons)*0.1 + 1
        self.bn_bias = torch.randn(1, self.layer1_neurons) * 0.1

        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2, self.bn_gain, self.bn_bias]
        for p in self.parameters:
            p.requires_grad = True
        self.num_params = sum(param.nelement() for param in self.parameters)

        if self.debug:
            print(f"Model parameter shapes: C={self.C.shape}, W1={self.W1.shape}, b1={self.b1.shape}, W2={self.W2.shape}, b2={self.b2.shape}, bn_gain={self.bn_gain.shape}, bn_bias={self.bn_bias.shape}")
            print(f"Total parameters: {self.num_params}")
    
    def train(self, training_epochs=10):
        n = self.batch_size
        # mini batch
        batch_idxs = torch.randint(0, len(self.X_train), (self.batch_size,)) # creates batch sized indexs array
        Xbatch = self.X_train[batch_idxs]
        Ybatch = self.Y_train[batch_idxs]

        self.xb = Xbatch
        # embeddings for Xbatch
        emb = self.C[Xbatch] # (batch_size, block_size, embedding_dim_size)
        emb_view = emb.view(len(Xbatch), -1) # (batch_size, block_size * embedding_dim_size)

        # Layer 1 
        # pre activations
        h_pre_bn = emb_view @ self.W1 + self.b1

        # batch norm - calculate every step explictly
        bn_mean_i = 1/n * h_pre_bn.sum(dim=0, keepdim=True) # size (1, layer_1_neurons)
        bn_diff = h_pre_bn - bn_mean_i # size (batch_size, layer_1_neurons)
        bn_diff_sqr = bn_diff ** 2 # size (batch_size, layer_1_neurons)
        bn_var = 1/(n-1) * bn_diff_sqr.sum(dim=0, keepdim=True) # size (1, layer_1_neurons)
        bn_var_inv = (bn_var + 1e-5)**-0.5 # inverse of standard deviation i.e, 1/sqrt(bn_var + eps). Size: (1, layer_1_neurons)
        bn_raw = bn_diff * bn_var_inv # size (batch_size, layer_1_neurons). This is z_score standarization. (xi-mean/std)
        h_post_bn = bn_raw * self.bn_gain + self.bn_bias # size (batch_size, layer_1_neurons)

        # apply non linearity
        h = torch.tanh(h_post_bn) # size (batch_size, layer_1_neurons)

        # Layer 2
        logits = h @ self.W2 + self.b2 # size (batch_size, vocab_size) [vocab_size = layer_2_neurons]

        # loss calculation (cross entropy)
        # mimick F.cross_entropy
        log_maxes = logits.max(dim=1, keepdim=True).values # size (batch_size, 1)
        norm_logs = logits - log_maxes # size (batch_size, vocab_size)
        
        # counts
        counts = norm_logs.exp() # size (batch_size, vocab_size)
        counts_sum = counts.sum(dim=1, keepdim=True) # size (batch_size, 1)
        counts_sum_inv = counts_sum ** -1 # just done for the sake of manual back prop # size: (batch_size, 1)
        probs = counts * counts_sum_inv
        log_probs = probs.log()  # size: (batch_size, vocab_size)
        loss = -log_probs[range(n), Ybatch].mean() # log_probs[batch_idxs, Ybatch] -> this picks Ybatch[i] element for each batch_idxs[i]


        # PyTorch back prop
        for p in self.parameters:
            p.grad = None
        
        self.intrm_tensors = [log_probs, probs, counts_sum_inv, counts, counts_sum,
                    norm_logs, log_maxes, logits, h, h_post_bn,
                    bn_raw, bn_var_inv, bn_var, bn_diff_sqr, bn_diff,
                    bn_mean_i, h_pre_bn, emb_view, emb]

        # retain grad for all intrim tensors later comparison purposes
        for t in self.intrm_tensors:
            t.retain_grad()
        
        loss.backward()

        self.ex_1(batch_idxs)
    
    def ex_1(self, batch_ids):
        #backprop through the whole thing manually, 
        # backpropagating through exactly all of the variables 
        # as they are defined in the forward pass above, one by one

        # d_loss = 1
        # d_log_probs -> d(loss)/d(log_probs). if l = (a + b + c )/3, then d(l)/d(a) = 1/3, d(l)/d(b) = 1/3, d(l)/d(c) = 1/3
        # so d_log_probs = [1/3, 1/3, 1/3, ...] (size: batch_size, vocab_size)
        log_probs = self.intrm_tensors[0]
        d_log_probs = torch.zeros_like(log_probs)
        d_log_probs[range(self.batch_size), self.Y_train[batch_ids]] = -1/self.batch_size
        self.cmp('log_probs', d_log_probs, log_probs)


        # d(loss)/d(probs) = d(loss)/d(log_probs) * d(probs) / d(log_probs)
        # but d(probs) / d(log_probs) = 1/probs since d(logx)/dx = 1/x
        # so d_probs = d_log_probs * (1/probs) = d_log_probs / probs
        probs = self.intrm_tensors[1]
        inv_probs = 1 / probs
        d_probs = d_log_probs * inv_probs # this is element wise multiplication, size is still (batch_size, vocab_size)
        self.cmp('probs', d_probs, probs)


        # Next, probs = counts * counts_sum_inv, 
        # so we need d(loss)/d(counts) and d(loss)/d(counts_sum_inv)
        # d(loss)/d(counts_sum_inv) = d(loss)/d(probs) * d(probs)/d(counts_sum_inv) [chain rule]
        # d(probs)/d(counts_sum_inv) = counts (since probs[i] = counts[i] * counts_sum_inv)
        counts_sum_inv = self.intrm_tensors[2]
        counts = self.intrm_tensors[3]
        
        # We would have to do a summation across dim 1 (across rows) here since counts is (32 x 27) matrix but counts_sum_inv is (32 x 1) 
        # When we do counts * counts_sum_inv, pytorch internally replicates counts_sum_inv across each row to make it (32 x 27) and does an element wise multiplication
        # When a element (or variable) is contributing to multiple operations, its gradient must be accumulated (via summation) across all those operations
        # Each element of counts_sum_inv contributes to all elements in the corresponding row of probs
        # So for each row i, d_counts_sum_inv[i] = sum_j(counts[i,j] * d_probs[i,j])
        # This gives us a (32 x 1) gradient for counts_sum_inv
    
        d_counts_sum_inv_replicated = (counts * d_probs) # this is the gradient of replicated counts_sum_inv size: 32x27
        d_counts_sum_inv = d_counts_sum_inv_replicated.sum(1, keepdim=True)  # sum across columns (dim 1) to get (32 x 1) gradient for counts_sum_inv
        self.cmp('counts_sum_inv', d_counts_sum_inv, counts_sum_inv)

        d_counts = d_probs * counts_sum_inv # this is not complete since counts is also part of another op, counts_sum = counts.sum(dim=1, keepdim=True)
        # self.cmp('counts', d_counts, counts)

        # Next, counts_sum_inv = counts_sum ** -1
        # so d(loss)/d(counts_sum) = d(loss)/d(counts_sum_inv) * d(counts_sum_inv)/d(counts_sum) [chain rule]
        # d(counts_sum_inv)/d(counts_sum) = -1 * counts_sum ** -2 = -1 / (counts_sum ** 2)
        counts_sum = self.intrm_tensors[4]
        d_counts_sum = d_counts_sum_inv * (-1 * (counts_sum ** -2)) # element wise multiplication, size is (32 x 1)
        self.cmp('counts_sum', d_counts_sum, counts_sum)

        # Next, counts_sum = counts.sum(dim=1, keepdim=True)
        # d(loss)/d(counts) = d(loss)/d(counts_sum) * d(counts_sum)/d(counts) [chain rule]
        # d(counts_sum)/d(counts) = 1 (since each element of counts contributes to exactly one element of counts_sum)
        # Therefore, d(loss)/d(counts) = d_counts_sum (broadcasted to match counts shape)
        d_counts += torch.ones_like(counts) * d_counts_sum
        self.cmp('d_counts', d_counts, counts)


        # Next, counts = norm_logs.exp()
        # d(loss)/d(norm_logs) = d(loss)/d(counts) * d(counts)/d(norm_logs) - [chain rule]
        # d(counts)/d(norm_logs) = counts (since counts = exp(norm_logs))
        norm_logs = self.intrm_tensors[5]
        d_norm_logs = d_counts * counts  # element-wise multiplication, size is (32 x 27)
        self.cmp('d_norm_logs', d_norm_logs, norm_logs)

        # Next, norm_logs = logits - log_maxes
        # d(loss)/d(logits) = d(loss)/d(norm_logs) * d(norm_logs)/d(logits) [chain rule]
        # d(norm_logs)/d(logits) = I (identity matrix, since each element of norm_logs only depends on the corresponding element of logits)
        # in other words d(norm_logs)/d(logits) = 1
        # Therefore, d(loss)/d(logits) = d_norm_logs
        d_logits = d_norm_logs.clone()  # size is (32 x 27)
    

        # d(loss)/d(log_maxes) = d(loss)/d(norm_logs) * d(norm_logs)/d(log_maxes) [chain rule]
        # d(norm_logs)/d(log_maxes) = -1 (since norm_logs = logits - log_maxes)
        log_maxes = self.intrm_tensors[6]
        d_log_maxes = (d_norm_logs * (-1)).sum(dim=1, keepdim=True)  # we will have to sum across columns since log_maxes would have been replicated by PyTorch broadcasting (each row has the same log_maxes value)
        self.cmp('d_log_maxes', d_log_maxes, log_maxes)


        # Next log_maxes = logits.max(dim=1, keepdim=True).values
        # d(loss)/d(logits) = d(loss)/d(log_maxes) * d(log_maxes)/d(logits)
        # d(log_maxes)/d(logits) is an zero array like logits with 1's in max positions
        logits = self.intrm_tensors[7]
        d_logits2 =  F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * d_log_maxes
        d_logits = d_logits + d_logits2
        self.cmp('d_logits', d_logits, logits)


        # Next, logits = h @ self.W2 + self.b2
        # d(loss)/d(h) = d(loss)/d(logits) * d(logits)/d(h) [chain rule]
        # d(logits)/d(h) = self.W2.T @ d(loss)/d(logits)
        h = self.intrm_tensors[8]
        d_h = d_logits @ self.W2.T
        self.cmp('d_h', d_h, h)

        d_w2 = h.T @ d_logits
        self.cmp('d_w2', d_w2, self.W2)

        d_b2 = d_logits.sum(dim=0)
        self.cmp('d_b2', d_b2, self.b2)

        # Next h = torch.tanh(h_post_bn)
        # d(loss)/d(h_post_bn) = d(loss)/d(h) * d(h)/h_post_bn
        # d(h)/h_post_bn = 1 - h**2
        h_post_bn = self.intrm_tensors[9]
        d_h_post_bn = (1 - h**2) * d_h
        self.cmp('d_h_post_bn', d_h_post_bn, h_post_bn)

        # Next h_post_bn = bn_raw * self.bn_gain + self.bn_bias
        # d(loss)/d(bn_gain) = d(loss)/d(h_post_bn) * d(h_post_bn)/d(bn_gain)
        # so it would be (bn_raw * h_post_bn).sum(1, kd=true)
        bn_raw = self.intrm_tensors[10]
        d_bn_gain = (bn_raw * d_h_post_bn).sum(dim=0, keepdim = True)
        self.cmp('d_bn_gain', d_bn_gain, self.bn_gain)

        d_bn_raw = self.bn_gain * d_h_post_bn # element wise multiply
        self.cmp('d_bn_raw', d_bn_raw, bn_raw)

        d_bn_bias = d_h_post_bn.sum(dim=0, keepdim=True)
        self.cmp('d_bn_bias', d_bn_bias, self.bn_bias)

        # Next bn_raw = bn_diff * bn_var_inv 
                      # (32, 64) (1, 64)
        bn_var_inv = self.intrm_tensors[11]
        bn_diff = self.intrm_tensors[14]
        d_bn_diff = bn_var_inv * d_bn_raw # not fully complete
        

        d_bn_var_inv = (bn_diff * d_bn_raw).sum(dim=0, keepdim=True)
        self.cmp('d_bn_var_inv', d_bn_var_inv, bn_var_inv)

        # Next, bn_var_inv = (bn_var + 1e-5)**-0.5
        bn_var = self.intrm_tensors[12]
        d_bn_var = d_bn_var_inv * (-0.5 * (bn_var + 1e-5)**-1.5)
        self.cmp('d_bn_var', d_bn_var, bn_var)

        # Next,  bn_var = 1/(n-1) * bn_diff_sqr.sum(dim=0, keepdim=True)
        bn_diff_sqr = self.intrm_tensors[13]
        d_bn_diff_sqr = (1.0 / (self.batch_size - 1)) * torch.ones_like(bn_diff_sqr) * d_bn_var
        self.cmp('d_bn_diff_sqr', d_bn_diff_sqr, bn_diff_sqr)

        # Next, bn_diff_sqr = bn_diff ** 2
        d_bn_diff += 2 * bn_diff * d_bn_diff_sqr
        self.cmp('d_bn_diff', d_bn_diff, bn_diff) 

        # Next, bn_diff = h_pre_bn - bn_mean_i
        h_pre_bn = self.intrm_tensors[16]
        d_h_pre_bn = d_bn_diff.clone() # not fully complete
        

        bn_mean_i = self.intrm_tensors[15]
        d_bn_mean_i = (-1 * d_bn_diff).sum(0, keepdim=True)
        self.cmp('d_bn_mean_i', d_bn_mean_i, bn_mean_i) 

        # Next, bn_mean_i = 1/n * h_pre_bn.sum(dim=0, keepdim=True)
        d_h_pre_bn += (1/self.batch_size) * torch.ones_like(h_pre_bn) * d_bn_mean_i
        self.cmp('d_h_pre_bn', d_h_pre_bn, h_pre_bn) 

        # Next, h_pre_bn = emb_view @ self.W1 + self.b1
        emb_view = self.intrm_tensors[17]
        emb = self.intrm_tensors[18]
        d_emb_view = d_h_pre_bn @ self.W1.T
        self.cmp('d_emb_view', d_emb_view, emb_view) 

        d_w1 = emb_view.T @ d_h_pre_bn
        self.cmp('d_w1', d_w1, self.W1) 

        d_b1 = d_h_pre_bn.sum(0, keepdim=True)
        self.cmp('d_b1', d_b1, self.b1)

        # Next, emb_view = emb.view(len(Xbatch), -1)
        # This is just a view representation, so just by changing the view rep of d_emb_view we would get the derivative of emd
        d_emb = d_emb_view.view(emb.shape) # d_emb would be (32, 3, 10)
        self.cmp('d_emb', d_emb, emb)
        
        # Next, emb = self.C[Xbatch]
        # Here we need to put all the correct d_emb dimension into C
        # For ever element in Xbatch we need to put the 10 dimensional embeding derviatie present in d_emb in the correct position of d_c
        d_c = torch.zeros_like(self.C)
        for i in range(self.xb.shape[0]):
            for j in range(self.xb.shape[1]):
                idx = self.xb[i][j] # gives the character index
                e = d_emb[i][j] # gives the 10 dimensionalembedding for that character
                d_c[idx] += e
        
        self.cmp('dc', d_c, self.C)



        






        
        



    # utility function we will use later when comparing manual gradients to PyTorch gradients
    def cmp(self, s, manualDT, pyTorchDT):
        ex = torch.all(manualDT == pyTorchDT.grad).item()
        app = torch.allclose(manualDT, pyTorchDT.grad)
        maxdiff = (manualDT - pyTorchDT.grad).abs().max().item()
        print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')
            
        



        

###################################################################### TEST ############################################################cd e####################


names = []
with open("./ex_llm/main/resources/indian_names_m.csv", "r", newline='') as file:
    reader = csv.DictReader(file)
    # get all the names from 'name' header
    for row in reader:
        name = row['name']
        # Filter to English characters only (ASCII letters) and spaces
        # english_name = ''.join(c for c in name if c.isalpha() and ord(c) < 128)
        english_name = ''.join(c for c in name if (c.isalpha() or c == ' ') and ord(c) < 128)
        if len(english_name.strip()) > 3:  # Keep names with at least 3 English characters
            names.append(english_name.lower())  # Convert to lowercase for consistency

print(f"Total English names loaded: {len(names)}")

# Create the model instance
model = ExplicitCharMLP(names, block_size=3, batch_size=32, debug=False)
model.layer1_neurons = 64
model.embedding_dim_size = 10
model.reset_parameters()
print("Total params = ", model.num_params)

model.train(1)
        

        
        