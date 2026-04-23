import csv
from math import sqrt
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 

# This builds an character level MLP to generate new names

class SimpleCharacterMLP:
    def __init__(self, training_set, block_size=3, training_split=80, dev_split=10, test_split=10, batch_size=32, debug=False):
        self.debug = debug
        self.block_size = block_size

        # build character-level vocabulary and mappings (c2i, i2c)
        all_chars = sorted(list(set(''.join(training_set))))
        unique_chars = len(all_chars)
        self.total_chars = unique_chars + 1 # extra spl char to denote start and end of the name

        self.c2i = {c:i+1 for i,c in enumerate(all_chars)}
        self.c2i['.'] = 0
        self.i2c = {i:c for c,i in self.c2i.items()} 

        # split the data according to the splits and create train,dev,test sets
        random.shuffle(training_set)
        total = len(training_set)
        train_end = int(training_split / 100 * total)
        dev_end = int((training_split + dev_split) / 100 * total)
        
        # build data sets
        self.X_train, self.Y_train = self._build_data_set(training_set[:train_end], block_size)
        self.X_dev, self.Y_dev = self._build_data_set(training_set[train_end:dev_end], block_size)
        self.X_test, self.Y_test = self._build_data_set(training_set[dev_end:], block_size)

        # set hyper parameters
        self.batch_size = batch_size if batch_size > 0 else len(self.X_train)
        self.embedding_dim_size = 2
        self.layer1_neurons = 100
        self.learning_rate = 0.1

        self._init_parms()
    
    def set_debug(self, debug_val):
        self.debug = debug_val
    
    # builds input X with block size element each - each element is a tensor of shape (block_size)
    # and corresponding target Y which is the next character
    def _build_data_set(self, words, block_size):
        X = [] # array of block_size'd arrays, providing sequence of chars
        Y = [] # Y[i] would indicate the next letter after the X[i]th letters
        for word in words:
            # create a block sized context arr for every word, start with '.'
            context = [0] * block_size
            for c in word+'.':
                cix = self.c2i[c]
                X.append(context)
                Y.append(cix)
                # print("".join([self.i2c[i] for i in context]), "===>", self.i2c[cix])
                context = context[1:] + [cix]
        
        return torch.tensor(X), torch.tensor(Y)
    
    def _init_parms(self):
        """
        Why multiply weights with special factors?
        - Random weights start too big or too small, making training unstable
        - Big weights -> neurons saturate (outputs always +1/-1), gradients vanish
        - Small weights -> gradients too tiny, learning is super slow
        - We scale weights to keep activations in a "sweet spot" where gradients flow well
        - Kaiming init scales based on layer size to maintain good gradient flow
        - Goal: Keep neuron outputs as unit variance gaussian (mean=0, std=1)
        - This ensures each layer maintains similar activation scales throughout training
        - instead of tuning manually we can also add a Batch normalization layer
        """
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
        self.b1 = torch.randn(self.layer1_neurons) * 0

        # create layer 2 with total_chars neurons and output size of layer 1 (as inputs)
        layer2_neurons = self.total_chars # we want one o/p for  each char
        self.W2 = torch.randn(self.layer1_neurons, layer2_neurons) * 0.01 #(inp_size, num_neurons) #(100, 27)
        self.b2 = torch.randn(self.total_chars) * 0

        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2]
        for p in self.parameters:
            p.requires_grad = True
        self.num_params = sum(param.nelement() for param in self.parameters)

        if self.debug:
            print(f"Model parameter shapes: C={self.C.shape}, W1={self.W1.shape}, b1={self.b1.shape}, W2={self.W2.shape}, b2={self.b2.shape}")
            print(f"Total parameters: {self.num_params}")
        
    def reset_parameters(self):
        self._init_parms()
    
    def train(self, training_epochs = 10, plot_graphs = ['tanh', 'loss']):
        stepi = []
        lossi = []
        for i in range(training_epochs):
            loss = self._forward_pass(plot_graphs=plot_graphs)
            self.current_loss = loss.item()
            stepi.append(i)
            lossi.append(loss.item())
            if i % 10000 == 0:
                print(f"Epoch {i}/{training_epochs}, Loss: {loss.item():.4f}")
            self._backward_pass(loss)
            self._update_parms()
       
        if 'loss' in plot_graphs:
            plt.plot(stepi, lossi)
            plt.title("Loss")
            plt.show()
        

    def _forward_pass(self, plot_graphs=[]):
        batch_idxs = torch.randint(0, len(self.X_train), (self.batch_size,)) # creates batch sized indexs array
        Xbatch = self.X_train[batch_idxs]

        emb = self.C[Xbatch] # creates a tensor of shape (batch_size, block_size, embedding_dim_size)
        
        emb_vie_col_size = self.block_size * self.embedding_dim_size

        # ex: if block_size = 3 , embedding_dim_size = 2, batch_size = 32, W1 = (6, 100) W2 = (100, 27)

        # layer 1
        # emb.view(len(Xbatch), emb_vie_col_size) => creates a vector of (32, 6) 
        h_pre_act = emb.view(len(Xbatch), emb_vie_col_size) @ self.W1 + self.b1 # should (32, 100) output 
        h = torch.tanh(h_pre_act)
        
        # layer 2
        logits = h @ self.W2 + self.b2 # should be (32, 27) output
        # this is equivalent to the below manual method but more efficient and numerically stable
        loss = F.cross_entropy(logits, self.Y_train[batch_idxs])
        # # get the counts 
        # counts = logits.exp()
        # softmax_probs = counts / counts.sum(dim=1, keepdim=True) # probs now has shape (32, 27) where each row sums to 1

        # # now we need to compute the loss - negative log likelihood loss
        # # for each example in the batch, we look at the probability of the correct next character
        # # and take the negative log of that probability

        # y_pred_probs = softmax_probs[idxs, self.Y_train[idxs]]
        # loss = -y_pred_probs.log().mean()
        # we can interprit -y_pred_probs.log() as the loss 
        # -y_pred_probs.log().mean() as the overall cost

        # debug logs
        if self.debug:
            print(f"emb shape: {emb.shape}")
            print(f"emb_view size {emb_vie_col_size}")
            print(f"h shape: {h.shape}")
            print(f"logits shape: {logits.shape}")
        
        if 'tanh' in plot_graphs:
            # plot pre activations and hist of tanh activations
            
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8))

          

            h_preact = h_pre_act.view(-1).tolist()
            # plt.subplot(1, 2, 2)
            ax1.hist(h_preact, bins=50)
            ax1.set_title("Distribution of pre-activation values")
            ax1.set_xlabel("Pre-activation value")
            ax1.set_ylabel("Frequency")

            # scatter plot if abs(h) >0.99
            # h.abs > 0.99 gives a boolean matrix of size (batch_size, layer1_neurons)
            # it plots black where its false and white if its true
            ax2.imshow(h.abs() > 0.99, cmap="gray", interpolation="nearest") 
            ax2.set_title("Neuron tahn outputs")

            h_list = h.view(-1).tolist() # converts to list of floats
            ax3.hist(h_list, bins=50)
            ax3.set_title("Distribution of tanh activations")
            ax3.set_xlabel("Activation value")
            ax3.set_ylabel("Frequency")

            plt.show()


        return loss

    def _backward_pass(self, loss):
        for p in self.parameters:
            p.grad = None
        
        loss.backward()
    
    def _update_parms(self):
        for p in self.parameters:
            p.data += -self.learning_rate *  p.grad
    
    def full_data_set_loss(self):
        return self.compute_loss(self.X_train, self.Y_train)
    
    def loss_on_dev_set(self):
        return self.compute_loss(self.X_dev, self.Y_dev)
    
    def loss_on_test_set(self):
        return self.compute_loss(self.X_test, self.Y_test)
    
    @torch.no_grad() # tells torch that it doesn't need to maintain the compute graph for the op's below
    def compute_loss(self, X, Y):
        # compute loss for given input X and target Y
        logits = self.C[X].view(-1, self.block_size * self.embedding_dim_size) @ self.W1 + self.b1
        logits = torch.tanh(logits)
        logits = logits @ self.W2 + self.b2
        loss = F.cross_entropy(logits, Y)
        return loss.item()
    
    def sample_words(self, size=10, start_with='.'):
        words = []
        for _ in range(size):
            word = []
            if start_with != '.':
                word.append(start_with)
            start_char_idx = self.c2i.get(start_with, 0)
            context = [0] * (self.block_size - 1) + [start_char_idx]
            while True:
                # get the embedding from C for '.'
                emb = self.C[torch.tensor(context)] # this is (1, block_size, embedding_dim_size)
                
                # get layer1 out
                h = torch.tanh(emb.view(1, -1) @ self.W1 + self.b1) # emb.view(1, -1) => (1, block_size * embedding_dim_size)

                # layer 2
                logits = h @ self.W2 + self.b2

                # sample from the logits
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs, num_samples=1).item()
                
                if ix == 0:
                    break
                # convert index back to character
                ch = self.i2c[ix]
                word.append(ch)
                context = context[1:] + [ix]
            
            words.append(''.join(word))

        return words

        
        

if __name__ == "__main__":
    import os
    names = []
    # print("Current directory:", os.getcwd())
    # print("Looking for indian_names_m.csv in:", os.path.join(os.getcwd(), "ex_llm", "main", "resources"))
    with open("./main/resources/indian_names_m.csv", "r", newline='') as file:
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
    model = SimpleCharacterMLP(names, block_size=3, batch_size=32)
    print("Total params = ", model.num_params)
    print("X_train shape:", model.X_train.shape)
    print("Y_train shape ", model.Y_train.shape)
    # model.train(training_epochs=10000)
    # print("Loss on mini batches: ", model.current_loss)
    # print("Final loss on full dataset:", model.full_data_set_loss())
