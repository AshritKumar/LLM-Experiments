# Mimicks the pyTorch classes for Layers - Liner, tanh, BatchNorm

import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Linear:
    """
        Defines a Linear layer
        fan_in: Number of inputs (Perticularly this is the number of features of the input)
        fan_out: Number of outputs (This is also equal to number of neurons in the layer)
        bias: Specifies whether to add a bias to the layer output
    """
    
    def __init__(self, fan_in, fan_out, bias=True):
        # kaiming init, divide by sqrt(fan_in)
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None # we initially want the bias to be zeros
    
    # runs the forward pass for given inputs

    #  Performs the core linear transformation by matrix multiplying the input tensor with the layer's 
    # weight matrix and optionally adding bias. The input tensor can have any number of dimensions, 
    # but the last dimension must equal fan_in (the layer's input size). The output will have the 
    # same shape as input except the last dimension becomes fan_out (number of neurons). For example, 
    # with input shape (batch_size, sequence_length, fan_in), the output shape will be 
    # (batch_size, sequence_length, fan_out). Each element in the output represents how strongly 
    # a particular neuron responds to a specific input position - think of each row as one input 
    # example's transformation across all neurons, and each column as one neuron's response pattern 
    # across all input examples. This allows the layer to learn different feature detectors where 
    # each neuron specializes in recognizing specific patterns in the input data.
    def forward(self, input: torch.Tensor):
        self.out = input @ self.weight
        if self.bias is not None:
            self.out += self.bias # in place operation
        # print(f"Linear: Input shape {input.shape}, weight shape = {self.weight.shape}, output shape = {self.out.shape}")
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    """
        dim: Number of dimensions or no of outputs (neurons) af the layer after which this batch norm is applied
        momentum: Used to calculate the exp moving avg of running_mean and running_variance
    """
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.momentum = momentum
        self.eps = eps
        self.dim = dim
        self.training = True

        # define the learnable parms gain(gamma) and bias(beta).
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        # buffers to maintain the running_mean and running_std. These will not be used as params in back prop
        self.running_mean = torch.ones(dim)
        self.running_var = torch.zeros(dim)

    # applies the batch norm
    # idea here is to normalize each neurons outputs to a gaussian distribution, scale it by gain and shift it by bias
    def forward(self, input: torch.Tensor):
        # print(f"BatchNorm: Input shape = {input.shape}, gain shape = {self.gamma.shape}, bias shape = {self.beta.shape}")
        # we need to use the mean and variance of the input in training mode, when in eval or sampling we need to use the running_mean and running_vae
        if self.training:
            # we are calculating this over 0 dimension(along the cols, i.e across all inputs for every neuron). So a (32,100) input will become (1, 100)
            cur_mean = input.mean(0, keepdim=True)
            cur_var = input.var(0, keepdim=True)
        else:
            cur_mean = self.running_mean
            cur_var = self.running_var
        
        # standardize input to unit variance (z_score standardization)
        iHat = (input - cur_mean)/torch.sqrt(cur_var + self.eps)
        # scale by the gain and add bias. At the initialization since gain is vector of 1's and bias is vector of 0's it out will be perfect gaussian dist with unit variance.
        # since gain(gamma) and bias(beta) are learnable parms they will be modified in the back prop and will be adjusted accordingly to minimize the loss
        self.out = self.gamma * iHat + self.beta

        # compute the runnign mean and var buffers only in training
        if self.training:
            with torch.no_grad(): # no need to track op's on running_mean and running_var
                # this will be exponential moving avg of mean and var
                self.running_mean = self.running_mean * (1-self.momentum) + cur_mean * self.momentum
                self.running_var = self.running_var * (1-self.momentum) + cur_var * self.momentum
        
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def forward(self, input):
        self.out = torch.tanh(input)
        return self.out

    def parameters(self):
        return []


class Embedding:
    """
    Creates an embedding lookup table, in forward pass creates an embedding for each input passed
    num_embeddings - Number of embeddings to create (could be equal to the size of the vocabulary)
    embedding_dim_size - Dimension of each embedding vector
    """
    def __init__(self, num_embeddings, embedding_dim_size):
        # Here weight is the C vector
        self.weight = torch.randn((num_embeddings, embedding_dim_size))
    
    def forward(self, input: torch.Tensor):
        self.out =  self.weight[input]
        # print(f"Embedding: Input shape {input.shape}, weight shape = {self.weight.shape}, output shape = {self.out.shape}")
        return self.out
    
    def parameters(self):
        return [self.weight]

class Flatten:
    def forward(self, input: torch.Tensor):
        self.out = input.view(input.shape[0], -1)
        # print(f"Flatten: Input shape {input.shape} output shape = {self.out.shape}")
        return self.out

    def parameters(self):
        return []

# Instead of maintaining a naked list of layers we can have an abstractions for the layers.
# This would do a forwardpass on all the layers
class Sequential:
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, input):
        out = input
        for layer in self.layers:
            out = layer.forward(out)
        self.out = out
        return self.out
    
    def parameters(self):
        # gets the parameters of all the layers
        return [p for layer in self.layers for p in layer.parameters()]

        
            
# NOTE: The last dimension of any layer out is can be called as number of neurons, no. of features or no. of channels


# defines a N layer MLP fo char model
class ComplexCharMLP:
    """
        training_set: Full word list
        block_size: block size of each ele in input if 3, the given 3 chars we predict 4th char
        training_split: % of training_set to be used of training
        dev_split: % of training_set to be used of dev
        test_split: % of training_set to be used of testing
        batch_size: mini batch size to be used in tarining
        layers: number of layers in MLP - Each layer will have (linear, tanh, batchnorm) except the last layer
    """
    def __init__(self, training_set, block_size=3, training_split=80, dev_split=10, test_split=10, batch_size=32, num_layers=6, debug=False):
        self.debug = debug
        self.block_size = block_size
        self.num_layers = num_layers

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
    

    def init_layers_params(self):
        # create the embedding looup vector C
        # self.C = torch.randn(self.total_chars, self.embedding_dim_size)
        # Instead of specificly creating the lookup now we can create embedding and flatten layers

        # initialize number of linear layers and one softmax layer
        self.layers = [
            Embedding(self.total_chars, self.embedding_dim_size),
            Flatten(),
            Linear(self.block_size * self.embedding_dim_size, self.layer1_neurons, bias=False), BatchNorm1d(self.layer1_neurons), Tanh() # no bias as we are using batch norm layer
        ]
        
        # add middle layers
        for _ in range(self.num_layers - 2):
            self.layers.extend([
                Linear(self.layer1_neurons, self.layer1_neurons, bias=False),
                BatchNorm1d(self.layer1_neurons),
                Tanh()
            ])
        
        # self.layers.extend([Linear(self.layer1_neurons, self.total_chars, bias=False), BatchNorm1d(self.total_chars)])
        self.layers.extend([Linear(self.layer1_neurons, self.total_chars)])

        # Wrap all the layers in sequential container
        self.model = Sequential(self.layers)

        # adjust the weights of layer

        with torch.no_grad():
            # make the last layer weights less as we want a uninform probability for the logits
            self.model.layers[-1].weight *= 0.1

            # # scale the linear layers before tanh by 5/3 for uniform preacts activations. NOT required if we have BN
            # for layer in self.layers[:-1]: # except the last softmax layer
            #     if isinstance(layer, Linear):
            #         layer.weight *= 5/3
        

        # parameters of the model
        self.parameters = self.model.parameters()
        for p in self.parameters:
            p.requires_grad = True
        
        self.num_params = sum(param.nelement() for param in self.parameters)
        print(f"Total parameters: {self.num_params}")
        
        # print debug logs of layer shapes
        if self.debug:
            
            print("Total layers ", len(self.model.layers))
            for i, layer in enumerate(self.model.layers):
                if isinstance(layer, Embedding):
                    print(f"Layer {i}: Embedding(shape={layer.weight.shape})")
                if isinstance(layer, Linear):
                    print(f"Layer {i}: Linear(shape={layer.weight.shape})")
                if isinstance(layer, BatchNorm1d):
                    print(f"Layer {i}: BatchNorm1d(shape={layer.gamma.shape})")
              

    def reset_parameters(self):
        self.init_layers_params()
    
    def train(self, training_epochs=1000):
        self.lossi = []
        self.parm_update_stats = [] # each element is an array of (lr*grad/p.data) scaler vals for each parameter
        for i in range(training_epochs):
            # construct mini batch
            batch_idxs = torch.randint(0, len(self.X_train), (self.batch_size,))
            Xbatch = self.X_train[batch_idxs] # creates a tensor of shape (batch_size, block_size, embedding_dim_size)
            
            # x = Xbatch # (32, 6)
            # # forward pass
            # # getting of embeddings for the Xbatch (embed the characters into vectors) would be taken care in the embedding loop
            # for layer in self.layers:
            #     x = layer.forward(x)
            
            # Directly call the model with the batch
            # use the final output as logits for loss computation
            logits = self.model(Xbatch)
            loss = F.cross_entropy(logits, self.Y_train[batch_idxs]) # this is loss function
            self.lossi.append(loss.item())

            if i % 10000 == 0:
                print(f"Epoch {i}/{training_epochs}, Loss: {loss.item():.4f}")
            
            if i%20000 ==0:
                self.monitor_batchnorm_stats()

             # backward pass
            for p in self.parameters:
                p.grad = None
            
            for layer in self.model.layers:
                layer.out.retain_grad() 
           
            loss.backward()

            # update
            lr = 0.01 if i > 100000 else self.learning_rate
            for p in self.parameters:
                p.data -= lr * p.grad
            
            # parameter update stats
            with torch.no_grad():
                update_stats = []
                for p in self.parameters:
                    update_magnitude = (lr*p.grad).std()
                    parm_data_magnitude = p.data.std()
                    update_stats.append((update_magnitude/parm_data_magnitude).log10().item()) # convert to log values for easier represnetation
                
                self.parm_update_stats.append(update_stats)


    def full_data_set_loss(self):
        return self.compute_loss(self.X_train, self.Y_train)
    
    def loss_on_dev_set(self):
        return self.compute_loss(self.X_dev, self.Y_dev)
    
    def loss_on_test_set(self):
        return self.compute_loss(self.X_test, self.Y_test)
    
    @torch.no_grad() # tells torch that it doesn't need to maintain the compute graph for the op's below
    def compute_loss(self, X, Y):
        # compute loss for given input X and target Y
        # emb = self.C[X]
        # x = emb.view(-1, self.block_size * self.embedding_dim_size)
        
        # x = X
        self.set_layers_training(False)
        # for layer in self.model.layers:
        #     x = layer.forward(x)
        
        logits = self.model(X)
        loss = F.cross_entropy(logits, Y)
        self.set_layers_training(True)
        return loss.item()
    
    def set_layers_training(self, is_training=True):
        for layer in self.model.layers:
            if hasattr(layer, 'training'):
                layer.training = is_training
    
    def monitor_batchnorm_stats(self):
        """
            Monitor BatchNorm running statistics for debugging
            Running var < 0: Major problem, indicates numerical instability
            Running var > 100: Activations have very high variance
            Gamma values > 10: BatchNorm is compensating heavily
            Running mean/var not changing: BatchNorm might not be updating
        """
        print("\n=== BatchNorm Running Stats ===")
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, BatchNorm1d):
                running_mean_stats = f"mean: {layer.running_mean.mean():.4f}, std: {layer.running_mean.std():.4f}"
                running_var_stats = f"mean: {layer.running_var.mean():.4f}, std: {layer.running_var.std():.4f}"
                gamma_stats = f"mean: {layer.gamma.mean():.4f}, std: {layer.gamma.std():.4f}"
                beta_stats = f"mean: {layer.beta.mean():.4f}, std: {layer.beta.std():.4f}"
                
                print(f"Layer {i} (BatchNorm1d):")
                print(f"  Running mean - {running_mean_stats}")
                print(f"  Running var  - {running_var_stats}")
                print(f"  Gamma (gain) - {gamma_stats}")
                print(f"  Beta (bias)  - {beta_stats}")
                
                # Check for potential issues
                if layer.running_var.min() < 0:
                    print("⚠️  WARNING: Negative running variance detected!")
                if layer.running_var.max() > 100:
                    print(f"  ⚠️  WARNING: Very large running variance: {layer.running_var.max():.2f}")
                if layer.gamma.abs().max() > 10:
                    print(f"  ⚠️  WARNING: Very large gamma values: {layer.gamma.abs().max():.2f}")
        print("=" * 35)
    
    # Plot diagnostic graphs
    @torch.no_grad()
    def plot_diagnostic_graphs(self):
        # 1. distribution of tanh activations
        print("Tanh Activations distribution")
        print("=" * 35)
        self.plot_tanh_acts_distribution()
        print("\n", "-"*100)

        # 2. distribution of tanh grads
        print("Tanh Grads distribution")
        print("=" * 35)
        self.plot_tanh_gradient_distribution()
        print("\n", "-"*100)

        # 3. Weights grad distribution
        print("Weights grad distribution")
        print("=" * 35)
        self.plot_gradient_weights_distribution()
        print("\n", "-"*100)

        # 4. Update to data ratio
        print("Parms Update to data ratio")
        print("=" * 35)
        self.plot_update_to_data_ratio()
        print("\n", "-"*100)

        # 5. Plot the loss graph
        print("Loss across epochs")
        self.plot_loss_over_steps()
        print("\n", "-"*100)

  
    
    def plot_tanh_acts_distribution(self):
        """
        What it visualizes: This plot shows the distribution of the activations (outputs) of Tanh layers across a batch of data.
        ideal case: 
        1. Gaussian-like distribution: Ideally, you want the activations to be roughly centered around zero with a standard deviation that isn't too far from 1
        2. Low saturation: The percentage of saturated neurons should be low. High saturation means many neurons are outputting values close to -1 or +1, where the gradient of the Tanh function is almost zero. This can lead to vanishing gradients in earlier layers.
        """
        legends = []
        plt.figure(figsize=(20,5))
        for i,layer in enumerate(self.model.layers[:-1]): # exclude last layer
            if isinstance(layer, Tanh):
                t = layer.out # this is a layer tanh output matrix
                sat_pct = ((abs(t) > 0.97).float().mean() * 100).item()
                print("Layer %d tanh output shape: %s, mean: %.4f, std: %.4f, saturation pct: %.2f%%" % (i, str(t.shape), t.mean(), t.std(), sat_pct))
                hy, hx = torch.histogram(t, density=True)
                plt.plot(hx[:-1].detach(), hy.detach(), label=f'Layer {i} Tanh')
                legends.append(f'layer {i} ({layer.__class__.__name__}) ({str(t.shape)})')
        plt.legend(legends)
        plt.title('Tanh activation distribution')
        plt.xlabel('Tanh Activation Value')
        plt.ylabel('Density')
        plt.show()
    
    def plot_tanh_gradient_distribution(self):
        """
        Plots the distribution of gradients flowing through Tanh layers.
        This helps visualize if gradients are vanishing or exploding.
        Ideal case:
        1. Gradients should generally be centered around zero
        2. Gradients should have a reasonable standard deviation (not too close to 0 or too large)
        """
        legends = []
        plt.figure(figsize=(20,5))
        for i,layer in enumerate(self.model.layers[:-1]):
            if isinstance(layer, Tanh):
                if layer.out.grad is not None: 
                    t_grad = layer.out.grad
                    print("Layer %d (%s) grad_shape: %s mean %.4f std: %.4f" % (i, layer.__class__.__name__, str(t_grad.shape), t_grad.mean(), t_grad.std()))
                    hy, hx = torch.histogram(t_grad, density=True)
                    # print("HY sum ",hy.sum())
                    plt.plot(hx[:-1].detach(), hy.detach(), label=f'Layer {i} Tanh Grad')
                    legends.append(f'layer {i} ({layer.__class__.__name__} grad) ({str(layer.out.shape)})')
                else:
                    print(f"Layer {i}: No gradients available")
        plt.legend(legends)
        plt.title('Tanh gradient distribution')
        plt.xlabel('Tanh Gradient Value')
        plt.ylabel('Density')
        plt.show()

    
    def plot_gradient_weights_distribution(self):
        """
        Plots the dist of grads of the weights of the layers (linear). Prints the mean/std of the grds and gard/data ratio of each layer weight.
        ideal case:
        1. Mean of the grads should be close to zero
        2. grad/data ratio: tells how much the gradient is causing the weight of the parm to change (% change in weight w.r.t gradient) 
        Should be around 1e-3 or -3 (log10 scale)
            too low: like 1e-6 - Vanishing gradients or LR too small
            too high: like 1e-1 - Exploding gradients or LR too high
        """
        legends = []
        plt.figure(figsize=(20,5))
        for i,p in enumerate(self.parameters):
            # only plot 2D parsms
            if p.ndim == 2:
                data = p.data
                grad = p.grad
                shape = str(p.shape)
                print("Layer %i weight shape: %s grad mean: %.4f gard std: %.4f grad/data ratio: %e" % (i, shape, grad.mean(), grad.std(), grad.std()/data.std()))
                hy, hx = torch.histogram(grad, density=True)
                # print("HY sum ",hy.sum())
                plt.plot(hx[:-1].detach(), hy.detach(), label=f'Layer {i} Weight Grad')
                legends.append(f'layer {i} ({shape})')
        plt.legend(legends)
        plt.title('Weight gradient distribution')
        plt.xlabel('Weight Gradient Value')
        plt.ylabel('Density')
        plt.show()

    
    def plot_update_to_data_ratio(self):
        """
        This plots the ratio of lr*grad to the data (on log10 scale) for all the parms across the traning epochs.
        ideal case:
        1. Stable lines around -3: Ideally, each line (representing a parameter) should hover stably around -3 (10e-3 or 0.001). 
        it suggests that the typical magnitude of the update being applied to that parameter in each optimization step (after backpropagation) is about 0.1% of the typical magnitude of the parameter's current value.
        3. No extreme spikes or drops: Sudden large spikes can indicate exploding gradients, while drops to very low values can suggest vanishing gradients

        """
        legends = []
        plt.figure(figsize=(20,5))
        for i,p in enumerate(self.parameters):
            if p.ndim == 2: # only weights with 2d matrix (i.e no tanh and batch norm or biases) 
                parm_ud_ratios = []
                for j in range(len(self.parm_update_stats)):
                    parm_ud_ratios.append(self.parm_update_stats[j][i])
                plt.plot(parm_ud_ratios)
                legends.append("Layer %i parm %s" % (i, str(p.shape)))
        plt.plot([0, len(self.parm_update_stats)], [-3, -3], 'k--', label='Ideal (-3)') # these ratios should be ~1e-3, indicate on plot
        plt.legend(legends)
        plt.title('Update to Data Ratio Over Training')
        plt.xlabel('Training Step')
        plt.ylabel('log10(Update/Data Ratio)')
        plt.show()

    def plot_loss_over_steps(self, avg_of_n_steps=1000):
        loss = self.lossi
        if (avg_of_n_steps > 0 and len(self.lossi) >= 2000):
            loss_per_n_steps = torch.tensor(self.lossi).view(-1, avg_of_n_steps) # creates a 2D array with 1000 consequitive loss elements in each row
            print("loss_per_n_steps shape ", loss_per_n_steps.shape)
            loss = loss_per_n_steps.mean(1)
        print("Self loss shape ", len(self.lossi))    
        print("\n Loss shape  = ", loss.shape)
        print("\n len shape  = ", len(loss))
        print("\n first 30 shape  = ", len(loss[:30]))
        plt.figure(figsize=(20,5))
        plt.plot(loss)
        plt.title('Loss Over Training Steps')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.show()



    
    def sample_words(self, size=10, start_with='.'):
        self.set_layers_training(False)
        words = []
        for _ in range(size):
            word = []
            if start_with != '.':
                word.append(start_with)
            start_char_idx = self.c2i.get(start_with, 0)
            context = [0] * (self.block_size - 1) + [start_char_idx]
            while True:
                # get the embedding from C for '.'
                # emb = self.C[torch.tensor(context)] # this is (1, block_size, embedding_dim_size)
                # x = emb.view(1, -1)
                # x = torch.tensor(context).unsqueeze(0) # size (1, block_size)
                # for layer in self.model.layers:
                #     x = layer.forward(x)
                logits = self.model(torch.tensor(context).unsqueeze(0))

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
                    