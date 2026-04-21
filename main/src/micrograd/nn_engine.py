import random
from back_prop_engine import Value

class Module:
    def parameters(self):
        return []

    def zero_grad(self): # resets all the param in the NN gradients to 0 
        for param in self.parameters():
            param.grad = 0.0

class Neuron(Module):
    def __init__(self, nin, label=""): # nin => number of inputs to the neuron
        # create weights array/list for the inputs wrapped in value objects
        self.label = label
        self.w = [Value(random.uniform(-1,1), label=f"{self.label} w{i}") for i in range(nin)] # list of weights
        self.b = Value(random.uniform(-1,1), label=f"{self.label} b") # bias term
    
    def parameters(self):
        return self.w + [self.b]
    
    def __call__(self, x): # forward pass
        # raw_act = sum((w1 * x1 for w1,x1 in zip(self.w, x))) + self.b # can be writtten like below, starts with bias and then adds weighted inputs. equivalent to: self.b + w1*x1 + w2*x2 + ...
        raw_act = sum((w1 * x1 for w1,x1 in zip(self.w, x)), self.b) # dot product of inputs and weights, with bias added
        out = raw_act.tanh()
        return out


# define a Layer, is a set of neurons not connected to each other but connect to the prev and next layer neurons
class Layer(Module):
    # nin => number of inputs to each neuron
    # num_neurons => number of neurons in the layer, this equals number of outputs in the layer
    def __init__(self, nin, num_neurons, label=""):
        self.label = label
        self.neurons = [Neuron(nin, f"{self.label} n{i}") for i in range(num_neurons)]

    def parameters(self):
        return [parms for n in self.neurons for parms in n.parameters()]
    
    def __call__(self, x): # x would be the inputs to the layer
        outs = [n(x) for n in self.neurons] # compute outputs for each neuron in the layer
        return outs[0] if len(outs) == 1 else outs


# define a multi layer perceptron, is a NN with set of layers
class MLP(Module):
    # takes number of inputs to the network -> nin
    # list of neuron sizes in each layer -> [hidden1_size, hidden2_size, ..., output_size]
    def __init__(self, nin, layer_sizes):
        # append input layer at begining so that we create first layer with nin inputs
        all_sizes = [nin] + layer_sizes
        self.layers = [Layer(all_sizes[i], all_sizes[i+1], f"l{i}") for i in range(len(layer_sizes))]
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __call__(self, x): # returns the o/p of last layer (output of the entire network) as a Value object
        for layer in self.layers:
            x = layer(x)
        return x