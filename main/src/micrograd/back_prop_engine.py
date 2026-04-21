import math

# This builds an inverted expression graph where root would be the final output and leaf nodes are scaler values
# This is a directed acyclic graph (DAG)
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __add__(self, other):
        # check if other is instance of value
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        # define a backward function which propagates the gradient from out to self and other.
        # NOTE: Here we are backpropagating, so the asumption is we would already have the gradient of out calculated.
        # We already went from (self,other) -> out in forward pass
        # Now we are going from out -> (self,other)


        def _backward():
            """ 
                computes the grad of self,other w.r.t to the global output
                grad of self,other w.r.t global o/p can be obtained by chain rule.

                for addition, applying the chain rule we just propgate the grad of out to self and other
                d(global)/d(self) = d(global)/d(out) * d(out)/d(self)
                d(global)/d(out) => this is present in out.grad
                d(out)/d(self) => for addition this is = 1
                so,  d(global)/d(self) = out.grad * 1

                We need to accumilate all the gradients for a multi variet case
            """
            self.grad += out.grad * 1
            other.grad += out.grad * 1
        
        out._backward = _backward
        return out
    
    def __radd__(self, other): # adds 2 + a
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self): # -self
        return self * -1
    
    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            
            """
                we have d(global)/d(out) as out.grad already, we need d(global)/d(self)
                out(gloabl) = some func
                out = self*other
                d(global)/d(self) = d(global)/d(out) * d(out)/d(self)
                d(global)/d(out) => this is present in out.grad
                d(out)/d(self) => other
                so,  d(global)/d(self) = out.grad * other.data

            """
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        
        out._backward = _backward

        return out
    
    def __pow__(self,other): #self^other
        assert isinstance(other, (int,float)), "2nd argument show be an int or float"
        out = Value(self.data**other, (self,), f"**{other}")
        def _backward():
            # d(x^a)/x = ax^a-1
            # we will not have other.grad here since other is not a value object its a scale int,float
            self.grad += out.grad * (other * self.data**(other-1))
        
        out._backward = _backward
        return out
    
    def __truediv__(self, other): # self / other
        other = other if isinstance(other, Value) else Value(other)
        return self * (other**-1)
    
    def __rtruediv__(self, other): # other/self
        other = other if isinstance(other, Value) else Value(other)
        return other * (self**-1)

    def exp(self):
        out = Value(math.exp(self.data), (self, ), 'exp')

        def _backward():
            # d(e^x)/x = e^x
            self.grad += out.grad * math.exp(self.data) # or out.data
        
        out._backward = _backward

        return out
    
    def _tanh(self):
        e2 = (self * 2).exp() # e^(2*self)
        t = (e2 - 1) / (e2 + 1)
        return t
    
    def tanh(self):
        #tanh(x) = e^2x - 1 / e^2x + 1
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            # d(tanh(x)/x) = 1 - tanh(x)^2
            self.grad += out.grad * (1 - t**2)
        
        out._backward = _backward

        return out
    

    def __repr__(self):
        return f"Value({self.label or '':} data={self.data}, grad={self.grad})"

    # does the topological sorting
    def _top_sort(self):
        visited = set()
        top_order = []
        def __dfs(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    __dfs(child)
                top_order.append(node)
        __dfs(self)
        return top_order
    
    def backward(self):
        # set the root grad to 1 as a base case
        self.grad = 1
        # get the topologicl ordering, this is orderd from leaf nodes (inputs/weights) to the final output node
        # [w2, x2, i2, w1, x1, i1, i, b, n, o]
        topo_order = self._top_sort()
        # we need to go in reverse topological order for backward pass
        # needed: [o, n, b, i, i1, x1, w1, i2, x2, w2]
        for node in reversed(topo_order):
            node._backward()