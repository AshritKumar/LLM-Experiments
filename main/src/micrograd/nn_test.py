from nn_engine import Neuron, Layer, MLP
from exp_visualzer import ExpViz
import torch

n = Neuron(2)
x = [1,2]
print(n(x))
ExpViz.viz_expr(n(x))

l = Layer(2, 3, "layer1") # layer with 3 neurons, each taking 2 inputs
i = [1,2]

o = l(i)
print(type(o[0]))
ExpViz.viz_expr(o[1])

i = [2, 3, 4]
m = MLP(3, [4, 4, 1]) # 3 inputs, 2 hidden layers of 4 neurons each, 1 output

o = m(i)
print(o)
ExpViz.viz_expr(o)

# m1 = MLP(27, [27])
# inp = torch.randn(5,27)
# # print(len(inp[0]))
# # print(inp.shape)

# out = []
# for i in inp:
#     r = m1(i)
#     # print(len(r))
#     out.append(r)

# print(len(out))
# print(len(out[0]))

# # print(len(m1.layers[0].neurons[0].w))
# # print(len(m1.layers[0].neurons[0].w))
# # print(len(out))
# # print(len(out[0]))
