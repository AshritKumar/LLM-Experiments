from back_prop_engine import Value
from exp_visualzer import ExpViz

# represents a single neuron computation node (similar to the torch version)
# This mirrors the manual computation from test_bp_torch.py
# Forward pass: o = tanh(x1*w1 + x2*w2 + b)

# inputs
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

# weights
w1 = Value(-3.0, label='w1')
# w1.data += -0.6
w2 = Value(1.0, label='w2')

# bias
b = Value(6.88, label='b')

#forward pass

i1 = x1 * w1
i1.label = 'i1'

i2 = x2 * w2
i2.label = 'i2'

i = i1 + i2
i.label = 'i'
# cell body
n = i + b
n.label = 'n'
# print(f"i1: {i1}, i2: {i2}, i: {i}, n: {n}")
o = n.tanh()
o.label = 'o'
# print(f"o: {o}")

ts = o._top_sort()
print("Topological sorted order\n", ts)

# now do the back prop
o.backward()

# trace
dot = ExpViz.viz_expr(o)

