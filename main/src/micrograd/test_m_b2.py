from back_prop_engine import Value
from exp_visualzer import ExpViz

# x = Value(2+1, label='x')
# c = 4*x**3 -12*x**2 + 2
# print(c)
# c.backward()
# ExpViz.viz_expr(c)

# 4x^3 - 12x^2 + 2 => 14 at x=2
# derivative: 12x^2 - 24x => 48 - 48 = 0 at x=2

# inputs
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

# weights
w1 = Value(-3.0, label='w1')
# w1.data += -0.6
w2 = Value(1.0, label='w2')

# bias
b = Value(6.88, label='b')

n = x1*w1 + x2*w2 + b
o = n.tanh()

print(o)
o.backward()

ExpViz.viz_expr(o)

