from torch import Tensor

# inputs
x1 = Tensor([2.0]).double()
x1.requires_grad = True
x2 = Tensor([0.0]).double()
x2.requires_grad = True

# weights
w1 = Tensor([-3.0]).double()
w1.requires_grad = True
# w1.data += -0.6
w2 = Tensor([1.0]).double()
w2.requires_grad = True

# bias
b = Tensor([6.88]).double()
b.requires_grad = True

n = x1 * w1 + x2 * w2 + b
o = n.tanh()

print(o.item())
o.backward()

print(f"x1.grad: {x1.grad.item()}")
print(f"x2.grad: {x2.grad.item()}")
print(f"w1.grad: {w1.grad.item()}")
print(f"w2.grad: {w2.grad.item()}")
print(f"b.grad: {b.grad.item()}")