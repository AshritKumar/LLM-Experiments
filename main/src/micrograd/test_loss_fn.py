from math import sqrt
import torch
def exp1():

    a = [
        [0, 1,0],
        [2, 3, 1],
        [0, 0,1]
    ]

    c = [
        [10, 20, 30, 35],
        [40, 50, 60, 45],
        [94, 88, 45, 66],
        [100, 98, 34, 11]
    ]

    a = torch.tensor(a)
    c = torch.tensor(c, dtype=torch.float32) # Ensure float for gradients

    c.requires_grad = True
    for _ in range(20):
        # `xx` is now part of the computation graph that tracks operations from `c`
        xx = c[a]

        # Let's define a simple scalar loss that depends on `xx`
        # For example, we can sum all elements of xx to create a scalar loss.
        loss = xx.sum()
        print(f"loss = {loss.item()}")

        # Now, call .backward() on the loss to compute gradients
        loss.backward()

        # Check the gradients for c
        print("Gradients of c after backward pass:")
        print(c.grad)

        # update c using gradient descent
        # with torch.no_grad():
        c.data -= 0.01 * c.grad
        c.grad.zero_()  # Reset gradients after update

def exp2():
    inp = torch.tensor([4,5], dtype=torch.float32)
    W = torch.randn((2,2)) # 2 neuron with 2 inputs
    target = torch.sqrt(torch.tensor([2.0, 4.0]))
    W.requires_grad = True
    for _ in range(3):
        # print(W)
        # print(inp)
        nout = inp @ W
        print("nout = ", nout)
        loss = (nout - target).pow(2).sum()  # make it scalar
        print("Loss = ",loss)
        
        W.grad = None
        loss.backward()
        # print("Grad = ",W.grad)
        W.data += -0.01 * W.grad

    print(W) # values should be close to [some values] since we want nout = 2
    expected = [sqrt(2.0), sqrt(4.0)]
    print(f"Expected values: nout = {expected}")
    # verify the final nout value for each output
    final_nout = inp @ W
    print(f"Final nout values: {final_nout}")
    print(f"Final mean nout value: {final_nout.mean().item():.4f}")
exp2()