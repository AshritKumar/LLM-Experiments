from nn_engine import MLP
from exp_visualzer import ExpViz
m = MLP(3, [4, 4, 1])
inp = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
req_op = [1.0, -1.0, -1.0, 1.0] # ground truths
# here the expectation is when we pass  
# [2.0, 3.0, -1.0] , the model should predict 1
# [3.0, -1.0, 0.5]  the model should predict -1
#  [0.5, 1.0, 1.0]  the model should predict -1
# [0.5, 1.0, 1.0]  the model should predict 1


# Implement a training loop
# Steps to train this simple neural network

# 1. **Forward Pass**: Pass the input data through the network to get predictions
# 2. **Calculate Loss**: Compute the mean squared error between predictions and actual values
# 3. **Backward Pass**: Calculate gradients using backpropagation
# 4. **Update Weights**: Adjust weights using gradient descent with a learning rate
# 5. **Repeat**: Go back to step 1 for multiple iterations until convergence

learning_rate = 0.1
training_epochs = 1000

for _ in range(training_epochs):
    # Forward pass
    predicted_vals = [m(x) for x in inp]
    
    # Calculate loss
    loss = sum((p - t)**2 for p, t in zip(predicted_vals, req_op))
    loss.label = "loss"
    
    # Backward pass
    loss.backward()
    
    # Update weights
    for p in m.parameters():
        p.data = p.data - learning_rate * p.grad
    
    # Zero gradients for next iteration
    m.zero_grad()
    
    
    if _ % 100 == 0:  # Print every 100 epochs
        print(f"Epoch {_}: Loss = {loss.data}")
        print(f"Predictions: {[f'{p.data:.4f}' for p in predicted_vals]}")
        print("-" * 50)

ExpViz.viz_expr(loss)

print("Final loss: " + str(loss.data))
print("Final predictions: " + str([f"{p.data:.4f}" for p in predicted_vals]))

