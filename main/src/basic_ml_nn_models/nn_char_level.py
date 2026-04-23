import csv
import torch
import torch.nn.functional as F

# 1. set up, 
# load the words, form c2i and i2c mappings

names = []
with open("./ex_llm/main/resources/indian_names_m.csv", "r", newline='') as file:
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

# build character-level vocabulary and mappings (c2i, i2c)
all_chars = sorted(list(set(''.join(names))))
unique_chars = len(all_chars)
total_chars = unique_chars + 1 # extra spl char to denote start and end of the name

c2i = {c:i+1 for i,c in enumerate(all_chars)}
c2i['.'] = 0
i2c = {i:c for c,i in c2i.items()} # c2i.items() => returns char, index

########

#2. create the training set of bi-grams as x,y vectors
# X contains the first char of bigram, Y contains the following character. Example: .amey.=> .a, am, me, ey, y. => X = [0, 1, 13, 5, 25] Y = [1, 13, 5, 25, 0]
# X vector would be the input to the NN
# Y would represent the labels
training_epochs = 1
learning_rate = 2

X = [] # 1D array of all the chars of the names
Y = [] # 1D array of all the chars following X[i] 

for name in names:
    chars = ['.'] + list(name) + ['.']
    # form bi-grams
    for c1,c2 in zip(chars, chars[1:]):
        x = c2i[c1]
        y = c2i[c2]
        X.append(x)
        Y.append(y)

# create tensor objects
X = torch.tensor(X)
Y = torch.tensor(Y)
num_ex = X.nelement()
print("Number of bi-gram examples = ", num_ex)

# for a neuron we need inputs and weights, and it does Wi*Xi
# we will have total_chars's number of neurons.
# here the column size represents number of neurons (each column is the weight vector for one neuron)
# row size represents the weight vector size (equal to input vector size)
W = torch.randn((total_chars, total_chars), requires_grad=True) 

# training loop

for _ in range(training_epochs):

    ######################################### FORWARD PASS ###############################################

    # do one hot encoding for X. Gives a 2D array of size [len(X) x 27]
    Xen = F.one_hot(X, num_classes=total_chars).float() # we will consider Xenc as inputs to the NN

    # This gives an o/p of (Xen Rows x total_chars(28) matrix) => each row is the o/p of one neuron (with 28 cols). Each col of the row represents the log counts the next character, for a given input char.
    logits = Xen @ W # This is doing Xi*Wi for all neurons, giving us logits for each output character. This is equal to W[row][Xi]

    print("Log counts shape ",logits.shape)

    # now to get the counts we raise these values to e (to convert log counts back to counts)
    # this now will have all positive numbers representing counts. Each row represents the counts for each possible next character given the input character (row index)
    # this is equal to the N count matrix we created in bi_gram model
    counts = logits.exp() 

    # normalize the counts to get the probabilities
    # Each row will now have the probability of each character given the input character
    # Equivalent to the P matrix we created in bi_gram model
    # This operation is called soft max => e^logits / sum(e^logits) for each row
    softmax_probs = counts / counts.sum(dim=1, keepdim=True)  # each row element is divided by the sum of that row

    # But above are some random probs generated using random weights
    # We need to train the weights to make these probs match the actual training data
    # This is done by defining a loss function and using gradient descent to minimize the loss
    # We'll use cross-entropy loss and backpropagation to update the weights

    # for each training example, we need to compute the negative log likelihood
    # NLL = -log(P_true_label)
    # P_true_label is the probability of the true next character given the input character
    # We can get this from softmax_probs by indexing with the true label Y

    # nlls = torch.zeros(len(X))
    # for i in range(len(X)):
    #     true_label = Y[i].item()
    #     prob_true_label = softmax_probs[i, true_label] 
    #     nlls[i] = -torch.log(prob_true_label)
    #     # input char index
    #     x = X[i].item()
    #     y = Y[i].item()
    #     print(f"Input char index: {x}('{i2c[x]}'), True next char index: {y}('{i2c[y]}'), Prob of true label: {prob_true_label.item():.4f}, NLL: {nlls[i].item():.4f}")

    # print("Avg NLL (loss)", nlls.mean().item())

    # Instead of iterating like above we can pluck the prob's at Y'th index like below
    rowIdxs = torch.arange(len(X)) #[0,1,2,3,4,5... len(X)]
    probs_at_true_labels = softmax_probs[rowIdxs, Y] # this picks the Y'th index value in each row of softmax_probs. returns a 1D arry with Y index values of softmax_probs
    reg_factor = 0.1
    # add a regularization component (L2 regularization) => reg_factor * (W ** 2).mean()
    loss = -probs_at_true_labels.log().mean() + reg_factor * (W ** 2).mean()

    print(f"loss = {loss.item():.4f}")

    ######################################### BACKWARD PASS ###############################################
    W.grad = None
    loss.backward()

    # update the params.
    # In this case we only have 1 layer with 27 neurons, and the weights are represents in W. So we can just update W.data
    W.data += -learning_rate * W.grad

# Now generate a name
# Start with '.'
# Get the logits for '.'
# Sample from the probability distribution
# Repeat until we hit '.'
# Print the name
generated_names = []
for _ in range(10):
    # Get the index for '.'
    start_char_idx = c2i['.']

    generated_name = ''
    current_char_idx = start_char_idx = c2i['.']

    while True:  # 0 represents '.'
        # Get logits for the current character
        logits = W[current_char_idx] # This is equal to doing one_hot(current_char_idx, len(X)) @ W
        
    # Convert logits to probabilities using softmax
        probabilities = logits.exp() / logits.exp().sum()
        
    # Sample from the probability distribution
        # torch.multinomial expects probabilities to sum to 1, which they do
        next_char_idx = torch.multinomial(probabilities, num_samples=1).item()
        if next_char_idx == 0:
            break
        
        # Add to generated name if not '.'
        if next_char_idx != 0:
            generated_name += i2c[next_char_idx]
        
        current_char_idx = next_char_idx
    
    generated_names.append(generated_name)

print(f"Generated names: {generated_names}")






