# character level language model for names
# read the names csv file
import csv
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
print(os.getcwd())

names = []
with open("./main/resources/indian_names_m.csv", "r", newline='') as file:
    reader = csv.DictReader(file)
    # get all the names from 'name' header
    for row in reader:
        name = row['name']
        # Filter to English characters only (ASCII letters) and spaces
        # english_name = ''.join(c for c in name if c.isalpha() and ord(c) < 128)
        english_name = ''.join(c for c in name if (c.isalpha() or c == ' ') and ord(c) < 128)
        if len(english_name.strip()) > 3:  # Keep names with at least 3 English characters
            names.append(english_name.lower())  # Convert to lowercase for consiste

# names = ['ashrit', 'amey', 'vaishnavi', 'adithya', 'kumar', 'rama', 'prasanna'] 
# names = ['ashrit', 'amey']

print(f"Total English names loaded: {len(names)}")


# # form bi grams of names
# bigram_dict = {}
# for name in names:
#     chars = ['<S>'] + list(name) + ['<E>'] # wrap with spl start and end chars
#     for c1,c2 in zip(chars, chars[1:]): 
#         bigram = (c1, c2)
#         bigram_dict[bigram] = bigram_dict.get(bigram, 0) + 1

# # get the most frequent bi grams, sort by count
# sorted_bigrams = sorted(bigram_dict.items(), key= lambda big_: -big_[1])
# print("Top 50 most frequent bigrams:")
# for bigram, count in sorted_bigrams[:50]:
#     print(f"  {bigram}: {count}")

all_chars = sorted(list(set(''.join(names))))
unique_chars = len(all_chars)



# create a char to index mapping
c2i = {c:i+1 for i,c in enumerate(all_chars)}
c2i['.'] = 0
i2c = {i:c for c,i in c2i.items()} # c2i.items() => returns char, index

# print(c2i)
# print(i2c)

N = torch.zeros((len(c2i), len(c2i)), dtype=torch.int32) 


# instead of storing bi grams in a dict, we will store them in a 2D array representation for easy manipulation
# represents a unique_chars*unique_chars array, each row represents the count of nammes starting with rowth char and col char
# example 
# N[1][1] => count of names startig with aa,  N[1][1]=> count of names stating with ab, ... N[1][25] => count of names starting with az
# N[25][1] => count of names startig with za, ... N[25][25] => count of names stating with zz

for name in names:
    chars = ['.'] + list(name) + ['.'] # wrap with spl start and end chars
    # produces consecutive char pairs. EX: ram => (<S>,r), (r,a), (a,m), (m,<E>). chars[1:] would give all chars starting from index 1 (i.e. all chars except the first one)
    for c1,c2 in zip(chars, chars[1:]): 
        # get the c1,c2 index
        c1idx = c2i[c1]
        c2idx = c2i[c2]
        N[c1idx, c2idx] += 1 # this represents the number of times c1, c2 appeared together in the sequence c1,c2

# create a probability matrix by normalizing the counts
P = N.float()
# we add fake counts to N so that we avoid zero probabilities for characters that never appear after a given character
# add 1 to all counts to avoid zero probabilities (Laplace smoothing)
P = P + 1 # this adds 1 to each count to avoid zero probabilities
# divides each row with the sum of rows
p_row_sum = P.sum(dim=1, keepdim=True) # sums each row and stores in a column vector to enable broadcasting for division. Creates a 27x1 vector
# we would be able to divide P with p_row_sum, since they are 27x27 and 27x1 respectively (broadcasting enables element-wise division where each element in a row gets divided by that row's sum)
P = P / p_row_sum # normalize each row to sum to 1


generated_names = []
# Now sample from this probability matrix to pick one character at a time
for _ in range(20):
    generated_name = []
    idx = 0
    while(True):
        char_idx = torch.multinomial(P[idx], num_samples=1, replacement=True).item()

        if char_idx == 0:
            break

        generated_name.append(i2c[char_idx])
        idx = char_idx
    if len(generated_name) > 0:
        generated_names.append("".join(generated_name))

print(generated_names)

# calculate the probabilty of name given the probility matrix
all_names = ['a', 'ashrit'] + generated_names
name_probabilities = []
log_probabilities = []
negative_log_probs = []
normalized_nlls = []

for name in all_names:
    chars = ['.'] + list(name) + ['.'] 
    name_prob = 1
    log_prob = 0.0
    for c1,c2 in zip(chars, chars[1:]): 
        # get the c1,c2 index
        c1idx = c2i[c1]
        c2idx = c2i[c2]
        prob = P[c1idx, c2idx]
        # log probability
        log_prob += torch.log(prob).item()
        name_prob *= prob
    negative_log_prob = -log_prob
    normalize_nll = negative_log_prob / len(name)
    
    # Store values for histogram
    name_probabilities.append(name_prob.item() if hasattr(name_prob, 'item') else name_prob)
    log_probabilities.append(log_prob)
    negative_log_probs.append(negative_log_prob)
    normalized_nlls.append(normalize_nll)
    
    print(f"Probability of name '{name}': {name_prob}, Log Probability: {log_prob}, {negative_log_prob=}, {normalize_nll=}")

# Plot histograms
def plot_probability_histograms():
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. All transition probabilities in matrix P
    all_matrix_probs = P[P > 0].flatten().numpy()  # Only non-zero probabilities
    axes[0, 0].hist(all_matrix_probs, bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('Distribution of All Transition Probabilities')
    axes[0, 0].set_xlabel('Probability')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_yscale('log')  # Log scale for better visibility
    
    # 2. Name probabilities with labels
    bars = axes[0, 1].bar(range(len(name_probabilities)), name_probabilities, alpha=0.7, color='green')
    axes[0, 1].set_title('Name Probabilities with Labels')
    axes[0, 1].set_xlabel('Names')
    axes[0, 1].set_ylabel('Probability')
    axes[0, 1].set_yscale('log')
    
    # Add name labels on x-axis (rotated for readability)
    axes[0, 1].set_xticks(range(len(all_names)))
    axes[0, 1].set_xticklabels([name[:10] + ('...' if len(name) > 10 else '') for name in all_names], 
                               rotation=45, ha='right', fontsize=8)
    
    # Add probability values on top of bars
    for i, (bar, prob) in enumerate(zip(bars, name_probabilities)):
        if prob > 0:  # Only show non-zero probabilities
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                           f'{prob:.2e}', ha='center', va='bottom', fontsize=6, rotation=90)
    
    # 3. Log probabilities with labels
    bars3 = axes[0, 2].bar(range(len(log_probabilities)), log_probabilities, alpha=0.7, color='red')
    axes[0, 2].set_title('Log Probabilities with Labels')
    axes[0, 2].set_xlabel('Names')
    axes[0, 2].set_ylabel('Log Probability')
    axes[0, 2].set_xticks(range(len(all_names)))
    axes[0, 2].set_xticklabels([name[:8] + ('...' if len(name) > 8 else '') for name in all_names], 
                               rotation=45, ha='right', fontsize=7)
    # Add values on top of bars
    for i, (bar, prob) in enumerate(zip(bars3, log_probabilities)):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       f'{prob:.1f}', ha='center', va='bottom', fontsize=5, rotation=90)
    
    # 4. Negative log probabilities with labels
    bars4 = axes[1, 0].bar(range(len(negative_log_probs)), negative_log_probs, alpha=0.7, color='orange')
    axes[1, 0].set_title('Negative Log Probabilities with Labels')
    axes[1, 0].set_xlabel('Names')
    axes[1, 0].set_ylabel('Negative Log Probability')
    axes[1, 0].set_xticks(range(len(all_names)))
    axes[1, 0].set_xticklabels([name[:8] + ('...' if len(name) > 8 else '') for name in all_names], 
                               rotation=45, ha='right', fontsize=7)
    # Add values on top of bars
    for i, (bar, prob) in enumerate(zip(bars4, negative_log_probs)):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                       f'{prob:.1f}', ha='center', va='bottom', fontsize=5, rotation=90)
    
    # 5. Normalized NLL with labels
    bars5 = axes[1, 1].bar(range(len(normalized_nlls)), normalized_nlls, alpha=0.7, color='purple')
    axes[1, 1].set_title('Normalized NLL with Labels')
    axes[1, 1].set_xlabel('Names')
    axes[1, 1].set_ylabel('Normalized NLL')
    axes[1, 1].set_xticks(range(len(all_names)))
    axes[1, 1].set_xticklabels([name[:8] + ('...' if len(name) > 8 else '') for name in all_names], 
                               rotation=45, ha='right', fontsize=7)
    # Add values on top of bars
    for i, (bar, nll) in enumerate(zip(bars5, normalized_nlls)):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                       f'{nll:.2f}', ha='center', va='bottom', fontsize=5, rotation=90)
    
    # 6. Log10 name probabilities with labels
    log_name_probs = [np.log10(p) for p in name_probabilities if p > 0]
    log_names = [name for i, name in enumerate(all_names) if name_probabilities[i] > 0]
    bars6 = axes[1, 2].bar(range(len(log_name_probs)), log_name_probs, alpha=0.7, color='brown')
    axes[1, 2].set_title('Log10(Name Probabilities) with Labels')
    axes[1, 2].set_xlabel('Names')
    axes[1, 2].set_ylabel('Log10(Name Probability)')
    axes[1, 2].set_xticks(range(len(log_names)))
    axes[1, 2].set_xticklabels([name[:8] + ('...' if len(name) > 8 else '') for name in log_names], 
                               rotation=45, ha='right', fontsize=7)
    # Add values on top of bars
    for i, (bar, log_prob) in enumerate(zip(bars6, log_name_probs)):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                       f'{log_prob:.1f}', ha='center', va='bottom', fontsize=5, rotation=90)
    
    plt.tight_layout()
    plt.show()

# Call the histogram plotting function
plot_probability_histograms()


def show_vizualization(N, P, show_probablity_viz=False, show_viz=True):
    if show_viz:
        bigram_viz = None
        prob_viz = None
        # visualize the bigram matrix and probablity matrix
        if show_probablity_viz:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            bigram_viz = ax1
            prob_viz = ax2
            ax1.imshow(N, cmap='Blues')
            ax1.set_title('Bigram Counts')
            ax2.imshow(P, cmap='Blues')
            ax2.set_title('Bigram Probabilities')
            plt.tight_layout()
        else:
            bigram_viz = plt
            plt.figure(figsize=(15,15))
            plt.imshow(N, cmap='Blues')

        # add character labels
        for i in range(len(c2i)):
            for j in range(len(c2i)):
                chstr = i2c[i]+i2c[j]
                bigram_viz.text(j, i, chstr,  ha='center', va='bottom', color='black')
                bigram_viz.text(j, i, str(N[i, j].item()), ha='center', va='top', fontsize=6, color='black')
                if show_probablity_viz:
                    prob_viz.text(j, i, chstr, ha='center', va='bottom', color='black', fontsize=8)
                    prob_value = P[i, j].item()
                    prob_viz.text(j, i, f'{prob_value:.2f}', ha='center', va='top', fontsize=6, color='black')
                    #prob_viz.text(j, i, f'{P[i, j].item():.2f}', ha='center', va='top', fontsize=6, color='white')


        bigram_viz.axis('off')
        bigram_viz.set_title('Bigram Character Frequency Matrix')
        
        if show_probablity_viz:
            prob_viz.axis('off')
            prob_viz.set_title('Bigram Character Probabilities Matrix')


    plt.show()

show_vizualization(N, P, show_probablity_viz=True, show_viz=True)