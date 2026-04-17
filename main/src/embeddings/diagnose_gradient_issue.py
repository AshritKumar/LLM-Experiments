"""
Diagnostic to understand why rare tokens change more than frequent ones.
This investigates gradient flow and updates to the embedding layer.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tokenization import SBTokenizer
from embeddings.train_embeddings import EmbeddingModel, WordDataSet

def get_data():
    resp = requests.get("https://www.gutenberg.org/files/35/35-0.txt")
    return resp.text

def main():
    print("="*80)
    print("GRADIENT FLOW DIAGNOSTIC")
    print("="*80)

    # Setup
    txt = get_data()
    print(f"\nText length: {len(txt):,} characters")

    tokzer = SBTokenizer()
    tokzer.train(txt, vocab_size=10000)
    vocab_size = len(tokzer.vocab)
    print(f"Vocabulary size: {vocab_size}")

    context_length = 8
    data_set = WordDataSet(tokzer, txt, context_length, stride=4)
    dl = DataLoader(data_set, batch_size=32, shuffle=True, drop_last=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    embedding_dim = 100
    model = EmbeddingModel(vocab_size, embedding_dim, context_length, device=device)

    # Count token frequencies
    print("\nCounting token frequencies...")
    all_tokens = tokzer.encode(txt)
    token_counts = {}
    for tok_id in all_tokens:
        token_counts[tok_id] = token_counts.get(tok_id, 0) + 1

    print(f"Total tokens: {len(all_tokens):,}")
    print(f"Unique tokens: {len(token_counts)}")

    # Track embeddings before training
    pre_train_embs = model.embedding.weight.detach().cpu().clone()

    # Train for just 1 epoch to see what happens
    print("\nTraining for 5 epochs with gradient tracking...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    loss_fn = torch.nn.NLLLoss().to(device)

    # Track which tokens get gradients
    token_gradient_counts = {i: 0 for i in range(vocab_size)}
    token_gradient_magnitudes = {i: [] for i in range(vocab_size)}

    model.train()

    for epoch in range(5):
        print(f"\nEpoch {epoch + 1}/5")
        batch_count = 0

        for inp, tg in dl:
            inp, tg = inp.to(device), tg.to(device)

            # Forward pass
            log_probs = model.forward(inp)
            loss = loss_fn(log_probs, tg[:, -1])

            # Backward pass
            model.zero_grad()
            loss.backward()

            # Track which tokens appeared in this batch and got gradients
            unique_tokens_in_batch = torch.unique(inp).cpu().numpy()

            # Get embedding gradients
            if model.embedding.weight.grad is not None:
                emb_grads = model.embedding.weight.grad.detach().cpu()

                for tok_id in unique_tokens_in_batch:
                    if tok_id < len(token_gradient_counts):
                        token_gradient_counts[tok_id] += 1
                        grad_norm = torch.norm(emb_grads[tok_id]).item()
                        token_gradient_magnitudes[tok_id].append(grad_norm)

            # Update
            optimizer.step()

            batch_count += 1
            if batch_count % 100 == 0:
                print(f"  Processed {batch_count}/{len(dl)} batches")

    # Get post-training embeddings
    post_train_embs = model.embedding.weight.detach().cpu().clone()

    # Calculate changes
    embedding_changes = torch.norm(post_train_embs - pre_train_embs, dim=1).numpy()

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    # Compute average gradient magnitudes
    avg_grad_magnitudes = {}
    for tok_id in range(vocab_size):
        if len(token_gradient_magnitudes[tok_id]) > 0:
            avg_grad_magnitudes[tok_id] = np.mean(token_gradient_magnitudes[tok_id])
        else:
            avg_grad_magnitudes[tok_id] = 0

    # Categorize tokens by frequency
    freq_bins = {
        'very_rare (1-5)': [],
        'rare (6-20)': [],
        'medium (21-100)': [],
        'common (101-500)': [],
        'very_common (>500)': []
    }

    for tok_id in range(vocab_size):
        freq = token_counts.get(tok_id, 0)
        change = embedding_changes[tok_id]
        grad_count = token_gradient_counts[tok_id]
        avg_grad = avg_grad_magnitudes[tok_id]

        if freq >= 1 and freq <= 5:
            freq_bins['very_rare (1-5)'].append((tok_id, freq, change, grad_count, avg_grad))
        elif freq <= 20:
            freq_bins['rare (6-20)'].append((tok_id, freq, change, grad_count, avg_grad))
        elif freq <= 100:
            freq_bins['medium (21-100)'].append((tok_id, freq, change, grad_count, avg_grad))
        elif freq <= 500:
            freq_bins['common (101-500)'].append((tok_id, freq, change, grad_count, avg_grad))
        elif freq > 500:
            freq_bins['very_common (>500)'].append((tok_id, freq, change, grad_count, avg_grad))

    print("\nEmbedding changes by frequency category:")
    print("-" * 80)
    for category, tokens in freq_bins.items():
        if len(tokens) > 0:
            changes = [t[2] for t in tokens]
            freqs = [t[1] for t in tokens]
            grad_counts = [t[3] for t in tokens]
            avg_grads = [t[4] for t in tokens]

            print(f"\n{category}: {len(tokens)} tokens")
            print(f"  Avg frequency: {np.mean(freqs):.1f}")
            print(f"  Avg embedding change: {np.mean(changes):.6f} ± {np.std(changes):.6f}")
            print(f"  Avg gradient updates: {np.mean(grad_counts):.1f}")
            print(f"  Avg gradient magnitude: {np.mean(avg_grads):.6f}")

    # Check specific examples
    print("\n" + "-" * 80)
    print("Examples from each category:")
    print("-" * 80)

    for category, tokens in freq_bins.items():
        if len(tokens) > 0:
            print(f"\n{category}:")
            # Show 3 examples
            for tok_id, freq, change, grad_count, avg_grad in tokens[:3]:
                try:
                    token_text = tokzer.decode([tok_id])
                    token_display = repr(token_text)[1:-1][:30]
                    print(f"  Token {tok_id:5d} ('{token_display}'): freq={freq:3d}, change={change:.4f}, "
                          f"grad_updates={grad_count:3d}, avg_grad={avg_grad:.6f}")
                except:
                    print(f"  Token {tok_id:5d}: freq={freq:3d}, change={change:.4f}, "
                          f"grad_updates={grad_count:3d}, avg_grad={avg_grad:.6f}")

    # Visualization
    print("\n" + "="*80)
    print("Creating visualizations...")
    print("="*80)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Frequency vs Change
    ax = axes[0, 0]
    freqs_plot = [token_counts.get(i, 0) for i in range(vocab_size)]
    ax.scatter(freqs_plot, embedding_changes, alpha=0.3, s=1)
    ax.set_xlabel('Token Frequency')
    ax.set_ylabel('Embedding L2 Change')
    ax.set_title('Frequency vs Change (Linear)')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Plot 2: Gradient updates vs Change
    ax = axes[0, 1]
    grad_counts_plot = [token_gradient_counts[i] for i in range(vocab_size)]
    ax.scatter(grad_counts_plot, embedding_changes, alpha=0.3, s=1)
    ax.set_xlabel('Number of Gradient Updates')
    ax.set_ylabel('Embedding L2 Change')
    ax.set_title('Gradient Updates vs Change')
    ax.grid(True, alpha=0.3)

    # Plot 3: Average gradient magnitude vs Change
    ax = axes[0, 2]
    avg_grads_plot = [avg_grad_magnitudes[i] for i in range(vocab_size)]
    ax.scatter(avg_grads_plot, embedding_changes, alpha=0.3, s=1)
    ax.set_xlabel('Avg Gradient Magnitude')
    ax.set_ylabel('Embedding L2 Change')
    ax.set_title('Gradient Magnitude vs Change')
    ax.grid(True, alpha=0.3)

    # Plot 4: Change distribution by frequency bin
    ax = axes[1, 0]
    bin_labels = []
    bin_changes = []
    for category in ['very_rare (1-5)', 'rare (6-20)', 'medium (21-100)', 'common (101-500)', 'very_common (>500)']:
        if len(freq_bins[category]) > 0:
            bin_labels.append(category)
            bin_changes.append([t[2] for t in freq_bins[category]])

    ax.boxplot(bin_changes, labels=bin_labels)
    ax.set_ylabel('Embedding L2 Change')
    ax.set_title('Change Distribution by Frequency')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 5: Gradient updates distribution
    ax = axes[1, 1]
    bin_grads = []
    for category in ['very_rare (1-5)', 'rare (6-20)', 'medium (21-100)', 'common (101-500)', 'very_common (>500)']:
        if len(freq_bins[category]) > 0:
            bin_grads.append([t[3] for t in freq_bins[category]])

    ax.boxplot(bin_grads, labels=bin_labels)
    ax.set_ylabel('Number of Gradient Updates')
    ax.set_title('Gradient Updates by Frequency')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 6: Correlations
    ax = axes[1, 2]
    # Calculate correlations for tokens that appeared
    appeared_tokens = [i for i in range(vocab_size) if token_counts.get(i, 0) > 0]
    freqs_appeared = [token_counts[i] for i in appeared_tokens]
    changes_appeared = [embedding_changes[i] for i in appeared_tokens]
    grads_appeared = [token_gradient_counts[i] for i in appeared_tokens]

    corr_freq_change = np.corrcoef(freqs_appeared, changes_appeared)[0, 1]
    corr_grad_change = np.corrcoef(grads_appeared, changes_appeared)[0, 1]

    correlations = {
        'Freq vs\nChange': corr_freq_change,
        'Grad Updates\nvs Change': corr_grad_change,
    }

    ax.bar(correlations.keys(), correlations.values())
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_ylabel('Correlation Coefficient')
    ax.set_title('Correlations')
    ax.set_ylim([-1, 1])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = '/Users/ashritkuma.samudrala/lnex/ex_llm_rag/main/src/embeddings/gradient_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.show()

    # Final diagnosis
    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)

    print(f"\nCorrelations:")
    print(f"  Frequency vs Change: {corr_freq_change:.4f}")
    print(f"  Gradient Updates vs Change: {corr_grad_change:.4f}")

    if corr_grad_change > 0.5:
        print("\n✅ Gradient updates correlate with changes (expected)")
    else:
        print("\n⚠️  Gradient updates don't strongly correlate with changes")

    if corr_freq_change < 0:
        print("\n❌ ISSUE: Rare tokens change more than frequent ones!")
        print("   This suggests a problem with:")
        print("   1. Optimizer momentum accumulation (AdamW)- rare tokens don't build momentum")
        print("   2. Weight decay affecting frequent tokens more")
        print("   3. Learning rate schedule")
        print("\n   HYPOTHESIS: AdamW's adaptive learning rates and weight decay")
        print("   cause frequent tokens to have smaller effective learning rates")
        print("   because they accumulate more weight decay over time.")
    else:
        print("\n✅ Frequent tokens change as expected")

    print("="*80)

if __name__ == "__main__":
    main()
