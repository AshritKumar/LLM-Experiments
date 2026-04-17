"""
Visualize how 'Time' and 'Machine' embeddings change during training.
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
from sklearn.decomposition import PCA
from tokenization import SBTokenizer
from embeddings.train_embeddings import EmbeddingModel, WordDataSet

def get_data():
    """Download The Time Machine from Project Gutenberg"""
    print("Downloading The Time Machine from Gutenberg.org...")
    resp = requests.get("https://www.gutenberg.org/files/35/35-0.txt")
    return resp.text

def main():
    print("="*80)
    print("VISUALIZING 'TIME' AND 'MACHINE' EMBEDDING CHANGES")
    print("="*80)

    # Setup
    txt = get_data()
    print(f"\nText length: {len(txt):,} characters")

    # Train tokenizer
    print("\nTraining tokenizer...")
    tokzer = SBTokenizer()
    tokzer.train(txt, vocab_size=10000)
    vocab_size = len(tokzer.vocab)
    print(f"Vocabulary size: {vocab_size}")

    # Check tokenization
    print("\nChecking tokenization:")
    time_tokens = tokzer.encode(" Time")
    machine_tokens = tokzer.encode(" Machine")
    print(f"  ' Time' -> {time_tokens}")
    print(f"  ' Machine' -> {machine_tokens}")

    if not time_tokens or not machine_tokens:
        print("ERROR: Could not tokenize words!")
        return

    time_id = time_tokens[0]
    machine_id = machine_tokens[0]
    print(f"  Using token IDs: Time={time_id}, Machine={machine_id}")

    # Create dataset
    print("\nCreating dataset...")
    context_length = 8
    data_set = WordDataSet(tokzer, txt, context_length, stride=4)
    dl = DataLoader(data_set, batch_size=32, shuffle=True, drop_last=True)
    print(f"Dataset size: {len(data_set):,} examples")

    # Create model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    embedding_dim = 100
    model = EmbeddingModel(vocab_size, embedding_dim, context_length, device=device)

    # Get pre-training embeddings
    print("\nCapturing pre-training embeddings...")
    pre_train_embs = model.embedding.weight.detach().cpu().clone()

    time_emb_pre = pre_train_embs[time_id].numpy()
    machine_emb_pre = pre_train_embs[machine_id].numpy()

    print(f"  'Time' embedding shape: {time_emb_pre.shape}")
    print(f"  'Time' norm: {np.linalg.norm(time_emb_pre):.4f}")
    print(f"  'Machine' norm: {np.linalg.norm(machine_emb_pre):.4f}")

    # Train model
    print("\nTraining model for 25 epochs...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    loss_fn = torch.nn.NLLLoss().to(device)

    loss_per_epoch = model.traiModel(dl, num_epochs=25, loss_function=loss_fn, optimizer=optimizer)

    print(f"  Loss: {loss_per_epoch[0].item():.6f} -> {loss_per_epoch[-1].item():.6f}")

    # Get post-training embeddings
    print("\nCapturing post-training embeddings...")
    post_train_embs = model.embedding.weight.detach().cpu().clone()

    time_emb_post = post_train_embs[time_id].numpy()
    machine_emb_post = post_train_embs[machine_id].numpy()

    print(f"  'Time' norm: {np.linalg.norm(time_emb_post):.4f}")
    print(f"  'Machine' norm: {np.linalg.norm(machine_emb_post):.4f}")

    # Calculate changes
    time_change = time_emb_post - time_emb_pre
    machine_change = machine_emb_post - machine_emb_pre

    time_change_norm = np.linalg.norm(time_change)
    machine_change_norm = np.linalg.norm(machine_change)

    print(f"\nChanges:")
    print(f"  'Time' L2 change: {time_change_norm:.6f}")
    print(f"  'Machine' L2 change: {machine_change_norm:.6f}")

    # Calculate cosine similarity
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    cos_pre = cosine_sim(time_emb_pre, machine_emb_pre)
    cos_post = cosine_sim(time_emb_post, machine_emb_post)

    print(f"\nCosine Similarity:")
    print(f"  Before: {cos_pre:.6f}")
    print(f"  After:  {cos_post:.6f}")
    print(f"  Change: {cos_post - cos_pre:.6f}")

    # Create comprehensive visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    fig = plt.figure(figsize=(20, 16))

    # Plot 1: First 20 dimensions - Time embedding
    ax1 = plt.subplot(4, 3, 1)
    x = np.arange(20)
    ax1.bar(x - 0.2, time_emb_pre[:20], width=0.4, label='Pre-training', alpha=0.7, color='red')
    ax1.bar(x + 0.2, time_emb_post[:20], width=0.4, label='Post-training', alpha=0.7, color='green')
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel('Value')
    ax1.set_title("'Time' Embedding: First 20 Dimensions")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: First 20 dimensions - Machine embedding
    ax2 = plt.subplot(4, 3, 2)
    ax2.bar(x - 0.2, machine_emb_pre[:20], width=0.4, label='Pre-training', alpha=0.7, color='red')
    ax2.bar(x + 0.2, machine_emb_post[:20], width=0.4, label='Post-training', alpha=0.7, color='green')
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('Value')
    ax2.set_title("'Machine' Embedding: First 20 Dimensions")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Change magnitude per dimension
    ax3 = plt.subplot(4, 3, 3)
    ax3.plot(np.abs(time_change), label='Time', alpha=0.7, linewidth=2)
    ax3.plot(np.abs(machine_change), label='Machine', alpha=0.7, linewidth=2)
    ax3.set_xlabel('Dimension')
    ax3.set_ylabel('Absolute Change')
    ax3.set_title('Change Magnitude per Dimension')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: 2D projection (PCA) - showing movement
    ax4 = plt.subplot(4, 3, 4)

    # Combine all embeddings for PCA
    all_embs = np.vstack([
        time_emb_pre, time_emb_post,
        machine_emb_pre, machine_emb_post
    ])

    pca = PCA(n_components=2)
    all_embs_2d = pca.fit_transform(all_embs)

    time_pre_2d = all_embs_2d[0]
    time_post_2d = all_embs_2d[1]
    machine_pre_2d = all_embs_2d[2]
    machine_post_2d = all_embs_2d[3]

    # Plot points
    ax4.scatter(*time_pre_2d, s=200, c='red', marker='o', label='Time (pre)', zorder=3, edgecolors='black', linewidth=2)
    ax4.scatter(*time_post_2d, s=200, c='darkred', marker='s', label='Time (post)', zorder=3, edgecolors='black', linewidth=2)
    ax4.scatter(*machine_pre_2d, s=200, c='blue', marker='o', label='Machine (pre)', zorder=3, edgecolors='black', linewidth=2)
    ax4.scatter(*machine_post_2d, s=200, c='darkblue', marker='s', label='Machine (post)', zorder=3, edgecolors='black', linewidth=2)

    # Draw arrows showing movement
    ax4.arrow(*time_pre_2d, *(time_post_2d - time_pre_2d),
              head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.5, linewidth=2)
    ax4.arrow(*machine_pre_2d, *(machine_post_2d - machine_pre_2d),
              head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.5, linewidth=2)

    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax4.set_title('2D Projection (PCA) - Embedding Movement')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Embedding norms
    ax5 = plt.subplot(4, 3, 5)
    words = ['Time', 'Machine']
    norms_pre = [np.linalg.norm(time_emb_pre), np.linalg.norm(machine_emb_pre)]
    norms_post = [np.linalg.norm(time_emb_post), np.linalg.norm(machine_emb_post)]

    x_pos = np.arange(len(words))
    width = 0.35
    ax5.bar(x_pos - width/2, norms_pre, width, label='Pre-training', color='red', alpha=0.7)
    ax5.bar(x_pos + width/2, norms_post, width, label='Post-training', color='green', alpha=0.7)
    ax5.set_ylabel('L2 Norm')
    ax5.set_title('Embedding Norms')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(words)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # Plot 6: Cosine similarity comparison
    ax6 = plt.subplot(4, 3, 6)
    similarities = [cos_pre, cos_post]
    labels = ['Pre-training', 'Post-training']
    colors = ['red', 'green']
    ax6.bar(labels, similarities, color=colors, alpha=0.7)
    ax6.set_ylabel('Cosine Similarity')
    ax6.set_title("'Time' - 'Machine' Cosine Similarity")
    ax6.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax6.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (label, sim) in enumerate(zip(labels, similarities)):
        ax6.text(i, sim + 0.01, f'{sim:.4f}', ha='center', va='bottom', fontweight='bold')

    # Plot 7: Dimension-wise comparison for Time
    ax7 = plt.subplot(4, 3, 7)
    ax7.scatter(time_emb_pre, time_emb_post, alpha=0.5, s=20)

    # Add diagonal line (no change)
    min_val = min(time_emb_pre.min(), time_emb_post.min())
    max_val = max(time_emb_pre.max(), time_emb_post.max())
    ax7.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=2, label='No change')

    ax7.set_xlabel('Pre-training Value')
    ax7.set_ylabel('Post-training Value')
    ax7.set_title("'Time': Dimension-wise Before vs After")
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.axis('equal')

    # Plot 8: Dimension-wise comparison for Machine
    ax8 = plt.subplot(4, 3, 8)
    ax8.scatter(machine_emb_pre, machine_emb_post, alpha=0.5, s=20)

    min_val = min(machine_emb_pre.min(), machine_emb_post.min())
    max_val = max(machine_emb_pre.max(), machine_emb_post.max())
    ax8.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=2, label='No change')

    ax8.set_xlabel('Pre-training Value')
    ax8.set_ylabel('Post-training Value')
    ax8.set_title("'Machine': Dimension-wise Before vs After")
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.axis('equal')

    # Plot 9: Histogram of changes
    ax9 = plt.subplot(4, 3, 9)
    ax9.hist(time_change, bins=30, alpha=0.5, label='Time', color='red')
    ax9.hist(machine_change, bins=30, alpha=0.5, label='Machine', color='blue')
    ax9.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax9.set_xlabel('Change Value')
    ax9.set_ylabel('Frequency')
    ax9.set_title('Distribution of Changes')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # Plot 10: Dot product before/after
    ax10 = plt.subplot(4, 3, 10)
    dot_pre = np.dot(time_emb_pre, machine_emb_pre)
    dot_post = np.dot(time_emb_post, machine_emb_post)

    metrics = ['Dot Product', 'Cosine Sim', 'L2 Dist']
    pre_vals = [
        dot_pre,
        cos_pre,
        np.linalg.norm(time_emb_pre - machine_emb_pre)
    ]
    post_vals = [
        dot_post,
        cos_post,
        np.linalg.norm(time_emb_post - machine_emb_post)
    ]

    x_pos = np.arange(len(metrics))
    width = 0.35
    ax10.bar(x_pos - width/2, pre_vals, width, label='Pre', color='red', alpha=0.7)
    ax10.bar(x_pos + width/2, post_vals, width, label='Post', color='green', alpha=0.7)
    ax10.set_ylabel('Value')
    ax10.set_title('Similarity Metrics')
    ax10.set_xticks(x_pos)
    ax10.set_xticklabels(metrics, rotation=45)
    ax10.legend()
    ax10.grid(True, alpha=0.3, axis='y')

    # Plot 11: Top 10 dimensions by absolute change - Time
    ax11 = plt.subplot(4, 3, 11)
    top_dims_time = np.argsort(np.abs(time_change))[-10:][::-1]
    top_changes_time = time_change[top_dims_time]

    ax11.barh(range(10), top_changes_time)
    ax11.set_yticks(range(10))
    ax11.set_yticklabels([f'Dim {d}' for d in top_dims_time])
    ax11.set_xlabel('Change Value')
    ax11.set_title("'Time': Top 10 Dimensions by Change")
    ax11.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax11.grid(True, alpha=0.3, axis='x')

    # Plot 12: Top 10 dimensions by absolute change - Machine
    ax12 = plt.subplot(4, 3, 12)
    top_dims_machine = np.argsort(np.abs(machine_change))[-10:][::-1]
    top_changes_machine = machine_change[top_dims_machine]

    ax12.barh(range(10), top_changes_machine)
    ax12.set_yticks(range(10))
    ax12.set_yticklabels([f'Dim {d}' for d in top_dims_machine])
    ax12.set_xlabel('Change Value')
    ax12.set_title("'Machine': Top 10 Dimensions by Change")
    ax12.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax12.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    output_path = '/Users/ashritkuma.samudrala/lnex/ex_llm_rag/main/src/embeddings/time_machine_embedding_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")
    plt.show()

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n'Time' embedding:")
    print(f"  Pre-training norm: {np.linalg.norm(time_emb_pre):.4f}")
    print(f"  Post-training norm: {np.linalg.norm(time_emb_post):.4f}")
    print(f"  L2 change: {time_change_norm:.6f}")
    print(f"  Mean change: {np.mean(time_change):.6f}")
    print(f"  Std change: {np.std(time_change):.6f}")

    print(f"\n'Machine' embedding:")
    print(f"  Pre-training norm: {np.linalg.norm(machine_emb_pre):.4f}")
    print(f"  Post-training norm: {np.linalg.norm(machine_emb_post):.4f}")
    print(f"  L2 change: {machine_change_norm:.6f}")
    print(f"  Mean change: {np.mean(machine_change):.6f}")
    print(f"  Std change: {np.std(machine_change):.6f}")

    print(f"\nRelationship between embeddings:")
    print(f"  Cosine similarity before: {cos_pre:.6f}")
    print(f"  Cosine similarity after: {cos_post:.6f}")
    print(f"  Change in similarity: {cos_post - cos_pre:.6f}")
    print(f"  L2 distance before: {np.linalg.norm(time_emb_pre - machine_emb_pre):.4f}")
    print(f"  L2 distance after: {np.linalg.norm(time_emb_post - machine_emb_post):.4f}")

    print("="*80)

if __name__ == "__main__":
    main()
