"""
Comprehensive test of embedding training using the full Time Machine novel.
This follows the exact steps specified by the user.
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
    """Download The Time Machine from Project Gutenberg"""
    print("Downloading The Time Machine from Gutenberg.org...")
    resp = requests.get("https://www.gutenberg.org/files/35/35-0.txt")
    return resp.text

def main():
    print("="*80)
    print("THE TIME MACHINE - FULL NOVEL EMBEDDING TRAINING TEST")
    print("="*80)

    # 1. Get the data
    print("\n[STEP 1] Downloading The Time Machine novel...")
    txt = get_data()
    print(f"   Text length: {len(txt):,} characters")
    print(f"   First 200 chars: {txt[:200]}")

    # 2. Train tokenizer
    print("\n[STEP 2] Training tokenizer with vocab_size=10000...")
    tokzer = SBTokenizer()
    tokzer.train(txt, vocab_size=10000)
    vocab_size = len(tokzer.vocab)
    print(f"   Actual vocabulary size: {vocab_size}")

    # Check how 'time' and 'machine' are tokenized
    print("\n[STEP 2a] Checking tokenization of key words...")
    test_words = ["time", " time", "Time", " Time", "machine", " machine", "Machine", " Machine"]
    token_map = {}
    for word in test_words:
        tokens = tokzer.encode(word)
        decoded = tokzer.decode(tokens) if tokens else ""
        token_map[word] = tokens
        print(f"   '{word}' -> {tokens} -> '{decoded}'")

    # 3. Create dataset and dataloader
    print("\n[STEP 3] Creating dataset and dataloader...")
    context_length = 8
    data_set = WordDataSet(tokzer, txt, context_length, stride=4)
    print(f"   Dataset size: {len(data_set):,} examples")

    dl = DataLoader(data_set, batch_size=32, shuffle=True, drop_last=True)
    print(f"   DataLoader batches per epoch: {len(dl)}")

    # 4. Create model
    print("\n[STEP 4] Creating embedding model...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"   Using device: {device}")

    embedding_dim = 100
    model = EmbeddingModel(vocab_size, embedding_dim, context_length, device=device)

    total_params = sum([p.nelement() for p in model.parameters()])
    print(f"   Total parameters: {total_params:,}")

    # Capture pre-training embeddings
    print("\n[STEP 4a] Capturing pre-training embeddings...")
    pre_train_embs = model.embedding.weight.detach().cpu().clone()
    print(f"   Pre-training embedding shape: {pre_train_embs.shape}")
    print(f"   Pre-training stats: mean={pre_train_embs.mean():.4f}, std={pre_train_embs.std():.4f}")

    # 5. Train the model
    print("\n[STEP 5] Training model for 25 epochs...")
    print("   (This will take several minutes with the full novel...)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    loss_fn = torch.nn.NLLLoss().to(device)

    try:
        loss_per_epoch = model.traiModel(dl, num_epochs=25, loss_function=loss_fn, optimizer=optimizer)

        print("\n[STEP 5a] Training completed!")
        print(f"   Initial loss: {loss_per_epoch[0].item():.6f}")
        print(f"   Final loss: {loss_per_epoch[-1].item():.6f}")
        print(f"   Loss reduction: {(loss_per_epoch[0] - loss_per_epoch[-1]).item():.6f}")

        # Check if loss decreased
        if loss_per_epoch[0].item() <= loss_per_epoch[-1].item():
            print("   ⚠️  WARNING: Loss did not decrease!")
        else:
            print(f"   ✅ Loss decreased by {((loss_per_epoch[0] - loss_per_epoch[-1])/loss_per_epoch[0]*100).item():.1f}%")

    except Exception as e:
        print(f"\n   ❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return

    # Capture post-training embeddings
    print("\n[STEP 6] Capturing post-training embeddings...")
    post_train_embs = model.embedding.weight.detach().cpu().clone()
    print(f"   Post-training embedding shape: {post_train_embs.shape}")
    print(f"   Post-training stats: mean={post_train_embs.mean():.4f}, std={post_train_embs.std():.4f}")

    # COMPREHENSIVE ANALYSIS
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS")
    print("="*80)

    # 1. Overall embedding change
    print("\n[ANALYSIS 1] Overall Embedding Changes")
    print("-" * 80)
    total_change = (post_train_embs - pre_train_embs).abs().sum().item()
    print(f"Total absolute change: {total_change:.2f}")

    embedding_changes = torch.norm(post_train_embs - pre_train_embs, dim=1).numpy()
    print(f"Mean L2 change per token: {embedding_changes.mean():.6f}")
    print(f"Std L2 change: {embedding_changes.std():.6f}")
    print(f"Max L2 change: {embedding_changes.max():.6f}")
    print(f"Min L2 change: {embedding_changes.min():.6f}")

    if total_change < 1.0:
        print("⚠️  WARNING: Very small embedding changes!")
    else:
        print("✅ Embeddings changed significantly")

    # 2. Check specific words: time, machine, travel
    print("\n[ANALYSIS 2] Changes in Specific Word Tokens")
    print("-" * 80)

    # Find which tokens to check based on the tokenization
    words_to_check = [
        (" Time", "Time with space"),
        (" Machine", "Machine with space"),
        (" time", "time lowercase with space"),
        (" machine", "machine lowercase with space"),
        (" travel", "travel"),
        (" Traveller", "Traveller"),
    ]

    word_changes = []
    for word, description in words_to_check:
        tokens = tokzer.encode(word)
        if tokens and len(tokens) > 0:
            token_id = tokens[0]

            pre_emb = pre_train_embs[token_id]
            post_emb = post_train_embs[token_id]

            l2_change = torch.norm(post_emb - pre_emb).item()
            word_changes.append((word, token_id, l2_change))

            print(f"'{word}' (token {token_id}):")
            print(f"  Full tokenization: {tokens}")
            print(f"  L2 change: {l2_change:.6f}")

    # 3. Cosine similarity between word pairs
    print("\n[ANALYSIS 3] Cosine Similarity Between Word Pairs")
    print("-" * 80)

    # Check Time-Machine similarity
    time_tokens = tokzer.encode(" Time")
    machine_tokens = tokzer.encode(" Machine")

    if time_tokens and machine_tokens and len(time_tokens) > 0 and len(machine_tokens) > 0:
        time_id = time_tokens[0]
        machine_id = machine_tokens[0]

        time_pre = pre_train_embs[time_id].numpy()
        machine_pre = pre_train_embs[machine_id].numpy()
        time_post = post_train_embs[time_id].numpy()
        machine_post = post_train_embs[machine_id].numpy()

        cos_sim_pre = np.dot(time_pre, machine_pre) / (np.linalg.norm(time_pre) * np.linalg.norm(machine_pre))
        cos_sim_post = np.dot(time_post, machine_post) / (np.linalg.norm(time_post) * np.linalg.norm(machine_post))

        print(f"'Time' - 'Machine' cosine similarity:")
        print(f"  Before training: {cos_sim_pre:.6f}")
        print(f"  After training:  {cos_sim_post:.6f}")
        print(f"  Change: {cos_sim_post - cos_sim_pre:.6f}")

        if abs(cos_sim_post - cos_sim_pre) < 0.001:
            print("  ⚠️  Similarity barely changed!")
        else:
            print(f"  ✅ Similarity changed by {abs(cos_sim_post - cos_sim_pre):.6f}")

    # 4. Token frequency analysis
    print("\n[ANALYSIS 4] Token Frequency in Training Data")
    print("-" * 80)

    all_tokens = tokzer.encode(txt)
    token_counts = {}
    for tok_id in all_tokens:
        token_counts[tok_id] = token_counts.get(tok_id, 0) + 1

    print(f"Total tokens in text: {len(all_tokens):,}")
    print(f"Unique tokens used: {len(token_counts):,} / {vocab_size}")

    print("\nFrequency of checked words:")
    for word, token_id, l2_change in word_changes:
        count = token_counts.get(token_id, 0)
        print(f"  '{word}' (token {token_id}): {count:,} occurrences, L2 change: {l2_change:.6f}")

    # 5. Top tokens by change
    print("\n[ANALYSIS 5] Top 20 Tokens by Embedding Change")
    print("-" * 80)

    top_changed_indices = np.argsort(embedding_changes)[-20:][::-1]
    for rank, idx in enumerate(top_changed_indices, 1):
        change = embedding_changes[idx]
        count = token_counts.get(int(idx), 0)
        try:
            token_text = tokzer.decode([int(idx)])
            # Escape special characters for display
            token_display = repr(token_text)[1:-1]  # Remove outer quotes
            print(f"  {rank:2d}. Token {idx:5d} ('{token_display[:30]}'): L2={change:.6f}, freq={count:,}")
        except:
            print(f"  {rank:2d}. Token {idx:5d}: L2={change:.6f}, freq={count:,}")

    # 6. Correlation between frequency and change
    print("\n[ANALYSIS 6] Frequency vs Change Correlation")
    print("-" * 80)

    freq_list = []
    change_list = []
    for tok_id in range(len(embedding_changes)):
        freq = token_counts.get(tok_id, 0)
        if freq > 0:  # Only consider tokens that appear
            freq_list.append(freq)
            change_list.append(embedding_changes[tok_id])

    if len(freq_list) > 0:
        correlation = np.corrcoef(freq_list, change_list)[0, 1]
        print(f"Correlation coefficient: {correlation:.4f}")
        if correlation > 0.5:
            print("✅ Strong positive correlation: frequent tokens change more")
        elif correlation > 0:
            print("⚠️  Weak positive correlation: some relationship exists")
        else:
            print("❌ Negative/no correlation: unexpected pattern")

    # 7. Visualizations
    print("\n[ANALYSIS 7] Generating visualizations...")
    print("-" * 80)

    fig = plt.figure(figsize=(20, 12))

    # Plot 1: Loss curve
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(loss_per_epoch.cpu().numpy(), linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Embedding change distribution
    ax2 = plt.subplot(3, 3, 2)
    ax2.hist(embedding_changes, bins=100, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('L2 Norm of Change')
    ax2.set_ylabel('Number of Tokens')
    ax2.set_title('Distribution of Embedding Changes')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Pre vs Post embedding value distribution
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(pre_train_embs.flatten().numpy(), bins=100, alpha=0.5, label='Pre', color='red', density=True)
    ax3.hist(post_train_embs.flatten().numpy(), bins=100, alpha=0.5, label='Post', color='green', density=True)
    ax3.set_xlabel('Embedding Value')
    ax3.set_ylabel('Density')
    ax3.set_title('Embedding Value Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Frequency vs Change scatter
    ax4 = plt.subplot(3, 3, 4)
    ax4.scatter(freq_list, change_list, alpha=0.3, s=1)
    ax4.set_xlabel('Token Frequency (log scale)')
    ax4.set_ylabel('L2 Change')
    ax4.set_xscale('log')
    ax4.set_title(f'Frequency vs Change (corr={correlation:.3f})')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Top 10 token changes
    ax5 = plt.subplot(3, 3, 5)
    top_10_idx = top_changed_indices[:10]
    top_10_changes = [embedding_changes[i] for i in top_10_idx]
    top_10_labels = [f"T{i}" for i in top_10_idx]
    ax5.barh(range(10), top_10_changes)
    ax5.set_yticks(range(10))
    ax5.set_yticklabels(top_10_labels)
    ax5.set_xlabel('L2 Change')
    ax5.set_title('Top 10 Tokens by Change')
    ax5.grid(True, alpha=0.3, axis='x')

    # Plot 6: Embedding norm before/after
    ax6 = plt.subplot(3, 3, 6)
    norms_pre = torch.norm(pre_train_embs, dim=1).numpy()
    norms_post = torch.norm(post_train_embs, dim=1).numpy()
    ax6.scatter(norms_pre, norms_post, alpha=0.3, s=1)
    ax6.plot([norms_pre.min(), norms_pre.max()], [norms_pre.min(), norms_pre.max()], 'r--', alpha=0.5)
    ax6.set_xlabel('Norm Before Training')
    ax6.set_ylabel('Norm After Training')
    ax6.set_title('Embedding Norms: Before vs After')
    ax6.grid(True, alpha=0.3)

    # Plot 7: Change by token position
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(embedding_changes[:min(1000, len(embedding_changes))])
    ax7.set_xlabel('Token ID')
    ax7.set_ylabel('L2 Change')
    ax7.set_title('Change by Token ID (first 1000)')
    ax7.grid(True, alpha=0.3)

    # Plot 8: Cumulative distribution of changes
    ax8 = plt.subplot(3, 3, 8)
    sorted_changes = np.sort(embedding_changes)
    cumulative = np.arange(1, len(sorted_changes) + 1) / len(sorted_changes)
    ax8.plot(sorted_changes, cumulative)
    ax8.set_xlabel('L2 Change')
    ax8.set_ylabel('Cumulative Probability')
    ax8.set_title('Cumulative Distribution of Changes')
    ax8.grid(True, alpha=0.3)

    # Plot 9: Specific word comparisons
    ax9 = plt.subplot(3, 3, 9)
    if len(word_changes) > 0:
        words_plot, ids_plot, changes_plot = zip(*word_changes[:8])  # Top 8 words
        ax9.barh(range(len(changes_plot)), changes_plot)
        ax9.set_yticks(range(len(changes_plot)))
        ax9.set_yticklabels([f"{w[:15]}" for w in words_plot])
        ax9.set_xlabel('L2 Change')
        ax9.set_title('Key Word Embedding Changes')
        ax9.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    output_path = '/Users/ashritkuma.samudrala/lnex/ex_llm_rag/main/src/embeddings/full_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved to: {output_path}")
    plt.show()

    # FINAL SUMMARY
    print("\n" + "="*80)
    print("FINAL DIAGNOSIS")
    print("="*80)

    # Determine if there's an issue
    issues_found = []
    if total_change < 1.0:
        issues_found.append("Embeddings changed very little")
    if (loss_per_epoch[0] - loss_per_epoch[-1]).item() < 0.01:
        issues_found.append("Loss barely decreased")
    if abs(cos_sim_post - cos_sim_pre) < 0.001:
        issues_found.append("Word similarities unchanged")

    if issues_found:
        print("❌ ISSUES DETECTED:")
        for issue in issues_found:
            print(f"   - {issue}")
        print("\nPOSSIBLE CAUSES:")
        print("   1. Learning rate too small (try 0.01 instead of 0.001)")
        print("   2. Model architecture issue (check gradient flow)")
        print("   3. Tokenizer/dataset issue (verify data is correct)")
        print("   4. Device/dtype mismatch")
        print("   5. Embeddings not receiving gradients")
    else:
        print("✅ TRAINING APPEARS SUCCESSFUL!")
        print(f"   - Loss decreased from {loss_per_epoch[0].item():.4f} to {loss_per_epoch[-1].item():.4f}")
        print(f"   - Total embedding change: {total_change:.2f}")
        print(f"   - Mean token change: {embedding_changes.mean():.6f}")
        print(f"   - Time-Machine similarity changed by: {abs(cos_sim_post - cos_sim_pre):.6f}")
        print("\n   The embeddings ARE learning from the text!")

    print("="*80)

if __name__ == "__main__":
    main()
