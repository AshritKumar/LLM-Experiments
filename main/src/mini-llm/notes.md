### BatchNorm vs LayerNorm

---
  What they normalize over:

  * BatchNorm1d (your torchfied_char_mlp_with_bn.py):
    * cur_mean = input.mean(0, keepdim=True)   # mean across the BATCH dimension
    * Input (32, 100) → mean shape (1, 100)
    * "For each neuron, what's the average activation across all 32 examples?"

  * LayerNorm (your gpt_dev.ipynb):
    * cur_mean = input.mean(1, keepdim=True)   # mean across the FEATURE dimension
    * Input (32, 100) → mean shape (32, 1)
    * "For each example, what's the average activation across all 100 features?"

  BatchNorm normalizes across the batch (per neuron).
  LayerNorm normalizes across the features (per sample).

  ---
  Why BatchNorm doesn't work well for transformers:

  1. Variable sequence lengths. Your batches have shape (B, CL, ED). At generation time, you pass sequences of length 1, 2, 3... up to context_len. BatchNorm would compute statistics over the batch AND sequence
  dimensions — those statistics change with sequence length, making behavior inconsistent between training and inference.

  2. Small/single-sample inference. BatchNorm's running stats (your running_mean/running_var) are an approximation. With batch size=1 at generation time, the batch statistics are meaningless — you'd just be normalizing
   a single number. LayerNorm doesn't have this problem since it normalizes within each sample independently.

  3. No running stats needed. Look at your LayerNorm.__call__ — it doesn't have a running_mean/running_var buffer. It always computes stats from the current input directly, so it behaves identically in train and eval.
  Your BatchNorm1d needs those buffers precisely because at eval time the batch might be too small to get reliable statistics.

  ---
  Summary:
  
  | Feature | BatchNorm | LayerNorm |
| :--- | :--- | :--- |
| **Normalizes over** | Batch dim (across examples) | Feature dim (within each example) |
| **Needs running stats** | Yes | No |
| **Works at batch size=1** | Poor (Fails during training) | Fine |
| **Works with variable seq len** | Poor | Fine |
| **Good for** | CNNs, MLPs | Transformers, RNNs |
  



### Residual connections

* Another importent addition to this is having residual connection
* This means, we add the input to the output of the transformer block, this is useful because addition distributes the gradinets equally to the nodes when doing back prop
* In the initilization the transformer block won't contribute much to the output, but as we train the model, the transformer block will learn to contribute more to the output. 
* So during the initial phases, the gradient directly flows to the input not impeded by the transformer block
* This helps normalize the gradients

### Claude explanation
 ---                                                                                                                                                                                                                     
  1. Residual connections (x = x + self.m_head_attn(x))                                                                                                                                                                 
                                                                                                                                                                                                                          
  The key intuition: at initialization, the transformer block produces near-random outputs. Without residuals, the gradient has to flow through the attention/FFN layers to reach the embeddings — through a noisy, random
   path. With residuals:

  x = x + block(x)

  The +x creates a highway for gradients to flow directly back to the embedding layer, bypassing the block entirely. Early in training, block(x) is noise, so the model essentially ignores it and relies on x alone (like
   a bigram). As training progresses, the block gradually learns to add useful corrections on top of x. The model learns incrementally rather than all-at-once.

  ---
  2. self.proj = nn.Linear(emb_dim, emb_dim) (the W0 projection)

  Each head operates in its own emb_dim//n_heads = 8 dimensional subspace. After concatenating all 4 heads you get back (B, CL, 32), but these are just 4 independent subspaces stacked — they haven't been mixed.

  The projection layer lets the model blend information across heads. Without it, head 0's output can never influence how head 1's output is used. The projection is essentially: "now that all heads have gathered their
  information, combine it into a coherent representation." It also makes the residual connection mathematically clean — the output is properly scaled for x + out.

  ---
  3. Expand → contract in FFN (emb_dim → 4*emb_dim → emb_dim)

  The attention layer is good at routing information (which tokens talk to which). But it's essentially a weighted average — limited in what computations it can express.

  The FFN is where the model does per-token computation. The expansion to 4*emb_dim creates a wider intermediate space where the ReLU can activate different combinations of features. Think of it as: attention decides
  what to look at, FFN decides what to do with it.

  The 4x factor comes from the original "Attention is All You Need" paper and is empirically found to work well — wide enough to have expressive power, contracted back so the residual addition stays in the same emb_dim
   space.

  Without the expansion (your earlier FeedForwardNetwork using emb_dim → emb_dim), the FFN has very limited capacity — it's nearly just a linear transformation with a ReLU, which is why GPTModel2 trains slower and
  worse than GPTModel3.




### Droupouts

Droupout is generally applied in 3 places
 ---
  1. Attention weights dropout (AttnHead_N, applied to w after softmax)
  w = self.dropout(w)  # w shape: (B, CL, CL)
  Here the "neurons" are attention connections — each element of w[b, i, j] is "how much token i attends to token j". Dropping these means: "token i won't gather
  information from token j this step." This prevents over-reliance on specific token relationships.

  ---
  2. Projection dropout (MultiHeadAttn_N, applied after self.proj)
  out = self.dropout(out)  # out shape: (B, CL, ED=32)
  Here — ythese are embedding dimensions. For each token position, some of the 32 emb_dim values are zeroed out. This is the closest to the classic "dropout on
  neurons" idea in MLPs.

  ---
  3. FFN dropout (FeedForwardNetwork_N, last in the sequential)
  nn.Dropout(dropout)  # applied to output shape: (B, CL, emb_dim=32)
  Same as above — embedding dimensions for each token position. Some of the 32 features computed by the FFN are zeroed out.

  ---
  Summary:

  ┌───────────────────┬──────────────────────────────────┬────────────────────────────────┐
  │     Location      │         What gets zeroed         │           Intuition            │
  ├───────────────────┼──────────────────────────────────┼────────────────────────────────┤
  │ Attention weights │ Token-to-token connections       │ Break specific attention paths │
  ├───────────────────┼──────────────────────────────────┼────────────────────────────────┤
  │ After projection  │ Embedding dimensions (B, CL, 32) │ Yes — "embedding neurons"      │
  ├───────────────────┼──────────────────────────────────┼────────────────────────────────┤
  │ After FFN         │ Embedding dimensions (B, CL, 32) │ Yes — "embedding neurons"      │
  └───────────────────┴──────────────────────────────────┴────────────────────────────────┘


#### Note on attention droupout

  w[b, i, j] = how much token i attends to token j. So each row i contains all attention scores for token i looking at every past token.

  ---
  "Wouldn't zeroing out high-affinity connections hurt?"

  Yes — intentionally. That's the entire point. Dropout forces the model to not over-rely on any single attention path.

  Think of it this way: if token i always has a very strong connection to token j, the model learns to depend entirely on that one path. If you randomly zero it out
  during training, the model is forced to also learn backup paths — other tokens that carry similar or complementary information.

  A real language analogy: if you're predicting the next word after "New York", attention might heavily focus on "York". Dropout occasionally blocks that, forcing the
   model to also learn from "New" or surrounding context. At inference time (dropout off), all paths are active and the model uses all of them.

  ---
  One important detail: after masking and softmax, dropout is applied to the already-normalized weights. So when you zero out some w[b, i, j], those positions don't
  get re-normalized — the remaining weights no longer sum to 1 for that row. To compensate, dropout scales the remaining values up by 1/(1-p) automatically. So token
  i still gets a strong aggregate signal, just from fewer tokens.

  ---
  The tradeoff: yes, occasionally a genuinely important connection gets dropped. But over thousands of batches, the model learns robust paths that don't depend on any
   single connection surviving — which is exactly what makes the model generalize better.



### Weight Initialization

Code: `gpt_token_level/gpt_model.py` → `GPTModel2EX1.__init__` and `_init_weights`

---
#### Why not use PyTorch defaults?

PyTorch's default for `nn.Linear` is Kaiming uniform — designed for ReLU networks. Transformers use GELU and have a very different signal flow (residual stream, attention, weight tying). Kaiming uniform can produce activations that are too large or too small at init for this architecture.

The GPT-2 paper and nanoGPT use `N(0, 0.02)` — empirically tuned to keep activations in a reasonable range across a wide variety of transformer sizes.

---
#### Base init: N(0, 0.02) for all Linear and Embedding weights

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

- All Linear weights start small and symmetric around 0.
- Biases start at 0 — no bias toward any direction at init.
- Embedding weights at std=0.02 means token embeddings start as small random vectors, not large ones that would dominate the residual stream.

---
#### Scaled Residual Init

```python
for pn, p in self.named_parameters():
    if pn.endswith('w0.weight') or pn.endswith('proj.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_transformer_blocks))
```

`w0` is the output projection in `MultiHeadAttention`. `proj` is the down-projection in `FFN`. These are the **last linear layer** before each residual addition in `TransformerBlock`:

```
x = x + attention(...w0...)   ← w0 controls magnitude added to x
x = x + ffn(...proj...)       ← proj controls magnitude added to x
```

With 6 transformer blocks, there are `2 * 6 = 12` total residual additions into `x`.

**The variance problem:**

If each addition has std σ, and additions are roughly independent, the variance of `x` after all blocks is:

```
Var(x_final) ≈ Var(x_embed) + 12 * σ²
```

With σ=0.02 this is already manageable, but scaling it down further by `sqrt(12)` ensures `x` at init is dominated by the embedding signal, not transformer noise.

**Why this matters:**

At init, we want the transformer to behave close to an identity function — input flows through mostly unchanged. The model then gradually learns to "use" its capacity. If residual additions are too large at init:
- The embedding signal is drowned out by noise
- Softmax in attention saturates (one-hot) → gradients vanish for most positions
- First optimizer step overshoots

With the scaled init, early training is stable and the model incrementally learns each block's contribution.

**Why std=1 would be catastrophic:**

std=1 → residual additions ~50x larger than std=0.02. After 12 additions the residual stream is pure noise. Attention scores `q @ k.T` would have std ~sqrt(head_dim) ≈ 8, causing softmax to collapse to one-hot. No learning signal for 99% of attention connections.



### Autocast and bfloat16

Code: `gpt_token_level/train_gpt.py` → `TrainGPTModel.__init__`, `train_model`, `estimate_loss`

---
#### What autocast does

`torch.autocast` automatically casts specific ops to bfloat16 at runtime while keeping weights and sensitive computations in float32. You don't manually cast anything — PyTorch has a built-in policy for each op.

```python
with torch.autocast(device_type='mps', dtype=torch.bfloat16):
    logits, loss = model(x, y)
# weights are still float32 after this block
```

---
#### float32 vs bfloat16 — what's the difference?

```
float32:   1 sign | 8 exponent | 23 mantissa  → high precision, range ±3.4e38
bfloat16:  1 sign | 8 exponent |  7 mantissa  → lower precision, SAME range as float32
float16:   1 sign | 5 exponent | 10 mantissa  → lower precision, small range ±65504
```

bfloat16 keeps the same exponent as float32 so it never overflows. float16 can overflow during forward pass and needs a GradScaler to compensate. bfloat16 is simpler and preferred on modern hardware (M3, A100).

---
#### PyTorch autocast op list for our transformer

**Cast to bfloat16** — compute-bound matmuls, safe to lose a little precision:

| Op | Where in our model |
|---|---|
| `nn.Linear` (forward) | QKV projection, w0, FFN fc, FFN proj, lm_head |
| `torch.matmul` / `@` | `q @ k.T`, `attn @ v` |
| `nn.Embedding` (lookup) | wte, wpe |
| `nn.GELU` | FFN activation (element-wise, runs in whatever dtype input is) |

**Stay in float32** — reductions and numerically sensitive ops:

| Op | Where in our model | Why float32 |
|---|---|---|
| `nn.LayerNorm` | ln_attn, ln_ffn, ln_final | Sums across 384 values; bfloat16 rounding accumulates |
| `F.softmax` | Attention weights | Exp + sum over all positions; overflow/underflow risk |
| `F.cross_entropy` | Loss computation | Sums over 50257 vocab logits; precision critical |
| `torch.log`, `torch.exp` | Inside softmax/cross_entropy | Numerically unstable in low precision |

---
#### Flow through one transformer block under autocast

```
x (float32, residual stream)
  │
  ├─ LayerNorm       → float32  (sensitive reduction)
  ├─ QKV Linear      → bfloat16 (matmul)
  ├─ q @ k.T         → bfloat16 (matmul)
  ├─ softmax         → float32  (sensitive)
  ├─ attn @ v        → bfloat16 (matmul)
  ├─ w0 Linear       → bfloat16 (matmul)
  └─ + x             → float32  (residual add, promotes back)

  ├─ LayerNorm       → float32
  ├─ FFN fc Linear   → bfloat16
  ├─ GELU            → bfloat16 (element-wise, inherits dtype)
  ├─ FFN proj Linear → bfloat16
  └─ + x             → float32

lm_head Linear       → bfloat16
cross_entropy loss   → float32
```

The residual stream `x` stays float32 throughout because additions promote to the higher dtype. The speedup comes entirely from the matmuls (QKV, FFN, lm_head) which dominate runtime — those run in bfloat16.

---
#### Why not manually cast specific layers yourself?

You could do `model.transformer_blocks.to(torch.bfloat16)` but then you'd have to manually insert `.to(torch.float32)` casts at every boundary (before LayerNorm, before cross_entropy, etc.). Autocast handles all of this automatically and correctly. The only reason to do manual casting is for inference where you want the entire model in bfloat16 permanently to save memory.