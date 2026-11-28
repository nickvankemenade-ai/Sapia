# Sapia architecture (v0)

Status: **implemented, superseded by the [v1 redesign](DESIGN_V1.md)** (1.045B parameters,
Muon, deeper and thinner) for the free-compute budget. Kept for the reasoning behind the shared parts.

Sapia is a ~1.2B-parameter decoder-only language model, pretrained from scratch on a curated English mix (see [DATA.md](DATA.md))
on a single GH200 (H100 GPU, 96 GB) with a budget of roughly 100 GPU-hours.

It starts from the 257M `gpt250m` model in `modal-lm`. That design is sound, so most of it is kept.
The changes below are the ones worth making at 1.2B on this hardware.

## Starting point: modal-lm `gpt250m`

| | gpt250m |
|---|---|
| Shape | 16 layers x 1024 wide, 16 heads (head dim 64), SwiGLU 2816 |
| Attention | Multi-head attention (one K/V head per query head), RoPE (base 10k), SDPA causal |
| Norms | Pre-norm RMSNorm (fp32 statistics), final RMSNorm |
| Embeddings | Tied input/output, GPT-2 BPE padded to 50,304 |
| Other | No biases, no dropout, GPT-2 scaled init on residual-writing layers |
| Context | 1024 |
| Params | 257,065,984 (embedding 20%) |
| Measured | ~61K tok/s, 34% MFU on an A100-80GB |

Weak points at larger scale: the GPT-2 tokenizer, generation without a KV cache, no protection
against attention-logit or output-logit blow-ups, and a PyTorch 2.3 ceiling.

## Kept as-is

- Decoder-only transformer, pre-norm residual blocks
- RMSNorm with fp32 statistics, no biases anywhere, no dropout
- RoPE (half-split rotation), SwiGLU feed-forward
- Tied embedding / unembedding (standard at ~1B: Llama 3.2 1B, SmolLM2 tie theirs)
- GPT-2 scaled init (std 0.02, residual-writing projections 0.02 / sqrt(2L))
- bf16 autocast, fused attention through `scaled_dot_product_attention`

## Proposed changes

### 1. Tokenizer: GPT-2 BPE -> a modern ~49K tokenizer
GPT-2's tokenizer is from 2019 and has no special tokens. Sapia should use a tokenizer trained
on similar data that has chat tokens built in. The candidate is the **SmolLM2 tokenizer**
(49,152 tokens, Apache 2.0, trained on a FineWeb-Edu-heavy mix, includes `<|im_start|>` /
`<|im_end|>`). The alternative is training our own 32K-49K byte-level BPE on a FineWeb-Edu sample.

- Keeps the vocab under 65,536, so token shards stay `uint16` (as in modal-lm).
- 49,152 is a multiple of 64, so no padding is needed.
- Avoid 128K vocabularies (Llama 3): the embedding would be 263M params (22% of the model)
  and shards would need `uint32`.
- Chat roles exist from day one instead of being bolted onto unused padded ids.

### 2. Grouped-query attention (16 query heads, 4 K/V heads)
- Cuts the KV cache 4x and the attention parameters by ~37%, at a negligible quality cost.
  Llama 3, Mistral, Qwen and TinyLlama all use it.
- Head dim 128 (16 heads x 128 = 2048), which suits FlashAttention well on Hopper GPUs.
- Implementation: `scaled_dot_product_attention(..., enable_gqa=True)` (PyTorch >= 2.5),
  falling back to `repeat_interleave` on older versions.

### 3. QK-norm
RMSNorm over each head's q and k (after projection, before RoPE). This is the main fix
for attention-logit growth, the most common cause of loss spikes. It allows a somewhat higher
learning rate and costs 256 params per layer. OLMo 2 and Gemma 3 use it.

### 4. z-loss on the output logits
Add `1e-4 * logsumexp(logits)^2` to the loss (PaLM). It keeps the softmax normalizer near zero,
which matters with bf16 and tied embeddings, and costs nothing. Logged separately from the
cross-entropy so validation loss stays comparable.

### 5. Shape: 24 layers x 2048
Deeper than gpt250m (16 layers), in line with ~1B models that trained well (TinyLlama: 22 x 2048, SwiGLU 5632;
SmolLM2-1.7B: 24 x 2048).

### 6. Context: 2048, with sequences packed and separated by EOS
- RoPE base 10,000 is fine up to 2048-4096.
- Attention costs ~8% of the FLOPs per token at 2048 (vs ~4% at 1024).
- Alternative if throughput is tight: pretrain at 1024 and switch to 2048 for the last ~10%.

### 7. Inference: KV cache
`generate` re-runs the whole context for every token in modal-lm. Sapia gets a KV cache from
the start: 24 layers x 2 x 4 heads x 128 x 2048 positions x 2 bytes = **48 MiB per sequence**
at full context in bf16, thanks to GQA.

### 8. PyTorch: drop the 2.3 ceiling
The GH200 is ARM (aarch64), so use NVIDIA's NGC PyTorch container or a recent aarch64 CUDA
wheel. That brings `enable_gqa`, FlexAttention and a mature `torch.compile`. **Turn compile on**
(off in modal-lm): typically 1.2-1.4x faster.

## Proposed config

```yaml
model:
  vocab_size: 49152       # SmolLM2 tokenizer
  block_size: 2048
  n_layer: 24
  d_model: 2048
  n_head: 16              # head dim 128
  n_kv_head: 4            # GQA, 4 query heads per K/V head
  mlp_hidden: 5632        # SwiGLU, 2.75 x d_model
  rope_base: 10000.0
  qk_norm: true
  z_loss: 1.0e-4
  tie_embeddings: true
  norm_eps: 1.0e-6
  dropout: 0.0
```

### Parameter count

| Part | Params |
|---|---|
| Embedding (tied) | 100,663,296 (8.5%) |
| Attention per layer (q 4.19M, kv 2.10M, out 4.19M) | 10,485,760 |
| SwiGLU per layer | 34,603,008 |
| Norms per layer (2 RMSNorm + QK-norm) | 4,352 |
| **Per layer** | **45,093,120** |
| 24 layers | 1,082,234,880 |
| Final norm | 2,048 |
| **Total** | **1,182,900,224** (~1.18B) |

### Compute budget on a GH200
- FLOPs per token: ~8.3 GFLOPs at context 2048 (6N + attention), ~7.7 GFLOPs at 1024.
- Throughput estimate: ~40-50K tokens/s at 34-40% MFU (modal-lm measured 34% on an A100).
- 90 usable hours -> **~13-16B tokens**, about 11-14 tokens per parameter. That is somewhat
  below compute-optimal (~20 tokens per parameter), which the quality of the data partly offsets.
  Do not go below ~10B tokens.
- Measure real tokens/s in a 30-minute trial and set `max_tokens` from it, since the LR
  schedule is defined over `max_tokens`.

## Considered and not adopted (for v0)

| Idea | Why not now |
|---|---|
| Untied embeddings | +100M params for little gain at this size |
| 128K vocabulary | 22% of params in the embedding, `uint32` shards |
| Document masking (no attention across packed documents) | Small gain; needs FlexAttention or varlen kernels. Candidate for v1 |
| FP8 training (Transformer Engine) | Possible 1.3x speed-up on Hopper, but extra risk on a one-shot budget |
| Multi-token prediction, MoE, sliding-window attention | Complexity not justified at 1.2B / 15B tokens |
| Muon optimizer | Promising, but AdamW is the known-good baseline; decide in the training design |

## Decisions

1. **Language: English only.** The data mix is English only (see DATA.md). This favours the SmolLM2
   tokenizer, which was trained on English, FineWeb-Edu-heavy data; a tokenizer of our own would
   gain little.
2. **Tokenizer: SmolLM2 (49,152 tokens), off the shelf.** A tokenizer of our own would compress
   FineWeb-Edu only a few percent better and brings its own bugs; SmolLM2's chat template also
   matches the SmolTalk data used for chat fine-tuning.
3. **Context: 2048 throughout.** A 1024 -> 2048 schedule would save under 1B tokens (~6%) but
   adds a mid-run phase switch (batch re-probe, recompile) to a one-shot, unattended run.

## Notes for the training design

- Monitoring: Weights & Biases for metrics (readable from a phone, survives the instance),
  plus ntfy push notifications on crash, NaN, loss spikes, milestones and completion.
- Run training under `tmux` or a `systemd` service so it outlives the SSH session.
