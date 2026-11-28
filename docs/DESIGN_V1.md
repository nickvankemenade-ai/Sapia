# Sapia v1: redesign for a small compute budget

Status: **implemented** (`configs/sapia.yaml`, `sapia/muon.py`). On a tiny 7M-parameter test with the
real data mix, Muon reached val loss 6.34 vs AdamW's 6.46 at the same 1.6M tokens. Supersedes the shape
and training choices in
[ARCHITECTURE.md](ARCHITECTURE.md) (v0); the data mix in [DATA.md](DATA.md) is unchanged.

## The constraint

- At least **1B parameters**.
- No paid compute: the realistic budget is **~5 x 10^19 FLOPs actually used** (e.g. a month of a
  free TPU v5e-8 at ~40% utilisation), maybe up to ~8 x 10^19 (Google Cloud's $300 credit on spot A100s).
- That is ~6-7 tokens per parameter for a 1B model. Compute-optimal for 5 x 10^19 would be ~650M
  parameters on ~13B tokens, so every choice below aims to **get more model quality per FLOP**.

## The decisions

### 1. Muon optimizer (the biggest win): ~2x compute efficiency
"Muon is Scalable for LLM Training" (Moonshot AI, 2025) reports Muon reaching AdamW's loss with
**~52% of the training FLOPs** in compute-optimal scaling-law runs, and an independent study of
30M-200M decoders found 48-52%. For Sapia that turns ~5 x 10^19 FLOPs into roughly the quality AdamW
would need ~10^20 for, i.e. **~7B real tokens that train like ~14B**: close to compute-optimal for 1B.

Recipe (from the Moonlight paper, which made Muon work at scale):
- Muon (momentum 0.95, Nesterov, 5 Newton-Schulz steps in bf16) for the **2D weights inside the
  blocks**: q/k/v, attention output, gate/up/down.
- **Weight decay inside Muon** and update RMS matched to AdamW (scale `0.2 * sqrt(max(rows, cols))`),
  so Muon and AdamW share one learning rate and weight decay. Without these, Muon's advantage shrank
  at larger scale.
- **AdamW** for the embedding (= output head, tied), the norms and other 1D parameters, as in the
  reference Muon implementation and modded-nanogpt.
- Peak learning rate 5e-4 (QK-norm tolerates it; SmolLM2-1.7B used 5e-4), warmup 2% of tokens, WSD decay.

Risk: the 2x figure comes mostly from the authors' own baselines. The fallback is one config flag
(`optimizer: adamw`), and the first hour of training can compare both.

### 2. Shape: deeper and thinner, 28 layers x 1792 (1.045B parameters)
MobileLLM (Meta, ICML 2024) found **depth matters more than width** for small models (their 1B:
54 layers x 1280), and Hugging Face's Smol Training Playbook found extra layers beat spending the
same parameters elsewhere (e.g. untied embeddings). Very deep stacks run less efficiently on GPUs
and TPUs (many small matrix multiplies), so v1 takes the middle: 28 layers, the depth of Qwen3-0.6B/1.7B.

| | v0 | **v1** |
|---|---|---|
| Layers x width | 24 x 2048 | **28 x 1792** |
| Query / key-value heads | 16 / 4, head size 128 | **16 / 4, head size 112** |
| SwiGLU hidden | 5632 | **4864** (2.71x) |
| Parameters | 1,182,900,224 | **1,045,145,472** |
| Embedding share | 8.5% | 8.4% |
| FLOPs/token (1024 / 2048 context) | 7.70G / 8.31G | **6.89G / 7.50G** |
| Tokens from 5 x 10^19 FLOPs | ~6.4B | **~7.2B** |

Grouped-query attention stays at 4 query heads per key/value head: SmolLM3's ablations found ratios
of 2, 4 and 8 all match full multi-head attention.

### 3. Context: 1024 for the main phase, 2048 for the final phase
Most documents in these datasets are shorter than 2K tokens (the Smol Playbook puts it at 80-90%).
Training the first 90% at 1024 tokens saves ~8% of the FLOPs (more tokens for the same budget);
the final anneal phase, already a separate data phase, switches to 2048 so the model still handles
2048-token contexts. RoPE base 10,000 covers both.

### 4. Kept from v0, now backed by more evidence
- **QK-norm** (Qwen3 uses it at every size; OLMo 2, Gemma 3) and **z-loss**: stability, higher learning rate.
- **Tied embeddings** (MobileLLM, SmolLM2/3, Qwen3 small, Llama 3.2): saves ~90M parameters for
  depth instead.
- **No weight decay on the embedding** (new; OLMo 2 and SmolLM3 removed it for stability).
- Pre-norm RMSNorm, SwiGLU, no biases, no dropout, SmolLM2 tokenizer, WSD schedule with the anneal
  data phase (SmolLM3 also uses WSD).

### 5. Batch size: 0.5M tokens per step (was 1M)
With only ~7B tokens, 1M-token steps would leave ~7,000 optimizer steps. Halving it doubles the
number of updates, which helps at small token budgets; Muon is reported to cope well with large
batches, so 0.5M is a safe middle.

## Considered and not adopted

| Idea | Why not |
|---|---|
| Intra-document attention masking (Llama 3, SmolLM3) | Helps long-context stability, little effect on short-context quality; needs FlexAttention, which free TPUs lack |
| NoPE in every 4th layer (SmolLM3) | Aimed at 64K-128K contexts; Sapia stops at 2048 |
| Mixture of experts | More parameters per FLOP, but more complexity, memory and instability than a 1-month free run should risk |
| MobileLLM-style layer sharing | Adds FLOPs without adding parameters: the opposite of what this budget needs |
| Very deep (54 x 1280, MobileLLM-1B) | Same FLOPs per token, lower hardware utilisation; MobileLLM trained it on 1T tokens |
| modded-nanogpt speedrun tricks (value embeddings, U-net skips, ReLU^2, logit soft-capping) | Tuned for 124M-parameter, few-hour runs; unproven at 1B |

## Expected result

At 5 x 10^19 FLOPs: ~7.2B tokens, which with Muon should train roughly like ~14B AdamW tokens:
about the compute-optimal point for a 1B model. Expect a validation loss somewhat better than a
same-budget AdamW run of v0 and clearly better than modal-lm's 257M model; still well behind
Qwen3/Llama 3.2-class 1B models, which saw trillions of tokens.

## Sources

- Muon is Scalable for LLM Training (Moonshot AI): https://arxiv.org/abs/2502.16982
- Muon: Training and Trade-offs with Latent Attention and MoE: https://arxiv.org/abs/2509.24406
- Muon reference implementation: https://github.com/KellerJordan/Muon (fork: https://github.com/yxlllc/Muon), modded-nanogpt: https://github.com/KellerJordan/modded-nanogpt
- MobileLLM (Meta, ICML 2024): https://arxiv.org/abs/2402.14905 ; MobileLLM-1B config: https://huggingface.co/facebook/MobileLLM-1B/blame/main/config.json
- SmolLM3 blog: https://huggingface.co/blog/smollm3 ; Smol Training Playbook reading note: https://jianyuh.github.io/training/2025/11/29/Smol-Train.html
- SmolLM2 paper: https://arxiv.org/abs/2502.02737
- Qwen3-0.6B-Base model card: https://huggingface.co/Qwen/Qwen3-0.6B-Base
