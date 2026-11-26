"""Sample text from a checkpoint.

    python -m sapia.sample --ckpt runs/tiny --prompt "The history of" --max_new_tokens 100

``--ckpt`` is a checkpoint file, a run directory or its ``checkpoints`` folder (the newest
checkpoint is used; add ``--best`` for ``best.pt``, the lowest-val-loss model). Only load checkpoints you created: they are regular pickled files.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

from .checkpoint import best_checkpoint_path, list_checkpoints, load_checkpoint, strip_compile_prefix
from .config import model_config_from_dict
from .model import GPT
from .precision import resolve_precision
from .tokenizer import get_tokenizer


def find_checkpoint(path: str, best: bool = False) -> Path:
    """Resolve a file, run dir or checkpoints dir to a checkpoint (newest, or best.pt if ``best``)."""
    p = Path(path)
    if p.is_file():
        return p
    if best:
        for d in (p / "checkpoints", p):
            if best_checkpoint_path(d).is_file():
                return best_checkpoint_path(d)
        raise FileNotFoundError(f"no best.pt found at {path} (it is written after the first evaluation)")
    for d in (p / "checkpoints", p):
        ckpts = list_checkpoints(d)
        if ckpts:
            return ckpts[-1][1]
    raise FileNotFoundError(f"no checkpoint found at {path}")


def load_model(ckpt_path: Path, device: torch.device) -> tuple[GPT, dict]:
    ckpt = load_checkpoint(ckpt_path)
    model = GPT(model_config_from_dict(ckpt["model_config"]))
    model.load_state_dict(strip_compile_prefix(ckpt["model"]))
    model.to(device).eval()
    meta = {k: ckpt.get(k) for k in ("step", "tokens_seen", "tokenizer", "eot_token", "last_val_loss", "precision",
                                     "chat_format")}
    return model, meta


def generate_text(model: GPT, tokenizer, prompt: str, *, max_new_tokens: int = 100, temperature: float = 0.8,
                  top_k: Optional[int] = 200, top_p: Optional[float] = None, seed: Optional[int] = None,
                  precision: str = "bf16", stop_at_eot: bool = True) -> str:
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt) if prompt else []
    if not ids:
        ids = [tokenizer.eot_token]  # unconditional sample: start from a document boundary
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    plan = resolve_precision(precision, device.type)
    with plan.autocast(device.type):
        out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
                             stop_token=tokenizer.eot_token if stop_at_eot else None, generator=generator)
    tokens = out[0].tolist()
    if stop_at_eot and len(tokens) > len(ids) and tokens[-1] == tokenizer.eot_token:
        tokens = tokens[:-1]
    return tokenizer.decode(tokens)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sapia.sample", description=__doc__.split("\n")[0])
    parser.add_argument("--ckpt", required=True, help="Checkpoint file, run directory or checkpoints directory.")
    parser.add_argument("--best", action="store_true",
                        help="With a run/checkpoints dir: use best.pt (lowest val loss) instead of the newest.")
    parser.add_argument("--prompt", default="", help="Prompt text (empty = unconditional).")
    parser.add_argument("--max_new_tokens", "--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 = greedy.")
    parser.add_argument("--top_k", "--top-k", type=int, default=200, help="0 disables top-k.")
    parser.add_argument("--top_p", "--top-p", type=float, default=None)
    parser.add_argument("--num_samples", "--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    parser.add_argument("--precision", default="bf16", help="bf16 (default) | fp16 | fp32; falls back automatically.")
    parser.add_argument("--tokenizer", default=None, help="Override the tokenizer recorded in the checkpoint.")
    args = parser.parse_args(argv)
    try:  # generated text can contain characters the Windows console code page cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" \
        else torch.device(args.device)
    ckpt_path = find_checkpoint(args.ckpt, best=args.best)
    model, meta = load_model(ckpt_path, device)
    tokenizer = get_tokenizer(args.tokenizer or meta.get("tokenizer") or "smollm2")
    print(f"checkpoint {ckpt_path} | step {meta['step']} | {model.num_params():,} params | "
          f"tokenizer {tokenizer.name} | device {device}", file=sys.stderr)
    for i in range(args.num_samples):
        seed = None if args.seed is None else args.seed + i
        text = generate_text(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_k=args.top_k or None, top_p=args.top_p,
                             seed=seed, precision=args.precision)
        if args.num_samples > 1:
            print(f"--- sample {i + 1} ---")
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
