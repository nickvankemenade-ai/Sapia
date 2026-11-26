"""Export weights only (bf16) from a training checkpoint: ~2.4 GB instead of ~19 GB.

    python -m sapia.export --ckpt runs/sapia --out final.pt          # newest checkpoint
    python -m sapia.export --ckpt runs/sapia --best --out best.pt    # lowest val loss

The result loads with ``python -m sapia.sample --ckpt final.pt`` and holds what is needed to
use or fine-tune the model, but not the optimizer state needed to continue pretraining.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

from .checkpoint import load_checkpoint, strip_compile_prefix
from .utils import replace_with_retry

KEEP = ("step", "tokens_seen", "model_config", "train_config", "tokenizer", "eot_token", "best_val_loss",
        "last_val_loss", "precision", "versions", "chat_format")


def export_weights(ckpt_path: str | Path, out_path: str | Path, dtype: torch.dtype = torch.bfloat16) -> Path:
    ckpt = load_checkpoint(Path(ckpt_path))
    weights = {k: (v.to(dtype) if v.is_floating_point() else v)
               for k, v in strip_compile_prefix(ckpt["model"]).items()}
    out = {k: ckpt.get(k) for k in KEEP if k in ckpt}
    out.update(model=weights, format="sapia-weights", weights_dtype=str(dtype).replace("torch.", ""))
    out_path = Path(out_path)
    tmp = out_path.with_name(out_path.name + ".tmp")
    torch.save(out, tmp)
    replace_with_retry(tmp, out_path)
    return out_path


def main(argv: Optional[list[str]] = None) -> int:
    from .sample import find_checkpoint

    p = argparse.ArgumentParser(prog="python -m sapia.export", description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", required=True, help="Checkpoint file, run directory or checkpoints directory.")
    p.add_argument("--best", action="store_true", help="Use best.pt instead of the newest checkpoint.")
    p.add_argument("--out", required=True, help="Output file, e.g. final.pt.")
    p.add_argument("--fp32", action="store_true", help="Keep float32 weights (twice the size).")
    args = p.parse_args(argv)
    src = find_checkpoint(args.ckpt, best=args.best)
    out = export_weights(src, args.out, torch.float32 if args.fp32 else torch.bfloat16)
    print(f"exported {src} -> {out} ({out.stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
