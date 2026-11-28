"""Kaggle helpers: join data prepared in several notebooks, and find the last checkpoint.

    python -m sapia.kaggle merge --out /tmp/sapia-mix          # joins every attached sapia-mix
    python -m sapia.kaggle latest                              # prints the newest checkpoint folder

``merge`` looks for ``mix.json`` files under ``/kaggle/input`` (each prep notebook output
holds one) and writes one combined ``mix.json`` whose source entries point (absolute paths)
into the read-only inputs, so nothing is copied, and the merged
directory lives outside ``/kaggle/working`` so it never counts towards the 20 GB output.

``latest`` finds the checkpoint with the highest step among the attached inputs (the
previous session's output) for ``--resume``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from .checkpoint import list_checkpoints
from .sources import PHASES

INPUT_ROOT = Path(os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input"))


def find_mixes(root: Path = INPUT_ROOT) -> list[Path]:
    """Every directory under ``root`` that contains a mix.json (depth-limited search)."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        depth = len(Path(dirpath).relative_to(root).parts)
        if "mix.json" in filenames:
            found.append(Path(dirpath))
            dirnames[:] = []
        elif depth >= 4:
            dirnames[:] = []
    return sorted(found)


def merge_mixes(mix_dirs: list[Path], out_dir: Path, log=print) -> dict:
    """Link each source's prepared folder from whichever input has it; write a combined mix.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mixes = [(d, json.loads((d / "mix.json").read_text(encoding="utf-8"))) for d in mix_dirs]
    if not mixes:
        raise FileNotFoundError("no mix.json found in the inputs; attach the prep notebooks' outputs")
    base = mixes[0][1]
    merged = {"version": base.get("version", 1), "total_tokens": base["total_tokens"],
              "anneal_fraction": base["anneal_fraction"], "tokenizer": base["tokenizer"], "phases": {}}
    missing = []
    for phase, specs in PHASES.items():
        entries = []
        for spec in specs:
            best = None
            for d, mix in mixes:
                for src in mix.get("phases", {}).get(phase, {}).get("sources", []):
                    if src["name"] == spec.name and (d / src["dir"] / "manifest.json").exists() and src["train_tokens"]:
                        if best is None or src["train_tokens"] > best[1]["train_tokens"]:
                            best = (d, src)
            if best is None:
                missing.append(f"{phase}/{spec.name}")
                continue
            d, src = best
            # An absolute "dir" points straight into the read-only input (Path(mix) / abs == abs).
            entries.append({**src, "dir": str((d / src["dir"]).resolve())})
            log(f"[merge] {phase}/{spec.name}: {src['train_tokens']:,} tokens from {d}")
        merged["phases"][phase] = {"tokens": base["phases"].get(phase, {}).get("tokens", 0), "sources": entries}
    (out_dir / "mix.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    if missing:
        log(f"[merge] WARNING: not prepared in any input: {', '.join(missing)} (their share goes to the others)")
    return merged


def latest_checkpoint(root: Path = INPUT_ROOT) -> Optional[Path]:
    """The checkpoints folder (anywhere under ``root``) holding the highest step."""
    best: Optional[tuple[int, Path]] = None
    for dirpath, dirnames, _ in os.walk(root, followlinks=True):
        if Path(dirpath).name == "checkpoints":
            ckpts = list_checkpoints(dirpath)
            if ckpts and (best is None or ckpts[-1][0] > best[0]):
                best = (ckpts[-1][0], Path(dirpath))
            dirnames[:] = []
        elif "mix.json" in os.listdir(dirpath):
            dirnames[:] = []                      # don't walk through gigabytes of shards
    return best[1] if best else None


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m sapia.kaggle")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge", help="join the attached sapia-mix outputs into one data directory")
    m.add_argument("--out", default="/tmp/sapia-mix")
    m.add_argument("--root", default=str(INPUT_ROOT))
    lt = sub.add_parser("latest", help="print the newest checkpoints folder among the inputs ('' if none)")
    lt.add_argument("--root", default=str(INPUT_ROOT))
    args = p.parse_args(argv)
    if args.cmd == "merge":
        dirs = find_mixes(Path(args.root))
        print(f"[merge] found {len(dirs)} prepared mix folder(s): {[str(d) for d in dirs]}")
        merge_mixes(dirs, Path(args.out))
        print(f"[merge] data ready in {args.out}")
    else:
        found = latest_checkpoint(Path(args.root))
        print(found or "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
