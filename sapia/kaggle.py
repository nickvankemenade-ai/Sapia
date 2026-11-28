"""Kaggle helpers: one-command training sessions, joining the data, finding the last checkpoint.

    python -m sapia.kaggle train --hours 8                     # a whole training session (see below)
    python -m sapia.kaggle merge --out /tmp/sapia-mix          # joins every attached sapia-mix
    python -m sapia.kaggle latest                              # prints the newest checkpoint folder

``train`` is everything a session needs, so the notebook is the same two lines every time:
read the Kaggle secrets, install what the TPU image lacks, join the attached data, find the
newest checkpoint (this notebook's own previous output, fetched with the Kaggle API, or any
attached input), and train with Sapia's fixed settings until ``--hours`` are used. It refuses
to start from zero when it cannot tell whether earlier progress exists.

``merge`` looks for ``mix.json`` files under ``/kaggle/input`` (each prep notebook output
holds one) and writes one combined ``mix.json`` whose source entries point (absolute paths)
into the read-only inputs, so nothing is copied, and the merged
directory lives outside ``/kaggle/working`` so it never counts towards the 20 GB output.

``latest`` finds the checkpoint with the highest step among the attached inputs (the
previous session's output) for ``--resume``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .checkpoint import list_checkpoints
from .sources import PHASES

INPUT_ROOT = Path(os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input"))
WORK_ROOT = Path(os.environ.get("KAGGLE_WORKING_ROOT", "/kaggle/working"))
REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL = "nickvankemenade/sapia-train"      # this training notebook: its last output holds the checkpoint
SECRETS = ("SAPIA_NTFY_TOPIC", "HF_TOKEN", "WANDB_API_KEY", "KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN")
POINTER = "session.json"                     # written at the start of each session: where it resumed from
# Sapia's fixed training settings for every Kaggle session.
TRAIN_ARGS = ["--config", "configs/sapia.yaml", "--max_tokens", "16e9", "--keep_last", "1", "--save_best", "false",
              "--ckpt_minutes", "45"]


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
        if entries:                 # a phase with no prepared data is left out (training then skips it)
            merged["phases"][phase] = {"tokens": base["phases"].get(phase, {}).get("tokens", 0), "sources": entries}
        else:
            log(f"[merge] WARNING: no data for the {phase} phase yet; attach it before training reaches that phase")
    if "main" not in merged["phases"]:
        raise FileNotFoundError("no main-phase data in the inputs")
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


# ----------------------------------------------------------------------------- one-command session

def load_secrets(log=print) -> list[str]:
    """Copy the notebook's Kaggle secrets into the environment (those not ticked are skipped)."""
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return [n for n in SECRETS if os.environ.get(n)]
    client, found = UserSecretsClient(), []
    for name in SECRETS:
        if not os.environ.get(name):
            try:
                value = client.get_secret(name)
            except Exception:  # noqa: BLE001 - not attached to this notebook
                value = None
            if value:
                os.environ[name] = value
        if os.environ.get(name):
            found.append(name)
    log(f"[kaggle] secrets: {', '.join(found) if found else 'none'}")
    return found


def ensure_packages(backend: str, log=print) -> None:
    """Install what this image lacks. The TPU check runs in a child process so it frees the TPU again."""
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    if backend == "tpu-jax":
        probe = subprocess.run([sys.executable, "-c", "import jax; assert jax.devices()[0].platform == 'tpu'"],
                               capture_output=True)
        if probe.returncode != 0:
            log("[kaggle] installing jax[tpu] ...")
            subprocess.run(pip + ["-U", "jax[tpu]", "-f",
                                  "https://storage.googleapis.com/jax-releases/libtpu_releases.html"], check=True)
    elif backend == "tpu" and importlib.util.find_spec("torch_xla") is None:
        log("[kaggle] installing torch_xla ...")
        subprocess.run(pip + ["torch~=2.6.0", "torch_xla[tpu]~=2.6.0", "-f",
                              "https://storage.googleapis.com/libtpu-releases/index.html"], check=True)
    if importlib.util.find_spec("torch") is None:
        log("[kaggle] installing PyTorch (CPU, for checkpoints) ...")
        subprocess.run(pip + ["torch", "--index-url", "https://download.pytorch.org/whl/cpu"], check=True)
    for module, package in (("yaml", "pyyaml"), ("tokenizers", "tokenizers"), ("kaggle", "kaggle")):
        if importlib.util.find_spec(module) is None:
            subprocess.run(pip + [package], check=True)


def fetch_previous_output(kernel: str, dest: Path, log=print) -> Optional[bool]:
    """Download the latest output of ``kernel`` into ``dest`` with the Kaggle API.

    Returns None without API credentials, True when downloaded (possibly empty: no earlier
    session), and raises when the download fails for another reason.
    """
    if not (os.environ.get("KAGGLE_API_TOKEN") or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))):
        return None
    dest.mkdir(parents=True, exist_ok=True)
    exe = shutil.which("kaggle")
    cmd = ([exe] if exe else [sys.executable, "-m", "kaggle.cli"]) + ["kernels", "output", kernel, "-p", str(dest)]
    log(f"[kaggle] fetching the previous output of {kernel} ...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    text = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        if "404" in text or "not found" in text.lower():
            log("[kaggle] no earlier output of this notebook (first session)")
            return True
        raise RuntimeError(f"could not fetch the previous output of {kernel}: {text[-500:]}")
    return True


def find_resume_point(roots: list[Path]) -> tuple[Optional[Path], int]:
    """(checkpoints folder with the highest step among ``roots``, that step), or (None, 0)."""
    best: tuple[Optional[Path], int] = (None, 0)
    for root in roots:
        found = latest_checkpoint(root) if root.exists() else None
        if found is not None:
            step = list_checkpoints(found)[-1][0]
            if step > best[1]:
                best = (found, step)
    return best


def expected_step(roots: list[Path]) -> int:
    """The step the last session started from (its pointer file), even if it saved nothing."""
    step = 0
    for root in roots:
        if root.exists():
            for f in root.rglob(POINTER):
                try:
                    step = max(step, int(json.loads(f.read_text(encoding="utf-8")).get("resumed_step", 0)))
                except (OSError, ValueError):
                    pass
    return step


def run_session(hours: float, backend: str = "tpu-jax", kernel: str = KERNEL, fresh: bool = False,
                extra: Optional[list[str]] = None, log=print) -> int:
    load_secrets(log)
    ensure_packages(backend, log)
    data_dir = Path("/tmp/sapia-mix")
    merge_mixes(find_mixes(INPUT_ROOT), data_dir, log=log)
    out_dir = WORK_ROOT / "run"
    prev = Path("/tmp/sapia-previous")
    try:
        fetched = fetch_previous_output(kernel, prev, log)
    except RuntimeError as exc:
        log(f"[kaggle] ERROR: {exc}")
        log("[kaggle] Not starting: earlier progress may exist. Fix the Kaggle API secrets (or attach the previous "
            "version of this notebook as an input) and run again.")
        return 2
    roots = [out_dir, prev, INPUT_ROOT]
    ckpt_dir, step = find_resume_point(roots)
    wanted = expected_step(roots)
    if ckpt_dir is None or step < wanted:
        if wanted > 0:
            log(f"[kaggle] ERROR: the last session started from step {wanted:,}, but no checkpoint at or after it was "
                "found (that session probably failed before saving). Attach the newest version of this notebook that "
                "has run/checkpoints as an input and run again.")
            return 2
        if ckpt_dir is None and fetched is None and not fresh:
            log("[kaggle] ERROR: no checkpoint found, and without the KAGGLE_USERNAME/KAGGLE_KEY (or KAGGLE_API_TOKEN) "
                "secrets this notebook cannot look at its own earlier output. Add those secrets, or attach the previous "
                "version as an input. (First session ever without the API? Add --fresh.)")
            return 2
    if ckpt_dir is not None:
        log(f"[kaggle] resuming from {ckpt_dir} (step {step:,})")
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("metrics.jsonl", "train.log"):          # keep the whole run's history in each output
            src = ckpt_dir.parent / name
            if src.is_file() and not (out_dir / name).exists():
                shutil.copy2(src, out_dir / name)
    else:
        log("[kaggle] no earlier checkpoint: starting a new run at step 0")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / POINTER).write_text(json.dumps({"resumed_step": step, "resumed_from": str(ckpt_dir or "")}),
                                   encoding="utf-8")
    cmd = [sys.executable, "-m", "sapia.train", "--backend", backend, *TRAIN_ARGS, "--data_dir", str(data_dir),
           "--out_dir", str(out_dir), "--resume", str(ckpt_dir) if ckpt_dir else "auto", "--max_hours", str(hours),
           *(extra or [])]
    log("[kaggle] " + " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m sapia.kaggle")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="run one training session (everything included)")
    t.add_argument("--hours", type=float, default=8.0, help="training time of this session (default 8)")
    t.add_argument("--backend", default="tpu-jax", choices=("tpu-jax", "tpu"))
    t.add_argument("--kernel", default=KERNEL, help="this notebook as user/slug (its last output has the checkpoint)")
    t.add_argument("--fresh", action="store_true", help="allow a new run when no checkpoint can be looked up")
    m = sub.add_parser("merge", help="join the attached sapia-mix outputs into one data directory")
    m.add_argument("--out", default="/tmp/sapia-mix")
    m.add_argument("--root", default=str(INPUT_ROOT))
    lt = sub.add_parser("latest", help="print the newest checkpoints folder among the inputs ('' if none)")
    lt.add_argument("--root", default=str(INPUT_ROOT))
    args, extra = p.parse_known_args(argv)
    if args.cmd == "train":
        return run_session(args.hours, args.backend, args.kernel, args.fresh, extra)
    if extra:
        p.error(f"unrecognized arguments: {' '.join(extra)}")
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
