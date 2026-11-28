"""Find the fastest ``--tpu-jax`` setup on this TPU: each candidate trains a few steps, timed.

    python -m sapia.bench_tpu --config configs/sapia.yaml --data_dir /tmp/sapia-mix

Each candidate (sharding strategy x activation remat) runs in its own process (only one process
can hold the TPU, and its memory is freed when the process ends), compiles, trains ``--steps``
steps without evals or checkpoints, and reports tokens/s and MFU, or that it ran out of memory.
The last line prints the flags of the fastest one. Takes ~3-4 minutes per candidate, mostly
compiling. Extra arguments are passed to every candidate (e.g. ``--tpu_tokens_per_chip 4096``).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from typing import Optional

DEFAULT_CANDIDATES = "dp:none,zero2:none,dp:dots,zero2:dots,fsdp:none"


def _child(argv: list[str]) -> int:
    from .config import parse_train_args
    from .train_jax import train_jax

    cfg = parse_train_args(argv)
    print("RESULT " + json.dumps(train_jax(cfg, _bench=True)), flush=True)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--child":
        return _child(argv[1:])
    p = argparse.ArgumentParser(prog="python -m sapia.bench_tpu", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", default=DEFAULT_CANDIDATES, help="comma-separated sharding:remat pairs")
    p.add_argument("--steps", type=int, default=10, help="timed optimizer steps per candidate")
    p.add_argument("--out_root", default="/tmp/sapia-bench")
    p.add_argument("--timeout_minutes", type=float, default=25.0)
    args, train_args = p.parse_known_args(argv)

    results = []
    for cand in [c.strip() for c in args.candidates.split(",") if c.strip()]:
        sharding, _, remat = cand.partition(":")
        remat = remat or "none"
        name = f"{sharding}-{remat}"
        cmd = [sys.executable, "-m", "sapia.bench_tpu", "--child", *train_args, "--backend", "tpu-jax",
               "--tpu_sharding", sharding, "--tpu_remat", remat, "--out_dir", f"{args.out_root}/{name}",
               "--resume", "never", "--max_steps", str(args.steps), "--log_interval", "2", "--ntfy_topic", ""]
        print(f"=== {name}: running {args.steps} steps ...", flush=True)
        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout_minutes * 60)
            out, err, code = r.stdout, r.stderr, r.returncode
        except subprocess.TimeoutExpired:
            out, err, code = "", "timeout", -1
        res = None
        for line in out.splitlines():
            if line.startswith("RESULT "):
                res = json.loads(line[7:])
        text = out + err
        if res and res.get("tok_per_sec"):
            mfu = res.get("mfu")
            print(f"    {res['tok_per_sec']:,.0f} tok/s" + (f" | MFU {100 * mfu:.1f}%" if mfu else "")
                  + f" | loss {res.get('loss', float('nan')):.3f} | {time.time() - t0:.0f}s", flush=True)
            results.append((res["tok_per_sec"], name, sharding, remat, mfu))
        elif "RESOURCE_EXHAUSTED" in text or "out of memory" in text.lower():
            print("    out of TPU memory", flush=True)
        else:
            print(f"    failed (exit {code}):\n" + "\n".join(text.strip().splitlines()[-15:]), flush=True)

    if not results:
        print("no candidate finished; see the errors above")
        return 1
    results.sort(reverse=True)
    tps, name, sharding, remat, mfu = results[0]
    hours = 16e9 / tps / 3600
    print(f"\nfastest: {name}: {tps:,.0f} tok/s" + (f", MFU {100 * mfu:.1f}%" if mfu else "")
          + f" -> 16B tokens in ~{hours:.0f} TPU hours")
    print(f"use: --tpu-jax --tpu_sharding {sharding} --tpu_remat {remat}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
