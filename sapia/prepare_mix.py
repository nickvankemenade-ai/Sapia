"""Prepare the whole Sapia pretraining mix (docs/DATA.md), one source at a time.

    python -m sapia.prepare_mix --total_tokens 15e9                 # the real run (~15B tokens)
    python -m sapia.prepare_mix --preset tiny                       # ~2M tokens, one file per source
    python -m sapia.prepare_mix --total_tokens 15e9 --only reddit   # (re)run selected sources

Layout::

    data/sapia-mix/
      mix.json                      # phases, shares, token targets and what was prepared
      main/<source>/                # one prepared dataset per source (manifest.json + shards)
      anneal/<source>/

Each source directory is an ordinary ``sapia.prepare_data`` output, so every source resumes on
its own: rerun the same command after an interruption. The train loader reads ``mix.json``
and samples each source in proportion to its share (see ``sapia.data.MixtureLoader``).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .config import ConfigError, _parse_int
from .data import load_manifest
from .prepare_data import PrepConfig, PrepError, run_prepare
from .sources import PHASES, SourceSpec
from .utils import atomic_write_json, fmt_count, fmt_duration

MIX_FILE = "mix.json"
MIX_VERSION = 1


@dataclass
class MixConfig:
    out_dir: str = "data/sapia-mix"
    total_tokens: int = 15_000_000_000
    anneal_fraction: float = 0.10
    phases: str = "main,anneal"
    only: Optional[str] = None            # comma-separated source names
    tokenizer: str = "smollm2"
    val_every: int = 500
    max_files: Optional[int] = None       # per source (tiny runs)
    num_workers: Optional[int] = None
    shard_tokens: Optional[int] = None

    def phase_tokens(self, phase: str) -> int:
        frac = self.anneal_fraction if phase == "anneal" else 1.0 - self.anneal_fraction
        return int(round(self.total_tokens * frac))

    def validate(self) -> None:
        if self.total_tokens <= 0:
            raise ConfigError("total_tokens must be positive")
        if not 0.0 <= self.anneal_fraction < 1.0:
            raise ConfigError("anneal_fraction must be in [0, 1)")
        unknown = set(self.selected_phases()) - set(PHASES)
        if unknown:
            raise ConfigError(f"unknown phase(s) {sorted(unknown)}; available: {sorted(PHASES)}")

    def selected_phases(self) -> list[str]:
        return [p.strip() for p in self.phases.split(",") if p.strip()]

    def selected_sources(self) -> Optional[set[str]]:
        return {s.strip() for s in self.only.split(",") if s.strip()} if self.only else None


MIX_PRESETS = {
    "full": dict(),
    "tiny": dict(out_dir="data/sapia-mix-tiny", total_tokens=2_000_000, val_every=20, max_files=1,
                 shard_tokens=200_000),
}


def source_dir(out_dir: str | Path, phase: str, name: str) -> Path:
    return Path(out_dir) / phase / name


def prep_config_for(spec: SourceSpec, cfg: MixConfig, phase: str, target: int) -> PrepConfig:
    """The PrepConfig for one source. A spec with an empty ``dataset`` reads local files (tests)."""
    hub = bool(spec.dataset)
    return PrepConfig(
        out_dir=str(source_dir(cfg.out_dir, phase, spec.name)),
        dataset=spec.dataset if hub else "",
        files=spec.files if hub else "",
        data_files=None if hub else spec.files,
        exclude=spec.exclude,
        file_mod=spec.file_mod,
        file_keep=",".join(str(k) for k in spec.file_keep),
        formatter=spec.formatter,
        formatter_args=json.dumps(spec.args, sort_keys=True),
        extra_columns=",".join(spec.extra_columns),
        key_column=spec.key_column,
        bucket=f"{spec.bucket[0]},{spec.bucket[1]}",
        tokenizer=cfg.tokenizer,
        target_tokens=max(1, target),
        val_every=cfg.val_every,
        max_files=cfg.max_files,
        num_workers=cfg.num_workers,
        shard_tokens=cfg.shard_tokens,
    )


def read_mix(out_dir: str | Path) -> dict:
    path = Path(out_dir) / MIX_FILE
    if not path.exists():
        raise PrepError(f"no {MIX_FILE} in {out_dir}; run python -m sapia.prepare_mix first")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_mix(cfg: MixConfig, phases: dict[str, list[SourceSpec]]) -> dict:
    """Describe every phase and source, with what is on disk so far."""
    out = {"version": MIX_VERSION, "total_tokens": cfg.total_tokens, "anneal_fraction": cfg.anneal_fraction,
           "tokenizer": cfg.tokenizer, "updated_unix": time.time(), "phases": {}}
    for phase, specs in phases.items():
        entries = []
        for spec in specs:
            d = source_dir(cfg.out_dir, phase, spec.name)
            entry = {"name": spec.name, "share": spec.share, "dir": f"{phase}/{spec.name}",
                     "target_tokens": int(round(spec.share * cfg.phase_tokens(phase))),
                     "spec": spec.to_dict(), "prepared": False, "train_tokens": 0, "val_tokens": 0}
            if (d / "manifest.json").exists():
                m = load_manifest(d, verify=False)
                entry.update(prepared=bool(m["complete"]), train_tokens=m["train_tokens"],
                             val_tokens=m["val_tokens"])
            entries.append(entry)
        out["phases"][phase] = {"tokens": cfg.phase_tokens(phase), "sources": entries}
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    atomic_write_json(Path(cfg.out_dir) / MIX_FILE, out)
    return out


def run_mix(cfg: MixConfig, phases: Optional[dict[str, list[SourceSpec]]] = None,
            log: Callable[[str], None] = lambda s: print(s, flush=True)) -> dict:
    """Prepare every selected source of every selected phase; returns the final mix.json."""
    cfg.validate()
    phases = phases if phases is not None else PHASES
    wanted_phases = cfg.selected_phases()
    only = cfg.selected_sources()
    t0 = time.time()
    failures: list[str] = []
    for phase in wanted_phases:
        for spec in phases[phase]:
            if only is not None and spec.name not in only:
                continue
            target = int(round(spec.share * cfg.phase_tokens(phase)))
            log(f"[mix] {phase}/{spec.name}: {spec.share:.1%} of {fmt_count(cfg.phase_tokens(phase))} "
                f"= {fmt_count(target)} tokens from {spec.dataset or spec.files}")
            try:
                manifest = run_prepare(prep_config_for(spec, cfg, phase, target),
                                       log=lambda s, n=spec.name: log(f"[{phase}/{n}] {s}"))
            except PrepError as e:
                failures.append(f"{phase}/{spec.name}: {e}")
                log(f"[mix] ERROR {phase}/{spec.name}: {e} (other sources continue; rerun to resume)")
                continue
            if manifest["train_tokens"] < target:
                log(f"[mix] WARNING {phase}/{spec.name}: source exhausted at "
                    f"{fmt_count(manifest['train_tokens'])} of {fmt_count(target)} tokens; the loader will "
                    "repeat it to keep its share")
            write_mix(cfg, {p: phases[p] for p in wanted_phases})
    mix = write_mix(cfg, {p: phases[p] for p in wanted_phases})
    log(f"[mix] finished in {fmt_duration(time.time() - t0)}; {cfg.out_dir}/{MIX_FILE} written")
    if failures:
        raise PrepError(f"{len(failures)} source(s) failed:\n  " + "\n  ".join(failures))
    return mix


def parse_mix_args(argv: Optional[list[str]] = None) -> MixConfig:
    p = argparse.ArgumentParser(prog="python -m sapia.prepare_mix",
                                description="Prepare the Sapia pretraining mix (see docs/DATA.md). Resumable.")
    p.add_argument("--preset", default="full", choices=sorted(MIX_PRESETS))
    p.add_argument("--out_dir")
    p.add_argument("--total_tokens", type=_parse_int, help="Tokens over both phases (e.g. 15e9).")
    p.add_argument("--anneal_fraction", type=float, help="Share of tokens in the final (anneal) phase.")
    p.add_argument("--phases", help="Comma-separated phases to prepare (main,anneal).")
    p.add_argument("--only", help="Comma-separated source names to prepare.")
    p.add_argument("--tokenizer")
    p.add_argument("--val_every", type=int)
    p.add_argument("--max_files", type=int)
    p.add_argument("--num_workers", type=int)
    p.add_argument("--shard_tokens", type=_parse_int)
    ns = vars(p.parse_args(argv))
    cfg = MixConfig(**MIX_PRESETS[ns.pop("preset")])
    for k, v in ns.items():
        if v is not None:
            setattr(cfg, k, v)
    cfg.validate()
    return cfg


def main(argv: Optional[list[str]] = None) -> None:
    run_mix(parse_mix_args(argv))


if __name__ == "__main__":
    main()
