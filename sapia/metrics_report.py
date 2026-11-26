"""``training_metrics.json``: one self-contained, regularly rewritten summary of a run.

``metrics.jsonl`` is the append-only log (one line per log/eval event). This file is the
same data as a single JSON document plus run information and live progress, so it can be
read in one go by the dashboard (``python -m sapia.dashboard``), downloaded from the
Modal Volume, or loaded into pandas/notebooks::

    {
      "version": 1,
      "status": "running" | "complete" | "time_limit" | "total_time_limit" | "interrupted" | "failed",
      "updated_unix": ...,
      "run": {preset, n_params, model, precision, device, gpu, batch plan, horizon_tokens, ...},
      "progress": {step, tokens_seen, horizon_tokens, fraction, tok_per_sec, mfu, eta_s, train_time_s},
      "last_train_loss": ..., "last_val_loss": ..., "best_val_loss": ..., "best_val_step": ...,
      "train": [ {step, tokens_seen, loss, lr, grad_norm, tok_per_sec, mfu, ...}, ... ],
      "eval":  [ {step, tokens_seen, val_loss, best_val_loss, ...}, ... ],
      "error": null | "message when status is failed"
    }

The file is rewritten atomically (temp file + rename), at most every ``min_interval``
seconds while training plus on every eval and at the end, so readers never see a
partial document.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from .utils import atomic_write_json

REPORT_NAME = "training_metrics.json"
REPORT_VERSION = 1


def _clean(value: Any) -> Any:
    """JSON has no NaN/Infinity: map non-finite floats to None (recursively)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


class TrainingMetrics:
    def __init__(self, path: str | Path, run_info: dict, existing: Iterable[dict] = (),
                 min_interval: float = 10.0) -> None:
        self.path = Path(path)
        self.run_info = dict(run_info)
        self.min_interval = float(min_interval)
        self.train: list[dict] = []
        self.eval: list[dict] = []
        self.status = "running"
        self.error: Optional[str] = None
        self.progress: dict = {}
        self._last_write = float("-inf")  # monotonic() can be small (fresh boot/container): first write always happens
        for record in existing:            # seeded from metrics.jsonl when a run resumes
            self.add(record)

    # ---- recording
    def add(self, record: dict) -> None:
        kind = record.get("kind")
        if kind == "train":
            self.train.append(dict(record))
        elif kind == "eval":
            self.eval.append(dict(record))

    def set_progress(self, **progress: Any) -> None:
        self.progress.update(progress)
        horizon = self.progress.get("horizon_tokens")
        seen = self.progress.get("tokens_seen")
        if horizon and seen is not None:
            self.progress["fraction"] = min(1.0, seen / horizon)

    # ---- output
    def to_dict(self) -> dict:
        losses = [r["loss"] for r in self.train if isinstance(r.get("loss"), float) and math.isfinite(r["loss"])]
        vals = [(r["val_loss"], r["step"]) for r in self.eval
                if isinstance(r.get("val_loss"), float) and math.isfinite(r["val_loss"])]
        best = min(vals) if vals else None
        return _clean({
            "version": REPORT_VERSION,
            "status": self.status,
            "updated_unix": time.time(),
            "run": self.run_info,
            "progress": self.progress,
            "last_train_loss": losses[-1] if losses else None,
            "last_val_loss": vals[-1][0] if vals else None,
            "best_val_loss": best[0] if best else None,
            "best_val_step": best[1] if best else None,
            "train": self.train,
            "eval": self.eval,
            "error": self.error,
        })

    def write(self, force: bool = False) -> bool:
        """Rewrite the file if ``force`` or ``min_interval`` passed. Returns True when written."""
        now = time.monotonic()
        if not force and now - self._last_write < self.min_interval:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path, self.to_dict())
        self._last_write = now
        return True

    def finish(self, status: str, error: Optional[str] = None, **progress: Any) -> None:
        self.status = status
        self.error = error
        if progress:
            self.set_progress(**progress)
        self.write(force=True)
