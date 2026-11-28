"""Push notifications to a phone via ntfy (https://ntfy.sh).

Install the ntfy app, subscribe to a hard-to-guess topic name, and pass that name as
``--ntfy_topic`` (or set ``SAPIA_NTFY_TOPIC``). Anyone who knows the topic can read the
messages, so use something random like ``sapia-7f3k9q2x``. Sending never blocks or breaks
training: messages go out on a background thread and failures are only logged.
"""
from __future__ import annotations

import math
import os
import threading
import time
import urllib.request
from typing import Callable, Optional

NTFY_SERVER = os.environ.get("SAPIA_NTFY_SERVER", "https://ntfy.sh")


class Notifier:
    def __init__(self, topic: Optional[str], run: str = "sapia", log: Callable[[str], None] = print):
        self.topic = topic or os.environ.get("SAPIA_NTFY_TOPIC") or None
        self.run = run
        self.log = log

    @property
    def enabled(self) -> bool:
        return bool(self.topic)

    def send(self, message: str, priority: str = "default") -> None:
        if not self.topic:
            return
        thread = threading.Thread(target=self._post, args=(message, priority), daemon=True)
        thread.start()

    def every(self, minutes: float) -> "ProgressPinger":
        return ProgressPinger(self, minutes)

    def _post(self, message: str, priority: str) -> None:
        req = urllib.request.Request(
            f"{NTFY_SERVER}/{self.topic}", data=message.encode("utf-8"), method="POST",
            headers={"Title": f"Sapia: {self.run}", "Priority": priority, "Tags": "robot"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10):
                pass
        except Exception as exc:  # noqa: BLE001 - notifications are best effort
            self.log(f"WARNING: ntfy notification failed ({type(exc).__name__}: {exc})")


def _short(n: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"


class ProgressPinger:
    """Sends a short update at most every ``minutes`` (silently: ntfy priority "low").

    Speed and MFU are averages since the previous message, over training compute time (evals
    and checkpoint saves are not counted), so they are comparable with the 50% MFU target.
    Call :meth:`maybe` at every log line with that window's tok/s and MFU.
    """

    def __init__(self, notifier: Notifier, minutes: float):
        self.notifier = notifier
        self.seconds = max(0.0, float(minutes or 0.0)) * 60
        self.last = time.time()
        self._prev_tokens: Optional[int] = None
        self._reset()

    def _reset(self) -> None:
        self._tokens = 0.0          # tokens trained since the last message
        self._time = 0.0            # compute seconds they took
        self._mfu_time = 0.0        # sum of MFU x seconds
        self._has_mfu = False

    def maybe(self, step: int, tokens: int, horizon: int, loss: float, tok_per_sec: float,
              val: Optional[float] = None, mfu: Optional[float] = None) -> bool:
        if self._prev_tokens is not None and tokens > self._prev_tokens and tok_per_sec > 0:
            dt = (tokens - self._prev_tokens) / tok_per_sec
            self._tokens += tokens - self._prev_tokens
            self._time += dt
            if mfu is not None and math.isfinite(mfu):
                self._mfu_time += mfu * dt
                self._has_mfu = True
        self._prev_tokens = tokens
        if not self.notifier.enabled or self.seconds <= 0 or time.time() - self.last < self.seconds:
            return False
        self.last = time.time()
        avg_tps = self._tokens / self._time if self._time > 0 else tok_per_sec
        avg_mfu = self._mfu_time / self._time if self._has_mfu and self._time > 0 else mfu
        self._reset()
        eta = (horizon - tokens) / avg_tps if avg_tps > 0 else math.inf
        eta_s = f"{eta / 3600:.1f}h" if math.isfinite(eta) else "n/a"
        msg = (f"loss {loss:.3f}" + (f" | val {val:.3f}" if val is not None else "")
               + (f"\nMFU {100 * avg_mfu:.1f}% (avg since last message)" if avg_mfu else "")
               + f"\n{_short(avg_tps)} tok/s avg | {_short(tokens)}/{_short(horizon)} tokens "
               + f"({100 * tokens / max(1, horizon):.1f}%), step {step} | {eta_s} of training left")
        self.notifier.send(msg, priority="low")
        return True
