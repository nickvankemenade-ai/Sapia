"""Push notifications to a phone via ntfy (https://ntfy.sh).

Install the ntfy app, subscribe to a hard-to-guess topic name, and pass that name as
``--ntfy_topic`` (or set ``SAPIA_NTFY_TOPIC``). Anyone who knows the topic can read the
messages, so use something random like ``sapia-7f3k9q2x``. Sending never blocks or breaks
training: messages go out on a background thread and failures are only logged.
"""
from __future__ import annotations

import os
import threading
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
