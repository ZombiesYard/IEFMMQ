"""
JSONL Event Store for recording and replaying SimTutor runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Union

from core.types import Event


class JsonlEventStore:
    def __init__(self, path: Union[str, Path], mode: str = "a"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open(mode, encoding="utf-8")

    def append(self, event: Event | dict) -> None:
        payload = event if isinstance(event, dict) else event.to_dict()
        self._fh.write(json.dumps(payload))
        self._fh.write("\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    @staticmethod
    def load(path: Union[str, Path]) -> List[dict]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        with p.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
