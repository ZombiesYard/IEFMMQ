"""
Mock environment adapter that replays scripted observations from JSON files.

Usage:
    env = MockEnvAdapter("C:/path/to/scenario.json")
    obs = env.get_observation()  # returns Observation or None when finished
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

from core.types import Observation


class MockEnvAdapter:
    def __init__(self, scenario_path: str):
        self.scenario_path = Path(scenario_path)
        self._script = self._load_script(self.scenario_path)
        self._cursor = 0

    @staticmethod
    def _load_script(path: Path) -> List[dict]:
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_file():
            raise ValueError(f"Scenario path is not a file: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Scenario must be a list of observation dicts")
        for idx, entry in enumerate(data):
            if not isinstance(entry, dict):
                raise ValueError(f"Scenario entry at index {idx} must be a dict")
        return data

    def reset(self) -> None:
        self._cursor = 0

    def get_observation(self) -> Optional[Observation]:
        """Return next Observation in the script or None if exhausted."""
        if self._cursor >= len(self._script):
            return None
        entry = self._script[self._cursor]
        self._cursor += 1
        return self._build_observation(entry)

    def _build_observation(self, entry: dict) -> Observation:
        payload = deepcopy(entry.get("payload", {}))
        tags = deepcopy(entry.get("tags", []))
        procedure_hint = entry.get("procedure_hint")
        source = entry.get("source", "mock_env")
        obs = Observation(
            source=source,
            payload=payload,
            tags=tags,
            procedure_hint=procedure_hint,
            metadata=deepcopy(entry.get("metadata", {})),
            attachments=deepcopy(entry.get("attachments", [])),
        )
        if "timestamp" in entry:
            obs.timestamp = entry["timestamp"]
        return obs

    def remaining(self) -> int:
        return len(self._script) - self._cursor
