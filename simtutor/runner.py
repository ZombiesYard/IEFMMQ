from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import yaml

from adapters.mock_env import MockEnvAdapter
from core.event_store import JsonlEventStore
from core.procedure import ProcedureEngine
from core.types import Event
from core.scoring import score_log, _load_taxonomy


def _load_steps(pack_path: Path):
    data = yaml.safe_load(pack_path.read_text(encoding="utf-8"))
    return data["steps"]


def run_simulation(pack_path: str, scenario_path: str, log_path: str | None = None) -> Path:
    pack_path = Path(pack_path)
    scenario_path = Path(scenario_path)
    steps = _load_steps(pack_path)
    engine = ProcedureEngine(steps)
    env = MockEnvAdapter(str(scenario_path))

    if log_path is None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_path = Path("logs") / f"run_{ts}.jsonl"
    else:
        log_path = Path(log_path)

    with JsonlEventStore(log_path, mode="w") as store:
        # initial activation
        before = len(engine.events)
        engine.activate_next()
        _log_new_engine_events(engine, store, before)

        while True:
            obs = env.get_observation()
            if obs is None:
                break
            store.append(Event(kind="observation", payload=obs.to_dict(), related_id=obs.observation_id))

            if not engine.active_step():
                before = len(engine.events)
                engine.activate_next()
                _log_new_engine_events(engine, store, before)

            active = engine.active_step()
            if obs.procedure_hint and obs.procedure_hint == active:
                before = len(engine.events)
                engine.complete_active()
                _log_new_engine_events(engine, store, before)

    return log_path


def _log_new_engine_events(engine: ProcedureEngine, store: JsonlEventStore, start_idx: int) -> None:
    for ev in engine.events[start_idx:]:
        store.append(Event(kind=ev["type"], payload=ev))


def replay_log(log_path: str, pack_path: str) -> Tuple[bool, str]:
    steps = _load_steps(Path(pack_path))
    order = [s["id"] for s in steps]
    status = {sid: "pending" for sid in order}
    active = None
    events = JsonlEventStore.load(log_path)
    for ev in events:
        kind = ev.get("kind") or ev.get("type")
        payload = ev.get("payload", {})
        if kind == "step_activated":
            sid = payload["step_id"]
            if status[sid] != "pending":
                return False, f"step {sid} activated from {status[sid]}"
            if active:
                return False, f"step {sid} activated while {active} active"
            status[sid] = "active"
            active = sid
        elif kind == "step_completed":
            sid = payload["step_id"]
            if active != sid:
                return False, f"complete {sid} but active {active}"
            status[sid] = "done"
            active = None
        elif kind == "step_blocked":
            sid = payload["step_id"]
            if active != sid:
                return False, f"block {sid} but active {active}"
            status[sid] = "blocked"
            active = None
        else:
            continue
    return True, "ok"


def score_run(log_path: str, pack_path: str, taxonomy_path: str) -> dict:
    events = JsonlEventStore.load(log_path)
    return score_log(events, pack_path, taxonomy_path)
