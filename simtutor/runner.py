"""
Simulation runner utilities for producing and validating event logs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import yaml

from adapters.mock_env import MockEnvAdapter
from core.event_store import JsonlEventStore
from adapters.event_store.telemetry_writer import TelemetryWriter
from core.procedure import ProcedureEngine
from core.types import Event
from core.scoring import score_log
from core.interaction_metrics import compute_interaction_metrics


def _load_steps(pack_path: Path) -> list[dict]:
    data = yaml.safe_load(pack_path.read_text(encoding="utf-8"))
    return data["steps"]


def run_simulation(pack_path: str, scenario_path: str, log_path: str | None = None) -> Path:
    pack_path = Path(pack_path)
    scenario_path = Path(scenario_path)
    steps = _load_steps(pack_path)
    engine = ProcedureEngine(steps)
    env = MockEnvAdapter(str(scenario_path))

    if log_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
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
    try:
        events = JsonlEventStore.load(log_path)
    except Exception as exc:
        return False, f"failed to load log: {exc}"
    for ev in events:
        kind = ev.get("kind") or ev.get("type")
        payload = ev.get("payload", {})
        if kind == "step_activated":
            sid = payload.get("step_id")
            if not sid:
                return False, "missing step_id for step_activated"
            if sid not in status:
                return False, f"unknown step_id {sid}"
            if status[sid] != "pending":
                return False, f"step {sid} activated from {status[sid]}"
            if active:
                return False, f"step {sid} activated while {active} active"
            next_pending = next((s for s in order if status[s] == "pending"), None)
            if next_pending and sid != next_pending:
                return False, f"step {sid} activated out of order; expected {next_pending}"
            status[sid] = "active"
            active = sid
        elif kind == "step_completed":
            sid = payload.get("step_id")
            if not sid:
                return False, "missing step_id for step_completed"
            if sid not in status:
                return False, f"unknown step_id {sid}"
            if active != sid:
                return False, f"complete {sid} but active {active}"
            status[sid] = "done"
            active = None
        elif kind == "step_blocked":
            sid = payload.get("step_id")
            if not sid:
                return False, "missing step_id for step_blocked"
            if sid not in status:
                return False, f"unknown step_id {sid}"
            if active != sid:
                return False, f"block {sid} but active {active}"
            status[sid] = "blocked"
            active = None
        else:
            continue
    if active:
        return False, f"ended with active {active}"
    return True, "ok"


def replay_telemetry(log_paths: list[str]) -> Tuple[bool, str]:
    for path in log_paths:
        last_seq = None
        last_t_wall = None
        try:
            for frame in TelemetryWriter.iter_frames(path):
                seq = frame.get("seq")
                t_wall = frame.get("t_wall")
                if not isinstance(seq, int):
                    return False, f"{path} missing seq"
                if not isinstance(t_wall, (int, float)):
                    return False, f"{path} missing t_wall"
                if last_seq is not None and seq <= last_seq:
                    return False, f"{path} non-monotonic seq at {seq}"
                if last_t_wall is not None and t_wall < last_t_wall:
                    return False, f"{path} non-monotonic t_wall at {t_wall}"
                last_seq = seq
                last_t_wall = t_wall
        except Exception as exc:
            return False, f"{path} failed to read: {exc}"
    return True, "ok"


def score_run(log_path: str, pack_path: str, taxonomy_path: str) -> dict:
    events = JsonlEventStore.load(log_path)
    return score_log(events, pack_path, taxonomy_path)


def batch_run(
    pack_path: str,
    scenarios: list[str],
    output_dir: str = "logs",
    taxonomy_path: str = "packs/fa18c_startup/taxonomy.yaml",
) -> list[dict]:
    results = []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen = {}
    for scenario in scenarios:
        scenario_path = Path(scenario)
        stem = scenario_path.stem
        if stem not in seen:
            seen[stem] = 0
            suffix = ""
        else:
            seen[stem] += 1
            suffix = f"_{seen[stem]}"
        log_name = f"{stem}{suffix}.jsonl"
        log_path = out_dir / log_name
        run_simulation(pack_path, scenario, str(log_path))
        score = score_run(str(log_path), pack_path, taxonomy_path)
        score["scenario"] = stem
        score["log_path"] = str(log_path)
        try:
            events = JsonlEventStore.load(log_path)
            score.update(compute_interaction_metrics(events).to_dict())
        except Exception:
            pass
        results.append(score)
    return results
