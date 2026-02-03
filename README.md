# SimTutor Core v0.1 F/A-18C Cold Start (Mock-first)

Simulator-agnostic tutoring backend with clean architecture (domain core + ports/adapters), mock environment, scoring, replayable event logs, and batch harness. Packs are derived from `Doc/Evaluation` sources.

## Repo Layout
- `core/` domain engines: procedure state machine, gating DSL, scoring, overlay planner, knowledge BM25, data types.
- `ports/` contracts for environment/model/knowledge/event store, etc.
- `adapters/` mock env, DCS UDP stub, model stub, local knowledge adapter.
- `packs/fa18c_startup/` procedure/taxonomy/ui_map + instruments.
- `simtutor/schemas/v1/` JSON Schemas for Observation/TutorRequest/TutorResponse/Event.
- `mock_scenarios/` scripted observation sequences.
- `tools/index_docs.py` offline indexer for `Doc/Evaluation` md/pdf.
- `logs/`, `artifacts/` run outputs.

## Quickstart
1) Python 3.10. Install deps:
   ```sh
   python -m pip install -e .
   ```
2) Build document index (defaults to `Doc/Evaluation`):
   ```sh
   python -m tools.index_docs --output Doc/Evaluation/index.json
   ```
3) Run a single mock scenario  log:
   ```sh
   python -m simtutor run --pack packs/fa18c_startup/pack.yaml \
       --scenario mock_scenarios/correct_process.json \
       --output logs/run_demo.jsonl
   ```
4) Replay trajectory check:
   ```sh
   python -m simtutor replay logs/run_demo.jsonl --pack packs/fa18c_startup/pack.yaml
   ```
5) Score a log:
   ```sh
   python -m simtutor score logs/run_demo.jsonl \
       --pack packs/fa18c_startup/pack.yaml \
       --taxonomy packs/fa18c_startup/taxonomy.yaml
   ```
6) Batch all scenarios, export CSV (requested example):
   ```sh
   python -m simtutor batch --pack packs/fa18c_startup/pack.yaml --output-dir artifacts
   ```
   Creates `artifacts/results.csv` plus per-scenario logs.
7) Schema validation for logs:
   ```sh
   python -m simtutor validate logs/run_demo.jsonl --schema event
   ```
8) Run all automated tests:
   ```sh
   pytest
   ```

## Data Contracts (v1)
- JSON Schemas in `simtutor/schemas/v1/*.schema.json`
- Python dataclasses in `core/types.py`
  - Observation, TutorRequest, TutorResponse, Event (versioned, UUID + ISO timestamps).

## Packs (fa18c_startup)
- `pack.yaml` steps S01CS10 mapped from `fa18c_startup_master.md` (phase, wording, preconditions placeholders).
- `taxonomy.yaml` error categories OM/CO/OR/PA/SV, weights, critical multiplier.
- `ui_map.yaml` abstract targets  DCS `pnt_*`.
- Instruments: NASATLX, quiz stored under `Doc/Evaluation`.

## Engines
- Procedure engine (`core/procedure.py`): pending/active/done/blocked, rewind, timeout, prompt counts; emits `step_activated|completed|blocked`.
- Gating (`core/gating.py`): DSL ops `var_gte`, `arg_in_range`, `flag_true`, `time_since`.
- Scoring (`core/scoring.py`): counts/errors per category + TotalErrorScore from event logs; uses pack/taxonomy.
- Overlay planner (`core/overlay.py`): maps abstract target to overlay action (highlight/clear/pulse).
- Knowledge BM25 (`core/knowledge.py` + `adapters/knowledge_local.py`): queries `Doc/Evaluation/index.json` (default).

## Adapters
- MockEnv (`adapters/mock_env.py`): replays `mock_scenarios/*.json` via `get_observation()`.
- DCS UDP stub (`adapters/dcs_adapter.py`): sends `HILITE pnt_XXX` / `CLEAR` to `127.0.0.1:7778`, compatible with `Scripts/Hooks/VRHilite.lua`; designed to map ui_map entries for future expansion.
- Model stub (`adapters/model_stub.py`): deterministic modes A/B/C; writes card info in `TutorResponse.metadata`.

## Experiment Harness
- Single run: `python -m simtutor run ...`
- Replay: `python -m simtutor replay ...`
- Score: `python -m simtutor score ...`
- Batch: `python -m simtutor batch --pack packs/fa18c_startup/pack.yaml --output-dir artifacts`
  - Optional: `--taxonomy packs/fa18c_startup/taxonomy.yaml`
  - Optional: `--scenarios mock_scenarios/*.json`

## Knowledge Retrieval
- Build index: `python -m tools.index_docs --output Doc/Evaluation/index.json`
- Query example (Python):
  ```python
  from adapters.knowledge_local import LocalKnowledgeAdapter
  kg = LocalKnowledgeAdapter()  # defaults Doc/Evaluation/index.json
  print(kg.query("battery on fire test", k=3))
  ```

## DCS Overlay Quick Test
- With `VRHilite.lua` running in DCS (listening 127.0.0.1:7778):
  ```python
  from adapters.dcs_adapter import DcsAdapter
  dcs = DcsAdapter()
  dcs.highlight("pnt_331")  # fire test switch
  dcs.clear()
  dcs.close()
  ```
  Mirrors `DCS/Scripts/HilteTest.py` / `HiliteClearTest.py`.

## DCS Export Injection (SimTutor.lua)
- Install SimTutor DCS scripts into Saved Games and patch `Export.lua`:
  ```sh
  python -m tools.install_dcs_hook --dcs-variant DCS
  ```
- Optional hook install (Saved Games `Scripts/Hooks/SimTutor.lua`):
  ```sh
  python -m tools.install_dcs_hook --use-hook
  ```

## DCS Telemetry (UDP -> Observation)
- SimTutor.lua sends JSON telemetry over UDP (default 127.0.0.1:7780, 20 Hz).
- PC receiver example:
  ```python
  from adapters.dcs.telemetry.receiver import DcsTelemetryReceiver
  with DcsTelemetryReceiver() as rx:
      obs = rx.get_observation()
      if obs:
          print(obs.payload)
  ```
- Record to JSONL event log:
  ```sh
  python -m tools.record_dcs_telemetry --output logs/dcs_telemetry.jsonl --duration 30 --print
  ```

## Source Documents (authoritative)
- `Doc/Evaluation/fa18c_startup_master.md`
- `Doc/Evaluation/Appendix - Training Task Syllabus.md`
- `Doc/Evaluation/fa18c_error_coding_guide.md`
- `Doc/Evaluation/fa18c_scoring_sheet_template.md`
- `Doc/Evaluation/fa18c_coldstart_quiz.md`
- `Doc/Evaluation/fa18c_nasatlx_vr.md`

## Notes
- Default doc root for indexing/retrieval: `Doc/Evaluation`.
- Event logs are JSONL; replayable and scorable.
- PyPDF2 deprecation warning is expected; no functional impact for now.
