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
- `telemetry_map.yaml` stable var mappings for gating/LLM/overlay (vars.*).
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
- Replay telemetry: `python -m simtutor replay --telemetry logs/telemetry_*.jsonl`
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

## DCS Overlay (SimTutorHighlight)
- Install hook: copy `DCS/Scripts/Hooks/SimTutorHighlight.lua` to
  `Saved Games\\DCS\\Scripts\\Hooks\\SimTutorHighlight.lua`.
- PC sender/ack example:
  ```python
  from adapters.dcs.overlay.ack_receiver import DcsOverlayAckReceiver
  from adapters.dcs.overlay.sender import DcsOverlaySender
  from core.overlay import OverlayIntent

  with DcsOverlayAckReceiver() as ack_rx, DcsOverlaySender(ack_receiver=ack_rx) as sender:
      intent = OverlayIntent(intent="highlight", target="battery", element_id="pnt_331")
      sender.send_intent(intent, expect_ack=True)
      sender.send_intent(OverlayIntent(intent="clear", target="battery", element_id="pnt_331"))
  ```

## DCS Export Injection (SimTutor.lua)
- Install SimTutor DCS scripts into Saved Games and patch `Export.lua`:
  ```sh
  python -m tools.install_dcs_hook --dcs-variant DCS
  ```
  This copies from `DCS/Scripts` in the repo to `Saved Games\DCS\Scripts`.

## DCS Capabilities Handshake
- DCS listens for HELLO on UDP `127.0.0.1:7793` and replies with caps.
- PC example:
  ```python
  from adapters.dcs.caps.handshake import negotiate, apply_caps_to_overlay_sender
  from adapters.dcs.overlay.sender import DcsOverlaySender

  caps = negotiate()
  sender = DcsOverlaySender()
  if caps:
      apply_caps_to_overlay_sender(sender, caps)
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

## DCS-BIOS Bridge (Hub Script -> SimTutor)
1) Generate catalog + Lua control list:
   ```sh
   python -m adapters.dcs_bios.catalog_loader \
       --aircraft FA-18C_hornet \
       --input DCS/Scripts/DCS-BIOS/doc/json/FA-18C_hornet.json \
       --output artifacts/dcs_bios_catalog_FA-18C_hornet.json \
       --controls-lua artifacts/dcs_bios_controls_FA-18C_hornet.lua
   ```
2) Load `DCS/Scripts/SimTutor/SimTutorDcsBiosHub.lua` into DCS-BIOS Hub.
   - Set `CONTROL_LIST_PATH` or `CONTROL_LIST_DIR` to the generated `artifacts/dcs_bios_controls_*.lua`.
   - Default UDP target: `127.0.0.1:7790`.
   - Set `SEND_DELTA_ONLY = true` to reduce bandwidth (receiver merges cache).
3) Receive on PC:
   ```python
   from adapters.dcs_bios.receiver import DcsBiosReceiver
   with DcsBiosReceiver() as rx:
       obs = rx.get_observation()
       if obs:
           print(obs.payload["bios"])
   ```
4) Raw DCS-BIOS export decode (if you see hex bytes):
   ```python
   from adapters.dcs_bios.receiver import DcsBiosRawReceiver
   with DcsBiosRawReceiver(
       aircraft="FA-18C_hornet",
       control_reference_dir="DCS/Scripts/DCS-BIOS/doc/json",
   ) as rx:
       obs = rx.get_observation()
       if obs:
           print(obs.payload["bios"])
   ```
   Or CLI:
   ```sh
   python -m tools.listen_dcs_bios_raw --aircraft FA-18C_hornet
   ```
   Note: the raw export stream is incremental; the first frame may be partial
   (e.g., `_ACFT_NAME` truncated). For a single fuller snapshot, wait for enough
   keys before exiting:
   ```sh
   python -m tools.listen_dcs_bios_raw --aircraft FA-18C_hornet \
       --once --wait 15 --min-keys 500 \
       --output artifacts/dcs_bios_frame_once.json
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
