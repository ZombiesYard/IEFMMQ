import argparse
from datetime import datetime, timezone
import json
import os
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Mapping, Tuple

from jsonschema import Draft202012Validator, FormatChecker

from simtutor.runner import replay_log, run_simulation


SCHEMA_INDEX = {
    "event": ("simtutor.schemas.v1", "event.schema.json"),
    "observation": ("simtutor.schemas.v1", "observation.schema.json"),
    "tutor_request": ("simtutor.schemas.v1", "tutor_request.schema.json"),
    "tutor_response": ("simtutor.schemas.v1", "tutor_response.schema.json"),
    "dcs_observation": ("simtutor.schemas.v2", "dcs_observation.json"),
    "dcs_bios_frame": ("simtutor.schemas.v2", "dcs_bios_frame.json"),
    "telemetry_frame": ("simtutor.schemas.v2", "telemetry_frame.json"),
    "dcs_overlay_command": ("simtutor.schemas.v2", "dcs_overlay_command.json"),
    "dcs_overlay_ack": ("simtutor.schemas.v2", "dcs_overlay_ack.json"),
    "dcs_hello": ("simtutor.schemas.v2", "dcs_hello.json"),
    "dcs_caps": ("simtutor.schemas.v2", "dcs_caps.json"),
}


def _load_schema(name: str) -> Mapping:
    if name not in SCHEMA_INDEX:
        raise FileNotFoundError(f"Unknown schema: {name}")
    schema_pkg, schema_file = SCHEMA_INDEX[name]
    try:
        schema_path = resources.files(schema_pkg) / schema_file
    except ModuleNotFoundError as exc:
        raise FileNotFoundError(f"Schema package not found: {schema_pkg}") from exc
    if not schema_path.is_file():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} invalid JSON: {exc}") from exc


def validate(files: list[str], schema_name: str) -> int:
    schema = _load_schema(schema_name)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    had_error = False

    for file in files:
        path = Path(file)
        if not path.exists():
            print(f"[ERROR] File not found: {file}")
            had_error = True
            continue
        for lineno, obj in _iter_jsonl(path):
            errors = sorted(validator.iter_errors(obj), key=lambda e: e.path)
            if errors:
                had_error = True
                for err in errors:
                    location = ".".join([str(p) for p in err.path])
                    print(f"[FAIL] {file}:{lineno} path={location or '<root>'} msg={err.message}")
            else:
                print(f"[OK]   {file}:{lineno}")
    return 1 if had_error else 0


def _new_replay_bios_log_path() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("logs") / f"replay_bios_{ts}.jsonl"


def _build_replay_model_from_args(args: argparse.Namespace) -> Any:
    from live_dcs import _build_model_from_args

    return _build_model_from_args(args)


def _run_replay_bios(args: argparse.Namespace) -> int:
    from adapters.action_executor import OverlayActionExecutor
    from core.event_store import JsonlEventStore
    from live_dcs import (
        CompositeHelpTrigger,
        LiveDcsTutorLoop,
        ReplayBiosReceiver,
        StdinHelpTrigger,
        UdpHelpTrigger,
    )

    output = Path(args.output) if args.output else _new_replay_bios_log_path()
    output.parent.mkdir(parents=True, exist_ok=True)

    source = None
    model = None
    loop = None
    stdin_trigger = None
    udp_trigger = None
    stats: dict[str, Any] = {}

    try:
        source = ReplayBiosReceiver(args.input, speed=args.speed)
        model = _build_replay_model_from_args(args)
        with JsonlEventStore(output, mode="w") as store:
            with OverlayActionExecutor(
                ui_map_path=args.ui_map,
                pack_path=args.pack,
                dry_run=bool(args.dry_run_overlay),
                session_id=args.session_id,
                event_sink=store.append,
            ) as executor:
                loop = LiveDcsTutorLoop(
                    source=source,
                    model=model,
                    action_executor=executor,
                    pack_path=args.pack,
                    ui_map_path=args.ui_map,
                    telemetry_map_path=args.telemetry_map,
                    bios_to_ui_path=args.bios_to_ui,
                    knowledge_index_path=args.knowledge_index,
                    rag_top_k=args.rag_top_k,
                    cooldown_s=args.cooldown_s,
                    session_id=args.session_id,
                    lang=args.lang,
                    event_sink=store.append,
                    dry_run_overlay=bool(args.dry_run_overlay),
                )

                stdin_trigger = StdinHelpTrigger() if args.stdin_help else None
                udp_trigger = (
                    UdpHelpTrigger(
                        host=args.help_udp_host,
                        port=args.help_udp_port,
                        timeout=args.help_udp_timeout,
                    )
                    if args.help_udp_port > 0
                    else None
                )
                triggers = []
                if stdin_trigger is not None:
                    stdin_trigger.start()
                    triggers.append(stdin_trigger)
                    print("[REPLAY_BIOS] stdin trigger enabled: press Enter/help/h/?")
                if udp_trigger is not None:
                    udp_trigger.start()
                    triggers.append(udp_trigger)
                    print(
                        f"[REPLAY_BIOS] udp trigger enabled: send 'help' to "
                        f"{args.help_udp_host}:{udp_trigger.bound_port}"
                    )
                help_trigger = None
                if len(triggers) == 1:
                    help_trigger = triggers[0]
                elif len(triggers) > 1:
                    help_trigger = CompositeHelpTrigger(triggers)

                stats = loop.run(
                    max_frames=args.max_frames,
                    duration_s=args.duration,
                    auto_help_on_first_frame=bool(args.auto_help_once),
                    auto_help_every_n_frames=args.auto_help_every_n_frames,
                    help_trigger=help_trigger,
                )
    finally:
        if stdin_trigger is not None:
            stdin_trigger.close()
        if udp_trigger is not None:
            udp_trigger.close()
        if loop is not None:
            loop.close()
        else:
            if source is not None and hasattr(source, "close"):
                source.close()
            if model is not None and hasattr(model, "close"):
                model.close()

    print(f"[REPLAY_BIOS] wrote events to {output}")
    print(f"[REPLAY_BIOS] stats={json.dumps(stats, ensure_ascii=False, sort_keys=True)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="simtutor", description="SimTutor CLI utilities")
    sub = parser.add_subparsers(dest="command")

    val = sub.add_parser("validate", help="Validate JSONL logs against schema")
    val.add_argument("files", nargs="+", help="One or more JSONL files")
    val.add_argument(
        "--schema",
        choices=sorted(SCHEMA_INDEX.keys()),
        default="event",
        help="Schema name to validate against (default: event)",
    )

    run = sub.add_parser("run", help="Run mock scenario and write event log")
    run.add_argument("--pack", required=True, help="Path to pack.yaml")
    run.add_argument("--scenario", required=True, help="Path to mock scenario JSON")
    run.add_argument("--output", help="Output log path (default logs/run_<ts>.jsonl)")

    rep = sub.add_parser("replay", help="Replay event log and validate trajectory")
    rep.add_argument("file", nargs="?", help="Event log JSONL")
    rep.add_argument("--pack", help="Path to pack.yaml for validation")
    rep.add_argument("--telemetry", nargs="*", help="Telemetry JSONL logs (validate seq/t_wall order)")

    score = sub.add_parser("score", help="Score an event log using taxonomy")
    score.add_argument("file", help="Event log JSONL")
    score.add_argument("--pack", required=True, help="Path to pack.yaml")
    score.add_argument("--taxonomy", required=True, help="Path to taxonomy.yaml")

    batch = sub.add_parser("batch", help="Run batch scenarios and export CSV")
    batch.add_argument("--pack", required=True, help="Path to pack.yaml")
    batch.add_argument("--taxonomy", default="packs/fa18c_startup/taxonomy.yaml", help="Path to taxonomy.yaml")
    batch.add_argument("--scenarios", nargs="*", help="Scenario JSON files (default: mock_scenarios/*.json)")
    batch.add_argument("--output-dir", default="logs", help="Directory to store logs/results.csv")

    sub.add_parser("model-config", help="Validate model provider env and print non-sensitive startup info")

    rep_bios = sub.add_parser("replay-bios", help="Replay DCS-BIOS JSONL through live tutor pipeline")
    rep_bios.add_argument("--input", required=True, help="Path to dcs_bios_raw.jsonl")
    rep_bios.add_argument("--speed", type=float, default=1.0, help="Replay speed (1.0 realtime, 0 max speed)")

    rep_bios.add_argument("--pack", default="packs/fa18c_startup/pack.yaml", help="pack.yaml path")
    rep_bios.add_argument("--ui-map", default="packs/fa18c_startup/ui_map.yaml", help="ui_map.yaml path")
    rep_bios.add_argument(
        "--telemetry-map",
        default="packs/fa18c_startup/telemetry_map.yaml",
        help="telemetry_map.yaml path",
    )
    rep_bios.add_argument(
        "--bios-to-ui",
        default="packs/fa18c_startup/bios_to_ui.yaml",
        help="bios_to_ui.yaml path",
    )
    rep_bios.add_argument("--knowledge-index", default="Doc/Evaluation/index.json", help="Grounding index.json path")
    rep_bios.add_argument("--rag-top-k", type=int, default=5, help="Grounding snippet top-k")

    rep_bios.add_argument("--output", help="Event log JSONL output path")
    rep_bios.add_argument("--session-id", default=None, help="Optional event session id")
    rep_bios.add_argument("--cooldown-s", type=float, default=4.0, help="Help cache cooldown seconds")
    rep_bios.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 means unlimited)")
    rep_bios.add_argument("--duration", type=float, default=0.0, help="Run duration seconds (0 means unlimited)")
    rep_bios.add_argument("--auto-help-once", action="store_true", help="Auto trigger one help after first frame")
    rep_bios.add_argument("--auto-help-every-n-frames", type=int, default=0, help="Auto help interval by frames")
    rep_bios.add_argument("--stdin-help", action="store_true", help="Enable stdin help trigger")
    rep_bios.add_argument("--help-udp-host", default="127.0.0.1", help="UDP host for help trigger")
    rep_bios.add_argument("--help-udp-port", type=int, default=0, help="UDP port for help trigger (0 disabled)")
    rep_bios.add_argument("--help-udp-timeout", type=float, default=0.2, help="UDP help trigger timeout seconds")
    rep_bios.add_argument(
        "--dry-run-overlay",
        dest="dry_run_overlay",
        action="store_true",
        default=True,
        help="Dry-run overlay (default enabled for replay safety)",
    )
    rep_bios.add_argument(
        "--no-dry-run-overlay",
        dest="dry_run_overlay",
        action="store_false",
        help="Disable dry-run overlay and send overlay commands",
    )

    rep_bios.add_argument("--model-provider", choices=["stub", "openai_compat", "ollama"], default="stub")
    rep_bios.add_argument("--model-name", default=os.getenv("SIMTUTOR_MODEL_NAME", "Qwen3-8B-Instruct"))
    rep_bios.add_argument("--model-base-url", default=os.getenv("SIMTUTOR_MODEL_BASE_URL", ""))
    rep_bios.add_argument("--model-timeout-s", type=float, default=float(os.getenv("SIMTUTOR_MODEL_TIMEOUT_S", "20")))
    rep_bios.add_argument("--model-api-key", default=os.getenv("SIMTUTOR_MODEL_API_KEY"))
    rep_bios.add_argument("--stub-mode", default="A", help="ModelStub mode (A/B/C)")
    rep_bios.add_argument("--lang", choices=["zh", "en"], default=os.getenv("SIMTUTOR_LANG", "zh"))
    rep_bios.add_argument(
        "--log-raw-llm-text",
        action="store_true",
        default=bool(int(os.getenv("SIMTUTOR_LOG_RAW_LLM_TEXT", "0"))),
        help="Log raw model text into tutor_response.metadata.raw_llm_text(_attempts)",
    )

    args = parser.parse_args()
    if args.command == "validate":
        return validate(args.files, args.schema)
    if args.command == "run":
        log_path = run_simulation(args.pack, args.scenario, args.output)
        print(f"[RUN] wrote {log_path}")
        return 0
    if args.command == "replay":
        if args.telemetry:
            from simtutor.runner import replay_telemetry

            ok, msg = replay_telemetry(args.telemetry)
        else:
            if not args.file:
                print("[REPLAY] missing event log file")
                return 1
            if not args.pack:
                print("[REPLAY] missing --pack for event replay")
                return 1
            ok, msg = replay_log(args.file, args.pack)
        print(f"[REPLAY] {msg}")
        return 0 if ok else 1
    if args.command == "score":
        from simtutor.runner import score_run

        result = score_run(args.file, args.pack, args.taxonomy)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "batch":
        from simtutor.runner import batch_run
        import csv
        import glob

        scenarios = args.scenarios or glob.glob("mock_scenarios/*.json")
        results = batch_run(args.pack, scenarios, args.output_dir, args.taxonomy)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "results.csv"
        if results:
            fieldnames = sorted({k for r in results for k in r.keys()})
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(results)
            print(f"[BATCH] wrote {csv_path} from {len(results)} scenarios")
        else:
            csv_path.write_text("", encoding="utf-8")
            print(f"[BATCH] no scenarios; wrote empty {csv_path}")
        return 0
    if args.command == "model-config":
        from simtutor.config import ModelConfigError, load_model_access_config

        try:
            cfg = load_model_access_config()
        except ModelConfigError as exc:
            print(f"[MODEL_CONFIG] error: {exc}")
            return 1
        print(f"[MODEL_CONFIG] {cfg.public_startup_info()}")
        return 0
    if args.command == "replay-bios":
        try:
            return _run_replay_bios(args)
        except Exception as exc:
            print(f"[REPLAY_BIOS] error: {exc}")
            return 1
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
