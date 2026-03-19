import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Tuple

from jsonschema import Draft202012Validator, FormatChecker

from adapters.pack_gates import DEFAULT_SCENARIO_PROFILE, SUPPORTED_SCENARIO_PROFILES
from adapters.vision_frames import DEFAULT_FRAME_CHANNEL
from adapters.vision_prompting import DEFAULT_LAYOUT_ID
from core.constants import ENV_COLD_START_PRODUCTION
from core.env_bool import parse_env_bool
from simtutor.cli_parsing import parse_env_int, parse_non_negative_int_arg
from simtutor.schemas import SCHEMA_INDEX, load_schema
from simtutor.runner import replay_log, run_simulation


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
    schema = load_schema(schema_name)
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


def _add_model_args(parser: argparse.ArgumentParser, *, default_provider: str, provider_choices: list[str]) -> None:
    parser.add_argument("--model-provider", choices=provider_choices, default=default_provider)
    parser.add_argument("--model-name", default=os.getenv("SIMTUTOR_MODEL_NAME", "Qwen3-8B-Instruct"))
    parser.add_argument("--model-base-url", default=os.getenv("SIMTUTOR_MODEL_BASE_URL", ""))
    parser.add_argument("--model-timeout-s", type=float, default=float(os.getenv("SIMTUTOR_MODEL_TIMEOUT_S", "20")))
    parser.add_argument(
        "--model-max-tokens",
        type=parse_non_negative_int_arg,
        default=parse_env_int("SIMTUTOR_MODEL_MAX_TOKENS", default=0, minimum=0),
        help="Max completion tokens for model providers that support it (0 uses provider default).",
    )
    parser.add_argument("--model-api-key", default=os.getenv("SIMTUTOR_MODEL_API_KEY"))
    model_multimodal_default = parse_env_bool("SIMTUTOR_MODEL_ENABLE_MULTIMODAL", default=False)
    model_multimodal_group = parser.add_mutually_exclusive_group()
    model_multimodal_group.add_argument(
        "--model-enable-multimodal",
        dest="model_enable_multimodal",
        action="store_true",
        help="Allow OpenAI-compatible models to send synchronized vision frames as multimodal image inputs.",
    )
    model_multimodal_group.add_argument(
        "--no-model-enable-multimodal",
        dest="model_enable_multimodal",
        action="store_false",
        help="Force text-only requests even when synchronized vision frames are available.",
    )
    parser.set_defaults(model_enable_multimodal=model_multimodal_default)
    parser.add_argument("--stub-mode", default="A", help="ModelStub mode (A/B/C)")
    parser.add_argument("--lang", choices=["zh", "en"], default=os.getenv("SIMTUTOR_LANG", "zh"))
    log_raw_default = parse_env_bool("SIMTUTOR_LOG_RAW_LLM_TEXT", default=False)
    log_raw_group = parser.add_mutually_exclusive_group()
    log_raw_group.add_argument(
        "--log-raw-llm-text",
        dest="log_raw_llm_text",
        action="store_true",
        help="Log raw model text into tutor_response.metadata.raw_llm_text(_attempts)",
    )
    log_raw_group.add_argument(
        "--no-log-raw-llm-text",
        dest="log_raw_llm_text",
        action="store_false",
        help="Disable raw model text logging even if SIMTUTOR_LOG_RAW_LLM_TEXT=1",
    )
    parser.set_defaults(log_raw_llm_text=log_raw_default)
    print_model_io_default = parse_env_bool("SIMTUTOR_PRINT_MODEL_IO", default=False)
    print_model_io_group = parser.add_mutually_exclusive_group()
    print_model_io_group.add_argument(
        "--print-model-io",
        dest="print_model_io",
        action="store_true",
        help="Print the full prompt text and decoded raw model reply to the terminal for debugging.",
    )
    print_model_io_group.add_argument(
        "--no-print-model-io",
        dest="print_model_io",
        action="store_false",
        help="Disable terminal model I/O debug printing even if SIMTUTOR_PRINT_MODEL_IO=1.",
    )
    parser.set_defaults(print_model_io=print_model_io_default)


def _run_replay_bios(args: argparse.Namespace) -> int:
    from adapters.action_executor import OverlayActionExecutor
    from core.event_store import JsonlEventStore
    from live_dcs import (
        CompositeHelpTrigger,
        LiveDcsTutorLoop,
        ReplayBiosReceiver,
        StdinHelpTrigger,
        UdpHelpTrigger,
        _build_vision_port_from_args,
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
        vision_port, vision_session_id, vision_sync_window_ms, vision_trigger_wait_ms = _build_vision_port_from_args(
            args,
            mode="replay",
        )
        with JsonlEventStore(output, mode="w") as store:
            with OverlayActionExecutor(
                ui_map_path=args.ui_map,
                pack_path=args.pack,
                max_targets=max(0, int(args.max_overlay_targets)),
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
                    cold_start_production=bool(args.cold_start_production),
                    knowledge_source_policy_path=args.knowledge_source_policy,
                    cooldown_s=args.cooldown_s,
                    session_id=args.session_id,
                    lang=args.lang,
                    scenario_profile=args.scenario_profile,
                    event_sink=store.append,
                    dry_run_overlay=bool(args.dry_run_overlay),
                    vision_port=vision_port,
                    vision_session_id=vision_session_id,
                    vision_mode="replay",
                    vision_sync_window_ms=vision_sync_window_ms,
                    vision_trigger_wait_ms=vision_trigger_wait_ms,
                    max_overlay_targets=max(0, int(args.max_overlay_targets)),
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


def _run_record_vlm(args: argparse.Namespace) -> int:
    from adapters.dcs_bios.receiver import DcsBiosReceiver
    from live_dcs import _build_vision_port_from_args

    if args.duration < 0:
        raise ValueError("record-vlm requires --duration >= 0")
    if args.max_frames < 0:
        raise ValueError("record-vlm requires --max-frames >= 0")
    if args.duration <= 0 and args.max_frames <= 0:
        raise ValueError("record-vlm requires --duration or --max-frames")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    vision_port, vision_session_id, _sync_window_ms, _trigger_wait_ms = _build_vision_port_from_args(
        args,
        mode="replay",
    )
    if vision_port is None or not vision_session_id:
        raise ValueError("record-vlm requires --vision-saved-games-dir plus --vision-session-id or --session-id")

    bios_frames = 0
    vision_frames = 0
    last_vision_frame_id: str | None = None
    start_monotonic = time.monotonic()

    with DcsBiosReceiver(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        merge_full_state=bool(args.merge_full_state),
    ) as source:
        vision_port.start(vision_session_id)
        try:
            with output.open("w", encoding="utf-8") as handle:
                while True:
                    for observation in vision_port.poll():
                        vision_frames += 1
                        last_vision_frame_id = observation.frame_id

                    obs = source.get_observation()
                    if obs is not None:
                        if not isinstance(obs.payload, dict):
                            raise ValueError("record-vlm observed non-object BIOS payload")
                        payload = obs.payload
                        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        handle.flush()
                        bios_frames += 1

                    if args.max_frames > 0 and bios_frames >= args.max_frames:
                        break
                    if args.duration > 0 and (time.monotonic() - start_monotonic) >= args.duration:
                        break
        except KeyboardInterrupt:
            pass
        finally:
            vision_port.stop()

    print(f"[RECORD_VLM] wrote bios frames to {output}")
    print(
        "[RECORD_VLM] stats="
        + json.dumps(
            {
                "bios_frames": bios_frames,
                "vision_frames": vision_frames,
                "vision_session_id": vision_session_id,
                "last_vision_frame_id": last_vision_frame_id,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _run_replay_eval(args: argparse.Namespace) -> int:
    from simtutor.replay_eval import ReplayEvalOracleModel, load_replay_eval_suite, run_replay_eval_suite

    suite = replace(load_replay_eval_suite(args.suite), lang=args.lang)
    provider_name = "replay_eval_oracle" if args.model_provider == "oracle" else args.model_provider

    def _model_factory(case) -> Any:
        if args.model_provider == "oracle":
            return ReplayEvalOracleModel(case, lang=args.lang)
        runtime_args = argparse.Namespace(**vars(args))
        runtime_args.session_id = case.session_id
        runtime_args.vision_saved_games_dir = str(case.vision.saved_games_dir) if case.vision is not None else None
        runtime_args.vision_session_id = case.vision.session_id if case.vision is not None else None
        runtime_args.vision_channel = case.vision.channel if case.vision is not None else DEFAULT_FRAME_CHANNEL
        runtime_args.vision_layout_id = case.vision.layout_id if case.vision is not None else DEFAULT_LAYOUT_ID
        return _build_replay_model_from_args(runtime_args)

    report = run_replay_eval_suite(
        suite,
        output_dir=args.output_dir,
        report_path=args.report,
        model_factory=_model_factory,
        provider_name=provider_name,
    )
    print(f"[REPLAY_EVAL] suite={suite.suite_id}")
    print(f"[REPLAY_EVAL] summary={json.dumps(report['summary'], ensure_ascii=False, sort_keys=True)}")
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

    sub.add_parser(
        "live-dcs",
        help="Run live DCS tutor loop against telemetry/vision sidecars",
    )

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
    rep_bios.add_argument(
        "--max-overlay-targets",
        type=int,
        default=1,
        help="Maximum number of overlay targets allowed per help cycle (default 1)",
    )
    replay_cold_start_default = parse_env_bool(ENV_COLD_START_PRODUCTION, default=False)
    replay_cold_start_group = rep_bios.add_mutually_exclusive_group()
    replay_cold_start_group.add_argument(
        "--cold-start-production",
        dest="cold_start_production",
        action="store_true",
        help="Enable cold-start production mode (requires valid knowledge source policy).",
    )
    replay_cold_start_group.add_argument(
        "--no-cold-start-production",
        dest="cold_start_production",
        action="store_false",
        help="Disable cold-start production mode even if env default is enabled.",
    )
    rep_bios.set_defaults(cold_start_production=replay_cold_start_default)
    rep_bios.add_argument(
        "--knowledge-source-policy",
        default=None,
        help=(
            "knowledge_source_policy.yaml path. In cold-start production mode, omitted path "
            "falls back to repository-checkout knowledge_source_policy.yaml when available. "
            "Providing this flag enables policy filtering in any mode."
        ),
    )

    rep_bios.add_argument("--output", help="Event log JSONL output path")
    rep_bios.add_argument("--session-id", default=None, help="Optional event session id")
    rep_bios.add_argument(
        "--vision-saved-games-dir",
        default=None,
        help="Saved Games/<variant> root for historical vision sidecar replay.",
    )
    rep_bios.add_argument(
        "--vision-session-id",
        default=None,
        help="Frame sidecar session id. Defaults to --session-id when omitted.",
    )
    rep_bios.add_argument("--vision-channel", default=DEFAULT_FRAME_CHANNEL, help="Vision frame channel name")
    rep_bios.add_argument(
        "--vision-layout-id",
        default=DEFAULT_LAYOUT_ID,
        help="Expected sidecar vision layout id",
    )
    rep_bios.add_argument(
        "--vision-sync-window-ms",
        type=parse_non_negative_int_arg,
        default=0,
        help="Frame selection sync window in milliseconds (0 uses replay default).",
    )
    rep_bios.add_argument(
        "--vision-trigger-wait-ms",
        type=parse_non_negative_int_arg,
        default=0,
        help="Reserved extra wait budget for trigger-frame arrival.",
    )
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

    _add_model_args(rep_bios, default_provider="stub", provider_choices=["stub", "openai_compat", "ollama"])
    rep_bios.add_argument(
        "--scenario-profile",
        choices=sorted(SUPPORTED_SCENARIO_PROFILES),
        default=DEFAULT_SCENARIO_PROFILE,
        help="Scenario profile to parameterize pack gate branches (default: airfield).",
    )

    rec_vlm = sub.add_parser("record-vlm", help="Record DCS-BIOS JSONL while monitoring VLM frame sidecar growth")
    rec_vlm.add_argument("--output", required=True, help="Path to dcs_bios_raw.jsonl")
    rec_vlm.add_argument("--host", default="0.0.0.0", help="DCS-BIOS UDP bind host (default 0.0.0.0)")
    rec_vlm.add_argument("--port", type=int, default=7790, help="DCS-BIOS UDP bind port (default 7790)")
    rec_vlm.add_argument("--timeout", type=float, default=0.2, help="Receiver socket timeout seconds")
    rec_vlm.add_argument("--session-id", default=None, help="Optional shared session id for BIOS log and frame sidecar")
    rec_vlm.add_argument(
        "--vision-saved-games-dir",
        required=True,
        help="Saved Games/<variant> root that contains SimTutor/frames/<session>/<channel>/frames.jsonl.",
    )
    rec_vlm.add_argument(
        "--vision-session-id",
        default=None,
        help="Frame sidecar session id. Defaults to --session-id when omitted.",
    )
    rec_vlm.add_argument("--vision-channel", default=DEFAULT_FRAME_CHANNEL, help="Vision frame channel name")
    rec_vlm.add_argument("--vision-layout-id", default=DEFAULT_LAYOUT_ID, help="Expected vision layout id")
    rec_vlm.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Record duration seconds (0 means use --max-frames).",
    )
    rec_vlm.add_argument(
        "--max-frames",
        type=parse_non_negative_int_arg,
        default=0,
        help="Max BIOS frames to record (0 means use --duration).",
    )
    merge_group = rec_vlm.add_mutually_exclusive_group()
    merge_group.add_argument(
        "--merge-full-state",
        dest="merge_full_state",
        action="store_true",
        help="Merge BIOS deltas to full state before writing each JSONL line (default: enabled).",
    )
    merge_group.add_argument(
        "--no-merge-full-state",
        dest="merge_full_state",
        action="store_false",
        help="Write delta-only BIOS state instead of a merged full-state payload.",
    )
    rec_vlm.set_defaults(merge_full_state=True)

    rep_eval = sub.add_parser("replay-eval", help="Run fixed replay regression suite and emit a stable report")
    rep_eval.add_argument(
        "--suite",
        default="replay_eval/fa18c_startup_v04/suite.yaml",
        help="Replay evaluation suite YAML path",
    )
    rep_eval.add_argument(
        "--output-dir",
        default="logs/replay_eval",
        help="Directory for per-case replay event logs and the report",
    )
    rep_eval.add_argument("--report", default=None, help="Optional explicit report JSON path")
    _add_model_args(rep_eval, default_provider="oracle", provider_choices=["oracle", "stub", "openai_compat", "ollama"])

    args, unknown_args = parser.parse_known_args()
    if args.command != "live-dcs" and unknown_args:
        parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")
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
    if args.command == "live-dcs":
        from live_dcs import main as _live_dcs_main
        try:
            return _live_dcs_main(list(unknown_args))
        except Exception as exc:
            print(f"[LIVE_DCS] {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    if args.command == "replay-bios":
        try:
            return _run_replay_bios(args)
        except Exception as exc:
            print(f"[REPLAY_BIOS] error: {exc}")
            return 1
    if args.command == "record-vlm":
        try:
            return _run_record_vlm(args)
        except Exception as exc:
            print(f"[RECORD_VLM] error: {exc}")
            return 1
    if args.command == "replay-eval":
        try:
            return _run_replay_eval(args)
        except Exception as exc:
            print(f"[REPLAY_EVAL] error: {exc}")
            return 1
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
