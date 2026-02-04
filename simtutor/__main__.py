import argparse
import json
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
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
