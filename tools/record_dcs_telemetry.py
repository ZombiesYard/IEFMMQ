from __future__ import annotations

import argparse
import time
from pathlib import Path

from adapters.dcs.telemetry.receiver import DcsTelemetryReceiver
from core.event_store import JsonlEventStore
from core.types import Event


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record DCS telemetry to JSONL event log.")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--host", default="0.0.0.0", help="UDP bind host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7780, help="UDP bind port (default 7780)")
    parser.add_argument("--duration", type=float, default=0, help="Seconds to record (0=until max-frames)")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to record (0=until duration)")
    parser.add_argument("--print", action="store_true", help="Print each frame payload to stdout")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    count = 0
    with DcsTelemetryReceiver(host=args.host, port=args.port) as rx, JsonlEventStore(
        output, mode="w"
    ) as store:
        while True:
            obs = rx.get_observation()
            if obs:
                store.append(
                    Event(kind="observation", payload=obs.to_dict(), related_id=obs.observation_id)
                )
                count += 1
                if args.print:
                    print(obs.payload)
            if args.max_frames and count >= args.max_frames:
                break
            if args.duration and (time.time() - start) >= args.duration:
                break
    print(f"[REC] wrote {count} frames to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

