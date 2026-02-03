from __future__ import annotations

import argparse
import json

from adapters.dcs.telemetry.receiver import DcsTelemetryReceiver


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Listen for DCS telemetry UDP frames.")
    parser.add_argument("--host", default="0.0.0.0", help="UDP bind host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7780, help="UDP bind port (default 7780)")
    parser.add_argument("--once", action="store_true", help="Read one frame and exit")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    with DcsTelemetryReceiver(host=args.host, port=args.port) as rx:
        if args.once:
            obs = rx.get_observation()
            print(json.dumps(obs.payload if obs else {"status": "no data"}, ensure_ascii=False))
            return 0

        print(f"listening on {args.host}:{args.port} ...")
        while True:
            obs = rx.get_observation()
            if obs:
                print(json.dumps(obs.payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
