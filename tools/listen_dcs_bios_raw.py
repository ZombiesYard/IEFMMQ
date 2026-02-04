from __future__ import annotations

import argparse
import json
import time

from adapters.dcs_bios.receiver import DcsBiosRawReceiver


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Listen to raw DCS-BIOS export stream and decode values.")
    parser.add_argument("--host", default="239.255.50.10", help="DCS-BIOS export host (default 239.255.50.10)")
    parser.add_argument("--port", type=int, default=5010, help="DCS-BIOS export port (default 5010)")
    parser.add_argument("--aircraft", required=True, help="Aircraft name (e.g., FA-18C_hornet)")
    parser.add_argument(
        "--control-dir",
        default="DCS/Scripts/DCS-BIOS/doc/json",
        help="Control reference directory (default DCS/Scripts/DCS-BIOS/doc/json)",
    )
    parser.add_argument("--once", action="store_true", help="Read one frame and exit")
    parser.add_argument(
        "--wait",
        type=float,
        default=5.0,
        help="Seconds to wait for a frame when --once is set (default 5.0)",
    )
    parser.add_argument(
        "--min-keys",
        type=int,
        default=200,
        help="Minimum bios keys to collect before exiting --once (default 200)",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=5,
        help="Exit if key count stops growing for N frames (default 5)",
    )
    parser.add_argument("--output", help="Write decoded frame(s) to file (JSON or JSONL)")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    output_fh = None
    try:
        if args.output:
            output_fh = open(args.output, "w", encoding="utf-8")
        with DcsBiosRawReceiver(
            host=args.host,
            port=args.port,
            aircraft=args.aircraft,
            control_reference_dir=args.control_dir,
        ) as rx:
            if args.once:
                deadline = time.time() + max(0.0, args.wait)
                obs = None
                last_payload = None
                last_key_count = 0
                stable_frames = 0
                while time.time() <= deadline:
                    obs = rx.get_observation()
                    if not obs:
                        continue
                    last_payload = obs.payload
                    bios = last_payload.get("bios", {})
                    key_count = len(bios) if isinstance(bios, dict) else 0
                    if key_count > last_key_count:
                        last_key_count = key_count
                        stable_frames = 0
                    else:
                        stable_frames += 1
                    if key_count >= args.min_keys:
                        break
                    if stable_frames >= args.stable_frames and last_key_count > 0:
                        break
                payload = last_payload if last_payload else {"status": "no data"}
                text = json.dumps(payload, ensure_ascii=False)
                print(text)
                if output_fh:
                    output_fh.write(text)
                    output_fh.write("\n")
                    output_fh.flush()
                return 0
            print(f"listening on {args.host}:{args.port} ...")
            try:
                while True:
                    obs = rx.get_observation()
                    if obs:
                        text = json.dumps(obs.payload, ensure_ascii=False)
                        print(text)
                        if output_fh:
                            output_fh.write(text)
                            output_fh.write("\n")
                            output_fh.flush()
            except KeyboardInterrupt:
                pass
        return 0
    finally:
        if output_fh:
            output_fh.close()


if __name__ == "__main__":
    raise SystemExit(main())
