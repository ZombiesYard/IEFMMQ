from __future__ import annotations

import argparse
import json
import socket
import time


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send fake DCS telemetry UDP frames.")
    parser.add_argument("--host", default="127.0.0.1", help="UDP target host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7780, help="UDP target port (default 7780)")
    parser.add_argument("--seq", type=int, default=1, help="Starting seq number")
    parser.add_argument("--count", type=int, default=1, help="Number of frames to send")
    parser.add_argument("--hz", type=int, default=20, help="Send rate (frames per second)")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    interval = 1.0 / max(1, args.hz)
    seq = args.seq
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for i in range(args.count):
            frame = {
                "schema_version": "v2",
                "seq": seq,
                "sim_time": float(i),
                "aircraft": "FA-18C",
                "cockpit": {"speed": 250.0 + i, "gear": "up"},
            }
            payload = json.dumps(frame).encode("utf-8")
            sock.sendto(payload, (args.host, args.port))
            seq += 1
            if i < args.count - 1:
                time.sleep(interval)
    print(f"sent {args.count} frame(s) to {args.host}:{args.port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

