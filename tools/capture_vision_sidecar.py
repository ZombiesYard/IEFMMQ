"""
Capture the frozen F/A-18C composite-panel screen region into the vision sidecar
manifest expected by live/replay VLM pipelines.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import socket
import time
from typing import Any, Callable

from PIL import Image

from adapters.vision_capture_trigger import (
    DEFAULT_VISION_CAPTURE_TRIGGER_HOST,
    DEFAULT_VISION_CAPTURE_TRIGGER_PORT,
    parse_capture_request_payload,
)
from adapters.vision_frames import DEFAULT_FRAME_MANIFEST_NAME, build_frame_filename, build_frame_id
from simtutor.schemas import validate_instance

DEFAULT_CAPTURE_FPS = 0.0


@dataclass(frozen=True)
class SidecarConfig:
    output_root: Path
    channel: str
    layout_id: str
    capture_width: int
    capture_height: int


@dataclass(frozen=True)
class CaptureStats:
    frames_written: int
    help_trigger_captures: int
    interval_captures: int
    last_frame_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_written": self.frames_written,
            "help_trigger_captures": self.help_trigger_captures,
            "interval_captures": self.interval_captures,
            "last_frame_id": self.last_frame_id,
        }


def _config_path(saved_games_dir: Path) -> Path:
    return saved_games_dir / "Scripts" / "SimTutor" / "SimTutorConfig.lua"


def load_sidecar_config(config_path: Path) -> SidecarConfig:
    content = config_path.read_text(encoding="utf-8")
    output_root = _match_config_text(content, r"output_root\s*=\s*\[\[(.*?)\]\]", "vision.output_root")
    channel = _match_config_text(content, r'channel\s*=\s*"([^"]+)"', "vision.channel")
    layout_id = _match_config_text(content, r'layout_id\s*=\s*"([^"]+)"', "vision.layout_id")
    capture_match = re.search(
        r"capture_resolution\s*=\s*\{\s*width\s*=\s*(\d+)\s*,\s*height\s*=\s*(\d+)\s*,\s*\}",
        content,
        flags=re.DOTALL,
    )
    if capture_match is None:
        raise ValueError(f"missing vision.capture_resolution in {config_path}")
    return SidecarConfig(
        output_root=Path(output_root),
        channel=channel,
        layout_id=layout_id,
        capture_width=int(capture_match.group(1)),
        capture_height=int(capture_match.group(2)),
    )


def _match_config_text(content: str, pattern: str, label: str) -> str:
    match = re.search(pattern, content, flags=re.DOTALL)
    if match is None or not match.group(1).strip():
        raise ValueError(f"missing {label} in SimTutorConfig.lua")
    return match.group(1).strip()


def capture_screen_region(*, width: int, height: int) -> Image.Image:
    try:
        from PIL import ImageGrab
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PIL.ImageGrab is required for live screen capture") from exc
    bbox = (0, 0, int(width), int(height))
    return ImageGrab.grab(bbox=bbox, all_screens=False)


class UdpCaptureRequestListener:
    def __init__(
        self,
        *,
        session_id: str,
        host: str = DEFAULT_VISION_CAPTURE_TRIGGER_HOST,
        port: int = DEFAULT_VISION_CAPTURE_TRIGGER_PORT,
        timeout: float = 0.1,
        sock: Any | None = None,
    ) -> None:
        if int(port) < 0:
            raise ValueError("port must be >= 0")
        self.session_id = str(session_id).strip()
        if not self.session_id:
            raise ValueError("session_id must be a non-empty string")
        self.host = str(host)
        self.port = int(port)
        self.timeout = max(0.01, float(timeout))
        self._sock: socket.socket | None = None
        if sock is not None:
            sock.settimeout(self.timeout)
            sock.bind((self.host, self.port))
            self._sock = sock
        elif self.port > 0:
            actual_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            actual_sock.settimeout(self.timeout)
            actual_sock.bind((self.host, self.port))
            self._sock = actual_sock

    @property
    def bound_port(self) -> int:
        if self._sock is None:
            return 0
        return int(self._sock.getsockname()[1])

    def poll(self) -> dict[str, Any] | None:
        if self._sock is None:
            return None
        try:
            payload, _addr = self._sock.recvfrom(4096)
        except socket.timeout:
            return None
        except OSError:
            return None
        return parse_capture_request_payload(payload, expected_session_id=self.session_id)

    def close(self) -> None:
        if self._sock is None:
            return
        try:
            self._sock.close()
        except OSError:
            pass
        self._sock = None


class VisionFrameSidecarWriter:
    def __init__(
        self,
        *,
        output_root: Path,
        session_id: str,
        channel: str,
        layout_id: str,
        capture_callable: Callable[[], Image.Image],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.output_root = Path(output_root).expanduser()
        self.session_id = str(session_id).strip()
        self.channel = str(channel).strip()
        self.layout_id = str(layout_id).strip()
        if not self.session_id:
            raise ValueError("session_id must be non-empty")
        if not self.channel:
            raise ValueError("channel must be non-empty")
        if not self.layout_id:
            raise ValueError("layout_id must be non-empty")
        self.capture_callable = capture_callable
        self.clock = clock
        self._frame_seq = 0

    @property
    def channel_dir(self) -> Path:
        return self.output_root / self.session_id / self.channel

    @property
    def manifest_path(self) -> Path:
        return self.channel_dir / DEFAULT_FRAME_MANIFEST_NAME

    def capture_frame(self, *, reason: str) -> dict[str, Any]:
        image = self.capture_callable()
        if not isinstance(image, Image.Image):
            raise TypeError("capture_callable must return a PIL.Image.Image")
        self.channel_dir.mkdir(parents=True, exist_ok=True)

        capture_wall_ms = int(round(float(self.clock()) * 1000.0))
        frame_seq = self._frame_seq
        self._frame_seq += 1
        frame_id = build_frame_id(capture_wall_ms=capture_wall_ms, frame_seq=frame_seq)
        filename = build_frame_filename(capture_wall_ms=capture_wall_ms, frame_seq=frame_seq)
        final_path = self.channel_dir / filename
        temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
        image.save(temp_path, format="PNG")
        temp_path.replace(final_path)

        entry = {
            "schema_version": "v2",
            "frame_id": frame_id,
            "capture_wall_ms": capture_wall_ms,
            "frame_seq": frame_seq,
            "channel": self.channel,
            "layout_id": self.layout_id,
            "image_path": str(final_path.resolve()),
            "width": int(image.width),
            "height": int(image.height),
            "source_session_id": self.session_id,
        }
        validate_instance(entry, "vision_frame_manifest_entry")
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            handle.flush()
        return {
            **entry,
            "capture_reason": reason,
            "timestamp": datetime.fromtimestamp(capture_wall_ms / 1000.0, tz=timezone.utc).isoformat(),
        }


class VisionCaptureSidecar:
    def __init__(
        self,
        *,
        writer: VisionFrameSidecarWriter,
        request_listener: UdpCaptureRequestListener | None = None,
        capture_fps: float = DEFAULT_CAPTURE_FPS,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.writer = writer
        self.request_listener = request_listener
        normalized_capture_fps = float(capture_fps)
        self.capture_interval_s = None if normalized_capture_fps <= 0 else 1.0 / normalized_capture_fps
        self.monotonic = monotonic
        self.sleep = sleep

    def run(
        self,
        *,
        duration_s: float = 0.0,
        max_frames: int = 0,
        idle_sleep_s: float = 0.02,
    ) -> CaptureStats:
        if duration_s < 0:
            raise ValueError("duration_s must be >= 0")
        if max_frames < 0:
            raise ValueError("max_frames must be >= 0")
        frames_written = 0
        help_trigger_captures = 0
        interval_captures = 0
        last_frame_id: str | None = None
        start = self.monotonic()
        next_interval_capture = start if self.capture_interval_s is not None else None

        while True:
            now = self.monotonic()
            if duration_s > 0 and (now - start) >= duration_s:
                break

            request = self.request_listener.poll() if self.request_listener is not None else None
            reason: str | None = None
            if request is not None:
                reason = str(request.get("reason") or "help")
            elif next_interval_capture is not None and now >= next_interval_capture:
                reason = "interval"

            if reason is not None:
                frame = self.writer.capture_frame(reason=reason)
                last_frame_id = str(frame["frame_id"])
                frames_written += 1
                if reason == "interval":
                    interval_captures += 1
                else:
                    help_trigger_captures += 1
                if self.capture_interval_s is not None:
                    next_interval_capture = self.monotonic() + self.capture_interval_s
                if max_frames > 0 and frames_written >= max_frames:
                    break
                continue

            if max_frames > 0 and frames_written >= max_frames:
                break
            self.sleep(max(0.0, idle_sleep_s))

        return CaptureStats(
            frames_written=frames_written,
            help_trigger_captures=help_trigger_captures,
            interval_captures=interval_captures,
            last_frame_id=last_frame_id,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture the SimTutor composite-panel sidecar frames on the DCS host.")
    parser.add_argument("--saved-games-dir", required=True, help="Saved Games/<variant> root containing Scripts/SimTutor/SimTutorConfig.lua.")
    parser.add_argument("--session-id", required=True, help="Vision sidecar session id written under SimTutor/frames/<session>/...")
    parser.add_argument("--config-path", default=None, help="Optional explicit SimTutorConfig.lua path.")
    parser.add_argument("--output-root", default=None, help="Optional override for vision.output_root.")
    parser.add_argument("--channel", default=None, help="Optional override for vision.channel.")
    parser.add_argument("--layout-id", default=None, help="Optional override for vision.layout_id.")
    parser.add_argument("--capture-width", type=int, default=None, help="Optional override for capture_resolution.width.")
    parser.add_argument("--capture-height", type=int, default=None, help="Optional override for capture_resolution.height.")
    parser.add_argument(
        "--capture-fps",
        type=float,
        default=DEFAULT_CAPTURE_FPS,
        help="Continuous low-fps capture rate; use 0 for help-trigger captures only.",
    )
    parser.add_argument("--duration", type=float, default=0.0, help="Optional run duration in seconds.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional max frame count before exit.")
    parser.add_argument("--trigger-host", default=DEFAULT_VISION_CAPTURE_TRIGGER_HOST, help="UDP host for help-trigger capture requests.")
    parser.add_argument("--trigger-port", type=int, default=DEFAULT_VISION_CAPTURE_TRIGGER_PORT, help="UDP port for help-trigger capture requests (0 disables).")
    parser.add_argument("--trigger-timeout", type=float, default=0.1, help="UDP receive timeout in seconds.")
    return parser


def _resolve_runtime_config(args: argparse.Namespace) -> tuple[SidecarConfig, Path]:
    saved_games_dir = Path(args.saved_games_dir).expanduser()
    config_path = Path(args.config_path).expanduser() if args.config_path else _config_path(saved_games_dir)
    loaded = load_sidecar_config(config_path)
    output_root = Path(args.output_root).expanduser() if args.output_root else loaded.output_root
    channel = str(args.channel).strip() if args.channel else loaded.channel
    layout_id = str(args.layout_id).strip() if args.layout_id else loaded.layout_id
    capture_width = int(args.capture_width) if args.capture_width is not None else loaded.capture_width
    capture_height = int(args.capture_height) if args.capture_height is not None else loaded.capture_height
    return (
        SidecarConfig(
            output_root=output_root,
            channel=channel,
            layout_id=layout_id,
            capture_width=capture_width,
            capture_height=capture_height,
        ),
        config_path,
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        config, config_path = _resolve_runtime_config(args)
        writer = VisionFrameSidecarWriter(
            output_root=config.output_root,
            session_id=args.session_id,
            channel=config.channel,
            layout_id=config.layout_id,
            capture_callable=lambda: capture_screen_region(
                width=config.capture_width,
                height=config.capture_height,
            ),
        )
        listener = UdpCaptureRequestListener(
            session_id=args.session_id,
            host=args.trigger_host,
            port=args.trigger_port,
            timeout=args.trigger_timeout,
        )
        try:
            sidecar = VisionCaptureSidecar(
                writer=writer,
                request_listener=listener,
                capture_fps=args.capture_fps,
            )
            print(
                "[CAPTURE_VISION] config="
                + json.dumps(
                    {
                        "config_path": str(config_path),
                        "output_root": str(config.output_root),
                        "channel": config.channel,
                        "layout_id": config.layout_id,
                        "capture_width": config.capture_width,
                        "capture_height": config.capture_height,
                        "session_id": args.session_id,
                        "trigger_host": args.trigger_host,
                        "trigger_port": listener.bound_port,
                        "capture_fps": args.capture_fps,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            stats = sidecar.run(duration_s=args.duration, max_frames=args.max_frames)
        finally:
            listener.close()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[CAPTURE_VISION] failed: {type(exc).__name__}: {exc}")
        return 1

    print(f"[CAPTURE_VISION] stats={json.dumps(stats.to_dict(), ensure_ascii=False, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
