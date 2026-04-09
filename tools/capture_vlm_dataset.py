"""
Help-triggered dataset capture tool for VLM fine-tuning.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

from PIL import Image

from adapters.vision_capture_trigger import (
    DEFAULT_VISION_CAPTURE_TRIGGER_HOST,
    DEFAULT_VISION_CAPTURE_TRIGGER_PORT,
)
from adapters.windows_global_help_trigger import (
    DEFAULT_GLOBAL_HELP_COOLDOWN_MS,
    WindowsGlobalHelpTrigger,
)
from adapters.vision_frames import (
    DEFAULT_ARTIFACT_SUFFIX,
    build_frame_filename,
    build_frame_id,
    render_vlm_ready_frame,
)
from simtutor.schemas import validate_instance
from tools.capture_vision_sidecar import (
    UdpCaptureRequestListener,
    capture_screen_region,
    load_sidecar_config,
)

DEFAULT_CAPTURE_FPS = 2.0
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / ".captures"


@dataclass(frozen=True)
class DatasetCaptureConfig:
    session_id: str
    output_root: Path
    channel: str
    layout_id: str
    screen_width: int
    screen_height: int
    render_vlm_artifacts: bool


@dataclass(frozen=True)
class DatasetCaptureStats:
    frames_written: int
    help_start_captures: int
    interval_captures: int
    started: bool
    last_frame_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_written": self.frames_written,
            "help_start_captures": self.help_start_captures,
            "interval_captures": self.interval_captures,
            "started": self.started,
            "last_frame_id": self.last_frame_id,
        }


def _default_config_path(saved_games_dir: Path) -> Path:
    return saved_games_dir / "Scripts" / "SimTutor" / "SimTutorConfig.lua"


class DatasetFrameWriter:
    def __init__(
        self,
        *,
        output_root: Path,
        session_id: str,
        channel: str,
        layout_id: str,
        capture_callable: Callable[[], Image.Image],
        render_vlm_artifacts: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.output_root = Path(output_root).expanduser().resolve()
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
        self.render_vlm_artifacts = bool(render_vlm_artifacts)
        self.clock = clock
        self._frame_seq = 0

    @property
    def session_dir(self) -> Path:
        return self.output_root / self.session_id

    @property
    def raw_dir(self) -> Path:
        return self.session_dir / "raw"

    @property
    def artifact_dir(self) -> Path:
        return self.session_dir / "artifacts"

    @property
    def manifest_path(self) -> Path:
        return self.session_dir / "frames.jsonl"

    @property
    def capture_index_path(self) -> Path:
        return self.session_dir / "capture_index.jsonl"

    def capture_frame(self, *, reason: str) -> dict[str, Any]:
        image = self.capture_callable()
        if not isinstance(image, Image.Image):
            raise TypeError("capture_callable must return a PIL.Image.Image")

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        if self.render_vlm_artifacts:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)

        capture_wall_ms = int(round(float(self.clock()) * 1000.0))
        frame_seq = self._frame_seq
        self._frame_seq += 1
        frame_id = build_frame_id(capture_wall_ms=capture_wall_ms, frame_seq=frame_seq)
        filename = build_frame_filename(capture_wall_ms=capture_wall_ms, frame_seq=frame_seq)

        raw_path = self.raw_dir / filename
        temp_path = raw_path.with_suffix(raw_path.suffix + ".tmp")
        image.save(temp_path, format="PNG")
        temp_path.replace(raw_path)

        artifact_path: Path | None = None
        artifact_metadata: dict[str, Any] | None = None
        if self.render_vlm_artifacts:
            artifact_path = self.artifact_dir / f"{raw_path.stem}{DEFAULT_ARTIFACT_SUFFIX}"
            artifact_metadata = render_vlm_ready_frame(raw_path, artifact_path)

        manifest_entry = {
            "schema_version": "v2",
            "frame_id": frame_id,
            "capture_wall_ms": capture_wall_ms,
            "frame_seq": frame_seq,
            "channel": self.channel,
            "layout_id": self.layout_id,
            "image_path": str(raw_path.resolve()),
            "width": int(image.width),
            "height": int(image.height),
            "source_session_id": self.session_id,
        }
        validate_instance(manifest_entry, "vision_frame_manifest_entry")
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
            handle.flush()

        index_entry = {
            "frame_id": frame_id,
            "capture_wall_ms": capture_wall_ms,
            "raw_image_path": str(raw_path.resolve()),
            "artifact_image_path": str(artifact_path.resolve()) if artifact_path is not None else None,
            "capture_reason": str(reason),
            "session_id": self.session_id,
            "channel": self.channel,
            "layout_id": self.layout_id,
        }
        with self.capture_index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(index_entry, ensure_ascii=False) + "\n")
            handle.flush()

        return {
            **manifest_entry,
            "capture_reason": str(reason),
            "artifact_image_path": index_entry["artifact_image_path"],
            "artifact_metadata": artifact_metadata,
            "timestamp": datetime.fromtimestamp(capture_wall_ms / 1000.0, tz=timezone.utc).isoformat(),
        }


class CaptureEventSource:
    def poll(self) -> dict[str, Any] | None:
        raise NotImplementedError

    def close(self) -> None:
        return


class GlobalHelpEventSource(CaptureEventSource):
    def __init__(
        self,
        *,
        hotkey: str,
        modifiers: str = "",
        cooldown_ms: int = DEFAULT_GLOBAL_HELP_COOLDOWN_MS,
        trigger: WindowsGlobalHelpTrigger | None = None,
    ) -> None:
        self.hotkey = str(hotkey).strip()
        self.modifiers = str(modifiers).strip()
        self.cooldown_ms = int(cooldown_ms)
        self._trigger = (
            trigger
            if trigger is not None
            else WindowsGlobalHelpTrigger(
                hotkey=self.hotkey,
                modifiers=self.modifiers,
                cooldown_ms=self.cooldown_ms,
            )
        )
        self.hotkey_label = getattr(self._trigger, "hotkey_label", self.hotkey)
        self._started = False

    def start(self) -> None:
        self._trigger.start()
        self._started = True

    def poll(self) -> dict[str, Any] | None:
        if hasattr(self._trigger, "poll") and self._trigger.poll():
            return {"reason": "help", "source": "global_hotkey", "hotkey_label": self.hotkey_label}
        return None

    def request_stop(self) -> None:
        if hasattr(self._trigger, "request_stop"):
            self._trigger.request_stop()

    def close(self) -> None:
        if not self._started:
            return
        self._trigger.close()


class UdpEventSource(CaptureEventSource):
    def __init__(self, listener: UdpCaptureRequestListener) -> None:
        self.listener = listener

    def poll(self) -> dict[str, Any] | None:
        payload = self.listener.poll()
        if payload is None:
            return None
        return {
            "reason": str(payload.get("reason") or "help"),
            "source": "udp",
        }

    def close(self) -> None:
        self.listener.close()


class HelpTriggeredDatasetCapture:
    def __init__(
        self,
        *,
        writer: DatasetFrameWriter,
        event_sources: list[CaptureEventSource] | None = None,
        capture_fps: float = DEFAULT_CAPTURE_FPS,
        start_on_launch: bool = False,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        print_frame: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        normalized_capture_fps = float(capture_fps)
        if normalized_capture_fps <= 0:
            raise ValueError("capture_fps must be > 0")
        self.writer = writer
        self.event_sources = list(event_sources or [])
        self.capture_interval_s = 1.0 / normalized_capture_fps
        self.start_on_launch = bool(start_on_launch)
        self.monotonic = monotonic
        self.sleep = sleep
        self.print_frame = print_frame

    def run(
        self,
        *,
        duration_s: float = 0.0,
        max_frames: int = 0,
        idle_sleep_s: float = 0.02,
    ) -> DatasetCaptureStats:
        if duration_s < 0:
            raise ValueError("duration_s must be >= 0")
        if max_frames < 0:
            raise ValueError("max_frames must be >= 0")

        frames_written = 0
        help_start_captures = 0
        interval_captures = 0
        last_frame_id: str | None = None
        start = self.monotonic()
        active = self.start_on_launch
        next_capture = start if active else None
        started = active

        while True:
            now = self.monotonic()
            if duration_s > 0 and (now - start) >= duration_s:
                break

            request = None
            for event_source in self.event_sources:
                request = event_source.poll()
                if request is not None:
                    break
            reason: str | None = None
            if request is not None and not active:
                active = True
                started = True
                reason = "help_start"
            elif active and next_capture is not None and now >= next_capture:
                reason = "interval"

            if reason is not None:
                frame = self.writer.capture_frame(reason=reason)
                last_frame_id = str(frame["frame_id"])
                if self.print_frame is not None:
                    self.print_frame(frame)
                frames_written += 1
                if reason == "help_start":
                    help_start_captures += 1
                else:
                    interval_captures += 1
                next_capture = self.monotonic() + self.capture_interval_s
                if max_frames > 0 and frames_written >= max_frames:
                    break
                continue

            if max_frames > 0 and frames_written >= max_frames:
                break
            self.sleep(max(0.0, idle_sleep_s))

        return DatasetCaptureStats(
            frames_written=frames_written,
            help_start_captures=help_start_captures,
            interval_captures=interval_captures,
            started=started,
            last_frame_id=last_frame_id,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture a VLM fine-tuning dataset after the first help trigger.")
    parser.add_argument("--session-id", required=True, help="Dataset capture session id.")
    parser.add_argument("--saved-games-dir", default=None, help="Saved Games/<variant> root containing SimTutorConfig.lua.")
    parser.add_argument("--config-path", default=None, help="Optional explicit SimTutorConfig.lua path.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Dataset capture output root.")
    parser.add_argument("--fps", type=float, default=DEFAULT_CAPTURE_FPS, help="Continuous capture FPS after help starts.")
    parser.add_argument(
        "--global-help-hotkey",
        default="X1",
        help="Optional Windows global help hotkey: ESC, F1-F24, X1/MOUSE4, X2/MOUSE5. Empty disables.",
    )
    parser.add_argument("--global-help-modifiers", default="", help="Optional modifiers joined by +, e.g. Ctrl+Shift.")
    parser.add_argument(
        "--global-help-cooldown-ms",
        type=int,
        default=DEFAULT_GLOBAL_HELP_COOLDOWN_MS,
        help="Debounce window for the Windows global help trigger in milliseconds.",
    )
    parser.add_argument("--help-trigger-host", default=DEFAULT_VISION_CAPTURE_TRIGGER_HOST, help="UDP host for help-trigger capture requests.")
    parser.add_argument("--help-trigger-port", type=int, default=DEFAULT_VISION_CAPTURE_TRIGGER_PORT, help="UDP port for help-trigger capture requests.")
    parser.add_argument("--trigger-timeout", type=float, default=0.1, help="UDP receive timeout in seconds.")
    parser.add_argument("--screen-width", type=int, default=None, help="Optional override for capture width.")
    parser.add_argument("--screen-height", type=int, default=None, help="Optional override for capture height.")
    parser.add_argument("--channel", default=None, help="Optional override for vision.channel.")
    parser.add_argument("--layout-id", default=None, help="Optional override for vision.layout_id.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional max frame count before exit.")
    parser.add_argument("--duration-s", type=float, default=0.0, help="Optional run duration in seconds.")
    parser.add_argument("--start-on-launch", action="store_true", help="Start capturing immediately without waiting for help.")
    parser.add_argument(
        "--no-render-vlm-artifacts",
        action="store_true",
        help="Disable VLM-ready artifact rendering and only keep raw frames.",
    )
    return parser


def _resolve_runtime_config(args: argparse.Namespace) -> tuple[DatasetCaptureConfig, Path]:
    if not args.config_path and not args.saved_games_dir:
        raise ValueError("either --saved-games-dir or --config-path must be provided")
    config_path = (
        Path(args.config_path).expanduser()
        if args.config_path
        else _default_config_path(Path(args.saved_games_dir).expanduser())
    )
    loaded = load_sidecar_config(config_path)
    screen_width = int(args.screen_width) if args.screen_width is not None else loaded.capture_width
    screen_height = int(args.screen_height) if args.screen_height is not None else loaded.capture_height
    if screen_width <= 0:
        raise ValueError("screen_width must be > 0")
    if screen_height <= 0:
        raise ValueError("screen_height must be > 0")
    return (
        DatasetCaptureConfig(
            session_id=str(args.session_id).strip(),
            output_root=Path(args.output_root).expanduser(),
            channel=str(args.channel).strip() if args.channel else loaded.channel,
            layout_id=str(args.layout_id).strip() if args.layout_id else loaded.layout_id,
            screen_width=screen_width,
            screen_height=screen_height,
            render_vlm_artifacts=not bool(args.no_render_vlm_artifacts),
        ),
        config_path,
    )


def _print_frame_event(frame: dict[str, Any]) -> None:
    print(
        "[CAPTURE_VLM_DATASET] frame="
        + json.dumps(
            {
                "frame_id": frame.get("frame_id"),
                "reason": frame.get("capture_reason"),
                "raw_image_path": frame.get("image_path"),
                "artifact_image_path": frame.get("artifact_image_path"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        config, config_path = _resolve_runtime_config(args)
        writer = DatasetFrameWriter(
            output_root=config.output_root,
            session_id=config.session_id,
            channel=config.channel,
            layout_id=config.layout_id,
            render_vlm_artifacts=config.render_vlm_artifacts,
            capture_callable=lambda: capture_screen_region(
                width=config.screen_width,
                height=config.screen_height,
            ),
        )
        event_sources: list[CaptureEventSource] = []
        global_hotkey_source: GlobalHelpEventSource | None = None
        if str(args.global_help_hotkey).strip():
            if sys.platform != "win32":
                raise RuntimeError("--global-help-hotkey is only supported on Windows")
            global_hotkey_source = GlobalHelpEventSource(
                hotkey=args.global_help_hotkey,
                modifiers=args.global_help_modifiers,
                cooldown_ms=args.global_help_cooldown_ms,
            )
            global_hotkey_source.start()
            event_sources.append(global_hotkey_source)
        listener: UdpCaptureRequestListener | None = None
        if int(args.help_trigger_port) > 0:
            listener = UdpCaptureRequestListener(
                session_id=config.session_id,
                host=args.help_trigger_host,
                port=args.help_trigger_port,
                timeout=args.trigger_timeout,
            )
            event_sources.append(UdpEventSource(listener))
        try:
            runner = HelpTriggeredDatasetCapture(
                writer=writer,
                event_sources=event_sources,
                capture_fps=args.fps,
                start_on_launch=args.start_on_launch,
                print_frame=_print_frame_event,
            )
            print(
                "[CAPTURE_VLM_DATASET] config="
                + json.dumps(
                    {
                        "config_path": str(config_path),
                        "output_root": str(config.output_root),
                        "session_id": config.session_id,
                        "channel": config.channel,
                        "layout_id": config.layout_id,
                        "screen_width": config.screen_width,
                        "screen_height": config.screen_height,
                        "fps": args.fps,
                        "global_help_hotkey": getattr(global_hotkey_source, "hotkey_label", None),
                        "trigger_host": args.help_trigger_host,
                        "trigger_port": listener.bound_port if listener is not None else 0,
                        "render_vlm_artifacts": config.render_vlm_artifacts,
                        "start_on_launch": bool(args.start_on_launch),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            if global_hotkey_source is not None:
                print(
                    f"[CAPTURE_VLM_DATASET] press {global_hotkey_source.hotkey_label} once to start capture, "
                    "then press Ctrl+C to stop"
                )
            stats = runner.run(duration_s=args.duration_s, max_frames=args.max_frames)
        finally:
            if listener is not None:
                listener.close()
            if global_hotkey_source is not None:
                global_hotkey_source.close()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[CAPTURE_VLM_DATASET] failed: {type(exc).__name__}: {exc}")
        return 1

    print(f"[CAPTURE_VLM_DATASET] stats={json.dumps(stats.to_dict(), ensure_ascii=False, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
