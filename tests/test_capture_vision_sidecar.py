from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from adapters.vision_capture_trigger import build_capture_request_payload, parse_capture_request_payload
from live_dcs import UdpVisionCaptureNotifier
from simtutor.schemas import validate_instance
from tools.capture_vision_sidecar import (
    UdpCaptureRequestListener,
    VisionCaptureSidecar,
    VisionFrameSidecarWriter,
    _resolve_runtime_config,
    build_arg_parser,
    load_sidecar_config,
)


class DummyDatagramSocket:
    def __init__(self) -> None:
        self.bound = ("127.0.0.1", 0)
        self.timeout = None
        self.sent: list[tuple[bytes, tuple[str, int]]] = []
        self.incoming: list[tuple[bytes, tuple[str, int]]] = []
        self.closed = False

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def bind(self, addr: tuple[str, int]) -> None:
        host, port = addr
        self.bound = (host, port if port > 0 else 37995)

    def getsockname(self) -> tuple[str, int]:
        return self.bound

    def sendto(self, payload: bytes, addr: tuple[str, int]) -> None:
        self.sent.append((payload, addr))

    def recvfrom(self, _size: int) -> tuple[bytes, tuple[str, int]]:
        if not self.incoming:
            raise TimeoutError("no datagram available")
        return self.incoming.pop(0)

    def close(self) -> None:
        self.closed = True


def test_load_sidecar_config_reads_generated_lua_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "Saved Games" / "DCS" / "Scripts" / "SimTutor" / "SimTutorConfig.lua"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "return {",
                "  vision = {",
                '    layout_id = "fa18c_composite_panel_v2",',
                '    channel = "composite_panel",',
                f"    output_root = [[{tmp_path / 'Saved Games' / 'DCS' / 'SimTutor' / 'frames'}]],",
                "    capture_resolution = {",
                "      width = 3440,",
                "      height = 1440,",
                "    },",
                "  },",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_sidecar_config(config_path)

    assert loaded.layout_id == "fa18c_composite_panel_v2"
    assert loaded.channel == "composite_panel"
    assert loaded.output_root == tmp_path / "Saved Games" / "DCS" / "SimTutor" / "frames"
    assert loaded.capture_width == 3440
    assert loaded.capture_height == 1440


def test_capture_request_payload_round_trip_filters_wrong_session() -> None:
    payload = build_capture_request_payload(session_id="sess-live", reason="help")

    assert parse_capture_request_payload(payload, expected_session_id="sess-live") == {
        "intent": "capture_vision",
        "session_id": "sess-live",
        "reason": "help",
    }
    assert parse_capture_request_payload(payload, expected_session_id="sess-other") is None


def test_udp_vision_capture_notifier_sends_help_request() -> None:
    sock = DummyDatagramSocket()
    notifier = UdpVisionCaptureNotifier(session_id="sess-live", host="127.0.0.1", port=7795, sock=sock)
    try:
        notifier.notify_help()
    finally:
        notifier.close()

    payload, addr = sock.sent[0]
    assert addr == ("127.0.0.1", 7795)
    assert parse_capture_request_payload(payload, expected_session_id="sess-live") == {
        "intent": "capture_vision",
        "session_id": "sess-live",
        "reason": "help",
    }


def test_vision_frame_sidecar_writer_writes_png_and_manifest(tmp_path: Path) -> None:
    writer = VisionFrameSidecarWriter(
        output_root=tmp_path / "Saved Games" / "DCS" / "SimTutor" / "frames",
        session_id="sess-live",
        channel="composite_panel",
        layout_id="fa18c_composite_panel_v2",
        capture_callable=lambda: Image.new("RGB", (640, 360), color=(15, 20, 24)),
        clock=lambda: 1772872444.902,
    )

    frame = writer.capture_frame(reason="interval")

    image_path = Path(frame["image_path"])
    assert image_path.exists()
    manifest_lines = writer.manifest_path.read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == 1
    manifest_entry = json.loads(manifest_lines[0])
    validate_instance(manifest_entry, "vision_frame_manifest_entry")
    assert manifest_entry["frame_id"] == "1772872444902_000000"
    assert manifest_entry["width"] == 640
    assert manifest_entry["height"] == 360
    assert manifest_entry["source_session_id"] == "sess-live"


def test_udp_capture_listener_and_sidecar_help_capture(tmp_path: Path) -> None:
    sock = DummyDatagramSocket()
    sock.incoming.append((build_capture_request_payload(session_id="sess-live"), ("127.0.0.1", 50123)))
    listener = UdpCaptureRequestListener(
        session_id="sess-live",
        host="127.0.0.1",
        port=0,
        timeout=0.05,
        sock=sock,
    )
    writer = VisionFrameSidecarWriter(
        output_root=tmp_path / "Saved Games" / "DCS" / "SimTutor" / "frames",
        session_id="sess-live",
        channel="composite_panel",
        layout_id="fa18c_composite_panel_v2",
        capture_callable=lambda: Image.new("RGB", (320, 180), color=(0, 0, 0)),
        clock=lambda: 1772872445.010,
    )
    sidecar = VisionCaptureSidecar(
        writer=writer,
        request_listener=listener,
        capture_fps=1000.0,
        sleep=lambda _seconds: None,
    )
    try:
        stats = sidecar.run(duration_s=0.2, max_frames=1)
    finally:
        listener.close()

    assert stats.frames_written == 1
    assert stats.help_trigger_captures == 1
    assert stats.interval_captures == 0
    assert stats.last_frame_id == "1772872445010_000000"


def test_sidecar_capture_fps_zero_disables_interval_capture(tmp_path: Path) -> None:
    writer = VisionFrameSidecarWriter(
        output_root=tmp_path / "Saved Games" / "DCS" / "SimTutor" / "frames",
        session_id="sess-live",
        channel="composite_panel",
        layout_id="fa18c_composite_panel_v2",
        capture_callable=lambda: Image.new("RGB", (320, 180), color=(0, 0, 0)),
        clock=lambda: 1772872445.010,
    )
    sidecar = VisionCaptureSidecar(
        writer=writer,
        request_listener=None,
        capture_fps=0.0,
        sleep=lambda _seconds: None,
    )

    stats = sidecar.run(duration_s=0.05, max_frames=0)

    assert stats.frames_written == 0
    assert stats.help_trigger_captures == 0
    assert stats.interval_captures == 0
    assert stats.last_frame_id is None
    assert writer.manifest_path.exists() is False


def test_sidecar_cli_defaults_to_help_trigger_only_capture() -> None:
    parser = build_arg_parser()

    args = parser.parse_args(["--saved-games-dir", "C:\\Saved Games\\DCS", "--session-id", "sess-live"])

    assert args.capture_fps == 0.0


def test_resolve_runtime_config_preserves_zero_sized_cli_override(tmp_path: Path) -> None:
    config_path = tmp_path / "Saved Games" / "DCS" / "Scripts" / "SimTutor" / "SimTutorConfig.lua"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "return {",
                "  vision = {",
                '    layout_id = "fa18c_composite_panel_v2",',
                '    channel = "composite_panel",',
                f"    output_root = [[{tmp_path / 'Saved Games' / 'DCS' / 'SimTutor' / 'frames'}]],",
                "    capture_resolution = {",
                "      width = 3440,",
                "      height = 1440,",
                "    },",
                "  },",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--saved-games-dir",
            str(tmp_path / "Saved Games" / "DCS"),
            "--session-id",
            "sess-live",
            "--config-path",
            str(config_path),
            "--capture-width",
            "0",
            "--capture-height",
            "0",
        ]
    )

    resolved, resolved_path = _resolve_runtime_config(args)

    assert resolved_path == config_path
    assert resolved.capture_width == 0
    assert resolved.capture_height == 0
