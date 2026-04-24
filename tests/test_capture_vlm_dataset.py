from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from adapters.vision_capture_trigger import build_capture_request_payload
from tools.capture_vision_sidecar import UdpCaptureRequestListener
from tools.capture_vlm_dataset import (
    DEFAULT_OUTPUT_ROOT,
    DatasetFrameWriter,
    GlobalHelpEventSource,
    HelpTriggeredDatasetCapture,
    UdpEventSource,
    _resolve_manual_plan,
    _resolve_runtime_config,
    _print_frame_event,
    build_arg_parser,
)


class DummyDatagramSocket:
    def __init__(self) -> None:
        self.bound = ("127.0.0.1", 0)
        self.timeout = None
        self.incoming: list[tuple[bytes, tuple[str, int]]] = []
        self.closed = False

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def bind(self, addr: tuple[str, int]) -> None:
        host, port = addr
        self.bound = (host, port if port > 0 else 37995)

    def getsockname(self) -> tuple[str, int]:
        return self.bound

    def recvfrom(self, _size: int) -> tuple[bytes, tuple[str, int]]:
        if not self.incoming:
            raise TimeoutError("no datagram available")
        return self.incoming.pop(0)

    def close(self) -> None:
        self.closed = True


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.value = float(start)

    def now(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += max(0.0, float(seconds))


class FakeTrigger:
    def __init__(self, *, hotkey_label: str = "X1") -> None:
        self.hotkey_label = hotkey_label
        self.started = False
        self.closed = False
        self.stop_requested = False
        self.events = 0

    def start(self) -> None:
        self.started = True

    def poll(self) -> bool:
        if self.events <= 0:
            return False
        self.events -= 1
        return True

    def request_stop(self) -> None:
        self.stop_requested = True

    def close(self) -> None:
        self.closed = True


def _write_lua_config(base_dir: Path) -> Path:
    config_path = base_dir / "Saved Games" / "DCS" / "Scripts" / "SimTutor" / "SimTutorConfig.lua"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "return {",
                "  vision = {",
                '    layout_id = "fa18c_composite_panel_v2",',
                '    channel = "composite_panel",',
                f"    output_root = [[{base_dir / 'Saved Games' / 'DCS' / 'SimTutor' / 'frames'}]],",
                "    capture_resolution = {",
                "      width = 1600,",
                "      height = 900,",
                "    },",
                "  },",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _capture_image() -> Image.Image:
    image = Image.new("RGB", (1600, 900), color=(20, 20, 24))
    for x in range(0, 540):
        for y in range(0, 900):
            image.putpixel((x, y), (12, 40, 52))
    return image


def test_dataset_capture_waits_for_help_before_writing_frames(tmp_path: Path) -> None:
    writer = DatasetFrameWriter(
        output_root=tmp_path / "captures",
        session_id="sess-live",
        channel="composite_panel",
        layout_id="fa18c_composite_panel_v2",
        capture_callable=_capture_image,
        clock=lambda: 1772872444.950,
    )
    clock = FakeClock()
    runner = HelpTriggeredDatasetCapture(
        writer=writer,
        event_sources=[],
        capture_fps=2.0,
        monotonic=clock.now,
        sleep=clock.sleep,
    )

    stats = runner.run(duration_s=0.3)

    assert stats.started is False
    assert stats.frames_written == 0
    assert writer.manifest_path.exists() is False
    assert writer.capture_index_path.exists() is False


def test_dataset_capture_starts_on_first_help_and_writes_artifact_and_index(tmp_path: Path) -> None:
    sock = DummyDatagramSocket()
    sock.incoming.append((build_capture_request_payload(session_id="sess-live"), ("127.0.0.1", 50123)))
    listener = UdpCaptureRequestListener(
        session_id="sess-live",
        host="127.0.0.1",
        port=0,
        timeout=0.05,
        sock=sock,
    )
    try:
        writer = DatasetFrameWriter(
            output_root=tmp_path / "captures",
            session_id="sess-live",
            channel="composite_panel",
            layout_id="fa18c_composite_panel_v2",
            capture_callable=_capture_image,
            clock=lambda: 1772872445.010,
        )
        clock = FakeClock()
        runner = HelpTriggeredDatasetCapture(
            writer=writer,
            event_sources=[UdpEventSource(listener)],
            capture_fps=2.0,
            monotonic=clock.now,
            sleep=clock.sleep,
        )

        stats = runner.run(duration_s=0.6, max_frames=2)
    finally:
        listener.close()

    assert stats.started is True
    assert stats.frames_written == 2
    assert stats.help_start_captures == 1
    assert stats.interval_captures == 1
    manifest_lines = writer.manifest_path.read_text(encoding="utf-8").splitlines()
    index_lines = writer.capture_index_path.read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == 2
    assert len(index_lines) == 2
    first_manifest = json.loads(manifest_lines[0])
    first_index = json.loads(index_lines[0])
    assert Path(first_manifest["image_path"]).exists()
    assert Path(first_index["artifact_image_path"]).exists()
    assert first_index["capture_reason"] == "help_start"
    assert first_index["frame_id"] == first_manifest["frame_id"]
    assert first_index["session_id"] == "sess-live"


def test_dataset_capture_starts_on_global_hotkey_in_single_terminal_mode(tmp_path: Path) -> None:
    fake_trigger = FakeTrigger(hotkey_label="X1")
    source = GlobalHelpEventSource(hotkey="X1", trigger=fake_trigger)
    source.start()
    fake_trigger.events = 1
    writer = DatasetFrameWriter(
        output_root=tmp_path / "captures",
        session_id="sess-live",
        channel="composite_panel",
        layout_id="fa18c_composite_panel_v2",
        capture_callable=_capture_image,
        clock=lambda: 1772872445.010,
    )
    clock = FakeClock()
    runner = HelpTriggeredDatasetCapture(
        writer=writer,
        event_sources=[source],
        capture_fps=2.0,
        monotonic=clock.now,
        sleep=clock.sleep,
    )

    stats = runner.run(duration_s=0.6, max_frames=2)
    source.close()

    assert fake_trigger.started is True
    assert fake_trigger.closed is True
    assert stats.started is True
    assert stats.help_start_captures == 1
    assert stats.interval_captures == 1


def test_print_frame_event_logs_paths(capsys) -> None:
    _print_frame_event(
        {
            "frame_id": "1772872445010_000000",
            "capture_reason": "help_start",
            "image_path": "/tmp/raw.png",
            "artifact_image_path": "/tmp/artifact.png",
        }
    )

    out = capsys.readouterr().out
    assert "1772872445010_000000" in out
    assert "help_start" in out
    assert "/tmp/raw.png" in out
    assert "/tmp/artifact.png" in out


def test_dataset_capture_index_and_output_dirs_match_expected_layout(tmp_path: Path) -> None:
    writer = DatasetFrameWriter(
        output_root=tmp_path / "captures",
        session_id="sess-live",
        channel="composite_panel",
        layout_id="fa18c_composite_panel_v2",
        capture_callable=_capture_image,
        clock=lambda: 1772872445.010,
    )

    frame = writer.capture_frame(reason="help_start")

    assert writer.raw_dir == tmp_path / "captures" / "sess-live" / "raw"
    assert writer.artifact_dir == tmp_path / "captures" / "sess-live" / "artifacts"
    assert writer.manifest_path == tmp_path / "captures" / "sess-live" / "frames.jsonl"
    assert writer.capture_index_path == tmp_path / "captures" / "sess-live" / "capture_index.jsonl"
    assert Path(frame["image_path"]).parent == writer.raw_dir
    assert Path(frame["artifact_image_path"]).parent == writer.artifact_dir


def test_dataset_capture_cli_defaults_and_runtime_config(tmp_path: Path) -> None:
    config_path = _write_lua_config(tmp_path)
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--session-id",
            "sess-live",
            "--config-path",
            str(config_path),
        ]
    )

    config, resolved_config_path = _resolve_runtime_config(args)

    assert args.fps == 2.0
    assert args.global_help_hotkey == "X1"
    assert args.output_root == str(DEFAULT_OUTPUT_ROOT)
    assert config.output_root == DEFAULT_OUTPUT_ROOT
    assert config.screen_width == 1600
    assert config.screen_height == 900
    assert config.render_vlm_artifacts is True
    assert resolved_config_path == config_path


def test_run005_manual_plan_shape_and_key_sequences() -> None:
    plan = _resolve_manual_plan("fa18c_run005_composition_rebalance_122")

    assert len(plan) == 122
    assert plan[0].seq == 1
    assert plan[0].total == 122
    assert plan[0].category_id == "default_TAC_HSI_MAP_BITROOT"
    assert "TAC" in plan[0].left_ddi_content
    assert "HSI" in plan[0].ampcd_content
    assert "BIT" in plan[0].right_ddi_content

    assert plan[86].seq == 87
    assert plan[86].category_id == "FCSMC_FINALGO_HSI_OK"
    assert "OK" in plan[86].ampcd_content

    assert plan[-1].seq == 122
    assert plan[-1].category_id == "HSI_TRANSITION_UNREADABLE"


def test_gitignore_ignores_tool_capture_output() -> None:
    gitignore_path = Path(__file__).resolve().parent.parent / ".gitignore"
    content = gitignore_path.read_text(encoding="utf-8")

    assert "/tools/.captures/" in content
