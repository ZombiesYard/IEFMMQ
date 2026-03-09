from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from adapters.vision_frames import (
    DEFAULT_FRAME_CHANNEL,
    DEFAULT_FRAME_MANIFEST_NAME,
    FrameDirectoryVisionPort,
    build_frame_channel_dir,
    build_frame_filename,
    build_frame_id,
    build_frame_manifest_path,
    render_vlm_ready_frame,
)
from adapters.vision_prompting import load_vision_layout, solve_layout_geometry


def _make_source_frame(path: Path, *, width: int, height: int) -> None:
    layout = load_vision_layout()
    solved = solve_layout_geometry(layout, output_width=width, output_height=height)
    strip = solved["strip_rect"]

    image = Image.new("RGB", (width, height), (200, 32, 32))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (
            strip["x"],
            strip["y"],
            strip["x"] + strip["width"],
            strip["y"] + strip["height"],
        ),
        fill=(16, 20, 22),
    )
    fills = {
        "left_ddi": (28, 41, 48),
        "ampcd": (24, 52, 58),
        "right_ddi": (28, 41, 48),
    }
    for region in solved["regions"]:
        draw.rectangle(
            (
                region["x"],
                region["y"],
                region["x"] + region["width"],
                region["y"] + region["height"],
            ),
            fill=fills[region["region_id"]],
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def _append_manifest_entry(channel_dir: Path, entry: dict[str, object]) -> None:
    manifest_path = build_frame_manifest_path(channel_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _manifest_entry(
    *,
    channel_dir: Path,
    capture_wall_ms: int,
    frame_seq: int,
    width: int,
    height: int,
    channel: str = DEFAULT_FRAME_CHANNEL,
    session_id: str = "sess-live",
    image_path: Path | None = None,
) -> dict[str, object]:
    image = image_path or channel_dir / build_frame_filename(capture_wall_ms=capture_wall_ms, frame_seq=frame_seq)
    return {
        "frame_id": build_frame_id(capture_wall_ms=capture_wall_ms, frame_seq=frame_seq),
        "capture_wall_ms": capture_wall_ms,
        "frame_seq": frame_seq,
        "channel": channel,
        "layout_id": "fa18c_composite_panel_v2",
        "image_path": str(image.resolve()),
        "width": width,
        "height": height,
        "source_session_id": session_id,
    }


@pytest.mark.parametrize(("width", "height"), [(1920, 1080), (3440, 1440), (3840, 1600)])
def test_render_vlm_ready_frame_crops_main_view_and_draws_region_guides(
    tmp_path: Path,
    width: int,
    height: int,
) -> None:
    source_path = tmp_path / f"source_{width}x{height}.png"
    output_path = tmp_path / f"source_{width}x{height}_vlm.png"
    _make_source_frame(source_path, width=width, height=height)

    metadata = render_vlm_ready_frame(source_path, output_path)

    with Image.open(source_path) as source_image:
        source_size = source_image.size
    with Image.open(output_path) as processed:
        processed_size = processed.size
        assert processed_size[0] < source_size[0]
        assert processed_size[1] >= source_size[1]
        pixels = processed.load()
        assert pixels is not None
        for y in range(processed_size[1]):
            for x in range(processed_size[0]):
                r, g, b = pixels[x, y]
                assert not (r > 180 and g < 80 and b < 80)
        left_border_pixel = processed.getpixel((metadata["regions"][0]["x"], metadata["regions"][0]["y"]))
        assert left_border_pixel != (16, 20, 22)

    assert metadata["source_size"] == {"width": width, "height": height}
    assert metadata["artifact_size"]["width"] == processed_size[0]
    assert metadata["crop_rect"]["width"] < width


def test_frame_directory_port_consumes_only_manifested_final_png_files(tmp_path: Path) -> None:
    saved_games_dir = tmp_path / "Saved Games" / "DCS"
    channel_dir = build_frame_channel_dir(
        saved_games_dir=saved_games_dir,
        session_id="sess-live",
        channel=DEFAULT_FRAME_CHANNEL,
    )
    temp_path = channel_dir / "1772872444901_000122.png.tmp"
    final_path = channel_dir / build_frame_filename(capture_wall_ms=1772872444902, frame_seq=123)
    _make_source_frame(temp_path, width=1920, height=1080)
    _make_source_frame(final_path, width=1920, height=1080)

    _append_manifest_entry(
        channel_dir,
        _manifest_entry(
            channel_dir=channel_dir,
            capture_wall_ms=1772872444901,
            frame_seq=122,
            width=1920,
            height=1080,
            image_path=temp_path,
        ),
    )
    _append_manifest_entry(
        channel_dir,
        _manifest_entry(
            channel_dir=channel_dir,
            capture_wall_ms=1772872444902,
            frame_seq=123,
            width=1920,
            height=1080,
        ),
    )

    port = FrameDirectoryVisionPort(saved_games_dir=saved_games_dir, channel=DEFAULT_FRAME_CHANNEL)
    port.start("sess-live")
    observations = port.poll()
    port.stop()

    assert [obs.frame_id for obs in observations] == ["1772872444902_000123"]
    obs = observations[0]
    assert obs.capture_wall_ms == 1772872444902
    assert obs.frame_seq == 123
    assert obs.source_session_id == "sess-live"
    assert obs.image_uri is not None
    assert Path(obs.image_uri).exists()
    assert obs.metadata["source_image_path"] == str(final_path.resolve())
    assert obs.metadata["manifest_path"] == str(build_frame_manifest_path(channel_dir).resolve())
    assert obs.metadata["artifact_kind"] == "vlm_ready"


def test_replay_can_reuse_same_manifest_and_artifact_directory(tmp_path: Path) -> None:
    saved_games_dir = tmp_path / "Saved Games" / "DCS"
    channel_dir = build_frame_channel_dir(
        saved_games_dir=saved_games_dir,
        session_id="sess-replay",
        channel=DEFAULT_FRAME_CHANNEL,
    )
    for capture_wall_ms, frame_seq in ((1772872444902, 123), (1772872445002, 124)):
        frame_path = channel_dir / build_frame_filename(capture_wall_ms=capture_wall_ms, frame_seq=frame_seq)
        _make_source_frame(frame_path, width=2560, height=1440)
        _append_manifest_entry(
            channel_dir,
            _manifest_entry(
                channel_dir=channel_dir,
                capture_wall_ms=capture_wall_ms,
                frame_seq=frame_seq,
                width=2560,
                height=1440,
                session_id="sess-replay",
            ),
        )

    live_port = FrameDirectoryVisionPort(saved_games_dir=saved_games_dir, channel=DEFAULT_FRAME_CHANNEL)
    live_port.start("sess-replay")
    live_obs = live_port.poll()
    live_port.stop()

    replay_port = FrameDirectoryVisionPort(saved_games_dir=saved_games_dir, channel=DEFAULT_FRAME_CHANNEL)
    replay_port.start("sess-replay")
    replay_obs = replay_port.poll()
    replay_port.stop()

    assert [item.frame_id for item in replay_obs] == [item.frame_id for item in live_obs]
    assert [item.image_uri for item in replay_obs] == [item.image_uri for item in live_obs]


def test_manifest_path_uses_frozen_frames_jsonl_name(tmp_path: Path) -> None:
    saved_games_dir = tmp_path / "Saved Games" / "DCS"
    channel_dir = build_frame_channel_dir(
        saved_games_dir=saved_games_dir,
        session_id="sess-manifest",
        channel=DEFAULT_FRAME_CHANNEL,
    )

    assert build_frame_manifest_path(channel_dir).name == DEFAULT_FRAME_MANIFEST_NAME
