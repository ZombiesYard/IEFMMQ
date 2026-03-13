"""
Frame-manifest ingestion and VLM-ready artifact rendering for the frozen
left-stack composite panel layout.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFont

from adapters.vision_prompting import DEFAULT_LAYOUT_ID, load_vision_layout, solve_layout_geometry
from core.types_v2 import VisionObservation
from ports.vision_port import VisionPort
from simtutor.schemas import validate_instance

DEFAULT_FRAME_CHANNEL = "composite_panel"
DEFAULT_FRAME_MANIFEST_NAME = "frames.jsonl"
DEFAULT_ARTIFACT_DIRNAME = "artifacts"
DEFAULT_ARTIFACT_SUFFIX = "_vlm.png"
DEFAULT_MIME_TYPE = "image/png"
_TEMP_FILE_SUFFIXES = (".tmp", ".part", ".partial")
_REGION_ACCENTS: dict[str, str] = {
    "left_ddi": "#74d1ff",
    "ampcd": "#6ee7c8",
    "right_ddi": "#74d1ff",
}
_LABEL_BACKGROUNDS: dict[str, str] = {
    "left_ddi": "#10252f",
    "ampcd": "#12312d",
    "right_ddi": "#10252f",
}
_CANVAS_BACKGROUND = "#0f1418"


def build_frames_root(saved_games_dir: str | Path) -> Path:
    return Path(saved_games_dir).expanduser().resolve() / "SimTutor" / "frames"


def build_frame_channel_dir(*, saved_games_dir: str | Path, session_id: str, channel: str) -> Path:
    return build_frames_root(saved_games_dir) / session_id / channel


def build_frame_filename(*, capture_wall_ms: int, frame_seq: int) -> str:
    capture = _require_non_negative_int(capture_wall_ms, "capture_wall_ms")
    seq = _require_non_negative_int(frame_seq, "frame_seq")
    return f"{capture}_{seq:06d}.png"


def build_frame_id(*, capture_wall_ms: int, frame_seq: int) -> str:
    return Path(build_frame_filename(capture_wall_ms=capture_wall_ms, frame_seq=frame_seq)).stem


def build_frame_manifest_path(channel_dir: str | Path) -> Path:
    return Path(channel_dir) / DEFAULT_FRAME_MANIFEST_NAME


def build_vlm_artifact_path(image_path: str | Path, *, artifact_dir_name: str = DEFAULT_ARTIFACT_DIRNAME) -> Path:
    source_path = Path(image_path)
    normalized_artifact_dir = _normalize_artifact_dir_name(artifact_dir_name)
    return source_path.parent / normalized_artifact_dir / f"{source_path.stem}{DEFAULT_ARTIFACT_SUFFIX}"


def is_final_frame_path(path_like: str | Path) -> bool:
    path = Path(path_like)
    name_lower = path.name.lower()
    if any(name_lower.endswith(suffix) for suffix in _TEMP_FILE_SUFFIXES):
        return False
    return path.suffix.lower() == ".png"


def render_vlm_ready_frame(
    source_path: str | Path,
    output_path: str | Path,
    *,
    layout: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_source = Path(source_path).expanduser().resolve()
    resolved_output = Path(output_path).expanduser().resolve()
    current_layout = dict(layout) if isinstance(layout, Mapping) else load_vision_layout()

    with Image.open(resolved_source) as source_image:
        image = source_image.convert("RGB")
        solved = solve_layout_geometry(current_layout, output_width=image.width, output_height=image.height)
        strip_rect = solved["strip_rect"]
        crop_box = (
            int(strip_rect["x"]),
            int(strip_rect["y"]),
            int(strip_rect["x"]) + int(strip_rect["width"]),
            int(strip_rect["y"]) + int(strip_rect["height"]),
        )
        cropped = image.crop(crop_box)

    margin = max(20, cropped.width // 24)
    border_width = max(4, cropped.width // 160)
    label_padding_x = max(10, cropped.width // 70)
    label_padding_y = max(6, cropped.height // 160)
    label_font = _load_font(max(18, cropped.height // 32))
    label_specs: list[dict[str, Any]] = []
    max_label_height = 0
    for region in solved["regions"]:
        region_id = str(region["region_id"])
        label_text = str(region["display_name_en"])
        accent = _REGION_ACCENTS.get(region_id)
        label_background = _LABEL_BACKGROUNDS.get(region_id)
        if accent is None or label_background is None:
            raise ValueError(f"unsupported vision layout region_id for VLM artifact rendering: {region_id!r}")
        label_bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), label_text, font=label_font)
        label_width = (label_bbox[2] - label_bbox[0]) + label_padding_x * 2
        label_height = (label_bbox[3] - label_bbox[1]) + label_padding_y * 2
        max_label_height = max(max_label_height, label_height)
        label_specs.append(
            {
                "region_id": region_id,
                "label_text": label_text,
                "accent": accent,
                "label_background": label_background,
                "label_width": label_width,
                "label_height": label_height,
            }
        )

    right_padding = max(margin * 2, max((int(spec["label_width"]) for spec in label_specs), default=0) + margin * 2)
    canvas = Image.new("RGB", (cropped.width + margin * 2 + right_padding, cropped.height + margin * 2), _CANVAS_BACKGROUND)
    canvas.paste(cropped, (margin, margin))
    draw = ImageDraw.Draw(canvas)

    region_metadata: list[dict[str, Any]] = []
    for region, label_spec in zip(solved["regions"], label_specs):
        local_x = margin + int(region["x"]) - int(strip_rect["x"])
        local_y = margin + int(region["y"]) - int(strip_rect["y"])
        local_w = int(region["width"])
        local_h = int(region["height"])
        region_id = str(label_spec["region_id"])
        label_text = str(label_spec["label_text"])
        accent = str(label_spec["accent"])
        label_background = str(label_spec["label_background"])
        label_width = int(label_spec["label_width"])
        label_height = int(label_spec["label_height"])

        draw.rounded_rectangle(
            (
                local_x,
                local_y,
                local_x + local_w,
                local_y + local_h,
            ),
            radius=max(16, local_w // 18),
            outline=accent,
            width=border_width,
        )

        label_x = min(
            local_x + local_w + border_width * 3,
            canvas.width - margin - label_width,
        )
        label_y = max(
            margin,
            min(
                local_y + (local_h // 2) - (label_height // 2),
                canvas.height - margin - label_height,
            ),
        )
        draw.rounded_rectangle(
            (
                label_x,
                label_y,
                label_x + label_width,
                label_y + label_height,
            ),
            radius=max(10, label_height // 3),
            fill=label_background,
            outline=accent,
            width=max(2, border_width // 2),
        )
        draw.text(
            (label_x + label_padding_x, label_y + label_padding_y),
            label_text,
            fill="#f4f7fb",
            font=label_font,
        )
        region_metadata.append(
            {
                "region_id": region_id,
                "label": label_text,
                "x": local_x,
                "y": local_y,
                "width": local_w,
                "height": local_h,
            }
        )

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(resolved_output)
    return {
        "crop_rect": {
            "x": int(strip_rect["x"]),
            "y": int(strip_rect["y"]),
            "width": int(strip_rect["width"]),
            "height": int(strip_rect["height"]),
        },
        "regions": region_metadata,
        "source_size": {"width": image.width, "height": image.height},
        "artifact_size": {"width": canvas.width, "height": canvas.height},
    }


class FrameDirectoryVisionPort(VisionPort):
    def __init__(
        self,
        *,
        saved_games_dir: str | Path,
        channel: str = DEFAULT_FRAME_CHANNEL,
        layout_id: str = DEFAULT_LAYOUT_ID,
        artifact_dir_name: str = DEFAULT_ARTIFACT_DIRNAME,
    ) -> None:
        self.saved_games_dir = Path(saved_games_dir).expanduser().resolve()
        self.channel = channel
        self.layout_id = _require_text(layout_id, "layout_id")
        self.artifact_dir_name = _normalize_artifact_dir_name(artifact_dir_name)
        self._layout = load_vision_layout()
        loaded_layout_id = _require_text(self._layout.get("layout_id"), "loaded layout_id")
        if self.layout_id != loaded_layout_id:
            raise ValueError(
                f"unsupported layout_id {self.layout_id!r}; current crop pipeline only supports {loaded_layout_id!r}"
            )
        self._manifest_offset = 0
        self._session_id: str | None = None
        self._channel_dir: Path | None = None
        self._manifest_path: Path | None = None

    def start(self, session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        self._session_id = session_id.strip()
        self._channel_dir = build_frame_channel_dir(
            saved_games_dir=self.saved_games_dir,
            session_id=self._session_id,
            channel=self.channel,
        )
        self._manifest_path = build_frame_manifest_path(self._channel_dir)
        self._manifest_offset = 0

    def poll(self) -> list[VisionObservation]:
        if self._channel_dir is None or self._manifest_path is None:
            raise RuntimeError("FrameDirectoryVisionPort.start(session_id) must be called before poll()")
        if not self._manifest_path.exists():
            return []

        observations: list[VisionObservation] = []
        with self._manifest_path.open("r", encoding="utf-8") as handle:
            handle.seek(self._manifest_offset)
            while True:
                line_start = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith("\n"):
                    handle.seek(line_start)
                    break
                next_offset = handle.tell()
                try:
                    entry_raw = json.loads(line)
                except json.JSONDecodeError:
                    handle.seek(line_start)
                    break
                self._manifest_offset = next_offset
                if not isinstance(entry_raw, dict):
                    raise ValueError(f"vision frame manifest line must be an object: {self._manifest_path}")
                validate_instance(entry_raw, "vision_frame_manifest_entry")
                entry = dict(entry_raw)
                image_path = self._resolve_image_path(entry["image_path"])
                if not is_final_frame_path(image_path):
                    continue
                if not image_path.exists():
                    self._manifest_offset = line_start
                    break
                observations.append(self._build_observation(entry, image_path=image_path))
        observations.sort(
            key=lambda observation: (
                observation.capture_wall_ms if isinstance(observation.capture_wall_ms, int) else -1,
                observation.frame_id,
            )
        )
        return observations

    def stop(self) -> None:
        self._manifest_offset = 0
        self._session_id = None
        self._channel_dir = None
        self._manifest_path = None

    def _resolve_image_path(self, raw_path: Any) -> Path:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("vision frame manifest image_path must be a non-empty string")
        if self._channel_dir is None:
            raise RuntimeError("channel directory is not initialized")
        candidate = Path(raw_path).expanduser()
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self._channel_dir / candidate).resolve()
        try:
            resolved.relative_to(self._channel_dir)
        except ValueError as exc:
            raise ValueError(
                f"vision frame manifest image_path escapes channel directory: {resolved}"
            ) from exc
        return resolved

    def _build_observation(self, entry: Mapping[str, Any], *, image_path: Path) -> VisionObservation:
        artifact_path = build_vlm_artifact_path(image_path, artifact_dir_name=self.artifact_dir_name)
        capture_wall_ms = _require_non_negative_int(entry["capture_wall_ms"], "capture_wall_ms")
        frame_seq = _require_non_negative_int(entry["frame_seq"], "frame_seq")
        manifest_layout_id = _require_text(entry["layout_id"], "layout_id")
        if manifest_layout_id != self.layout_id:
            raise ValueError(
                f"vision frame manifest layout_id mismatch: expected {self.layout_id!r}, got {manifest_layout_id!r}"
            )
        manifest_channel = _require_text(entry["channel"], "channel")
        if manifest_channel != self.channel:
            raise ValueError(
                f"vision frame manifest channel mismatch: expected {self.channel!r}, got {manifest_channel!r}"
            )
        expected_frame_id = build_frame_id(capture_wall_ms=capture_wall_ms, frame_seq=frame_seq)
        expected_filename = build_frame_filename(capture_wall_ms=capture_wall_ms, frame_seq=frame_seq)
        frame_id = _require_text(entry["frame_id"], "frame_id")
        if frame_id != expected_frame_id:
            raise ValueError(
                f"vision frame manifest frame_id mismatch: expected {expected_frame_id!r}, got {frame_id!r}"
            )
        if image_path.name != expected_filename:
            raise ValueError(
                f"vision frame manifest image_path mismatch: expected filename {expected_filename!r}, "
                f"got {image_path.name!r}"
            )

        artifact_meta = render_vlm_ready_frame(image_path, artifact_path, layout=self._layout)
        source_size = artifact_meta["source_size"]
        expected_width = _require_positive_int(entry["width"], "width")
        expected_height = _require_positive_int(entry["height"], "height")
        if source_size["width"] != expected_width or source_size["height"] != expected_height:
            raise ValueError(
                "vision frame manifest dimensions do not match source image: "
                f"manifest=({expected_width}x{expected_height}) image=({source_size['width']}x{source_size['height']})"
            )
        source_session_id = _require_text(entry["source_session_id"], "source_session_id")

        observation = VisionObservation(
            frame_id=frame_id,
            timestamp=datetime.fromtimestamp(capture_wall_ms / 1000.0, tz=timezone.utc).isoformat(),
            source="vision_frame_manifest",
            session_id=source_session_id,
            channel=self.channel,
            layout_id=self.layout_id,
            image_uri=str(artifact_path.resolve()),
            source_image_path=str(image_path.resolve()),
            mime_type=DEFAULT_MIME_TYPE,
            capture_wall_ms=capture_wall_ms,
            frame_seq=frame_seq,
            width=expected_width,
            height=expected_height,
            source_session_id=source_session_id,
            metadata={
                "artifact_kind": "vlm_ready",
                "manifest_path": str(self._manifest_path.resolve()) if self._manifest_path is not None else None,
                "source_image_path": str(image_path.resolve()),
                **artifact_meta,
            },
        )
        validate_instance(observation.to_dict(), "vision_observation")
        return observation


@lru_cache(maxsize=16)
def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    candidates = (
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "Arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _normalize_artifact_dir_name(value: str | Path) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("artifact_dir_name must be a non-empty simple directory name")
    path = Path(text)
    if path.is_absolute():
        raise ValueError("artifact_dir_name must be relative to the frame channel directory")
    if len(path.parts) != 1 or path.parts[0] in {"", ".", ".."}:
        raise ValueError("artifact_dir_name must be a simple directory name without path traversal")
    return path.parts[0]


def _require_non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


__all__ = [
    "DEFAULT_ARTIFACT_DIRNAME",
    "DEFAULT_FRAME_CHANNEL",
    "DEFAULT_FRAME_MANIFEST_NAME",
    "FrameDirectoryVisionPort",
    "build_frame_channel_dir",
    "build_frame_filename",
    "build_frame_id",
    "build_frame_manifest_path",
    "build_frames_root",
    "build_vlm_artifact_path",
    "is_final_frame_path",
    "render_vlm_ready_frame",
]
