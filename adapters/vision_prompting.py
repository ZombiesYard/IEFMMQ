"""
Utilities for the frozen native-viewport layout, scaling, prompt snippets, and
SVG sample generation.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

VISION_REGION_ORDER: tuple[str, ...] = (
    "left_ddi",
    "ampcd",
    "right_ddi",
)
DEFAULT_LAYOUT_ID = "fa18c_composite_panel_v1"

_REGION_FILL_BY_ID: dict[str, str] = {
    "left_ddi": "#18272d",
    "ampcd": "#1b3138",
    "right_ddi": "#18272d",
}
_REGION_ACCENT_BY_ID: dict[str, str] = {
    "left_ddi": "#74d1ff",
    "ampcd": "#6ee7c8",
    "right_ddi": "#74d1ff",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_vision_layout_path() -> Path:
    return _repo_root() / "packs" / "fa18c_startup" / "vision_layout.yaml"


def _resolve_vision_layout_path(path: str | Path | None) -> Path:
    return Path(path) if path is not None else default_vision_layout_path()


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _require_mapping(data, f"vision layout YAML at {path}")


def _validate_region(region: Mapping[str, Any], *, canvas_width: int, canvas_height: int) -> dict[str, Any]:
    normalized = {
        "region_id": region.get("region_id"),
        "display_name_zh": region.get("display_name_zh"),
        "display_name_en": region.get("display_name_en"),
        "x": region.get("x"),
        "y": region.get("y"),
        "width": region.get("width"),
        "height": region.get("height"),
        "intended_contents": region.get("intended_contents"),
        "background_style": region.get("background_style"),
        "ocr_priority": region.get("ocr_priority"),
    }
    region_id = normalized["region_id"]
    if not isinstance(region_id, str) or not region_id:
        raise ValueError("vision layout region_id must be a non-empty string")
    if not isinstance(normalized["display_name_zh"], str) or not normalized["display_name_zh"]:
        raise ValueError(f"vision layout region {region_id} missing display_name_zh")
    if not isinstance(normalized["display_name_en"], str) or not normalized["display_name_en"]:
        raise ValueError(f"vision layout region {region_id} missing display_name_en")
    for key in ("x", "y", "width", "height"):
        normalized[key] = _require_positive_int(normalized[key], f"vision layout region {region_id}.{key}")
    if not isinstance(normalized["intended_contents"], list) or not normalized["intended_contents"]:
        raise ValueError(f"vision layout region {region_id} intended_contents must be a non-empty list")
    if not all(isinstance(item, str) and item for item in normalized["intended_contents"]):
        raise ValueError(f"vision layout region {region_id} intended_contents must contain non-empty strings")
    if not isinstance(normalized["background_style"], str) or not normalized["background_style"]:
        raise ValueError(f"vision layout region {region_id} missing background_style")
    if normalized["ocr_priority"] not in {"high", "medium", "low"}:
        raise ValueError(f"vision layout region {region_id} ocr_priority must be high/medium/low")
    if normalized["x"] + normalized["width"] > canvas_width:
        raise ValueError(f"vision layout region {region_id} exceeds canvas width")
    if normalized["y"] + normalized["height"] > canvas_height:
        raise ValueError(f"vision layout region {region_id} exceeds canvas height")
    return normalized


def _check_non_overlapping_regions(regions: list[dict[str, Any]]) -> None:
    for idx, current in enumerate(regions):
        left_a = current["x"]
        top_a = current["y"]
        right_a = left_a + current["width"]
        bottom_a = top_a + current["height"]
        for other in regions[idx + 1 :]:
            left_b = other["x"]
            top_b = other["y"]
            right_b = left_b + other["width"]
            bottom_b = top_b + other["height"]
            overlaps = left_a < right_b and right_a > left_b and top_a < bottom_b and bottom_a > top_b
            if overlaps:
                raise ValueError(
                    f"vision layout regions overlap: {current['region_id']} vs {other['region_id']}"
                )


def _normalize_layout(data: Mapping[str, Any]) -> dict[str, Any]:
    layout_id = data.get("layout_id")
    if not isinstance(layout_id, str) or not layout_id:
        raise ValueError("vision layout missing layout_id")
    if layout_id != DEFAULT_LAYOUT_ID:
        raise ValueError(f"unsupported vision layout id: {layout_id}")

    canvas = _require_mapping(data.get("canvas"), "vision layout canvas")
    canvas_width = _require_positive_int(canvas.get("width"), "vision layout canvas.width")
    canvas_height = _require_positive_int(canvas.get("height"), "vision layout canvas.height")

    background = _require_mapping(data.get("background"), "vision layout background")
    if not isinstance(background.get("style"), str) or not background["style"]:
        raise ValueError("vision layout background.style must be a non-empty string")
    if not isinstance(background.get("fill"), str) or not background["fill"]:
        raise ValueError("vision layout background.fill must be a non-empty string")
    if not isinstance(background.get("description"), str) or not background["description"]:
        raise ValueError("vision layout background.description must be a non-empty string")

    raw_regions = data.get("regions")
    if not isinstance(raw_regions, list) or not raw_regions:
        raise ValueError("vision layout regions must be a non-empty list")

    normalized_regions = [
        _validate_region(
            _require_mapping(region, "vision layout region"),
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
        for region in raw_regions
    ]
    region_order = tuple(region["region_id"] for region in normalized_regions)
    if region_order != VISION_REGION_ORDER:
        raise ValueError(f"vision layout region order must be {VISION_REGION_ORDER!r}")
    _check_non_overlapping_regions(normalized_regions)

    return {
        "layout_id": layout_id,
        "title": data.get("title") if isinstance(data.get("title"), str) else layout_id,
        "description": data.get("description") if isinstance(data.get("description"), str) else "",
        "canvas": {"width": canvas_width, "height": canvas_height},
        "background": {
            "style": background["style"],
            "fill": background["fill"],
            "description": background["description"],
        },
        "regions": normalized_regions,
    }


@lru_cache(maxsize=4)
def _cached_layout(path_str: str) -> dict[str, Any]:
    return _normalize_layout(_load_yaml(Path(path_str)))


def load_vision_layout(path: str | Path | None = None) -> dict[str, Any]:
    resolved = _resolve_vision_layout_path(path).resolve()
    return _cached_layout(str(resolved))


def scale_layout_regions(
    layout: Mapping[str, Any],
    *,
    output_width: int,
    output_height: int,
) -> dict[str, Any]:
    target_width = _require_positive_int(output_width, "output_width")
    target_height = _require_positive_int(output_height, "output_height")
    canvas = _require_mapping(layout.get("canvas"), "layout canvas")
    base_width = _require_positive_int(canvas.get("width"), "layout canvas.width")
    base_height = _require_positive_int(canvas.get("height"), "layout canvas.height")

    scale = min(target_width / base_width, target_height / base_height)
    offset_x = int(round((target_width - base_width * scale) / 2))
    offset_y = int(round((target_height - base_height * scale) / 2))

    scaled_regions: list[dict[str, Any]] = []
    for region in layout.get("regions", []):
        region_map = _require_mapping(region, "layout region")
        scaled_regions.append(
            {
                "region_id": region_map["region_id"],
                "display_name_zh": region_map["display_name_zh"],
                "display_name_en": region_map["display_name_en"],
                "x": offset_x + int(round(int(region_map["x"]) * scale)),
                "y": offset_y + int(round(int(region_map["y"]) * scale)),
                "width": int(round(int(region_map["width"]) * scale)),
                "height": int(round(int(region_map["height"]) * scale)),
            }
        )

    return {
        "layout_id": layout.get("layout_id"),
        "canvas": {"width": target_width, "height": target_height},
        "scale": scale,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "regions": scaled_regions,
    }


def build_vlm_region_prompt(layout: Mapping[str, Any] | None = None, *, lang: str = "zh") -> str:
    current_layout = layout or load_vision_layout()
    segments: list[str] = []
    for region in current_layout["regions"]:
        region_id = region["region_id"]
        label = region["display_name_zh"] if lang == "zh" else region["display_name_en"]
        segments.append(f"{region_id}={label}")
    if lang == "zh":
        return (
            "固定原生视口区域命名（按顺序引用，禁止使用任何显示器别名）："
            + "; ".join(segments)
            + "。引用时必须使用 region_id。"
        )
    return (
        "Frozen native-viewport region names (reference in order; do not use any display aliases): "
        + "; ".join(segments)
        + ". Always cite the region_id verbatim."
    )


def render_layout_svg(layout: Mapping[str, Any] | None = None) -> str:
    current_layout = layout or load_vision_layout()
    canvas = current_layout["canvas"]
    width = canvas["width"]
    height = canvas["height"]
    background = current_layout["background"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"  <title id=\"title\">{current_layout['title']}</title>",
        f"  <desc id=\"desc\">{background['description']}</desc>",
        f"  <rect width=\"{width}\" height=\"{height}\" fill=\"{background['fill']}\" />",
        "  <g font-family=\"DejaVu Sans Mono, monospace\">",
    ]
    for region in current_layout["regions"]:
        region_id = region["region_id"]
        fill = _REGION_FILL_BY_ID.get(region_id, "#22303a")
        accent = _REGION_ACCENT_BY_ID.get(region_id, "#a5d6ff")
        x = region["x"]
        y = region["y"]
        w = region["width"]
        h = region["height"]
        title_y = y + 62
        subtitle_y = y + 112
        content_y = y + h - 44
        lines.extend(
            [
                f"    <g id=\"{region_id}\">",
                f"      <rect x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"{h}\" rx=\"26\" ry=\"26\" fill=\"{fill}\" stroke=\"{accent}\" stroke-width=\"6\" />",
                f"      <text x=\"{x + 28}\" y=\"{title_y}\" fill=\"#f4f7fb\" font-size=\"40\" font-weight=\"700\">{region['display_name_en']}</text>",
                f"      <text x=\"{x + 28}\" y=\"{subtitle_y}\" fill=\"{accent}\" font-size=\"28\">{region_id} · {region['display_name_zh']}</text>",
                f"      <text x=\"{x + 28}\" y=\"{content_y}\" fill=\"#d0dae4\" font-size=\"24\">OCR priority: {region['ocr_priority']}</text>",
                "    </g>",
            ]
        )
    lines.extend(
        [
            "  </g>",
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_LAYOUT_ID",
    "VISION_REGION_ORDER",
    "build_vlm_region_prompt",
    "default_vision_layout_path",
    "load_vision_layout",
    "render_layout_svg",
    "scale_layout_regions",
]
