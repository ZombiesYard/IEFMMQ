"""
Utilities for the normalized native-viewport layout, geometry solving, prompt
snippets, and SVG preview generation.
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
DEFAULT_LAYOUT_ID = "fa18c_composite_panel_v2"
FULLSCREEN_PREVIEW_WIDTH = 1600
FULLSCREEN_PREVIEW_HEIGHT = 900

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


def _require_norm_float(value: Any, label: str, *, allow_zero: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    normalized = float(value)
    if allow_zero:
        if normalized < 0.0 or normalized > 1.0:
            raise ValueError(f"{label} must be within [0, 1]")
    elif normalized <= 0.0 or normalized > 1.0:
        raise ValueError(f"{label} must be within (0, 1]")
    return normalized


def _require_positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0.0:
        raise ValueError(f"{label} must be a positive number")
    return float(value)


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _require_mapping(data, f"vision layout YAML at {path}")


def _validate_strip_norm(strip: Mapping[str, Any]) -> dict[str, Any]:
    anchor = strip.get("anchor")
    if anchor != "left":
        raise ValueError("vision layout strip_norm.anchor must be 'left'")
    normalized = {
        "anchor": anchor,
        "x_norm": _require_norm_float(strip.get("x_norm"), "vision layout strip_norm.x_norm"),
        "y_norm": _require_norm_float(strip.get("y_norm"), "vision layout strip_norm.y_norm"),
        "height_norm": _require_norm_float(
            strip.get("height_norm"),
            "vision layout strip_norm.height_norm",
            allow_zero=False,
        ),
        "target_aspect_ratio": _require_positive_float(
            strip.get("target_aspect_ratio"),
            "vision layout strip_norm.target_aspect_ratio",
        ),
        "min_main_view_width_px": _require_positive_int(
            strip.get("min_main_view_width_px"),
            "vision layout strip_norm.min_main_view_width_px",
        ),
    }
    if normalized["x_norm"] != 0.0:
        raise ValueError("vision layout strip_norm.x_norm must currently be 0.0")
    if normalized["y_norm"] != 0.0:
        raise ValueError("vision layout strip_norm.y_norm must currently be 0.0")
    if normalized["height_norm"] != 1.0:
        raise ValueError("vision layout strip_norm.height_norm must currently be 1.0")
    return normalized


def _validate_region(region: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {
        "region_id": region.get("region_id"),
        "display_name_zh": region.get("display_name_zh"),
        "display_name_en": region.get("display_name_en"),
        "x_norm": _require_norm_float(region.get("x_norm"), "vision layout region.x_norm"),
        "y_norm": _require_norm_float(region.get("y_norm"), "vision layout region.y_norm"),
        "width_norm": _require_norm_float(
            region.get("width_norm"),
            "vision layout region.width_norm",
            allow_zero=False,
        ),
        "height_norm": _require_norm_float(
            region.get("height_norm"),
            "vision layout region.height_norm",
            allow_zero=False,
        ),
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
    if normalized["x_norm"] + normalized["width_norm"] > 1.0:
        raise ValueError(f"vision layout region {region_id} exceeds strip width")
    if normalized["y_norm"] + normalized["height_norm"] > 1.0:
        raise ValueError(f"vision layout region {region_id} exceeds strip height")
    if not isinstance(normalized["intended_contents"], list) or not normalized["intended_contents"]:
        raise ValueError(f"vision layout region {region_id} intended_contents must be a non-empty list")
    if not all(isinstance(item, str) and item for item in normalized["intended_contents"]):
        raise ValueError(f"vision layout region {region_id} intended_contents must contain non-empty strings")
    if not isinstance(normalized["background_style"], str) or not normalized["background_style"]:
        raise ValueError(f"vision layout region {region_id} missing background_style")
    if normalized["ocr_priority"] not in {"high", "medium", "low"}:
        raise ValueError(f"vision layout region {region_id} ocr_priority must be high/medium/low")
    return normalized


def _check_non_overlapping_regions(regions: list[dict[str, Any]]) -> None:
    for idx, current in enumerate(regions):
        left_a = current["x_norm"]
        top_a = current["y_norm"]
        right_a = left_a + current["width_norm"]
        bottom_a = top_a + current["height_norm"]
        for other in regions[idx + 1 :]:
            left_b = other["x_norm"]
            top_b = other["y_norm"]
            right_b = left_b + other["width_norm"]
            bottom_b = top_b + other["height_norm"]
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

    logical_canvas = _require_mapping(data.get("logical_canvas"), "vision layout logical_canvas")
    logical_width = _require_positive_int(logical_canvas.get("width"), "vision layout logical_canvas.width")
    logical_height = _require_positive_int(logical_canvas.get("height"), "vision layout logical_canvas.height")

    background = _require_mapping(data.get("background"), "vision layout background")
    if not isinstance(background.get("style"), str) or not background["style"]:
        raise ValueError("vision layout background.style must be a non-empty string")
    if not isinstance(background.get("fill"), str) or not background["fill"]:
        raise ValueError("vision layout background.fill must be a non-empty string")
    if not isinstance(background.get("description"), str) or not background["description"]:
        raise ValueError("vision layout background.description must be a non-empty string")

    strip_norm = _validate_strip_norm(_require_mapping(data.get("strip_norm"), "vision layout strip_norm"))
    raw_regions = data.get("regions")
    if not isinstance(raw_regions, list) or not raw_regions:
        raise ValueError("vision layout regions must be a non-empty list")
    normalized_regions = [_validate_region(_require_mapping(region, "vision layout region")) for region in raw_regions]
    if tuple(region["region_id"] for region in normalized_regions) != VISION_REGION_ORDER:
        raise ValueError(f"vision layout region order must be {VISION_REGION_ORDER!r}")
    _check_non_overlapping_regions(normalized_regions)

    return {
        "layout_id": layout_id,
        "title": data.get("title") if isinstance(data.get("title"), str) else layout_id,
        "description": data.get("description") if isinstance(data.get("description"), str) else "",
        "logical_canvas": {"width": logical_width, "height": logical_height},
        "background": {
            "style": background["style"],
            "fill": background["fill"],
            "description": background["description"],
        },
        "strip_norm": strip_norm,
        "regions": normalized_regions,
    }


@lru_cache(maxsize=4)
def _cached_layout(path_str: str) -> dict[str, Any]:
    return _normalize_layout(_load_yaml(Path(path_str)))


def load_vision_layout(path: str | Path | None = None) -> dict[str, Any]:
    resolved = _resolve_vision_layout_path(path).resolve()
    return _cached_layout(str(resolved))


def solve_layout_geometry(
    layout: Mapping[str, Any],
    *,
    output_width: int,
    output_height: int,
) -> dict[str, Any]:
    target_width = _require_positive_int(output_width, "output_width")
    target_height = _require_positive_int(output_height, "output_height")
    strip_norm = _require_mapping(layout.get("strip_norm"), "layout strip_norm")
    strip_x = int(round(target_width * float(strip_norm["x_norm"])))
    strip_y = int(round(target_height * float(strip_norm["y_norm"])))
    strip_height = int(round(target_height * float(strip_norm["height_norm"])))
    strip_width = int(round(strip_height * float(strip_norm["target_aspect_ratio"])))
    min_main_view_width = int(strip_norm["min_main_view_width_px"])
    max_strip_width = target_width - min_main_view_width
    if max_strip_width <= 0:
        raise ValueError("screen width is too small for the normalized left-stack layout")
    if strip_width > max_strip_width:
        strip_width = max_strip_width
    if strip_x + strip_width > target_width:
        raise ValueError("resolved strip exceeds output width")
    if strip_y + strip_height > target_height:
        raise ValueError("resolved strip exceeds output height")
    main_view_x = strip_x + strip_width
    main_view_width = target_width - main_view_x
    if main_view_width < min_main_view_width:
        raise ValueError("resolved main view width is below the configured minimum")

    resolved_regions: list[dict[str, Any]] = []
    for region in layout.get("regions", []):
        region_map = _require_mapping(region, "layout region")
        resolved_regions.append(
            {
                "region_id": region_map["region_id"],
                "display_name_zh": region_map["display_name_zh"],
                "display_name_en": region_map["display_name_en"],
                "x": strip_x + int(round(strip_width * float(region_map["x_norm"]))),
                "y": strip_y + int(round(strip_height * float(region_map["y_norm"]))),
                "width": int(round(strip_width * float(region_map["width_norm"]))),
                "height": int(round(strip_height * float(region_map["height_norm"]))),
            }
        )

    return {
        "layout_id": layout.get("layout_id"),
        "canvas": {"width": target_width, "height": target_height},
        "strip_rect": {"x": strip_x, "y": strip_y, "width": strip_width, "height": strip_height},
        "main_view_rect": {"x": main_view_x, "y": 0, "width": main_view_width, "height": target_height},
        "regions": resolved_regions,
    }


def scale_layout_regions(
    layout: Mapping[str, Any],
    *,
    output_width: int,
    output_height: int,
) -> dict[str, Any]:
    return solve_layout_geometry(layout, output_width=output_width, output_height=output_height)


def build_vlm_region_prompt(layout: Mapping[str, Any] | None = None, *, lang: str = "zh") -> str:
    current_layout = layout or load_vision_layout()
    segments: list[str] = []
    for region in current_layout["regions"]:
        region_id = region["region_id"]
        label = region["display_name_zh"] if lang == "zh" else region["display_name_en"]
        segments.append(f"{region_id}={label}")
    if lang == "zh":
        return "固定原生视口区域命名（按顺序引用，禁止使用任何显示器别名）：" + "; ".join(segments) + "。引用时必须使用 region_id。"
    return (
        "Frozen native-viewport region names (reference in order; do not use any display aliases): "
        + "; ".join(segments)
        + ". Always cite the region_id verbatim."
    )


def render_layout_svg(layout: Mapping[str, Any] | None = None) -> str:
    current_layout = layout or load_vision_layout()
    logical_canvas = current_layout["logical_canvas"]
    width = logical_canvas["width"]
    height = logical_canvas["height"]
    background = current_layout["background"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'  <title id="title">{current_layout["title"]} Strip Preview</title>',
        f'  <desc id="desc">{background["description"]}</desc>',
        f'  <rect width="{width}" height="{height}" fill="{background["fill"]}" />',
        '  <g font-family="DejaVu Sans Mono, monospace">',
        '    <text x="28" y="48" fill="#d9e2ea" font-size="24" font-weight="700">Normalized Export Strip</text>',
        '    <text x="28" y="80" fill="#8ba1b3" font-size="18">Three native display exports resolved inside the left stack.</text>',
    ]
    for region in current_layout["regions"]:
        region_id = region["region_id"]
        fill = _REGION_FILL_BY_ID[region_id]
        accent = _REGION_ACCENT_BY_ID[region_id]
        ocr_priority = region["ocr_priority"]
        x = int(round(width * float(region["x_norm"])))
        y = int(round(height * float(region["y_norm"])))
        w = int(round(width * float(region["width_norm"])))
        h = int(round(height * float(region["height_norm"])))
        lines.extend(
            [
                f'    <g id="{region_id}">',
                f'      <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="26" ry="26" fill="{fill}" stroke="{accent}" stroke-width="6" />',
                f'      <text x="{x + 28}" y="{y + 62}" fill="#f4f7fb" font-size="40" font-weight="700">{region["display_name_en"]}</text>',
                f'      <text x="{x + 28}" y="{y + 112}" fill="{accent}" font-size="28">{region_id}</text>',
                f'      <text x="{x + 28}" y="{y + h - 44}" fill="#d0dae4" font-size="24">OCR priority: {ocr_priority}</text>',
                "    </g>",
            ]
        )
    lines.extend(["  </g>", "</svg>"])
    return "\n".join(lines) + "\n"


def render_fullscreen_layout_svg(
    layout: Mapping[str, Any] | None = None,
    *,
    output_width: int = FULLSCREEN_PREVIEW_WIDTH,
    output_height: int = FULLSCREEN_PREVIEW_HEIGHT,
) -> str:
    current_layout = layout or load_vision_layout()
    solved = solve_layout_geometry(current_layout, output_width=output_width, output_height=output_height)
    strip_rect = solved["strip_rect"]
    main_rect = solved["main_view_rect"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{output_width}" height="{output_height}" viewBox="0 0 {output_width} {output_height}" role="img" aria-labelledby="title desc">',
        '  <title id="title">F/A-18C Normalized Full-Screen Layout Diagram</title>',
        '  <desc id="desc">Generic full-screen diagram showing the normalized export strip on the left and the main simulator view on the right.</desc>',
        '  <rect width="100%" height="100%" fill="#0d1216" />',
        f'  <rect x="{main_rect["x"]}" y="{main_rect["y"]}" width="{main_rect["width"]}" height="{main_rect["height"]}" fill="#101d28" stroke="#415564" stroke-width="2" />',
        f'  <rect x="{strip_rect["x"]}" y="{strip_rect["y"]}" width="{strip_rect["width"]}" height="{strip_rect["height"]}" fill="#101416" stroke="#8ca4b8" stroke-width="3" />',
        '  <g font-family="DejaVu Sans Mono, monospace">',
        f'    <text x="{main_rect["x"] + 28}" y="46" fill="#f0f4f8" font-size="24" font-weight="700">Main Simulator View</text>',
        f'    <text x="{main_rect["x"] + 28}" y="76" fill="#8aa0b2" font-size="18">The playable camera occupies the remaining screen space.</text>',
        f'    <text x="{strip_rect["x"] + 24}" y="46" fill="#f0f4f8" font-size="24" font-weight="700">Export Strip</text>',
        f'    <text x="{strip_rect["x"] + 24}" y="76" fill="#8aa0b2" font-size="18">VLM input is cut from this strip first, then split into three regions.</text>',
        f'    <line x1="{main_rect["x"] + 40}" y1="{int(output_height * 0.55)}" x2="{output_width - 60}" y2="{int(output_height * 0.55)}" stroke="#4f6676" stroke-width="2" stroke-dasharray="8 8" />',
        f'    <line x1="{main_rect["x"] + 40}" y1="{int(output_height * 0.70)}" x2="{output_width - 60}" y2="{int(output_height * 0.52)}" stroke="#38515e" stroke-width="2" />',
    ]
    for region in solved["regions"]:
        region_id = region["region_id"]
        fill = _REGION_FILL_BY_ID[region_id]
        accent = _REGION_ACCENT_BY_ID[region_id]
        x = region["x"]
        y = region["y"]
        w = region["width"]
        h = region["height"]
        lines.extend(
            [
                f'    <g id="fullscreen_{region_id}">',
                f'      <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="20" ry="20" fill="{fill}" stroke="{accent}" stroke-width="4" />',
                f'      <text x="{x + 18}" y="{y + 42}" fill="#f4f7fb" font-size="22" font-weight="700">{region["display_name_en"]}</text>',
                f'      <text x="{x + 18}" y="{y + 70}" fill="{accent}" font-size="16">{region_id}</text>',
                "    </g>",
            ]
        )
    lines.extend(["  </g>", "</svg>"])
    return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_LAYOUT_ID",
    "FULLSCREEN_PREVIEW_HEIGHT",
    "FULLSCREEN_PREVIEW_WIDTH",
    "VISION_REGION_ORDER",
    "build_vlm_region_prompt",
    "default_vision_layout_path",
    "load_vision_layout",
    "render_fullscreen_layout_svg",
    "render_layout_svg",
    "scale_layout_regions",
    "solve_layout_geometry",
]
