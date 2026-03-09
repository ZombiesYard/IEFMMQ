from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from adapters.vision_prompting import (
    DEFAULT_LAYOUT_ID,
    VISION_REGION_ORDER,
    build_vlm_region_prompt,
    load_vision_layout,
    render_fullscreen_layout_svg,
    render_layout_svg,
    solve_layout_geometry,
)


def _layout_path() -> Path:
    return Path(__file__).resolve().parent.parent / "packs" / "fa18c_startup" / "vision_layout.yaml"


def _doc_path() -> Path:
    return Path(__file__).resolve().parent.parent / "Doc" / "Vision" / "fa18c_composite_panel_ZH_v2.md"


def _strip_asset_path() -> Path:
    return Path(__file__).resolve().parent.parent / "Doc" / "Vision" / "assets" / "fa18c_composite_panel_v2.svg"


def _fullscreen_asset_path() -> Path:
    return Path(__file__).resolve().parent.parent / "Doc" / "Vision" / "assets" / "fa18c_composite_panel_fullscreen_v2.svg"


def test_load_vision_layout_returns_normalized_v2_contract() -> None:
    layout = load_vision_layout()

    assert layout["layout_id"] == DEFAULT_LAYOUT_ID
    assert layout["logical_canvas"] == {"width": 880, "height": 1440}
    assert layout["strip_norm"] == {
        "anchor": "left",
        "x_norm": 0.0,
        "y_norm": 0.0,
        "height_norm": 1.0,
        "target_aspect_ratio": 0.6111111111,
        "min_main_view_width_px": 640,
    }
    assert [region["region_id"] for region in layout["regions"]] == list(VISION_REGION_ORDER)


def test_normalized_regions_fit_strip_and_do_not_overlap() -> None:
    layout = load_vision_layout()
    regions = layout["regions"]

    for region in regions:
        assert 0.0 <= region["x_norm"] <= 1.0
        assert 0.0 <= region["y_norm"] <= 1.0
        assert region["x_norm"] + region["width_norm"] <= 1.0
        assert region["y_norm"] + region["height_norm"] <= 1.0

    for idx, current in enumerate(regions):
        for other in regions[idx + 1 :]:
            overlaps = (
                current["x_norm"] < other["x_norm"] + other["width_norm"]
                and current["x_norm"] + current["width_norm"] > other["x_norm"]
                and current["y_norm"] < other["y_norm"] + other["height_norm"]
                and current["y_norm"] + current["height_norm"] > other["y_norm"]
            )
            assert overlaps is False, (current["region_id"], other["region_id"])


@pytest.mark.parametrize(
    ("output_width", "output_height", "expected_strip", "expected_main"),
    [
        (1920, 1080, {"x": 0, "y": 0, "width": 660, "height": 1080}, {"x": 660, "y": 0, "width": 1260, "height": 1080}),
        (2560, 1440, {"x": 0, "y": 0, "width": 880, "height": 1440}, {"x": 880, "y": 0, "width": 1680, "height": 1440}),
        (3440, 1440, {"x": 0, "y": 0, "width": 880, "height": 1440}, {"x": 880, "y": 0, "width": 2560, "height": 1440}),
        (3840, 1600, {"x": 0, "y": 0, "width": 978, "height": 1600}, {"x": 978, "y": 0, "width": 2862, "height": 1600}),
        (3840, 1080, {"x": 0, "y": 0, "width": 660, "height": 1080}, {"x": 660, "y": 0, "width": 3180, "height": 1080}),
        (2160, 1440, {"x": 0, "y": 0, "width": 880, "height": 1440}, {"x": 880, "y": 0, "width": 1280, "height": 1440}),
    ],
)
def test_solve_layout_geometry_handles_reference_screen_sizes(
    output_width: int,
    output_height: int,
    expected_strip: dict[str, int],
    expected_main: dict[str, int],
) -> None:
    solved = solve_layout_geometry(load_vision_layout(), output_width=output_width, output_height=output_height)

    assert solved["canvas"] == {"width": output_width, "height": output_height}
    assert solved["strip_rect"] == expected_strip
    assert solved["main_view_rect"] == expected_main
    assert [region["region_id"] for region in solved["regions"]] == list(VISION_REGION_ORDER)
    for region in solved["regions"]:
        assert region["x"] >= expected_strip["x"]
        assert region["y"] >= expected_strip["y"]
        assert region["x"] + region["width"] <= expected_strip["x"] + expected_strip["width"]
        assert region["y"] + region["height"] <= expected_strip["y"] + expected_strip["height"]


def test_solve_layout_geometry_rejects_too_narrow_screen() -> None:
    with pytest.raises(ValueError, match="too small"):
        solve_layout_geometry(load_vision_layout(), output_width=639, output_height=1080)


def test_vlm_region_prompt_uses_frozen_region_ids_and_order() -> None:
    zh_prompt = build_vlm_region_prompt(lang="zh")
    en_prompt = build_vlm_region_prompt(lang="en")

    zh_positions = [zh_prompt.index(region_id) for region_id in VISION_REGION_ORDER]
    en_positions = [en_prompt.index(region_id) for region_id in VISION_REGION_ORDER]
    assert zh_positions == sorted(zh_positions)
    assert en_positions == sorted(en_positions)


def test_strip_svg_asset_matches_rendered_layout() -> None:
    assert _strip_asset_path().read_text(encoding="utf-8") == render_layout_svg(load_vision_layout())


def test_fullscreen_svg_asset_matches_rendered_layout() -> None:
    assert _fullscreen_asset_path().read_text(encoding="utf-8") == render_fullscreen_layout_svg(load_vision_layout())


def test_svg_assets_are_english_and_have_expected_labels() -> None:
    strip_svg = _strip_asset_path().read_text(encoding="utf-8")
    fullscreen_svg = _fullscreen_asset_path().read_text(encoding="utf-8")

    assert "Left DDI" in strip_svg
    assert "left_ddi" in strip_svg
    assert "左 DDI" not in strip_svg
    assert "Main Simulator View" in fullscreen_svg
    assert "Export Strip" in fullscreen_svg


def test_pack_metadata_and_doc_share_same_priority_step_lists() -> None:
    pack_path = Path(__file__).resolve().parent.parent / "packs" / "fa18c_startup" / "pack.yaml"
    pack = yaml.safe_load(pack_path.read_text(encoding="utf-8"))
    metadata = pack["metadata"]
    doc = _doc_path().read_text(encoding="utf-8")

    assert metadata["vision_layout_id"] == DEFAULT_LAYOUT_ID
    for step_id in metadata["vision_priority_steps"]:
        assert f"`{step_id}`" in doc
    for step_id in metadata["bios_priority_steps"]:
        assert f"`{step_id}`" in doc
    for step_id in metadata["manual_or_out_of_layout_steps"]:
        assert f"`{step_id}`" in doc


def test_layout_yaml_declares_required_v2_fields() -> None:
    data = yaml.safe_load(_layout_path().read_text(encoding="utf-8"))

    assert data["layout_id"] == DEFAULT_LAYOUT_ID
    assert set(data["strip_norm"]) >= {
        "anchor",
        "x_norm",
        "y_norm",
        "height_norm",
        "target_aspect_ratio",
        "min_main_view_width_px",
    }
    assert len(data["regions"]) == 3
    for region in data["regions"]:
        assert set(region) >= {
            "region_id",
            "display_name_zh",
            "display_name_en",
            "x_norm",
            "y_norm",
            "width_norm",
            "height_norm",
            "intended_contents",
            "background_style",
            "ocr_priority",
        }
