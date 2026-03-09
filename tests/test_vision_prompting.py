from __future__ import annotations

from pathlib import Path

import yaml

from adapters.vision_prompting import (
    DEFAULT_LAYOUT_ID,
    VISION_REGION_ORDER,
    build_vlm_region_prompt,
    load_vision_layout,
    render_layout_svg,
    scale_layout_regions,
)


def _layout_path() -> Path:
    return Path(__file__).resolve().parent.parent / "packs" / "fa18c_startup" / "vision_layout.yaml"


def _doc_path() -> Path:
    return Path(__file__).resolve().parent.parent / "Doc" / "Vision" / "fa18c_composite_panel_ZH_v1.md"


def _asset_path() -> Path:
    return Path(__file__).resolve().parent.parent / "Doc" / "Vision" / "assets" / "fa18c_composite_panel_v1.svg"


def test_load_vision_layout_returns_frozen_contract() -> None:
    layout = load_vision_layout()

    assert layout["layout_id"] == DEFAULT_LAYOUT_ID
    assert layout["canvas"] == {"width": 880, "height": 1440}
    assert [region["region_id"] for region in layout["regions"]] == list(VISION_REGION_ORDER)
    assert [
        (region["x"], region["y"], region["width"], region["height"])
        for region in layout["regions"]
    ] == [
        (216, 24, 448, 448),
        (216, 496, 448, 448),
        (216, 968, 448, 448),
    ]


def test_layout_regions_fit_canvas_and_do_not_overlap() -> None:
    layout = load_vision_layout()
    canvas_width = layout["canvas"]["width"]
    canvas_height = layout["canvas"]["height"]
    regions = layout["regions"]

    for region in regions:
        assert region["x"] >= 0
        assert region["y"] >= 0
        assert region["x"] + region["width"] <= canvas_width
        assert region["y"] + region["height"] <= canvas_height

    for idx, current in enumerate(regions):
        for other in regions[idx + 1 :]:
            overlaps = (
                current["x"] < other["x"] + other["width"]
                and current["x"] + current["width"] > other["x"]
                and current["y"] < other["y"] + other["height"]
                and current["y"] + current["height"] > other["y"]
            )
            assert overlaps is False, (current["region_id"], other["region_id"])


def test_scale_layout_regions_preserves_order_and_bounds() -> None:
    layout = load_vision_layout()

    for output_width, output_height in ((1920, 1080), (3840, 2160)):
        scaled = scale_layout_regions(layout, output_width=output_width, output_height=output_height)
        assert scaled["canvas"] == {"width": output_width, "height": output_height}
        assert [region["region_id"] for region in scaled["regions"]] == list(VISION_REGION_ORDER)
        for region in scaled["regions"]:
            assert region["x"] >= 0
            assert region["y"] >= 0
            assert region["x"] + region["width"] <= output_width
            assert region["y"] + region["height"] <= output_height
        for idx, current in enumerate(scaled["regions"]):
            for other in scaled["regions"][idx + 1 :]:
                overlaps = (
                    current["x"] < other["x"] + other["width"]
                    and current["x"] + current["width"] > other["x"]
                    and current["y"] < other["y"] + other["height"]
                    and current["y"] + current["height"] > other["y"]
                )
                assert overlaps is False, (output_width, output_height, current["region_id"], other["region_id"])


def test_vlm_region_prompt_uses_frozen_region_ids_and_order() -> None:
    zh_prompt = build_vlm_region_prompt(lang="zh")
    en_prompt = build_vlm_region_prompt(lang="en")

    zh_positions = [zh_prompt.index(region_id) for region_id in VISION_REGION_ORDER]
    en_positions = [en_prompt.index(region_id) for region_id in VISION_REGION_ORDER]
    assert zh_positions == sorted(zh_positions)
    assert en_positions == sorted(en_positions)
    assert [segment.split("=")[0] for segment in zh_prompt.split("：", 1)[1].split("。", 1)[0].split("; ")] == list(
        VISION_REGION_ORDER
    )
    assert [segment.split("=")[0] for segment in en_prompt.split(": ", 1)[1].split(". ", 1)[0].split("; ")] == list(
        VISION_REGION_ORDER
    )


def test_svg_asset_matches_rendered_layout() -> None:
    svg = render_layout_svg(load_vision_layout())
    assert _asset_path().read_text(encoding="utf-8") == svg


def test_svg_contains_viewbox_and_all_region_labels() -> None:
    svg = _asset_path().read_text(encoding="utf-8")

    assert 'viewBox="0 0 880 1440"' in svg
    for expected in (
        "Left DDI",
        "AMPCD",
        "Right DDI",
    ):
        assert expected in svg


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


def test_layout_yaml_declares_required_region_fields() -> None:
    data = yaml.safe_load(_layout_path().read_text(encoding="utf-8"))
    regions = data["regions"]

    assert len(regions) == 3
    for region in regions:
        assert set(region) >= {
            "region_id",
            "display_name_zh",
            "display_name_en",
            "x",
            "y",
            "width",
            "height",
            "intended_contents",
            "background_style",
            "ocr_priority",
        }
