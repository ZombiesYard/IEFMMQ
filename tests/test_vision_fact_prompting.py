from __future__ import annotations

from adapters.vision_fact_prompting import build_vision_fact_prompt
from core.vision_facts import load_vision_facts_config


def test_vision_fact_prompt_requires_top_level_facts_object_in_zh() -> None:
    prompt = build_vision_fact_prompt(
        vision={"frame_ids": ["1772872445010_000123"], "frame_id": "1772872445010_000123"},
        lang="zh",
        config=load_vision_facts_config(),
    )

    assert "顶层只允许 facts 字段" in prompt
    assert "只输出 facts 数组" not in prompt


def test_vision_fact_prompt_example_matches_extractor_response_shape_in_en() -> None:
    prompt = build_vision_fact_prompt(
        vision={"frame_ids": ["1772872445010_000123"], "frame_id": "1772872445010_000123"},
        lang="en",
        config=load_vision_facts_config(),
    )

    example_line = prompt.strip().splitlines()[-1]
    assert "contain only the facts field" in prompt
    assert "Return facts array only" not in prompt
    assert '"expires_after_ms"' not in example_line
    assert example_line == (
        '{"facts":[{"fact_id":"fcs_page_visible","state":"uncertain",'
        '"source_frame_id":"1772872445010_000123","confidence":0.0,"evidence_note":"..."}]}'
    )
