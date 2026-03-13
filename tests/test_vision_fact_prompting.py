from __future__ import annotations

from adapters.vision_fact_prompting import build_vision_fact_prompt


def _minimal_config() -> dict:
    return {
        "facts_by_id": {
            "left_ddi_fcs_page_button_visible": {},
            "fcs_page_visible": {},
            "bit_page_visible": {},
            "bit_root_page_visible": {},
            "bit_page_failure_visible": {},
            "right_ddi_fcsmc_page_visible": {},
            "right_ddi_in_test_visible": {},
            "fcs_bit_result_visible": {},
            "fcs_reset_seen": {},
        },
        "step_bindings": {
            "S08": {
                "all_of": ["fcs_page_visible"],
                "any_of": ["bit_page_visible", "bit_root_page_visible", "bit_page_failure_visible"],
            },
            "S18": {
                "all_of": ["fcs_bit_result_visible"],
                "any_of": [],
            },
        },
    }


def test_vision_fact_prompt_requires_top_level_facts_object_in_zh() -> None:
    prompt = build_vision_fact_prompt(
        vision={"frame_ids": ["1772872445010_000123"], "frame_id": "1772872445010_000123"},
        lang="zh",
        config=_minimal_config(),
    )

    assert "顶层只允许 facts 字段" in prompt
    assert "只输出 facts 数组" not in prompt


def test_vision_fact_prompt_example_matches_extractor_response_shape_in_en() -> None:
    prompt = build_vision_fact_prompt(
        vision={"frame_ids": ["1772872445010_000123"], "frame_id": "1772872445010_000123"},
        lang="en",
        config=_minimal_config(),
    )

    example_line = prompt.strip().splitlines()[-1]
    assert "contain only the facts field" in prompt
    assert "Return facts array only" not in prompt
    assert '"expires_after_ms"' not in example_line
    assert example_line == (
        '{"facts":[{"fact_id":"fcs_page_visible","state":"uncertain",'
        '"source_frame_id":"1772872445010_000123","confidence":0.0,"evidence_note":"..."}]}'
    )


def test_vision_fact_prompt_respects_explicit_empty_config() -> None:
    prompt = build_vision_fact_prompt(
        vision={"frame_ids": ["1772872445010_000123"], "frame_id": "1772872445010_000123"},
        lang="en",
        config={},
    )

    assert '"facts":[]' in prompt
    assert '"step_bindings":{}' in prompt


def test_vision_fact_prompt_mentions_ddi_menu_and_in_test_navigation_rules_in_zh() -> None:
    prompt = build_vision_fact_prompt(
        vision={"frame_ids": ["1772872445010_000123"], "frame_id": "1772872445010_000123"},
        lang="zh",
        config=_minimal_config(),
    )

    assert "PB18" in prompt
    assert "根菜单/顶层菜单页" in prompt
    assert "FCS 按钮本身" in prompt
    assert "FCS-MC" in prompt
    assert "IN TEST" in prompt


def test_vision_fact_prompt_explicitly_distinguishes_fcs_button_from_real_fcs_page() -> None:
    prompt = build_vision_fact_prompt(
        vision={"frame_ids": ["1772872445010_000123"], "frame_id": "1772872445010_000123"},
        lang="zh",
        config=_minimal_config(),
    )

    assert "并不等于 fcs_page_visible" in prompt
    assert "LEF/TEF/AIL/RUD" in prompt
    assert "SV1/SV2" in prompt
    assert "大量 X" in prompt


def test_vision_fact_prompt_explicitly_distinguishes_fcs_button_from_real_fcs_page_in_en() -> None:
    prompt = build_vision_fact_prompt(
        vision={"frame_ids": ["1772872445010_000123"], "frame_id": "1772872445010_000123"},
        lang="en",
        config=_minimal_config(),
    )

    assert "does not imply fcs_page_visible" in prompt
    assert "LEF/TEF/AIL/RUD" in prompt
    assert "SV1/SV2" in prompt
    assert "many X marks remain" in prompt


def test_vision_fact_prompt_treats_bit_failures_as_valid_s08_bit_page_evidence() -> None:
    prompt = build_vision_fact_prompt(
        vision={"frame_ids": ["1772872445010_000123"], "frame_id": "1772872445010_000123"},
        lang="en",
        config=_minimal_config(),
    )

    assert "BIT FAILURES line. The BIT FAILURES page is the BIT root page itself" in prompt
    assert '"any_of":["bit_page_visible","bit_root_page_visible","bit_page_failure_visible"]' in prompt


def test_vision_fact_prompt_treats_fcsa_and_fcsb_go_as_final_s18_go_evidence() -> None:
    zh_prompt = build_vision_fact_prompt(
        vision={"frame_ids": ["1772872445010_000123"], "frame_id": "1772872445010_000123"},
        lang="zh",
        config=_minimal_config(),
    )
    en_prompt = build_vision_fact_prompt(
        vision={"frame_ids": ["1772872445010_000123"], "frame_id": "1772872445010_000123"},
        lang="en",
        config=_minimal_config(),
    )

    assert "若能同时明确读到 FCSA=GO 与 FCSB=GO，可直接视为最终 GO 结果成立" in zh_prompt
    assert "both FCSA=GO and FCSB=GO at the same time" in en_prompt
