from __future__ import annotations

from adapters.vision_fact_prompting import build_vision_fact_prompt


def _minimal_config() -> dict:
    return {
        "facts_by_id": {
            "tac_page_visible": {},
            "supt_page_visible": {},
            "fcs_page_visible": {},
            "fcs_page_x_marks_visible": {},
            "bit_root_page_visible": {},
            "fcsmc_page_visible": {},
            "fcsmc_intermediate_result_visible": {},
            "fcsmc_in_test_visible": {},
            "fcsmc_final_go_result_visible": {},
            "hsi_page_visible": {},
            "hsi_map_layer_visible": {},
            "ins_grnd_alignment_text_visible": {},
            "ins_ok_text_visible": {},
        },
        "step_bindings": {
            "S08": {
                "all_of": ["fcs_page_visible", "bit_root_page_visible"],
                "any_of": [],
            },
            "S18": {
                "all_of": ["fcsmc_final_go_result_visible"],
                "any_of": [],
            },
        },
    }


def test_vision_fact_prompt_requires_summary_and_facts_top_level_only_in_zh() -> None:
    prompt = build_vision_fact_prompt(
        vision={"frame_ids": ["1772872445010_000123"], "frame_id": "1772872445010_000123"},
        lang="zh",
        config=_minimal_config(),
    )

    assert "顶层只允许包含：summary、facts" in prompt
    assert "严禁输出 source_frame_id" in prompt


def test_vision_fact_prompt_example_matches_aligned_response_shape_in_en() -> None:
    prompt = build_vision_fact_prompt(
        vision={"frame_ids": ["1772872445010_000123"], "frame_id": "1772872445010_000123"},
        lang="en",
        config=_minimal_config(),
    )

    example_line = prompt.strip().splitlines()[-1]
    assert "The top-level object may contain only: summary, facts." in prompt
    assert "Do NOT output source_frame_id." in prompt
    assert '"expires_after_ms"' not in example_line
    assert '"source_frame_id"' not in example_line
    assert example_line == (
        '{"summary":"one short sentence","facts":[{"fact_id":"tac_page_visible",'
        '"state":"seen","evidence_note":"short evidence"}]}'
    )


def test_vision_fact_prompt_respects_explicit_empty_config() -> None:
    prompt = build_vision_fact_prompt(
        vision={"frame_ids": ["1772872445010_000123"], "frame_id": "1772872445010_000123"},
        lang="en",
        config={},
    )

    assert '"facts":[]' in prompt
    assert '"step_bindings":{}' in prompt


def test_vision_fact_prompt_mentions_tac_supt_and_real_fcs_page_boundaries() -> None:
    prompt = build_vision_fact_prompt(
        vision={"frame_ids": ["1772872445010_000123"], "frame_id": "1772872445010_000123"},
        lang="zh",
        config=_minimal_config(),
    )

    assert "tac_page_visible" in prompt
    assert "supt_page_visible" in prompt
    assert "fcs_page_visible" in prompt
    assert "页面选项标签不等于页面本身" in prompt
    assert "PB18" in prompt
    assert "PB15" in prompt
    assert "LEF/TEF/AIL/RUD/STAB/SV1/SV2/CAS" in prompt


def test_vision_fact_prompt_explicitly_distinguishes_bit_root_and_fcsmc_states() -> None:
    prompt = build_vision_fact_prompt(
        vision={"frame_ids": ["1772872445010_000123"], "frame_id": "1772872445010_000123"},
        lang="en",
        config=_minimal_config(),
    )

    assert "bit_root_page_visible" in prompt
    assert "fcsmc_page_visible" in prompt
    assert "PBIT GO is intermediate, not final GO." in prompt
    assert "IN TEST is running, not final GO." in prompt
    assert "fcsmc_final_go_result_visible requires final GO results" in prompt


def test_vision_fact_prompt_lists_all_13_aligned_fact_ids() -> None:
    prompt = build_vision_fact_prompt(
        vision={"frame_ids": ["1772872445010_000123"], "frame_id": "1772872445010_000123"},
        lang="en",
        config=_minimal_config(),
    )

    for fact_id in _minimal_config()["facts_by_id"]:
        assert fact_id in prompt
