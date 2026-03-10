"""
Prompt builder for structured visual-fact extraction.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from adapters.vision_prompting import build_vlm_region_prompt
from core.vision_facts import load_vision_facts_config


def build_vision_fact_prompt(
    *,
    vision: Mapping[str, Any],
    lang: str = "zh",
    config: Mapping[str, Any] | None = None,
) -> str:
    current_config = config or load_vision_facts_config()
    fact_specs = current_config.get("facts_by_id", {})
    fact_ids = [fact_id for fact_id in fact_specs.keys() if isinstance(fact_id, str) and fact_id]
    frame_ids = [item for item in vision.get("frame_ids", []) if isinstance(item, str) and item]
    bindings = current_config.get("step_bindings", {})

    payload = {
        "frame_ids": frame_ids,
        "vision_status": vision.get("status"),
        "primary_frame_id": vision.get("frame_id"),
        "step_bindings": {
            step_id: list(required_facts)
            for step_id, required_facts in bindings.items()
        },
        "facts": [
            {
                "fact_id": fact_id,
                "sticky": bool(fact_specs.get(fact_id, {}).get("sticky")),
                "expires_after_ms": fact_specs.get(fact_id, {}).get("expires_after_ms"),
                "intended_regions": list(fact_specs.get(fact_id, {}).get("intended_regions", [])),
            }
            for fact_id in fact_ids
        ],
    }

    if lang == "zh":
        rules = [
            "你是 SimTutor 的视觉事实抽取器。只能输出一个 JSON 对象，禁止输出 tutor answer、markdown 或自然语言段落。",
            "你必须输出一个 JSON 对象，且顶层只允许 facts 字段；不要输出其他顶层键。",
            "facts 必须是数组；每个 fact_id 只能出现一次。",
            "state 只能是 seen、not_seen、uncertain。",
            "只有在图像证据明确时才能输出 seen；模糊、遮挡、页面不完整时必须输出 uncertain。",
            "source_frame_id 必须引用提供的 frame_ids 之一。",
            "evidence_note 必须简短描述可验证的视觉依据，不得编造 BIOS 值。",
            "fcs_reset_seen 只有在 FCS 页面可见且重置后的叉/故障标记已消失时才可为 seen。",
            "fcs_bit_interaction_seen 只有在右 DDI 的 BIT/FCS 页面证据支持 FCS BIT 正在执行时才可为 seen。",
            "fcs_bit_result_visible 只有在 BIT 结果明确可读时才可为 seen；若只看到页面但结果不清楚，返回 uncertain。",
            "takeoff_trim_seen 只有在 FCS 页面上能看到按下 TAKEOFF TRIM 后的配平/舵面指示变化时才可为 seen。",
            "ins_go 只有在 AMPCD 对准信息明确显示到 GO/0.5 附近时才可为 seen。",
            build_vlm_region_prompt(lang="zh"),
        ]
    else:
        rules = [
            "You are SimTutor's visual fact extractor. Output exactly one JSON object only.",
            "The top-level object must contain only the facts field; do not output any other top-level keys.",
            "facts must be an array, and each fact_id must appear at most once.",
            "state must be one of seen, not_seen, uncertain.",
            "Use seen only when the image evidence is clear; if blurry, occluded, or incomplete, use uncertain.",
            "source_frame_id must cite one of the provided frame_ids.",
            "evidence_note must briefly describe verifiable visual evidence and must not invent BIOS values.",
            "fcs_reset_seen may be seen only when the FCS page is visible and the post-reset X/fault marks are gone.",
            "fcs_bit_interaction_seen may be seen only when the right DDI BIT/FCS page supports that FCS BIT is actively being run.",
            "fcs_bit_result_visible may be seen only when the BIT result is readable; if page is visible but result is unclear, use uncertain.",
            "takeoff_trim_seen may be seen only when the FCS page shows trim/control-surface changes after TAKEOFF TRIM.",
            "ins_go may be seen only when the AMPCD alignment info clearly reaches GO or about 0.5.",
            build_vlm_region_prompt(lang="en"),
        ]

    rendered_rules = "\n".join(f"- {rule}" for rule in rules)
    return (
        f"Rules:\n{rendered_rules}\n"
        "Context JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)}\n"
        "Output JSON shape exactly:\n"
        '{"facts":[{"fact_id":"fcs_page_visible","state":"uncertain","source_frame_id":"1772872445010_000123","confidence":0.0,"evidence_note":"..."}]}'
    )


__all__ = ["build_vision_fact_prompt"]
