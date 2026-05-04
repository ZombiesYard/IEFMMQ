"""
Prompt builder for structured visual-fact extraction.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from adapters.vision_prompting import build_vlm_region_prompt
from core.vision_facts import load_vision_facts_config


def build_vision_fact_prompt(
    *,
    vision: Mapping[str, Any],
    lang: str = "zh",
    config: Mapping[str, Any] | None = None,
) -> str:
    current_config = config if config is not None else load_vision_facts_config()
    fact_specs = current_config.get("facts_by_id", {})
    fact_ids = [fact_id for fact_id in fact_specs.keys() if isinstance(fact_id, str) and fact_id]
    frame_ids = [item for item in vision.get("frame_ids", []) if isinstance(item, str) and item]
    bindings = current_config.get("step_bindings", {})

    payload = {
        "frame_ids": frame_ids,
        "vision_status": vision.get("status"),
        "primary_frame_id": vision.get("frame_id"),
        "step_bindings": {
            step_id: {
                "all_of": list(required_facts.get("all_of", [])) if isinstance(required_facts, Mapping) else [],
                "any_of": list(required_facts.get("any_of", [])) if isinstance(required_facts, Mapping) else [],
            }
            for step_id, required_facts in bindings.items()
        },
        "facts": fact_ids,
    }

    if lang == "en":
        rules = [
            "You are the SimTutor visual fact extractor for the F/A-18C cold-start flow.",
            "The input contains one or two composite-panel images with fixed top-to-bottom regions: left_ddi, ampcd, right_ddi.",
            "If two images are provided, they are ordered as pre-trigger frame first and trigger frame second.",
            "Inspect only the provided image or images and label the configured visual facts.",
            "If the image is blurry, obstructed, or a fact cannot be confirmed from the provided image or images, use state='uncertain'.",
            "The top-level object may contain only: summary, facts.",
            "Each fact object may contain only: fact_id, state, evidence_note.",
            "Do NOT output frame_id.",
            "Do NOT output source_frame_id.",
            "Do NOT output confidence, expires_after_ms, sticky, actions, or tutor advice.",
            "Except for tac_page_visible and supt_page_visible, page option labels are not pages. Do not mark a page visible just because its name appears as a pushbutton/menu option.",
            "tac_page_visible means a small TAC/MENU navigation label is visible.",
            "supt_page_visible means a small SUPT option label is visible.",
            "If the left DDI is on TAC, PB18 is the navigation step toward SUPT. If the left DDI is already on SUPT, PB15 is the step toward the real FCS page.",
            "fcs_page_visible is the dedicated flight-control page with readable LEF/TEF/AIL/RUD/STAB/SV1/SV2/CAS grid labels. A page label or menu option is not enough.",
            "fcs_page_x_marks_visible only means literal X/fault fills inside FCS page channel boxes.",
            "bit_root_page_visible is the BIT FAILURES/root page; an FCS-MC entry label is not the FCS-MC subpage.",
            "fcsmc_page_visible requires a readable FCS-MC title/subpage with MC1, MC2, FCSA, and FCSB rows.",
            "PBIT GO is intermediate, not final GO.",
            "IN TEST is running, not final GO.",
            "fcsmc_intermediate_result_visible is for readable intermediate FCS-MC result states such as PBIT GO, but not the final completed GO result.",
            "fcsmc_final_go_result_visible requires final GO results for MC1, MC2, FCSA, and FCSB together.",
            "hsi_page_visible means an HSI navigation/POS page is visible; it does not imply INS alignment text or INS OK.",
            "hsi_map_layer_visible requires a colored topographic/chart MAP background; black HSI symbology alone is not MAP.",
            "ins_grnd_alignment_text_visible requires a literal GRND alignment block with QUAL or TIME/countdown text.",
            "ins_ok_text_visible requires clearly readable OK near the INS alignment block, usually near QUAL.",
            build_vlm_region_prompt(lang="en"),
        ]
        example = (
            '{"summary":"one short sentence","facts":[{"fact_id":"tac_page_visible",'
            '"state":"seen","evidence_note":"short evidence"}]}'
        )
    else:
        rules = [
            "你是 SimTutor 的视觉事实抽取器，负责 F/A-18C 冷启动流程的视觉事实判断。",
            "输入可能包含一张或两张组合面板图，固定从上到下依次是：left_ddi、ampcd、right_ddi。",
            "如果提供两张图，顺序是 pre_trigger_frame 在前、trigger_frame 在后。",
            "只根据当前提供的图像，为配置中的视觉 facts 输出标注。",
            "如果当前图像模糊、遮挡、页面不完整、或者仅凭当前提供的图像无法确认，请使用 state='uncertain'。",
            "顶层只允许包含：summary、facts。",
            "facts 数组里的每个对象只允许包含：fact_id、state、evidence_note。",
            "严禁输出 frame_id。",
            "严禁输出 source_frame_id。",
            "严禁输出 confidence、expires_after_ms、sticky、actions 或 tutor advice。",
            "除了 tac_page_visible 和 supt_page_visible 之外，页面选项标签不等于页面本身。不要因为某个 pushbutton/menu option 写着页面名就标 seen。",
            "tac_page_visible 就是单独小的 TAC/MENU 导航标签可见。",
            "supt_page_visible 就是单独小的 SUPT 选项标签可见。",
            "如果左 DDI 还在 TAC 页，下一步导航应是 PB18 进入 SUPT；如果左 DDI 已在 SUPT 页，下一步导航应是 PB15 进入真正的 FCS 页面。",
            "fcs_page_visible 是专用飞控 FCS 页面，需要能读到 LEF/TEF/AIL/RUD/STAB/SV1/SV2/CAS grid 等标签；菜单项或页面名不算。",
            "fcs_page_x_marks_visible 只表示 FCS 页面通道格子内的字面 X/fault fills。",
            "bit_root_page_visible 是 BIT FAILURES/root 页面；FCS-MC 入口标签不等于 FCS-MC 子页面。",
            "fcsmc_page_visible 需要能读到 FCS-MC 标题/子页面，并有 MC1、MC2、FCSA、FCSB 等行。",
            "PBIT GO 是中间态，不是 final GO。",
            "IN TEST 是运行中，不是 final GO。",
            "fcsmc_intermediate_result_visible 表示能明确读到 PBIT GO 等 FCS-MC 中间结果，而不是最终完成结果。",
            "fcsmc_final_go_result_visible 必须是 MC1、MC2、FCSA、FCSB 同页可读的最终 GO 结果。",
            "hsi_page_visible 是 HSI navigation/POS 页面可见；它本身不代表 INS 对准文字或 INS OK 可见。",
            "hsi_map_layer_visible 需要彩色地形/航图 MAP 背景；黑底 HSI 符号本身不算 MAP。",
            "ins_grnd_alignment_text_visible 需要字面 GRND 对准文字块，并伴随 QUAL 或 TIME/countdown。",
            "ins_ok_text_visible 需要在 INS 对准文字块附近清楚读到 OK，通常在 QUAL 附近。",
            build_vlm_region_prompt(lang="zh"),
        ]
        example = (
            '{"summary":"一句短总结","facts":[{"fact_id":"tac_page_visible",'
            '"state":"seen","evidence_note":"简短证据"}]}'
        )

    rendered_rules = "\n".join(f"- {rule}" for rule in rules)
    return (
        f"Rules:\n{rendered_rules}\n"
        "Context JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)}\n"
        "Output JSON shape exactly:\n"
        f"{example}"
    )


__all__ = ["build_vision_fact_prompt"]
