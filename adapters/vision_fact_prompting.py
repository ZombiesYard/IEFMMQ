"""
Prompt builder for structured visual-fact extraction.
"""

from __future__ import annotations

from typing import Any, Mapping

from adapters.vision_prompting import build_vlm_region_prompt
from core.vision_facts import load_vision_facts_config


def _render_fact_list(fact_ids: list[str], *, empty_label: str) -> str:
    if not fact_ids:
        return empty_label
    return "\n".join(f"{index}. {fact_id}" for index, fact_id in enumerate(fact_ids, start=1))


def build_vision_fact_prompt(
    *,
    vision: Mapping[str, Any] | None = None,
    lang: str = "zh",
    config: Mapping[str, Any] | None = None,
) -> str:
    del vision  # intentionally unused; kept for backward compat
    current_config = config if config is not None else load_vision_facts_config()
    fact_specs = current_config.get("facts_by_id", {})
    fact_ids = [fact_id for fact_id in fact_specs.keys() if isinstance(fact_id, str) and fact_id]
    _include_instruction = (
        "Include all configured facts. When unsure, use state='uncertain'."
        if fact_ids
        else "No facts are configured; output an empty facts array."
    )

    if lang == "en":
        facts_block = _render_fact_list(fact_ids, empty_label="(none configured)")
        if fact_ids:
            example = (
                '{"summary":"one short sentence","facts":[{"fact_id":"tac_page_visible",'
                '"state":"seen","evidence_note":"short evidence"}]}'
            )
        else:
            example = '{"summary":"one short sentence","facts":[]}'
        return (
            "You are the SimTutor visual fact extractor for the F/A-18C cold-start flow.\n"
            "The input contains one or two composite-panel images with fixed top-to-bottom regions: left_ddi, ampcd, right_ddi.\n"
            "If two images are provided, they are ordered as pre-trigger frame first and trigger frame second.\n"
            "\n"
            "Task:\n"
            "Inspect only the provided image or images and label the configured visual facts below.\n"
            "If an image is blurry, obstructed, incomplete, or a fact cannot be confirmed from the provided image or images alone, use state='uncertain'.\n"
            "Use only visible evidence from the provided image or images.\n"
            "Do not infer from procedure history or expected step.\n"
            "Ignore runtime metadata and system expectations.\n"
            "\n"
            "Facts to label:\n"
            f"{facts_block}\n"
            "\n"
            "Key decision boundaries:\n"
            "- Page option labels are not pages. Do not mark a page visible just because its name appears as a pushbutton/menu option.\n"
            "- tac_page_visible requires the actual TAC/TAC MENU page; a small TAC/MENU navigation label alone is not enough.\n"
            "- supt_page_visible requires the actual SUPT/SUPT MENU page; a small SUPT option label alone is not enough.\n"
            "- fcs_page_visible is the dedicated flight-control page with readable LEF/TEF/AIL/RUD/STAB/SV1/SV2/CAS grid labels. A page label or menu option is not enough.\n"
            "- fcs_page_x_marks_visible only means literal X/fault fills inside FCS page channel boxes.\n"
            "- bit_root_page_visible is the BIT FAILURES/root page; an FCS-MC entry label is not the FCS-MC subpage.\n"
            "- fcsmc_page_visible requires a readable FCS-MC title/subpage with MC1, MC2, FCSA, and FCSB rows.\n"
            "- PBIT GO is intermediate, not final GO.\n"
            "- IN TEST is running, not final GO.\n"
            "- fcsmc_intermediate_result_visible is for readable intermediate FCS-MC result states such as PBIT GO, but not the final completed GO result.\n"
            "- fcsmc_final_go_result_visible requires final GO results for MC1, MC2, FCSA, and FCSB together.\n"
            "- hsi_page_visible means an HSI navigation/POS page is visible; it does not imply INS alignment text or INS OK.\n"
            "- hsi_map_layer_visible requires a colored topographic/chart MAP background; black HSI symbology alone is not MAP.\n"
            "- ins_grnd_alignment_text_visible requires a literal GRND alignment block with QUAL or TIME/countdown text.\n"
            "- ins_ok_text_visible requires clearly readable OK near the INS alignment block, usually near QUAL.\n"
            f"- {build_vlm_region_prompt(lang='en')}\n"
            "\n"
            "Output requirements:\n"
            "- Output exactly one JSON object.\n"
            "- The top-level object may contain only: summary, facts.\n"
            "- Each fact object may contain only: fact_id, state, evidence_note.\n"
            "- Do NOT output frame_id.\n"
            "- Do NOT output source_frame_id.\n"
            "- Do NOT output confidence, expires_after_ms, sticky, actions, or tutor advice.\n"
            "- Do NOT wrap the JSON in markdown or any extra prose.\n"
            f"- {_include_instruction}\n"
            "\n"
            "Return format:\n"
            f"{example}"
        )
    else:
        facts_block = _render_fact_list(fact_ids, empty_label="（当前没有配置 fact）")
        if fact_ids:
            example = (
                '{"summary":"一句短总结","facts":[{"fact_id":"tac_page_visible",'
                '"state":"seen","evidence_note":"简短证据"}]}'
            )
            zh_include = "配置中的 facts 都要覆盖；拿不准时用 state='uncertain'。"
        else:
            example = '{"summary":"一句短总结","facts":[]}'
            zh_include = "当前没有配置 fact；输出空的 facts 数组。"
        return (
            "你是 SimTutor 的视觉事实抽取器，负责 F/A-18C 冷启动流程的视觉事实判断。\n"
            "输入可能包含一张或两张组合面板图，固定从上到下依次是：left_ddi、ampcd、right_ddi。\n"
            "如果提供两张图，顺序是 pre_trigger_frame 在前、trigger_frame 在后。\n"
            "\n"
            "任务：\n"
            "只根据当前提供的图像，为下面配置中的视觉 facts 输出标注。\n"
            "如果当前图像模糊、遮挡、页面不完整、或者仅凭当前提供的图像无法确认，请使用 state='uncertain'。\n"
            "只使用当前图像里可见的证据。\n"
            "不要根据流程历史或预期步骤去推断。\n"
            "忽略运行时元数据和系统预期。\n"
            "\n"
            "需要标注的 fact：\n"
            f"{facts_block}\n"
            "\n"
            "关键判别边界：\n"
            "- 页面选项标签不等于页面本身。不要因为某个 pushbutton/menu option 写着页面名就标 seen。\n"
            "- tac_page_visible 需要实际 TAC/TAC MENU 页面；单独小的 TAC/MENU 导航标签不够。\n"
            "- supt_page_visible 需要实际 SUPT/SUPT MENU 页面；单独小的 SUPT 选项标签不够。\n"
            "- fcs_page_visible 是专用飞控 FCS 页面，需要能读到 LEF/TEF/AIL/RUD/STAB/SV1/SV2/CAS grid 等标签；菜单项或页面名不算。\n"
            "- fcs_page_x_marks_visible 只表示 FCS 页面通道格子内的字面 X/fault fills。\n"
            "- bit_root_page_visible 是 BIT FAILURES/root 页面；FCS-MC 入口标签不等于 FCS-MC 子页面。\n"
            "- fcsmc_page_visible 需要能读到 FCS-MC 标题/子页面，并有 MC1、MC2、FCSA、FCSB 等行。\n"
            "- PBIT GO 是中间态，不是 final GO。\n"
            "- IN TEST 是运行中，不是 final GO。\n"
            "- fcsmc_intermediate_result_visible 表示能明确读到 PBIT GO 等 FCS-MC 中间结果，而不是最终完成结果。\n"
            "- fcsmc_final_go_result_visible 必须是 MC1、MC2、FCSA、FCSB 同页可读的最终 GO 结果。\n"
            "- hsi_page_visible 是 HSI navigation/POS 页面可见；它本身不代表 INS 对准文字或 INS OK 可见。\n"
            "- hsi_map_layer_visible 需要彩色地形/航图 MAP 背景；黑底 HSI 符号本身不算 MAP。\n"
            "- ins_grnd_alignment_text_visible 需要字面 GRND 对准文字块，并伴随 QUAL 或 TIME/countdown。\n"
            "- ins_ok_text_visible 需要在 INS 对准文字块附近清楚读到 OK，通常在 QUAL 附近。\n"
            f"- {build_vlm_region_prompt(lang='zh')}\n"
            "\n"
            "输出要求：\n"
            "- 只允许输出一个 JSON 对象。\n"
            "- 顶层只允许包含：summary、facts。\n"
            "- facts 数组里的每个对象只允许包含：fact_id、state、evidence_note。\n"
            "- 严禁输出 frame_id。\n"
            "- 严禁输出 source_frame_id。\n"
            "- 严禁输出 confidence、expires_after_ms、sticky、actions 或 tutor advice。\n"
            "- 不要输出 markdown，也不要输出额外解释。\n"
            f"- {zh_include}\n"
            "\n"
            "返回格式：\n"
            f"{example}"
        )


__all__ = ["build_vision_fact_prompt"]
