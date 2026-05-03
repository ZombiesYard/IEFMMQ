# F/A-18C Run-003 V3 采集计划与视觉 Facts 说明

本文档记录 `fa18c_run003_v3_220` 的手动截图采集计划，以及后续 VLM 预标注、人工复核、SFT 导出和报告撰写时使用的视觉 facts 解释。

## 采集目的

Run-003 V3 不是为了复现真实冷启动流程中各页面出现的自然时间分布，也不是严格类别均衡数据集。它是一个 small-sample, stratified, boundary-focused, hard-negative-enriched 采集计划。

目标是验证：少量人工复核的领域样本，能否让通用 VLM 通过微调学会 F/A-18C 冷启动中的特定视觉边界，并减少关键 false positives，尤其是完成状态和相似页面之间的误判。

## 记录顺序说明

截图文件名里的 frame suffix 只表示实际写入文件的顺序，例如 `_000220`。经过中途修复、删除或补采以后，它可能不连续，也可能和采集计划序号不同。

后续数据处理应以以下字段为准：

| 文件 | 权威字段 | 含义 |
|---|---|---|
| `capture_plan.csv` | `seq` | 计划中的目标截图序号 |
| `capture_plan_progress.csv` | `seq` | 实际截图对应的计划序号 |
| `capture_plan_progress.jsonl` | `seq` | 实际截图对应的计划序号 |
| `capture_plan_progress.*` | `captured_frame_id` | 该计划项实际绑定的 frame id |
| `capture_index.jsonl` | `frame_id` | raw/artifact 图像路径索引 |

因此，文件名序号不连续不影响训练或标注。只要 `capture_plan_progress.*`、`capture_index.jsonl`、`frames.jsonl` 中的 `captured_frame_id/frame_id` 和图像路径一致即可。

## 标注状态

每个视觉 fact 使用三分类状态：

| state | 含义 |
|---|---|
| `seen` | 当前组合屏图像中清楚可见该视觉证据 |
| `not_seen` | 当前组合屏图像中没有看到该视觉证据 |
| `uncertain` | 图像模糊、遮挡、过渡、裁切或文字不可读，无法可靠判断 |

不要让模型输出 `confidence`、`source_frame_id`、`frame_id` 或 `sticky`。这些字段不是图像视觉证据，不应进入训练目标。

## 建议视觉 Facts

Run-003 V3 面向新的低层视觉 facts。它们用于结构化视觉证据提取，不直接等同于最终步骤完成判断。

| fact_id | 视觉含义 | 备注 |
|---|---|---|
| `tac_page_visible` | 任意 DDI/AMPCD 上可见 TAC / TAC MENU 页面 | 用于导航恢复，PB18 可回到 TAC/SUPT 路径 |
| `supt_page_visible` | 任意 DDI/AMPCD 上可见 SUPT 页面 | SUPT 上可进入 BIT、FCS、HSI 等页面 |
| `fcs_page_visible` | 可见真正的 FCS 飞控页面 | 注意它不是 FCS-MC BIT 子页面 |
| `fcs_page_x_marks_visible` | FCS 页面上的 SV1/SV2 等通道格子内可见明显 X/fault fills | FCS reset 前状态的低层视觉证据 |
| `bit_root_page_visible` | 可见 BIT FAILURES / BIT root 页面 | 合并旧 `bit_root_page_visible` 与 `bit_page_failure_visible` |
| `fcsmc_page_visible` | 可见 FCS-MC BIT 子页面 | 不绑定右 DDI，任意显示器上出现均为 seen |
| `fcsmc_intermediate_result_visible` | FCS-MC 页面中可见 PBIT GO 等中间状态 | PBIT GO 不是最终 GO |
| `fcsmc_in_test_visible` | FCS-MC 页面中明确可见 IN TEST | 运行中状态，不是完成 |
| `fcsmc_final_go_result_visible` | FCS-MC 页面中可见最终 GO 结果 | 完成类视觉证据，需严谨标注 |
| `hsi_page_visible` | 可见 HSI / INS alignment 相关页面 | 不绑定 AMPCD，但默认常见于 AMPCD |
| `hsi_map_layer_visible` | HSI 上可见 MAP overlay/地图叠加层 | INS 字符被遮挡或干扰时尤其重要 |
| `ins_grnd_alignment_text_visible` | 可见 GRND/QUAL/countdown 等对准运行文字 | 表示对准进行中，不等于 OK |
| `ins_ok_text_visible` | 可见 INS/HSI 对准 OK 文字 | 完成类视觉证据，替代旧 `ins_go` |

可由下游逻辑派生、但不建议作为 VLM 直接输出的事实：

| derived item | 推导方式 |
|---|---|
| `no_relevant_page_visible` | 上述目标页 facts 均不为 `seen` 时，由后处理推导 |
| `need_navigation_recovery` | 目标页不可见且当前步骤需要特定页面时，由下游程序状态推导 |

## Category 到 Facts 的预期关系

`category_id` 是采集计划中的主目标，不是单标签分类。由于输入是组合屏，一张图中可以同时有多个 facts 为 `seen`，例如 FCS 页旁边自然出现 SUPT 页。人工复核时应按图像内容照实标注。

| category_id | 主要 positive facts | 主要 hard negatives / 注意点 |
|---|---|---|
| `tac_page` | `tac_page_visible=seen` | 如果另一个屏幕自然出现 SUPT，也应标 `supt_page_visible=seen` |
| `supt_page` | `supt_page_visible=seen` | TAC/SUPT 可同时出现，按图像照实标注 |
| `other_non_target_page` | 通常无目标页 facts 为 seen | 训练 none-of-target，不要求识别 EW/SA/RADAR/STORES/MIDS 等具体页面名 |
| `fcs_page_with_x` | `fcs_page_visible=seen`, `fcs_page_x_marks_visible=seen` | 不要把 FCS 页面误判为 FCS-MC 页面 |
| `fcs_page_without_obvious_x` | `fcs_page_visible=seen`, `fcs_page_x_marks_visible=not_seen` | 单帧只判断 X/fault fills 是否清楚可见，不根据历史推断 reset 是否完成 |
| `bit_root_failures` | `bit_root_page_visible=seen` | `fcsmc_page_visible=not_seen`，除非图中另一个屏幕真的显示 FCS-MC |
| `fcsmc_pbit_go` | `fcsmc_page_visible=seen`, `fcsmc_intermediate_result_visible=seen` | `fcsmc_final_go_result_visible=not_seen`; PBIT GO 不是 final GO |
| `fcsmc_in_test` | `fcsmc_page_visible=seen`, `fcsmc_in_test_visible=seen` | `fcsmc_final_go_result_visible=not_seen`; IN TEST 是运行中 |
| `fcsmc_final_go` | `fcsmc_page_visible=seen`, `fcsmc_final_go_result_visible=seen` | 不应同时标 PBIT/IN TEST 中间状态，除非图中确实同时可见 |
| `hsi_map_overlay` | `hsi_page_visible=seen`, `hsi_map_layer_visible=seen` | `ins_ok_text_visible` 若不可读应为 `uncertain` 或 `not_seen`，不要猜 OK |
| `hsi_alignment_running_map_off` | `hsi_page_visible=seen`, `hsi_map_layer_visible=not_seen`, `ins_grnd_alignment_text_visible=seen` | `ins_ok_text_visible=not_seen`; countdown/QUAL 运行中不是 OK |
| `hsi_ok` | `hsi_page_visible=seen`, `ins_ok_text_visible=seen` | 只有 OK 文字清楚可见才标 seen |
| `unreadable_transition` | 视图像决定 | 模糊、遮挡、过渡、裁切时优先 `uncertain`，不要猜测 seen |

## 完整截图采集表

| seq range | category_id | count | target_display | left_ddi_content | ampcd_content | right_ddi_content | expected_key_facts |
|---|---:|---:|---|---|---|---|---|
| 001-008 | `tac_page` | 8 | `left_ddi` | TAC / TAC MENU | Natural cold-start context; HSI/MAP is acceptable. | Natural context; avoid BIT root or FCS-MC if practical. | `tac_page_visible=seen; supt_page_visible=not_seen; fcs_page_visible=not_seen` |
| 009-010 | `tac_page` | 2 | `right_ddi` | Natural non-target or blank page. | Natural cold-start context; HSI/MAP is acceptable. | TAC / TAC MENU | `tac_page_visible=seen; supt_page_visible=not_seen; fcs_page_visible=not_seen` |
| 011-018 | `supt_page` | 8 | `left_ddi` | SUPT page | Natural cold-start context; HSI/MAP is acceptable. | Natural context; avoid BIT root or FCS-MC if practical. | `supt_page_visible=seen; tac_page_visible=not_seen; fcs_page_visible=not_seen` |
| 019-020 | `supt_page` | 2 | `right_ddi` | Natural non-target or blank page. | Natural cold-start context; HSI/MAP is acceptable. | SUPT page | `supt_page_visible=seen; tac_page_visible=not_seen; fcs_page_visible=not_seen` |
| 021-026 | `other_non_target_page` | 6 | `left_ddi` | EW/RADAR/STORES/MIDS/SA/DATA/FPAS/CHKLIST or another non-target page | Non-HSI page, blank, or map-free non-target if practical. | Non-target or blank page. | all target page facts should be `not_seen` unless the image is genuinely unreadable |
| 027-032 | `other_non_target_page` | 6 | `right_ddi` | Non-target or blank page. | Non-HSI page, blank, or map-free non-target if practical. | EW/RADAR/STORES/MIDS/SA/DATA/FPAS/CHKLIST or another non-target page | all target page facts should be `not_seen` unless the image is genuinely unreadable |
| 033-036 | `other_non_target_page` | 4 | `ampcd` | Non-target or blank page. | Non-HSI/non-INS page if practical; otherwise blank/dark stable display. | Non-target or blank page. | `hsi_page_visible=not_seen; other target page facts=not_seen` |
| 037-044 | `other_non_target_page` | 8 | `all` | EW/RADAR/STORES/MIDS/SA/DATA/FPAS/CHKLIST or another non-target page | Non-HSI page, blank, or stable non-target display. | EW/RADAR/STORES/MIDS/SA/DATA/FPAS/CHKLIST or another non-target page | all target page facts should be `not_seen` |
| 045-048 | `other_non_target_page` | 4 | `all` | Non-target or stable blank. | Non-HSI non-target or stable blank. | Non-target or stable blank. | all target page facts should be `not_seen` |
| 049-060 | `fcs_page_with_x` | 12 | `left_ddi` | FCS page with obvious X/fault fills | Natural cold-start context; HSI/MAP is acceptable. | Natural context; avoid BIT root or FCS-MC if practical. | `fcs_page_visible=seen; fcs_page_x_marks_visible=seen; fcsmc_page_visible=not_seen` |
| 061-064 | `fcs_page_with_x` | 4 | `right_ddi` | Natural non-target or blank page. | Natural cold-start context; HSI/MAP is acceptable. | FCS page with obvious X/fault fills | `fcs_page_visible=seen; fcs_page_x_marks_visible=seen; fcsmc_page_visible=not_seen` |
| 065-074 | `fcs_page_without_obvious_x` | 10 | `left_ddi` | FCS page without obvious X/fault fills | Natural cold-start context; HSI/MAP is acceptable. | Natural context; avoid BIT root or FCS-MC if practical. | `fcs_page_visible=seen; fcs_page_x_marks_visible=not_seen; fcsmc_page_visible=not_seen` |
| 075-078 | `fcs_page_without_obvious_x` | 4 | `right_ddi` | Natural non-target or blank page. | Natural cold-start context; HSI/MAP is acceptable. | FCS page without obvious X/fault fills | `fcs_page_visible=seen; fcs_page_x_marks_visible=not_seen; fcsmc_page_visible=not_seen` |
| 079-092 | `bit_root_failures` | 14 | `right_ddi` | Natural non-target or SUPT only if needed for navigation. | Natural cold-start context; HSI/MAP is acceptable. | BIT FAILURES / BIT root page | `bit_root_page_visible=seen; fcsmc_page_visible=not_seen` |
| 093-096 | `bit_root_failures` | 4 | `left_ddi` | BIT FAILURES / BIT root page | Natural cold-start context; HSI/MAP is acceptable. | Natural non-target or blank page. | `bit_root_page_visible=seen; fcsmc_page_visible=not_seen` |
| 097-112 | `fcsmc_pbit_go` | 16 | `right_ddi` | Natural non-target or blank page. | Natural cold-start context; HSI/MAP is acceptable. | FCS-MC page with FCSA/FCSB PBIT GO | `fcsmc_page_visible=seen; fcsmc_intermediate_result_visible=seen; fcsmc_final_go_result_visible=not_seen` |
| 113-114 | `fcsmc_pbit_go` | 2 | `left_ddi` | FCS-MC page with FCSA/FCSB PBIT GO | Natural cold-start context; HSI/MAP is acceptable. | Natural non-target or blank page. | `fcsmc_page_visible=seen; fcsmc_intermediate_result_visible=seen; fcsmc_final_go_result_visible=not_seen` |
| 115-130 | `fcsmc_in_test` | 16 | `right_ddi` | Natural non-target or blank page. | Natural cold-start context; HSI/MAP is acceptable. | FCS-MC page showing IN TEST | `fcsmc_page_visible=seen; fcsmc_in_test_visible=seen; fcsmc_final_go_result_visible=not_seen` |
| 131-132 | `fcsmc_in_test` | 2 | `left_ddi` | FCS-MC page showing IN TEST | Natural cold-start context; HSI/MAP is acceptable. | Natural non-target or blank page. | `fcsmc_page_visible=seen; fcsmc_in_test_visible=seen; fcsmc_final_go_result_visible=not_seen` |
| 133-148 | `fcsmc_final_go` | 16 | `right_ddi` | Natural non-target or blank page. | Natural cold-start context; HSI/MAP is acceptable. | FCS-MC page with final GO results | `fcsmc_page_visible=seen; fcsmc_final_go_result_visible=seen; fcsmc_intermediate_result_visible=not_seen; fcsmc_in_test_visible=not_seen` |
| 149-150 | `fcsmc_final_go` | 2 | `left_ddi` | FCS-MC page with final GO results | Natural cold-start context; HSI/MAP is acceptable. | Natural non-target or blank page. | `fcsmc_page_visible=seen; fcsmc_final_go_result_visible=seen; fcsmc_intermediate_result_visible=not_seen; fcsmc_in_test_visible=not_seen` |
| 151-170 | `hsi_map_overlay` | 20 | `ampcd` | Natural non-target or procedure context. | HSI with MAP overlay; INS/QUAL text may be hard to read. | Natural non-target or procedure context. | `hsi_page_visible=seen; hsi_map_layer_visible=seen; ins_ok_text_visible=not_seen/uncertain` |
| 171-172 | `hsi_map_overlay` | 2 | `ampcd` | Natural non-target, TAC/SUPT navigation context, or stable blank page. | HSI with MAP overlay; use a different range/zoom/clutter state if practical. | Natural non-target, TAC/SUPT navigation context, or stable blank page. | `hsi_page_visible=seen; hsi_map_layer_visible=seen; ins_ok_text_visible=not_seen/uncertain` |
| 173-174 | `hsi_map_overlay` | 2 | `ampcd` | Natural non-target, TAC/SUPT navigation context, or stable blank page. | HSI with MAP overlay; prefer hard-to-read INS/QUAL text, clutter, or similar-looking map symbols. | Natural non-target, TAC/SUPT navigation context, or stable blank page. | `hsi_page_visible=seen; hsi_map_layer_visible=seen; ins_ok_text_visible=not_seen/uncertain` |
| 175-192 | `hsi_alignment_running_map_off` | 18 | `ampcd` | Natural non-target or procedure context. | HSI with MAP off; GRND/QUAL/countdown visible; no OK. | Natural non-target or procedure context. | `hsi_page_visible=seen; hsi_map_layer_visible=not_seen; ins_grnd_alignment_text_visible=seen; ins_ok_text_visible=not_seen` |
| 193-194 | `hsi_alignment_running_map_off` | 2 | `left_ddi` | HSI with MAP off; GRND/QUAL/countdown visible; no OK. | Natural non-HSI or blank if practical. | Natural non-target or blank page. | `hsi_page_visible=seen; hsi_map_layer_visible=not_seen; ins_grnd_alignment_text_visible=seen; ins_ok_text_visible=not_seen` |
| 195-196 | `hsi_alignment_running_map_off` | 2 | `right_ddi` | Natural non-target or blank page. | Natural non-HSI or blank if practical. | HSI with MAP off; GRND/QUAL/countdown visible; no OK. | `hsi_page_visible=seen; hsi_map_layer_visible=not_seen; ins_grnd_alignment_text_visible=seen; ins_ok_text_visible=not_seen` |
| 197-206 | `hsi_ok` | 10 | `ampcd` | Natural non-target or procedure context. | HSI with clear OK text. | Natural non-target or procedure context. | `hsi_page_visible=seen; ins_ok_text_visible=seen` |
| 207 | `hsi_ok` | 1 | `left_ddi` | HSI with clear OK text. | Natural non-HSI or blank if practical. | Natural non-target or blank page. | `hsi_page_visible=seen; ins_ok_text_visible=seen` |
| 208 | `hsi_ok` | 1 | `right_ddi` | Natural non-target or blank page. | Natural non-HSI or blank if practical. | HSI with clear OK text. | `hsi_page_visible=seen; ins_ok_text_visible=seen` |
| 209-212 | `unreadable_transition` | 4 | `any` | Any display may be changing pages, blurred, or partly unreadable. | Any display may be changing pages, blurred, or partly unreadable. | Any display may be changing pages, blurred, or partly unreadable. | ambiguous target facts should be `uncertain` rather than guessed `seen` |
| 213-216 | `unreadable_transition` | 4 | `any` | Stable blank/dark or partly unreadable display. | Stable blank/dark or partly unreadable display. | Stable blank/dark or partly unreadable display. | target facts should be `not_seen` when truly absent, `uncertain` when unreadable |
| 217-220 | `unreadable_transition` | 4 | `any` | Target or non-target page may be cropped/obscured. | Target or non-target page may be cropped/obscured. | Target or non-target page may be cropped/obscured. | ambiguous target facts should be `uncertain` rather than guessed `seen` |

## 报告中建议写法

Run-003 V3 可以在报告中描述为：

> Run-003 V3 uses a stratified, boundary-focused, hard-negative-enriched capture plan rather than a natural temporal distribution. The plan deliberately includes non-target DDI/AMPCD pages, intermediate FCS-MC states, HSI map-overlay hard negatives, alignment-running states without OK, and unreadable transitions. This design targets the dominant false-positive modes observed in the previous heldout benchmark.

中文表述：

> Run-003 V3 不是自然时间分布采样，而是一个分层的视觉边界采集计划。它有意加入非目标页面、FCS-MC 中间状态、HSI 地图叠加层 hard negatives、无 OK 的对准运行状态以及不可读过渡帧，用于针对上一轮 holdout benchmark 中暴露出的关键 false positives。
