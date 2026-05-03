# F/A-18C Run-005 Composition-Rebalance 截图操作指南

本文档对应截图工具内置计划：

```text
fa18c_run005_composition_rebalance_122
```

Run-005 不替代 Run-003，而是作为补采数据加入训练。目标是修正 Run-003 中多个关键 facts 的 `not_seen` 先验过强问题，同时补充真实冷启动中常见的多屏 fact 共现。

## 采集目标

Run-005 是 composition-balanced supplemental set，不是自然频率采样，也不是 50/50 标签均衡数据集。

目标包括：

- 补 `bit_root_page_visible`、`fcs_page_visible`、`fcsmc_page_visible` 的正例和真实共现。
- 补 FCS-MC `PBIT GO`、`IN TEST`、`final GO` 的小字状态正例。
- 补 `FCS-MC final GO + HSI no OK` hard negatives，防止模型把 FCS 完成误推成 INS 完成。
- 补少量 `ins_ok_text_visible` 正例，但不让 OK 正例过多。
- 保留少量 non-target 和 transition/unreadable 样本。

## 自然页规则

如果表格写 `non-target` 或 `stable blank`，优先使用：

```text
EW
SA
RADAR / ATTACK RADAR
STORES
MIDS
DATA
FPAS
CHKLIST
HUD page
稳定空白/暗屏
```

除非表格明确要求，不要把以下页面放在自然页位置：

```text
TAC
SUPT
FCS page
BIT root
FCS-MC
HSI
HSI MAP
HSI GRND/QUAL/TIME
HSI OK
```

如果实际操作中无法避免目标页出现在自然页位置，人工标注时必须按图像真实内容标注，不要为了符合采集表而故意标错。

## 截图表

| seq range | count | category_id | left DDI | AMPCD | right DDI | 主要目标 |
|---|---:|---|---|---|---|---|
| 001-012 | 12 | `default_TAC_HSI_MAP_BITROOT` | TAC / TAC MENU | HSI + MAP overlay, no OK | BIT FAILURES / BIT root | 默认冷启动共现 |
| 013-022 | 10 | `SUPT_HSI_MAP_BITROOT` | SUPT page | HSI + MAP overlay, no OK | BIT FAILURES / BIT root | SUPT + BIT root 共现 |
| 023-032 | 10 | `FCS_X_HSI_MAP_BITROOT` | FCS page with obvious X/fault fills | HSI + MAP overlay, no OK | BIT FAILURES / BIT root | FCS X + BIT root 共现 |
| 033-040 | 8 | `FCS_NOX_HSI_GRND_BITROOT` | FCS page without obvious X/fault fills | HSI MAP off, GRND/QUAL/TIME visible, no OK | BIT FAILURES / BIT root | FCS no-X + GRND + BIT root |
| 041-048 | 8 | `FCS_X_HSI_GRND_FCSMC_PBIT` | FCS page with X/fault fills | HSI MAP off, GRND/QUAL/TIME visible, no OK | FCS-MC with PBIT GO | PBIT 正例，final GO hard negative |
| 049-055 | 7 | `SUPT_HSI_GRND_FCSMC_PBIT` | SUPT page | HSI MAP off, GRND/QUAL/TIME visible, no OK | FCS-MC with PBIT GO | SUPT + PBIT 共现 |
| 056-063 | 8 | `TAC_HSI_GRND_FCSMC_INTEST` | TAC / TAC MENU | HSI MAP off, GRND/QUAL/TIME visible, no OK | FCS-MC showing IN TEST | IN TEST 正例 |
| 064-070 | 7 | `FCS_NOX_HSI_GRND_FCSMC_INTEST` | FCS page without obvious X/fault fills | HSI MAP off, GRND/QUAL/TIME visible, no OK | FCS-MC showing IN TEST | FCS + IN TEST 共现 |
| 071-079 | 9 | `SUPT_HSI_GRND_FCSMC_FINALGO` | SUPT page | HSI MAP off, GRND/QUAL/TIME visible, no OK | FCS-MC final GO | final GO 正例，INS OK hard negative |
| 080-086 | 7 | `FCS_NOX_HSI_GRND_FCSMC_FINALGO` | FCS page without obvious X/fault fills | HSI MAP off, GRND/QUAL/TIME visible, no OK | FCS-MC final GO | FCS final GO 共现 |
| 087-092 | 6 | `FCSMC_FINALGO_HSI_OK` | non-target page or stable blank | HSI with clear OK near QUAL/GRND | FCS-MC final GO | 双完成少量正例 |
| 093-096 | 4 | `TAC_HSI_OK_NO_FCSMC` | TAC / TAC MENU | HSI with clear OK near QUAL/GRND | non-target page or stable blank | INS OK 正例，不绑定 FCS-MC |
| 097-100 | 4 | `SUPT_HSI_OK_NO_FCSMC` | SUPT page | HSI with clear OK near QUAL/GRND | non-target page or stable blank | INS OK + SUPT |
| 101-103 | 3 | `BITROOT_HSI_MAP_NO_FCSMC` | BIT FAILURES / BIT root | HSI + MAP overlay, no OK | non-target page or stable blank | BIT root + HSI MAP；FCS-MC 不可同屏时的实际采集版 |
| 104-107 | 4 | `NONBITROOT_HSI_GRND_FCSMC_INTEST` | TAC/SUPT/FCS/non-target page; avoid BIT root | HSI MAP off, GRND/QUAL/TIME visible, no OK | FCS-MC showing IN TEST | FCS-MC IN TEST + HSI GRND；不含 BIT root |
| 108-115 | 8 | `MIXED_NON_TARGET` | EW/SA/RADAR/STORES/MIDS/DATA/FPAS/CHKLIST or blank | non-HSI page or blank | EW/SA/RADAR/STORES/MIDS/DATA/FPAS/CHKLIST or blank | 全目标 facts hard negative |
| 116-118 | 3 | `FCS_TRANSITION_UNREADABLE` | FCS page changing/blurred/cropped/unreadable | non-target or blank | non-target or blank | FCS uncertain |
| 119 | 1 | `BITROOT_TRANSITION_UNREADABLE` | non-target or blank | non-target or blank | BIT root changing/blurred/cropped/unreadable | BIT root uncertain |
| 120 | 1 | `FCSMC_TRANSITION_UNREADABLE` | non-target or blank | non-target or blank | FCS-MC changing/blurred/cropped/unreadable | FCS-MC uncertain |
| 121-122 | 2 | `HSI_TRANSITION_UNREADABLE` | non-target or blank | HSI/INS text changing/blurred/cropped/unreadable | non-target or blank | HSI/INS uncertain |

## 标注注意事项

`PBIT GO`：

```text
fcsmc_page_visible = seen
fcsmc_intermediate_result_visible = seen
fcsmc_final_go_result_visible = not_seen
```

`IN TEST`：

```text
fcsmc_page_visible = seen
fcsmc_in_test_visible = seen
fcsmc_final_go_result_visible = not_seen
```

FCS-MC final GO 但 HSI 没 OK：

```text
fcsmc_final_go_result_visible = seen
ins_ok_text_visible = not_seen
```

HSI GRND/QUAL/TIME no OK：

```text
hsi_page_visible = seen
ins_grnd_alignment_text_visible = seen
ins_ok_text_visible = not_seen
```

纯黑屏 / 稳定空白：

```text
目标 facts = not_seen
```

只有页面正在切换、模糊、遮挡、文字不可读时才标 `uncertain`。

## 预期标签分布

Run-005 本身 122 张的预期分布：

| fact_id | seen | not_seen | uncertain | seen ratio |
|---|---:|---:|---:|---:|
| `tac_page_visible` | 24 | 98 | 0 | 19.7% |
| `supt_page_visible` | 30 | 92 | 0 | 24.6% |
| `fcs_page_visible` | 40 | 81 | 1 | 32.8% |
| `fcs_page_x_marks_visible` | 18 | 103 | 1 | 14.8% |
| `bit_root_page_visible` | 43 | 78 | 1 | 35.2% |
| `fcsmc_page_visible` | 56 | 64 | 2 | 45.9% |
| `fcsmc_intermediate_result_visible` | 15 | 105 | 2 | 12.3% |
| `fcsmc_in_test_visible` | 19 | 101 | 2 | 15.6% |
| `fcsmc_final_go_result_visible` | 22 | 98 | 2 | 18.0% |
| `hsi_page_visible` | 107 | 12 | 3 | 87.7% |
| `hsi_map_layer_visible` | 35 | 84 | 3 | 28.7% |
| `ins_grnd_alignment_text_visible` | 72 | 47 | 3 | 59.0% |
| `ins_ok_text_visible` | 14 | 105 | 3 | 11.5% |

Run-003 + Run-005 合并后，总计 342 张图：

| fact_id | seen | not_seen | uncertain | seen ratio |
|---|---:|---:|---:|---:|
| `tac_page_visible` | 70 | 272 | 0 | 20.5% |
| `supt_page_visible` | 50 | 292 | 0 | 14.6% |
| `fcs_page_visible` | 70 | 271 | 1 | 20.5% |
| `fcs_page_x_marks_visible` | 34 | 307 | 1 | 9.9% |
| `bit_root_page_visible` | 61 | 280 | 1 | 17.8% |
| `fcsmc_page_visible` | 111 | 229 | 2 | 32.5% |
| `fcsmc_intermediate_result_visible` | 33 | 307 | 2 | 9.6% |
| `fcsmc_in_test_visible` | 37 | 303 | 2 | 10.8% |
| `fcsmc_final_go_result_visible` | 40 | 300 | 2 | 11.7% |
| `hsi_page_visible` | 213 | 126 | 3 | 62.3% |
| `hsi_map_layer_visible` | 114 | 225 | 3 | 33.3% |
| `ins_grnd_alignment_text_visible` | 130 | 209 | 3 | 38.0% |
| `ins_ok_text_visible` | 26 | 311 | 5 | 7.6% |

训练时建议将 Run-005 过采样一次，即传两遍 Run-005 EN/ZH，以抵消 Run-003 的 strong `not_seen` prior。

## PowerShell 启动命令

```powershell
cd "L:\Documents\files\Yu Zhang TU Clausthal\Thesis\IEFMMQ"

python -m tools.capture_vlm_dataset `
  --session-id fa18c-coldstart-run-005-composition-rebalance `
  --saved-games-dir "C:\Users\15423\Saved Games\DCS" `
  --output-root tools\.captures `
  --manual-plan fa18c_run005_composition_rebalance_122 `
  --global-help-hotkey X2 `
  --global-help-cooldown-ms 1000 `
  --help-trigger-port 0
```

`X2` 通常是鼠标前进侧键，等价于 `MOUSE5`。如果你的鼠标映射相反，可改成 `X1`。
