# F/A-18C 组合面板图版式 v1

`layout_id`: `fa18c_composite_panel_v1`

本文件冻结 v0.4 首版视觉输入版式。目标是让 Qwen3.5 更稳定地识别显示页、字符和灯态，不追求真实座舱截图的还原感。

## 固定覆盖范围

- `left_ddi`
- `ampcd`
- `right_ddi`
- `warning_panel`
- `ufc`
- `ifei`
- `standby_hud`

## 明确不包含

- 外景
- 座舱大面积背景
- 无关控制台
- 非 pack 当前需要的额外局部裁切

## 区域命名规则

- 后续 VLM prompt、manifest、文档和调试输出只允许使用上述 7 个 `region_id`
- 禁止别名漂移，例如把 `ampcd` 写成 `mpcd`
- 区域顺序固定为：`left_ddi -> ampcd -> right_ddi -> warning_panel -> ufc -> ifei -> standby_hud`

## 区域定义

| region_id | 中文名 | 位置与尺寸 (x,y,w,h) | 主要内容 |
| --- | --- | --- | --- |
| `left_ddi` | 左 DDI | `32,32,768,768` | 左 DDI 页面，如 FCS/HSI/ENG |
| `ampcd` | AMPCD | `896,32,768,768` | 中央 AMPCD 页面文字与符号 |
| `right_ddi` | 右 DDI | `1760,32,768,768` | 右 DDI 页面，如 BIT/FCS |
| `warning_panel` | 告警灯区 | `32,832,848,248` | `MASTER CAUTION`、告警/注意/提示灯、火警相关灯 |
| `ufc` | UFC | `912,832,736,248` | UFC 显示与 COMM 相关读数 |
| `ifei` | IFEI | `1680,832,848,248` | 发动机/燃油与 BINGO 相关显示 |
| `standby_hud` | 备用仪表/HUD | `912,1112,736,296` | standby 仪表与 pack 所需的 HUD 小区域 |

## 步骤优先级边界

优先依赖组合图：

- `S02`
- `S07`
- `S08`
- `S09`
- `S15`
- `S18`
- `S21`
- `S22`
- `S23`
- `S24`
- `S25`

仍优先依赖 DCS-BIOS：

- `S01`
- `S03`
- `S04`
- `S06`
- `S10`
- `S12`
- `S13`

当前不归入首版组合图优先范围，属于手工确认或布局外：

- `S05`
- `S11`
- `S14`
- `S16`
- `S17`
- `S19`
- `S20`

## 样例图

- 资产路径：`Doc/Vision/assets/fa18c_composite_panel_v1.svg`
- 样例图按冻结版式生成，仅用于验证区域命名、裁切稳定性和人工可读性

## 三屏真视口 PoC（首轮）

- 首轮只验证 `left_ddi`、`ampcd`、`right_ddi`
- 这三块通过 DCS 原生 `LEFT_MFCD`、`CENTER_MFCD`、`RIGHT_MFCD` monitor viewport 导出
- 组合画布固定放在主屏右侧扩展桌面，尺寸固定为 `2560x1440`
- 生成安装命令：

```bash
python -m tools.install_dcs_monitor_setup --dcs-variant DCS --mode extended-right --main-width 1920 --main-height 1080
```

- 安装后需要在 DCS Options 中选择 `SimTutor_FA18C_CompositePanel_v1`
- 推荐总分辨率为 `main_width + 2560` 乘以 `max(main_height, 1440)`
- 本轮不包含 `warning_panel`、`ufc`、`ifei`、`standby_hud` 的真视口导出
- 如果只有一块屏幕，可改用单屏模式：

```bash
python -m tools.install_dcs_monitor_setup --dcs-variant DCS --mode single-monitor --main-width 1920 --main-height 1080
```

- 单屏模式会把三块 MFCD/AMPCD 缩放排在屏幕上半区，主视角放在下半区
- 如果是 `3440x1440` 这类 21:9 屏幕，更推荐超宽屏模式：

```bash
python -m tools.install_dcs_monitor_setup --dcs-variant DCS --mode ultrawide-left-stack --main-width 3440 --main-height 1440
```

- 该模式会把三块导出屏竖着排在最左侧多出来的窄条里，右侧保留完整 `2560x1440` 的 `16:9` 主视角
