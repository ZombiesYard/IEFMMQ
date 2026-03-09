# F/A-18C 原生视口布局 v1

`layout_id`: `fa18c_composite_panel_v1`

本文件冻结 v0.4 当前实际使用的视觉输入方案：只保留 DCS 原生可稳定导出的 3 块显示器视口，不再把 `warning_panel`、`ufc`、`ifei`、`standby_hud` 放进视觉 contract。

## 固定覆盖范围

- `left_ddi`
- `ampcd`
- `right_ddi`

## 明确不再纳入视觉输入

- `MASTER CAUTION` 与左右告警/注意/提示灯整块
- `IFEI` 全部数字与 BINGO 读数
- `UFC` scratchpad、option windows、COMM1/COMM2 显示
- 备用仪表与 HUD 小裁切
- 外景、主视角、座舱大面积背景、无关控制台

上面这些信息在当前方案中优先走 DCS-BIOS，不再要求 VLM 从图像里识别。

## 区域命名规则

- 后续 VLM prompt、manifest、文档和调试输出只允许使用这 3 个 `region_id`
- 禁止别名漂移，例如把 `ampcd` 写成 `mpcd`
- 区域顺序固定为：`left_ddi -> ampcd -> right_ddi`

## 区域定义

画布尺寸固定为 `880x1440`，对应当前实际部署里从超宽屏左侧导出带裁出来的 3 屏竖排区域。

| region_id | 中文名 | 位置与尺寸 (x,y,w,h) | 对应 DCS 原生视口 | 主要内容 |
| --- | --- | --- | --- | --- |
| `left_ddi` | 左 DDI | `216,24,448,448` | `LEFT_MFCD` | 左 DDI 页面，如 FCS/HSI |
| `ampcd` | AMPCD | `216,496,448,448` | `CENTER_MFCD` | 中央 AMPCD 页面文字与符号 |
| `right_ddi` | 右 DDI | `216,968,448,448` | `RIGHT_MFCD` | 右 DDI 页面，如 BIT/FCS |

## 步骤优先级边界

优先依赖原生视口：

- `S08`
- `S15`
- `S18`

仍优先依赖 DCS-BIOS：

- `S01`
- `S03`
- `S04`
- `S05`
- `S06`
- `S07`
- `S09`
- `S10`
- `S11`
- `S12`
- `S13`
- `S21`

当前不归入首版原生视口优先范围，属于手工确认或布局外：

- `S02`
- `S14`
- `S16`
- `S17`
- `S19`
- `S20`
- `S22`
- `S23`
- `S24`
- `S25`

## 样例图

- 资产路径：`Doc/Vision/assets/fa18c_composite_panel_v1.svg`
- 样例图只用于验证 3 个 `region_id` 的命名、顺序和边框，不表示主视角布局

## 当前实际导出方案

当前冻结的实际显示方案是 `3440x1440` 单屏超宽模式：

- DCS monitor setup 使用 `ultrawide-left-stack`
- 左侧 `880px` 窄条竖排导出 3 块原生视口
- 右侧保留完整 `2560x1440` 的 `16:9` 正常游戏画面
- VLM 侧只应消费左侧这块 `880x1440` 导出带，不能把右侧主视角送进模型
- 对 `16:9` 单屏，`single-monitor` 现在也使用同一套左栈布局语义，只是按屏幕高度缩放左侧导出带，避免后续代码维护两套单屏几何

安装命令：

```bash
python -m tools.install_dcs_monitor_setup --dcs-variant DCS --mode ultrawide-left-stack --main-width 3440 --main-height 1440
```

安装后：

- 在 DCS Options 中选择 `SimTutor_FA18C_CompositePanel_v1`
- 分辨率设为 `3440x1440`
- 如果是 `16:9` 单屏，可使用：

```bash
python -m tools.install_dcs_monitor_setup --dcs-variant DCS --mode single-monitor --main-width 1920 --main-height 1080
```

- `single-monitor` 与 `ultrawide-left-stack` 共享同一套左栈三视口布局；`extended-right` 仅保留为调试模式
