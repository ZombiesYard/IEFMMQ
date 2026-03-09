# F/A-18C 归一化原生视口布局 v2

`layout_id`: `fa18c_composite_panel_v2`

本文件冻结当前 v0.4 视觉输入 contract 的 v2 版本。v2 不再以像素坐标作为权威，而是以“整屏上的左侧导出带 + 导出带内部 3 个 region”的归一化几何作为唯一真源。

## 固定视觉范围

- `left_ddi`
- `ampcd`
- `right_ddi`

其余证据仍优先走 DCS-BIOS，包括：

- `MASTER CAUTION` 与左右告警/注意/提示灯
- `IFEI` 数字与 BINGO
- `UFC` scratchpad、option displays、COMM1/COMM2 显示
- standby / HUD

## v2 几何规则

- 第一步：按整屏归一化 `strip` 切出左侧导出带
- 第二步：按 `strip` 内部归一化 `regions` 切出 3 块视口
- 后续 VLM 不允许直接按全屏像素硬切 `left_ddi / ampcd / right_ddi`

`strip_norm` 当前冻结为：

- `anchor: left`
- `x_norm: 0.0`
- `y_norm: 0.0`
- `height_norm: 1.0`
- `target_aspect_ratio: 880 / 1440`
- `min_main_view_width_px: 640`

## region 顺序

顺序固定为：

- `left_ddi`
- `ampcd`
- `right_ddi`

禁止任何别名漂移，例如把 `ampcd` 写成 `mpcd`。

## 参考资产

- strip 预览：`Doc/Vision/assets/fa18c_composite_panel_v2.svg`
- 全屏结构图：`Doc/Vision/assets/fa18c_composite_panel_fullscreen_v2.svg`

说明：

- 两张 SVG 都使用英文标签
- strip 预览图只展示导出带内部结构
- 全屏结构图只展示“左侧导出带 + 右侧模拟器主画面”的归一化关系，不绑定具体物理分辨率

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

## Monitor Setup 行为

- `single-monitor` 与 `ultrawide-left-stack` 共享同一套 normalized left-stack solver
- `extended-right` 保留为调试模式，但其 3 块视口也由同一套 normalized region 几何求解
- 这意味着后续采帧、切图、VLM 引用都只需要维护一套几何逻辑

## 帧工件规范

- 默认交换方式固定为：磁盘目录落盘 + `frames.jsonl`
- 目录固定为：`<Saved Games>/<DCS variant>/SimTutor/frames/<session_id>/<channel>/`
- 帧文件名固定为：`<capture_wall_ms>_<frame_seq:06d>.png`，例如 `1772872444902_000123.png`
- manifest 固定文件名：同目录 `frames.jsonl`
- 每行至少包含：
  - `frame_id`
  - `capture_wall_ms`
  - `frame_seq`
  - `channel`
  - `layout_id`
  - `image_path`
  - `width`
  - `height`
  - `source_session_id`

## 原子落盘与 Python 侧消费规则

- DCS 侧必须先写临时文件，再原子 rename 成最终 `.png`
- Python 侧只消费 manifest 中声明且已经存在的最终 `.png`
- `.tmp / .part / .partial` 一律视为未完成工件，不进入 `VisionObservation`
- `VisionObservation.image_uri` 固定指向 Python 生成的 VLM-ready 最终工件
- 原始截图路径保留在 `VisionObservation.source_image_path` 与 metadata 中，靠 `frame_id` 统一关联日志、重放、模型请求和调试工具

## Python 裁剪与标注规则

- Python 先按整屏 solver 切掉右侧模拟器主画面，只保留左侧导出带
- 然后在导出带上叠加 VLM-friendly 边框与英文标签
- 标签顺序固定为：
  - `Left DDI`
  - `AMPCD`
  - `Right DDI`
- 该过程必须适配不同分辨率和宽高比，只能依赖当前冻结的 normalized layout 计算 crop，不允许写死像素
