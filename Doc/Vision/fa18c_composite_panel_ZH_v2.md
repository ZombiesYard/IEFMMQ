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

## DCS 侧最小配置模板

推荐直接使用安装器生成 Saved Games 配置，而不是手工新建文件：

```bash
python -m tools.install_dcs_hook \
  --dcs-variant DCS \
  --install-composite-panel \
  --monitor-mode extended-right
```

若不传 `--main-width` / `--main-height`，安装器只会在 Windows 上自动探测当前主屏分辨率并据此生成 Monitor Setup；非 Windows shell 需要手工显式传入宽高。

执行后，最少会涉及这些位置：

- `Saved Games/<DCS variant>/Scripts/Export.lua`
  - 若尚未接入 SimTutor，安装器会追加 hook 入口
- `Saved Games/<DCS variant>/Scripts/SimTutor/SimTutor.lua`
- `Saved Games/<DCS variant>/Scripts/SimTutor/SimTutor Function.lua`
- `Saved Games/<DCS variant>/Scripts/SimTutor/SimTutorConfig.lua`
  - 这里是 v0.4 组合面板最小模板，包含 `caps.vlm_frame = true`
  - 也包含 `vision.output_root`、`layout_id`、`channel`、背景色和推荐输出分辨率
  - 现在也包含 `overlay.command_host/command_port/ack_host/ack_port`，默认仍是单机 `127.0.0.1`
- `Saved Games/<DCS variant>/Config/MonitorSetup/SimTutor_FA18C_CompositePanel_v1.lua`
  - 这是 DCS 原生视口导出排版文件

如果你只想生成 Monitor Setup，不改 hook，可单独执行：

```bash
python -m tools.install_dcs_monitor_setup \
  --dcs-variant DCS \
  --mode extended-right
```

## 正式支持的部署拓扑

v0.4 当前正式支持两种部署方式：

- 单机：`DCS + simtutor + Qwen/vLLM` 都在同一台机器
- 双机：`DCS + simtutor` 同机，`Qwen/vLLM` 远程部署

双机场景下，若 `live_dcs.py` 仍与 DCS 同机运行，则只需要把 Python 侧
`--model-base-url` 指向远程 Qwen/vLLM，例如：

```bash
python live_dcs.py \
  --model-provider openai_compat \
  --model-base-url http://10.0.0.42:8000
```

此时 `SimTutorConfig.lua` 里的 `telemetry` / `handshake` / `overlay` 地址通常仍保持
`127.0.0.1` 即可。只有当后续把 Python 主控进程迁到另一台机器时，才需要进一步调整：

- `telemetry.host`：改成 Python 主控机地址
- `overlay.ack_host`：改成 Python 主控机地址
- `handshake.host`：改成 DCS 机上可绑定的地址，例如 `0.0.0.0`
- `overlay.command_host`：改成 DCS 机上可绑定的地址，例如 `0.0.0.0`

## 如何把多个关键区域排成一张组合图

首版冻结布局不是主视口截图，而是 DCS 原生导出的 3 个 viewport：

- `LEFT_MFCD` -> 左 DDI
- `CENTER_MFCD` -> AMPCD
- `RIGHT_MFCD` -> 右 DDI

布局规则：

- 3 块区域固定在屏幕左侧，按上到下排列
- 右侧保留给模拟器主画面
- 背景建议保持深色纯色，默认模板为 `rgb(15, 20, 24)`
- `extended-right` 模式会在主画面右侧追加固定 `2560x1440` 调试画布
- `single-monitor` / `ultrawide-left-stack` 会把左侧导出带和右侧主视口一起解到同一屏幕

也就是说，DCS 负责“把 3 个关键面板稳定导出到固定区域”，而后续 sidecar/frame writer 再按固定 contract 把这块组合图落到磁盘并写 `frames.jsonl`。

## 首版纳入与不纳入范围

首版优先依赖组合面板导出的区域：

- 左 DDI
- AMPCD
- 右 DDI

这些步骤优先依赖原生视口证据：

- `S08`
- `S15`
- `S18`

这些区域或步骤暂不纳入首版组合图优先范围，仍优先走 DCS-BIOS 或人工确认：

- `MASTER CAUTION` 与左右告警/提示灯
- `IFEI`
- `UFC`
- standby 仪表
- HUD
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

## 最小联调检查清单

### 1. 安装配置

- 执行 `python -m tools.install_dcs_hook --install-composite-panel ...`
- 默认仅在 Windows 上自动探测当前主屏分辨率；非 Windows shell 请直接补 `--main-width` / `--main-height`
- 检查 `Saved Games/<variant>/Scripts/SimTutor/SimTutorConfig.lua`
- 确认其中至少有：
  - `caps.vlm_frame = true`
  - `vision.output_root = <Saved Games>/<variant>/SimTutor/frames`
  - `vision.channel = "composite_panel"`
  - `vision.layout_id = "fa18c_composite_panel_v2"`
  - `overlay.command_host = "127.0.0.1"`（默认单机）
  - `overlay.ack_host = "127.0.0.1"`（默认单机）

### 2. DCS Options

- 打开 DCS -> `Options` -> `System`
- 选择 Monitor Setup：`SimTutor_FA18C_CompositePanel_v1`
- 分辨率设置为安装器打印的 recommended resolution
- 首次联调建议先用 `extended-right`

### 3. 启动后检查 DCS 侧

- `Saved Games/<variant>/Scripts/Export.lua` 中已有 SimTutor hook 入口
- DCS 启动后 capability handshake 返回 `vlm_frame=true`
- 若 sidecar/frame writer 已部署，`<Saved Games>/<variant>/SimTutor/frames/<session_id>/composite_panel/` 下会持续出现：
  - `<capture_wall_ms>_<frame_seq>.png`
  - `frames.jsonl`

### 4. Python 侧检查

- 启动 `python live_dcs.py`
- 确认 Python 侧能看到 `VisionObservation`
- 若视觉 sidecar 尚未就绪，也必须只出现 `vision_unavailable` 降级，而不是中断 telemetry 主链路

## 数据集采集工具

若当前目标不是 live tutor，而是为后续 VLM/SFT 准备截图数据，可直接使用：

```powershell
python .\tools\capture_vlm_dataset.py `
  --session-id fa18c-coldstart-run-001 `
  --saved-games-dir "$env:USERPROFILE\Saved Games\DCS"
```

当前工具行为：

- 默认输出到 `tools/.captures/<session_id>/`
- 保存原始整屏截图到 `raw/`
- 同步生成按当前组合图 contract 裁剪和加标注后的 VLM-ready 工件到 `artifacts/`
- 写出：
  - `frames.jsonl`
  - `capture_index.jsonl`
- 在 Windows 上默认监听全局 help 侧键 `X1` / `MOUSE4`
- 启动后先 idle；第一次按 help 侧键后开始持续采集
- 默认采样频率为 `2 fps`
- 用 `Ctrl+C` 结束

若你的 help 侧键是第二个鼠标侧键，可改为：

```powershell
python .\tools\capture_vlm_dataset.py `
  --session-id fa18c-coldstart-run-001 `
  --saved-games-dir "$env:USERPROFILE\Saved Games\DCS" `
  --global-help-hotkey X2
```

说明：

- 这个工具面向“截图数据采集”，不参与 tutor 主链路
- 它复用了当前 `fa18c_composite_panel_v2` 的相同截图和裁剪流程，确保训练输入形态和线上一致
- `tools/.captures/` 已默认加入 `.gitignore`

### 5. 失败时优先排查

- Monitor Setup 没选对，导致 `LEFT_MFCD/CENTER_MFCD/RIGHT_MFCD` 没有排到约定位置
- DCS 分辨率没按推荐值设置，导致组合图几何漂移
- `SimTutorConfig.lua` 中 `vision.output_root` 指到错误目录
- sidecar/frame writer 没有按最终文件名写 `.png` 或没有原子 rename
- Python 读取到的 `frames.jsonl` 中 `layout_id/channel/source_session_id` 不匹配
