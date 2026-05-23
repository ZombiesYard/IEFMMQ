# SimTutor Experiment Operation Manual

本文档是面向实验执行者的操作手册。目标是让除开发者本人之外的实验助手，也能按同一流程完成 Quest 3 / DCS / SimTutor 正式实验、导出数据，并记录异常。

本手册默认研究问题为：

> 在 Quest 3 虚拟座舱训练环境中，foundation-model-driven SimTutor 是否能提高新手完成 F/A-18C cold-start 流程的完成率、步骤准确度和帮助后的恢复能力？

Quest 3 是固定实验平台，不作为被比较的自变量。正式比较的是 `without_tutor` 与 `with_tutor`。

实验用表格见 [Experiment forms](experiment-forms.md)，包括 condition assignment、trial sheet、NASA Raw TLX、usability/trust questionnaire 和 export quality checklist。

## 1. 实验角色

每次实验至少需要一名实验执行者。

| 角色 | 职责 |
|---|---|
| Experimenter | 安装检查、启动 DCS/SimTutor、读统一说明、记录异常、导出数据 |
| Participant | 按条件完成 F/A-18C cold-start 任务 |
| Optional observer | 记录明显误操作、卡住时间点、VR 不适和技术问题 |

实验执行者不能在任务中口头提示冷启动步骤。只能处理安全、VR 舒适、DCS 崩溃、设备失效等技术问题。

## 2. 实验条件

正式实验只比较 tutor 是否存在，不比较 VR 和非 VR。

| Condition | Platform | Assistance | Participant instruction |
|---|---|---|---|
| `without_tutor` | Quest 3 / DCS | 传统资料，例如纸质/电子 checklist；SimTutor 不给帮助 | “请尽量按 checklist 完成冷启动。” |
| `with_tutor` | Quest 3 / DCS | SimTutor Chinese help + cockpit overlay | “卡住或不确定时按 X1 请求帮助，按提示和高亮操作。” |

推荐主实验采用 between-subject design：每名参与者只进入一个条件，避免冷启动学习效应污染第二次 trial。

如果参与者数量太少，可以做 within-subject pilot，但必须在论文中明确说明它是 exploratory，因为第二轮会受到练习效应影响。

## 3. 主要指标

实验数据导出后主要看以下指标：

| Metric | Meaning |
|---|---|
| CompletionRate | trial 是否完成到 S33 |
| StepCompletionAccuracy | S01-S33 完成比例 |
| CriticalStepSuccessRate | critical steps 完成比例 |
| TaskTime_sec | trial 用时 |
| HelpRequests | 参与者请求 help 的次数 |
| RecoveryAfterHelpCandidateRate | help 后是否出现恢复/推进候选 |
| OverlayRejectionRate | overlay target 被系统拒绝的比例 |
| FallbackRate | harness repair/fallback 介入比例 |
| VLMCalls | VLM fact extractor 调用次数 |
| VLMNotRequiredLeakageCount | 非视觉步骤误调用 VLM 的泄漏计数 |
| NASA-TLX / SUS / confidence | 主观负荷、可用性、自信程度 |

5 类错误类型可以作为人工编码补充，不要作为系统必须实时识别的主 claim。

## 4. 文件和命名规范

正式实验开始前先确定 `study_id`，例如：

```text
fa18c_quest3_simitutor_2026
```

每名参与者使用匿名 ID：

```text
P01, P02, P03, ...
```

每次 trial 使用：

```text
T01, T02, ...
```

推荐目录：

```text
logs/
recordings/
questionnaires/
artifacts/experiments/<study_id>/
artifacts/analysis/<study_id>/
```

推荐 log 名称：

```text
logs/P01_T01_with_tutor.jsonl
logs/P02_T01_without_tutor.jsonl
```

如果 `--output` 已存在，live loop 会自动写入带时间戳或数字后缀的新文件。实验结束后必须记录控制台打印的实际 resolved output path。

## 5. 实验前一次性准备

### 5.1 软件和硬件

确认以下项目可用：

- Windows simulator host
- DCS World
- F/A-18C module
- DCS-BIOS
- Quest 3，Quest Link 或 Air Link
- 鼠标、键盘、HOTAS 或实验指定输入设备
- Python 环境和本仓库代码
- 远程或本地 model endpoint
- 能访问 `simtutor-base` 和 `simtutor-vision`

### 5.2 安装 DCS hook 和 composite panel

在 simulator host 的 PowerShell 中运行：

```powershell
python -m tools.install_dcs_hook `
  --dcs-variant DCS `
  --saved-games-dir "C:\Users\<USER>\Saved Games\DCS" `
  --install-composite-panel `
  --monitor-mode single-monitor
```

如果使用 ultrawide：

```powershell
python -m tools.install_dcs_hook `
  --dcs-variant DCS `
  --saved-games-dir "C:\Users\<USER>\Saved Games\DCS" `
  --install-composite-panel `
  --monitor-mode ultrawide-left-stack `
  --main-width 3440 `
  --main-height 1440
```

安装后检查：

- `Saved Games\DCS\Scripts\Export.lua` 包含 SimTutor snippet
- `Saved Games\DCS\Scripts\SimTutor\SimTutor.lua` 存在
- `Saved Games\DCS\Scripts\SimTutor\SimTutor Function.lua` 存在
- `Saved Games\DCS\Scripts\Hooks\SimTutorHighlight.lua` 存在
- `Saved Games\DCS\Scripts\SimTutor\SimTutorConfig.lua` 存在
- `SimTutorConfig.lua` 中 `overlay.hilite_ids` 至少包含 4 个 id

### 5.3 Quest 3 / DCS VR 设置

在 DCS 中确认：

- VR enabled
- Quest 3 画面清晰且能稳定运行
- `Options -> VR -> VR Mirror Options -> Use DCS System Resolution`
- DCS resolution 使用 monitor setup 文件推荐的 canvas size
- 进入座舱后能看到 cockpit、鼠标光标和 overlay 高亮
- 参与者能坐稳，能随时要求暂停

Quest 3 不是本实验自变量，因此所有条件都必须使用相同 VR 设置。

### 5.4 模型 endpoint

推荐正式实验使用 launcher 管理 SSH tunnel。若手动启动 tunnel，先确保本机有 SSH key authentication，不要在实验中输入密码。

PowerShell 示例：

```powershell
ssh -N -L 16324:127.0.0.1:6324 yz50@cloud-247.rz.tu-clausthal.de
```

然后检查：

```powershell
curl http://127.0.0.1:16324/health
curl http://127.0.0.1:16324/v1/models
```

必须能看到：

- `simtutor-base`
- `simtutor-vision`

如果无法连接 endpoint，不要开始正式 trial。

## 6. 推荐运行方式：SimTutor launcher

当前推荐用 GUI launcher 执行正式实验，因为它会统一管理 preflight、tunnel、vision sidecar、live-dcs、export 和 analysis。

在仓库根目录运行：

```powershell
python .\tools\simtutor_launcher.py
```

### 6.1 Launcher 必填字段

每个 trial 开始前填写：

| Field | Example |
|---|---|
| participant_id | `P01` |
| condition | `with_tutor` 或 `without_tutor` |
| trial_id | `T01` |
| study_id | `fa18c_quest3_simitutor_2026` |
| participant_group | `novice` |
| experimenter_id | `E01` |
| questionnaire_ref | `questionnaires/P01_T01.json` |
| recording_ref | `recordings/P01_T01.mp4` |
| model_profile_mode | `remote_tunnel` |
| model_base_url | `http://127.0.0.1:16324` |
| text_model_name | `simtutor-base` |
| vision_model_name | `simtutor-vision` |
| vr_setup | `Quest 3 Link` 或实际配置 |
| monitor_setup | `fa18c_composite_panel_v2` |

`with_tutor` 条件：

- `model_enable_multimodal = true`
- `max_overlay_targets = 4`
- `language = zh`
- `scenario_profile = airfield`

`without_tutor` 条件：

- 不向参与者提供 X1 help 操作。
- 如果需要 passive logging，可仍启动 launcher，但必须记录“participant did not use tutor help”。
- baseline export 会把 DCS-BIOS observation、derived telemetry vars 和 pack completion gates 作为隐藏 passive evidence，用于自动填充可审计的 step completion。
- passive logging 不会调用 LLM/VLM，不显示 overlay，也不向参与者暴露 step hint。
- `step_coding.csv` 中 `Completed`、`EvidenceRefs`、`AutoCodingNotes` 可包含自动 gate evidence，例如 `completed_from_passive_gate`、`passive_gate:S01:s01_requires_battery_on`、`telemetry_var:vars.battery_on`。
- `HelpRequests`、`LLMTriggers`、`VLMCalls`、`OverlayExecuted` 在未请求帮助的 baseline trial 中应保持 0。
- 人工/录像编码仍用于审计和修正，尤其是不可观测动作、视频可见但 telemetry 不足的步骤，以及 `Error_*` 人工错误类型列。

### 6.2 Launcher 操作顺序

每次 trial：

1. 点击 `Load` 或填写本 trial 设置。
2. 点击 `Save`。
3. 点击 `Install`，确认 hook/config 已安装。
4. 点击 `Preflight`，必须没有 error。
5. 点击 `Validate Model`，必须能连接 model endpoint 且模型名正确。
6. 点击 `Dry Run`，检查命令和 log path。
7. 打开 DCS，加载固定 mission，进入 cold and dark cockpit。
8. 打开录像。
9. 点击 `Start`。
10. 确认 launcher log 显示 sidecar 和 live-dcs 已启动。
11. 开始 participant task。
12. trial 结束后点击 `Stop`。
13. 等 launcher 自动运行 export 和 analysis。
14. 检查导出的 `quality_gate.json`，必须 `passed=true`。

不要在一个 participant 的 trial 中间修改 settings。

## 7. 手动 CLI 运行方式

如果 launcher 不可用，使用本节命令。所有命令都在仓库根目录执行。

### 7.1 启动 vision sidecar

PowerShell 窗口 1：

```powershell
python .\tools\capture_vision_sidecar.py `
  --saved-games-dir "C:\Users\<USER>\Saved Games\DCS" `
  --session-id "sess-P01-T01" `
  --capture-fps 0 `
  --trigger-host "127.0.0.1" `
  --trigger-port 7795
```

### 7.2 启动 live-dcs

PowerShell 窗口 2：

```powershell
python -m simtutor live-dcs `
  --bios-source raw `
  --raw-bios-host 239.255.50.10 `
  --raw-bios-port 5010 `
  --raw-bios-aircraft FA-18C_hornet `
  --raw-bios-control-dir "DCS\Scripts\DCS-BIOS\doc\json" `
  --timeout 0.5 `
  --session-id "sess-P01-T01" `
  --vision-saved-games-dir "C:\Users\<USER>\Saved Games\DCS" `
  --vision-session-id "sess-P01-T01" `
  --vision-trigger-wait-ms 4000 `
  --vision-capture-trigger-host "127.0.0.1" `
  --vision-capture-trigger-port 7795 `
  --global-help-hotkey "X1" `
  --global-help-cooldown-ms 800 `
  --model-provider openai_compat `
  --model-base-url "http://127.0.0.1:16324" `
  --model-name "simtutor-base" `
  --vision-model-name "simtutor-vision" `
  --model-timeout-s 60 `
  --max-overlay-targets 4 `
  --knowledge-index "Doc\Evaluation\index.json" `
  --rag-top-k 5 `
  --model-enable-multimodal `
  --no-cold-start-production `
  --lang zh `
  --scenario-profile airfield `
  --log-raw-llm-text `
  --print-model-io `
  --output "logs\P01_T01_with_tutor.jsonl"
```

### 7.3 导出 trial 数据

trial 结束后运行：

```powershell
python -m simtutor experiment-export "logs\P01_T01_with_tutor.jsonl" `
  --study-id "fa18c_quest3_simitutor_2026" `
  --participant-id "P01" `
  --condition "with_tutor" `
  --group "novice" `
  --trial-id "T01" `
  --experimenter-id "E01" `
  --questionnaire "questionnaires\P01_T01.json" `
  --recording-ref "recordings\P01_T01.mp4" `
  --output-dir "artifacts\experiments\fa18c_quest3_simitutor_2026" `
  --pack "packs\fa18c_startup\pack.yaml" `
  --taxonomy "packs\fa18c_startup\taxonomy.yaml" `
  --ui-map "packs\fa18c_startup\ui_map.yaml" `
  --bios-to-ui "packs\fa18c_startup\bios_to_ui.yaml" `
  --model-provider openai_compat `
  --model-name simtutor-base `
  --vision-model-name simtutor-vision `
  --prompt-version launcher-v0.4 `
  --scenario-profile airfield `
  --dcs-mission "fa18c_cold_start_fixed.miz" `
  --dcs-aircraft FA-18C_hornet `
  --vr-setup "Quest 3 Link" `
  --monitor-setup fa18c_composite_panel_v2 `
  --strict
```

如果同一 participant/trial 需要重导出，加 `--overwrite`，但必须先确认原始 log 没有被误删。

### 7.4 汇总分析

所有 trial 导出后运行：

```powershell
python -m simtutor experiment-analyze "artifacts\experiments\fa18c_quest3_simitutor_2026" `
  --output-dir "artifacts\analysis\fa18c_quest3_simitutor_2026"
```

如果只需要 CSV：

```powershell
python -m simtutor experiment-analyze "artifacts\experiments\fa18c_quest3_simitutor_2026" `
  --output-dir "artifacts\analysis\fa18c_quest3_simitutor_2026" `
  --no-figures
```

## 8. 每个 participant 的完整流程

### 8.1 到场前

实验执行者完成：

- DCS 可启动
- Quest 3 可用
- DCS-BIOS raw telemetry 有输出
- model endpoint 可用
- launcher preflight pass
- 空白 questionnaire 文件准备好
- 录像路径准备好
- participant ID 已分配
- condition 已随机分配或按预注册顺序分配

### 8.2 欢迎和 consent

实验执行者说明：

1. 本实验评估虚拟座舱训练系统，不评估个人能力。
2. 参与者可随时停止。
3. 录像、日志和问卷会匿名保存。
4. 如果 VR 不适，可以暂停或终止。
5. 实验过程中除技术问题外，实验执行者不会给步骤提示。

确保参与者完成 consent。

### 8.3 背景问卷

记录：

- 年龄段，可选
- DCS 经验
- 飞行模拟经验
- VR 经验
- 英文/航空术语熟悉程度
- F/A-18C 冷启动熟悉程度
- 自信程度，1 到 7 分

### 8.4 训练说明

给所有参与者相同的设备说明：

- 如何佩戴 Quest 3
- 如何重置视角
- 如何使用鼠标/控制器点击 cockpit
- 如何暂停
- 如何说出“我不舒服”或“我要停止”

不要讲 F/A-18C 冷启动步骤。

### 8.5 条件说明

`without_tutor`：

```text
你将在 Quest 3 中完成 F/A-18C 冷启动任务。
你可以使用提供的 checklist。
实验者不会告诉你下一步该做什么。
如果你认为已经完成，或无法继续，请告诉实验者。
```

`with_tutor`：

```text
你将在 Quest 3 中完成 F/A-18C 冷启动任务。
当你不确定下一步、卡住、或想确认当前状态时，按 X1 请求 SimTutor 帮助。
系统会用中文说明当前状态和下一步，并在座舱中高亮相关控件。
请按你理解的提示操作。
如果你认为已经完成，或无法继续，请告诉实验者。
```

### 8.6 Trial 开始

实验执行者：

1. DCS 加载固定 mission。
2. 确认 aircraft 是 cold and dark。
3. 确认 cockpit 视角正常。
4. 开启录像。
5. 启动 launcher 或 CLI。
6. 宣布：“任务开始。”
7. 记录 start time。

### 8.7 Trial 中

实验执行者只允许说：

- “你可以继续。”
- “请按照你自己的判断操作。”
- “如果不舒服可以暂停。”
- “这是技术问题，我需要暂停实验处理。”

实验执行者不允许说：

- “下一步是打开电瓶。”
- “你漏了 fire test。”
- “按这个按钮。”
- “系统说错了，你应该这样。”

如果 SimTutor 明显输出错误，实验执行者不要现场纠正，除非会导致任务无法继续。记录 request_id、时间和现象。

### 8.8 Trial 结束条件

任一条件满足即结束：

- 完成 S33 / taxi-ready 状态
- 达到最大时间，例如 25 或 30 分钟
- 参与者放弃或无法继续
- DCS / Quest / model / overlay 发生不可恢复技术故障
- 参与者 VR 不适

结束时记录：

- Completed: yes/no
- EndReason: completed / timeout / participant_stop / technical_failure / experimenter_stop
- Notes: 简短说明

### 8.9 结束后问卷

记录：

- NASA-TLX 或简化 workload
- SUS 或简化 usability
- 操作信心
- 任务难度
- 系统帮助是否清楚
- overlay 是否看得清
- 是否信任系统提示
- 最有帮助和最困惑的地方

## 9. 数据导出和质量检查

每个 trial 必须产生：

```text
raw_events.jsonl
session.json
quality_gate.json
help_cycles.csv
action_timeline.csv
step_coding.csv
trial_summary.csv
summary.json
```

检查 `quality_gate.json`：

- `passed` 必须为 `true`
- 如果为 `false`，不要删除文件；记录错误并修复导出或元数据

检查 `trial_summary.csv`：

- ParticipantID 正确
- TrialID 正确
- Condition 正确
- Completed 合理
- HelpRequests 合理
- VLMNotRequiredLeakageCount 为 0 或已解释
- OverlayRejectionRate 不异常

检查 `step_coding.csv`：

- S01-S33 都存在
- 自动编码明显错误时，不直接覆盖 raw data；在人工作业列中修正
- 5 类错误若使用，只作为人工 coding 字段

## 10. 人工编码建议

最小人工编码必须确认：

- 每步是否完成
- trial 是否完成
- 关键失败点
- 是否有实验者介入
- 是否有技术故障

如果时间允许，再编码 5 类错误：

| Error | Meaning |
|---|---|
| OM | omission，漏做 |
| CO | commission，做了不该做的 |
| OR | order，顺序错误 |
| PA | parameter/action，操作方向或参数错误 |
| SV | system/visual/verification，状态确认或视觉判断错误 |

正式论文主分析不要依赖系统实时错误分类。用人工编码作为补充分析更稳。

## 11. 异常处理

### 11.1 Model endpoint 失败

现象：

- help 很久无响应
- console 出现 500、400、timeout
- `Validate Model` fail

处理：

1. 停止 trial。
2. 保存当前 log。
3. 记录 `technical_failure=model_endpoint`。
4. 检查 SSH tunnel 和 `/v1/models`。
5. 不要把故障 trial 当作正式完成数据，除非预先定义为 technical failure。

### 11.2 VLM 在非视觉步骤被调用

导出后看：

- `VLMNotRequiredLeakageCount`
- `help_cycles.csv` 中 `vlm_call_status`

如果大于 0：

1. 记录为 system issue。
2. 保留 log。
3. 用 request_id 提取 fixture。

```powershell
python -m simtutor extract-live-fixture `
  --input "logs\P01_T01_with_tutor.jsonl" `
  --request-id "<request-id>" `
  --output-dir "artifacts\live_fixtures"
```

### 11.3 Overlay 错误或不可见

处理：

1. 不现场纠正步骤。
2. 记录 request_id、参与者看到的内容、实际应高亮目标。
3. 检查 `SimTutorConfig.lua` 中 `overlay.hilite_ids` 是否至少 4 个。
4. 检查 DCS hook 是否是最新文件。

### 11.4 Quest 3 不适

处理：

1. 立即暂停。
2. 让参与者摘下 HMD。
3. 询问是否继续。
4. 如果停止，记录 `participant_stop_vr_discomfort`。

### 11.5 Baseline 无法自动导出完整步骤

baseline 条件可能没有 help cycles，因此自动 `step_coding.csv` 可能不完整。处理方式：

1. 保留 raw log 和录像。
2. 用录像人工编码 S01-S33。
3. 在 `experimenter_notes` 中写明 baseline 使用 video/manual coding。
4. 分析时把 automatic coding 和 human coding 区分开。

## 12. 实验当天一页清单

正式 trial 前：

- [ ] Participant ID 已分配
- [ ] Condition 已分配
- [ ] Consent 完成
- [ ] Pre-questionnaire 完成
- [ ] Quest 3 清晰且舒适
- [ ] DCS 固定 mission 已加载
- [ ] Aircraft cold and dark
- [ ] Recorder 已启动
- [ ] Launcher `Preflight` pass
- [ ] Launcher `Validate Model` pass
- [ ] Log path 已记录
- [ ] 实验者不提供步骤提示

正式 trial 后：

- [ ] Stop live session
- [ ] Export 自动或手动完成
- [ ] `quality_gate.json` passed
- [ ] `trial_summary.csv` 检查
- [ ] `step_coding.csv` 检查
- [ ] Post-questionnaire 完成
- [ ] 异常和 request_id 已记录
- [ ] 录像路径写入 `recording_ref`
- [ ] 数据备份

## 13. 最小 pilot 计划

正式实验前至少做 2 个 pilot：

| Pilot | Purpose |
|---|---|
| Pilot A | 完整跑通 Quest 3、DCS、launcher、model、export、analysis |
| Pilot B | 让非开发者按本手册执行一次，检查说明是否足够 |

Pilot 通过标准：

- 能从 cold and dark 开始并结束 trial
- log 自动保存且不覆盖旧文件
- export strict 通过
- analysis 生成 CSV
- 实验助手能独立找到输出目录和质量报告
- 参与者理解何时可以请求 help

## 14. 论文中如何使用这些数据

May self-test logs 可以作为 system validation 和 development trajectory：

- harness 改造后完成率提升
- VLM leakage 下降
- export quality gate 稳定
- overlay rejection 低

不要把 May self-test logs 当作 novice user study 证据，因为它们来自熟练开发者。

正式 participant 数据用于支持或反驳主 claim：

> 在 Quest 3 虚拟座舱中，foundation-model SimTutor 是否提升 novice procedural task performance。

最稳的论文结果结构：

1. System implementation and reliability
2. VLM fact extraction validation
3. Harness/runtime reliability validation
4. User study: `with_tutor` vs `without_tutor`
5. Qualitative participant feedback

## 15. 数据归档

每个 participant/trial 完成后，将以下内容归档：

```text
logs/<trial>.jsonl
recordings/<trial>.mp4
questionnaires/<trial>.json
artifacts/experiments/<study_id>/<participant_id>/<trial_id>/
```

每晚备份：

```text
artifacts/experiments/<study_id>/
artifacts/analysis/<study_id>/
logs/
questionnaires/
recordings/
```

不要把姓名、邮箱、学生号等个人身份信息写入 log、participant_id、trial_id 或 GitHub issue。

## 16. 最后确认

只有当以下条件都满足时，才开始正式收 participant：

- 至少 2 个 pilot 已完成
- launcher 或 CLI 流程稳定
- Quest 3 setup 稳定
- model endpoint 稳定
- export strict 通过
- analysis 输出可读
- 实验助手能按本手册独立跑完 trial
- 已确定正式 condition label 和 participant ID 规则
- 已确定 baseline 是否使用人工录像编码
