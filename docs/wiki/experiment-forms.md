# SimTutor Experiment Forms

本页提供正式实验可直接复制、打印或转成问卷系统的 Markdown 表格。默认研究设计为 Quest 3 固定平台下的 `with_tutor` vs `without_tutor`。

可填写并导出 JSON/CSV 的 HTML 版本见：[Experiment forms HTML](experiment-forms.html)。

## 1. 条件顺序建议

### 推荐主设计：between-subject

正式论文主实验建议使用 between-subject design：

| Participant | Trial | Condition | Order issue |
|---|---|---|---|
| P01 | T01 | `with_tutor` 或 `without_tutor` | 无先后顺序问题 |
| P02 | T01 | `with_tutor` 或 `without_tutor` | 无先后顺序问题 |

每名参与者只做一次正式 F/A-18C cold-start trial。条件由实验前随机分配或按平衡表分配。

这样最干净，因为 F/A-18C cold-start 是强学习任务。同一个人第二次做时会记住电瓶、APU、DDI、INS、FCS、engine start 等步骤，练习效应无法通过短休息完全消除。

### 不推荐主设计：同一人先无 tutor 再有 tutor

如果让同一个人：

```text
without_tutor -> with_tutor
```

第二次表现变好可能来自：

- SimTutor 有效
- 第一次已经学过流程
- 更熟悉 Quest 3 / DCS 操作
- 更熟悉 cockpit layout

因此不能作为强因果证据。

### 如果必须 within-subject

如果参与者数量不足，必须让同一个人做两个 trial，则使用 counterbalanced exploratory design：

| Group | Trial 1 | Rest | Trial 2 |
|---|---|---|---|
| A | `without_tutor` | 15-20 min | `with_tutor` |
| B | `with_tutor` | 15-20 min | `without_tutor` |

注意：

- 15-20 分钟只能降低疲劳，不能清除 procedural learning。
- 结果应作为 pilot/exploratory。
- 分析时必须报告 order effect。
- 如果有条件，两个 trial 应使用等价但不同的 procedural task；如果还是同一个 F/A-18C cold-start，必须把学习效应写成 limitation。

### 最小可行正式方案

一个月内推荐：

```text
Between-subject, one formal trial per participant.
Condition labels: without_tutor, with_tutor.
Platform: Quest 3 for both conditions.
Baseline help: VR-accessible static checklist, not paper-only external manual.
```

## 2. Condition Assignment Sheet

正式实验前准备一张随机分配表。不要实验开始后按结果临时改条件。

| ParticipantID | AssignedCondition | Group | TrialID | RandomizationMethod | ExperimenterID | Notes |
|---|---|---|---|---|---|---|
| P01 |  | novice | T01 |  |  |  |
| P02 |  | novice | T01 |  |  |  |
| P03 |  | novice | T01 |  |  |  |
| P04 |  | novice | T01 |  |  |  |
| P05 |  | novice | T01 |  |  |  |
| P06 |  | novice | T01 |  |  |  |
| P07 |  | novice | T01 |  |  |  |
| P08 |  | novice | T01 |  |  |  |
| P09 |  | novice | T01 |  |  |  |
| P10 |  | novice | T01 |  |  |  |
| P11 |  | novice | T01 |  |  |  |
| P12 |  | novice | T01 |  |  |  |

推荐平衡分配：

| ParticipantID | SuggestedCondition |
|---|---|
| P01 | with_tutor |
| P02 | without_tutor |
| P03 | without_tutor |
| P04 | with_tutor |
| P05 | with_tutor |
| P06 | without_tutor |
| P07 | without_tutor |
| P08 | with_tutor |
| P09 | with_tutor |
| P10 | without_tutor |
| P11 | without_tutor |
| P12 | with_tutor |

## 3. Participant Background Questionnaire

在 trial 前填写。

| Field | Answer |
|---|---|
| ParticipantID |  |
| Date |  |
| Condition |  |
| Age range | 18-24 / 25-34 / 35-44 / 45+ / prefer not to say |
| Dominant hand | left / right / both / prefer not to say |
| DCS experience | none / very little / occasional / frequent |
| Flight simulator experience | none / very little / occasional / frequent |
| VR experience | none / very little / occasional / frequent |
| Quest 3 experience | none / very little / occasional / frequent |
| HOTAS or cockpit control experience | none / very little / occasional / frequent |
| F/A-18C experience | none / watched videos / tried before / experienced |
| F/A-18C cold-start experience | none / watched videos / tried before / can complete |
| Familiarity with aviation English | 1 / 2 / 3 / 4 / 5 / 6 / 7 |
| Confidence before task | 1 / 2 / 3 / 4 / 5 / 6 / 7 |
| VR discomfort history | no / mild / moderate / severe |
| Notes |  |

Scale meaning for 1-7 questions:

| Score | Meaning |
|---|---|
| 1 | not at all |
| 4 | neutral / medium |
| 7 | very high |

## 4. Trial Run Sheet

实验执行者每个 trial 填一份。

| Field | Value |
|---|---|
| StudyID |  |
| ParticipantID |  |
| TrialID |  |
| Condition | `with_tutor` / `without_tutor` |
| ExperimenterID |  |
| Date |  |
| StartTime |  |
| EndTime |  |
| DCS mission |  |
| DCS aircraft | FA-18C_hornet |
| VR setup | Quest 3 Link / Quest 3 Air Link / other |
| Monitor setup | fa18c_composite_panel_v2 |
| Model endpoint |  |
| Text model | simtutor-base |
| Vision model | simtutor-vision |
| Git commit |  |
| Git dirty | yes / no |
| Launcher used | yes / no |
| Raw log path |  |
| Recording path |  |
| Questionnaire path |  |
| Completed | yes / no |
| End reason | completed / timeout / participant_stop / vr_discomfort / technical_failure / experimenter_stop |
| Technical issue occurred | yes / no |
| Experimenter intervention | none / technical only / procedural hint given |
| Notes |  |

## 5. Trial Observation Notes

实验中不要给步骤提示，只记录现象。

| Time | RequestID | Step/Phase | Observation | Participant behavior | System behavior | Severity | Notes |
|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  | low / medium / high |  |
|  |  |  |  |  |  | low / medium / high |  |
|  |  |  |  |  |  | low / medium / high |  |

常见记录项：

- help response slow
- overlay target wrong
- overlay invisible
- participant ignored correct help
- participant misunderstood wording
- model message wrong but validator output correct
- VLM called unexpectedly
- DCS/Quest/controller issue

## 6. Post-Trial NASA Raw TLX

建议使用 Raw NASA-TLX：参与者对 6 个维度各打 0-100 分，最后取平均。分数越高表示主观负荷越高。

填写说明：

```text
请根据刚才这一次任务的主观感受打分。0 表示左侧描述，100 表示右侧描述。可以填写任意 0 到 100 的整数。
```

| Dimension | 中文说明 | 0 | 100 | Score 0-100 |
|---|---|---|---|---|
| Mental Demand | 心理需求：任务需要多少思考、记忆、判断、搜索和注意力？ | 非常低 | 非常高 |  |
| Physical Demand | 身体需求：任务需要多少身体操作、转头、点击、移动或姿态控制？ | 非常低 | 非常高 |  |
| Temporal Demand | 时间压力：你是否觉得任务节奏紧、需要赶时间？ | 非常低 | 非常高 |  |
| Performance | 表现负担：你对自己的任务表现有多不满意？ | 非常满意/成功 | 非常不满意/失败 |  |
| Effort | 努力程度：你为了完成任务付出了多少努力？ | 非常低 | 非常高 |  |
| Frustration | 挫败感：你感到多烦躁、沮丧、紧张或受挫？ | 非常低 | 非常高 |  |

Raw TLX 计算：

```text
RawTLX = (MentalDemand + PhysicalDemand + TemporalDemand + Performance + Effort + Frustration) / 6
```

| ParticipantID | TrialID | Condition | MentalDemand | PhysicalDemand | TemporalDemand | Performance | Effort | Frustration | RawTLX |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
|  |  |  |  |  |  |  |  |  |  |

## 7. Post-Trial Usability and Trust Questionnaire

所有参与者都可以填写。`without_tutor` 条件中与 SimTutor 相关的题目可标记为 N/A。

Scale:

| Score | Meaning |
|---|---|
| 1 | strongly disagree |
| 2 | disagree |
| 3 | slightly disagree |
| 4 | neutral |
| 5 | slightly agree |
| 6 | agree |
| 7 | strongly agree |

| Item | Statement | Score 1-7 | N/A |
|---|---|---:|---|
| U1 | I understood what I was expected to do during the task. |  |  |
| U2 | The cockpit procedure felt difficult to follow. |  |  |
| U3 | I felt confident about my next action. |  |  |
| U4 | I could find the relevant cockpit controls. |  |  |
| U5 | The Quest 3 setup was comfortable enough for this task. |  |  |
| T1 | The tutor's instruction was easy to understand. |  |  |
| T2 | The highlighted control matched the written instruction. |  |  |
| T3 | The tutor helped me recover when I was unsure. |  |  |
| T4 | I trusted the tutor's guidance. |  |  |
| T5 | The tutor interrupted the task flow. |  |  |
| T6 | I would prefer using this tutor over a static checklist in VR. |  |  |

## 8. Open Questions

| Question | Answer |
|---|---|
| What was the most difficult part of the task? |  |
| What helped you the most? |  |
| What confused you the most? |  |
| Did the VR setup make the task easier or harder? Why? |  |
| If you used SimTutor, did any help message feel wrong or unclear? |  |
| If you used a checklist, was it easy to access in VR? |  |
| Any other comments? |  |

## 9. Export Quality Checklist

实验结束后由实验执行者填写。

| Check | Pass/Fail | Notes |
|---|---|---|
| Raw log exists |  |  |
| Recording exists |  |  |
| Questionnaire exists |  |  |
| `experiment-export --strict` passed |  |  |
| `quality_gate.json` has `passed=true` |  |  |
| `trial_summary.csv` condition is correct |  |  |
| `trial_summary.csv` participant/trial IDs are correct |  |  |
| `step_coding.csv` contains S01-S33 |  |  |
| `help_cycles.csv` looks plausible |  |  |
| `action_timeline.csv` exists |  |  |
| `VLMNotRequiredLeakageCount` acceptable |  |  |
| Technical issues recorded |  |  |
| Export directory backed up |  |  |

## 10. Baseline-Specific Notes

For `without_tutor`:

| Item | Answer |
|---|---|
| Checklist format | VR kneeboard / in-VR static page / paper / other |
| Checklist pages |  |
| Participant could access checklist without removing Quest 3 | yes / no |
| Participant used checklist | yes / no / partially |
| Manual coding required | yes / no |
| S18/S19 visual review completed | yes / no / N/A |
| S18/S19 review source | video / offline VLM / both / N/A |
| Notes |  |

为了避免弱证据，正式 baseline 推荐使用 VR 内可访问 checklist，而不是要求参与者摘下头显查看外部 PDF。
S18/S19 依赖右 DDI FCS-MC 页面和 final GO 视觉确认；如果 `step_coding.csv` 标记 `visual_step_requires_manual_review`，需要用录像或 trial 后离线 VLM 复核，不能把缺少 passive visual evidence 直接视为确认遗漏。

## 11. Tutor-Specific Notes

For `with_tutor`:

| Item | Answer |
|---|---|
| Help hotkey | X1 |
| Participant understood help rule | yes / no |
| Overlay visible in VR | yes / no / partially |
| Multi-target overlay visible | yes / no / partially |
| Tutor message language | zh / en / mixed |
| Any wrong help observed | yes / no |
| Any wrong highlight observed | yes / no |
| Notable request IDs |  |
