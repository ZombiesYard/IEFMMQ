## 变更摘要

<!-- 3-6 行：说明这次 PR 解决的问题与核心改动 -->

## 关联 Issue

- Issue:
- 范围边界（本 PR 不做什么）:

## 根因与设计决策

### 根因

<!-- 说明为什么会出现问题，避免只描述现象 -->

### 设计决策

<!-- 列出关键 trade-off，为什么这样做 -->

## 文件清单

<!-- 每个文件一行，写清职责变化 -->
- [ ] `path/to/file_a.py`:
- [ ] `path/to/file_b.md`:

## 行为变化（Before/After）

### Before

### After

## 安全与架构自检

- [ ] 未引入自动点击/动作注入（仅 overlay）
- [ ] overlay target 仍经过 allowlist + evidence 校验
- [ ] `core/` 未引入 DCS/UDP/Lua/机型按钮细节
- [ ] 未破坏 `schemas/v1/*`
- [ ] 新字段为可选字段或已给出迁移策略
- [ ] 日志未输出 API key 或完整 prompt 文本
- [ ] 失败路径具备 deterministic fallback

## 测试证据

### 本地执行命令

```bash
python tools/ci_guard.py
pytest -q
```

### 结果摘要

- `ci_guard`:
- `pytest`:

### 新增/修改测试

- [ ] 单元测试:
- [ ] 集成测试:
- [ ] 手工冒烟测试步骤（如涉及 DCS/vLLM）:

## 可观测性与回放

- [ ] 事件链完整：`tutor_request -> tutor_response -> overlay_*`
- [ ] metadata 诊断字段已补齐（如 `retry_count/repair_applied/fallback_overlay_*`）
- [ ] replay 日志可复现关键路径

## 风险与降级策略

<!-- 明确失败时系统如何安全退化 -->

## Reviewer 检查重点（建议）

1. 架构边界是否被破坏（core purity）。
2. 安全边界是否被破坏（no automation + allowlist/evidence）。
3. 契约兼容是否安全（v1/v2 schema 语义稳定）。
4. 测试是否覆盖真实失败路径而非仅 happy path。

