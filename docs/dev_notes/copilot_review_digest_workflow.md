# Copilot Review Digest Workflow

这个小工具用于在 **Copilot review 已经完成之后**，由你手动触发一次命令，把当前 PR 里的 Copilot review comments 抓出来，并整理成适合继续喂给 Codex 的文本。

工具位置：

- `tools/copilot_review_digest.py`

## 常用命令

### 1. 当前分支对应 PR，输出成适合喂给 Codex 的中文 prompt

```bash
python -m tools.copilot_review_digest
```

### 2. 指定 PR 编号

```bash
python -m tools.copilot_review_digest --pr 205
```

### 3. 输出到文件，方便直接复制或让其他工具读取

```bash
python -m tools.copilot_review_digest --pr 205 --output .tmp/copilot_review_prompt.md
```

### 4. 看 markdown 摘要，而不是 Codex prompt

```bash
python -m tools.copilot_review_digest --pr 205 --format markdown
```

### 5. 看原始 JSON 结构

```bash
python -m tools.copilot_review_digest --pr 205 --format json
```

## 设计目标

这个工具默认：

- 用 `gh` 读取当前 repo / 当前 PR
- 优先抓取 **Copilot** 的 review comments
- 默认忽略已经 resolved 的 review threads
- 输出时保留：
  - 文件路径
  - 行号
  - comment 内容
  - comment URL
  - 创建时间

## 适合的使用方式

推荐流程：

1. 你或自动规则触发 Copilot review
2. 等 Copilot review 完成
3. 运行这个工具
4. 把输出直接给 Codex
5. Codex 根据这些 review comments 修复问题，并说明接受/拒绝理由

## 已知限制

- 它依赖本地 `gh` 已正确登录
- 它只负责“读 review comments”，不自动修改代码
- 它默认只看 **Copilot** comments，不会汇总所有人类 reviewer 的评论
- 它目前基于 GitHub GraphQL `reviewThreads` 抓取 review thread 信息

---

## 第二步：更完整的 PR 修复包

如果你想把 **PR 基本信息 + CI/checks + Copilot comments** 一次性整理给 Codex，可以用：

- `tools/copilot_review_loop.py`

### 常用命令

```bash
python3 -m tools.copilot_review_loop
```

这条命令会默认：

- 自动定位当前分支对应的 PR
- 读取 PR 标题、分支、review decision
- 读取 `gh pr checks` 输出
- 读取当前未 resolved 的 Copilot review threads
- 输出成适合继续喂给 Codex 的中文修复提示

### 输出到文件

```bash
python3 -m tools.copilot_review_loop --output .tmp/copilot_review_bundle.md
```

如果当前 branch 没有对应 PR，或者 `gh` 无法自动推断，就显式指定：

```bash
python3 -m tools.copilot_review_loop --pr 205 --output .tmp/copilot_review_bundle.md
```

### 重新请求 Copilot review

如果你已经 push 了修复，并且想显式再请求一次 Copilot review，可以用：

```bash
python3 -m tools.copilot_review_loop --request-copilot-review
```

这个动作会在生成修复包的同时，执行：

- `gh pr edit <PR> --add-reviewer @copilot`

### 推荐使用方式

推荐的半自动流程：

1. 你等待 Copilot review 完成
2. 运行：

   ```bash
   python3 -m tools.copilot_review_loop --output .tmp/copilot_review_bundle.md
   ```

3. 把输出直接给 Codex
4. Codex 修改代码并跑测试
5. 你 push 新 commit
6. 运行：

   ```bash
   python3 -m tools.copilot_review_loop --request-copilot-review
   ```

7. 等待下一轮 review

### 把失败的 CI 日志也一起打包

如果你希望 Codex 同时看到当前 PR 的失败 Actions 日志，可以加这个开关：

```bash
python3 -m tools.copilot_review_loop --pr 205 --include-failed-run-logs --output .tmp/copilot_review_bundle.md
```

默认行为：

- 按 PR 当前 head commit 查找 workflow runs
- 选取最近的失败 runs
- 抓取 `gh run view --log-failed` 输出
- 对每个失败日志做截断，避免 bundle 过大

默认限制：

- 最多包含 2 个失败 runs
- 每个失败日志最多 12000 个字符

如果你想调大或调小：

```bash
python3 -m tools.copilot_review_loop \
  --pr 205 \
  --include-failed-run-logs \
  --max-failed-runs 1 \
  --max-log-chars 8000 \
  --output .tmp/copilot_review_bundle.md
```

---

## 第三步：直接调用 Codex 自动修复

如果你已经想把“生成修复包 + 调用 Codex 非交互执行”串起来，可以用：

- `tools/copilot_review_autofix.py`

### 最常用命令

```bash
python3 -m tools.copilot_review_autofix --pr 205 --include-failed-run-logs
```

这条命令会：

- 先生成一个整理好的修复 prompt
- 写到 `.tmp/copilot_review_bundle.md`
- 再调用 `codex exec`
- 把 Codex 的最后总结写到 `.tmp/copilot_autofix_last_message.md`

### 先只看命令，不真正执行

```bash
python3 -m tools.copilot_review_autofix --pr 205 --include-failed-run-logs --dry-run
```

这个模式适合先检查：

- 生成的 prompt 是否合理
- `codex exec` 命令是否符合你的预期

### 可调参数

如果你想指定模型、profile 或 sandbox：

```bash
python3 -m tools.copilot_review_autofix \
  --pr 205 \
  --include-failed-run-logs \
  --codex-model gpt-5.5 \
  --codex-profile default \
  --codex-sandbox workspace-write \
  --codex-approval-policy never
```

### 当前边界

这个脚本当前会自动：

- 生成 bundle
- 调用 Codex 修复

但它**不会**自动做这些远程动作：

- `git push`
- `gh pr edit --add-reviewer @copilot`
- `gh pr merge`

也就是说，它现在是“自动修代码”，不是“自动合并机器人”。

### 第四步：把修复后的 push / re-review 也串起来

现在 `tools/copilot_review_autofix.py` 已经支持可选的闭环动作，但默认仍然是保守模式。

如果你想在 Codex 修完后：

- 运行一个或多个后置验证命令
- 自动 `git add -A && git commit`
- 自动 `git push`
- 自动重新请求 Copilot review

可以这样用：

```bash
python3 -m tools.copilot_review_autofix \
  --pr 205 \
  --include-failed-run-logs \
  --post-command "source ~/venvs/iefmmq-wsl/bin/activate && PYTHONPATH=/usr/lib/python3/dist-packages python -m pytest -q tests/test_live_dcs.py" \
  --git-commit \
  --commit-message "fix: address Copilot review feedback on PR #205" \
  --git-push \
  --request-copilot-review
```

说明：

- `--post-command` 可重复传入多次
- `--git-commit` 会执行 `git add -A`
- 如果本轮没有新改动可提交，它会跳过 commit
- `--git-push` 目前要求本轮确实创建了 commit，避免误 push
- `--request-copilot-review` 目前要求同时开启 `--git-push`

### 推荐的逐步放权方式

如果你想稳一点，建议按这个顺序来：

1. 先只用 `--dry-run`
2. 再只跑自动 Codex 修复
3. 再加 `--post-command`
4. 再加 `--git-commit`
5. 最后才加 `--git-push --request-copilot-review`
