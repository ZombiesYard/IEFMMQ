# VLM Dataset Pipeline

This page describes the visual dataset pipeline used for Qwen3.5 VLM fine-tuning.

## Capture Fine-Tuning Screenshots

PowerShell example on the simulator host:

```powershell
python .\tools\capture_vlm_dataset.py `
  --session-id fa18c-coldstart-run-001 `
  --saved-games-dir "<saved-games-dir>"
```

The tool waits for the configured help hotkey by default. After the first trigger it captures continuous composite-panel frames at the configured FPS and writes outputs under:

```text
tools/.captures/<session-id>/
```

The raw capture cache is ignored by Git and should not be pushed as a normal repository artifact.

## Generate VLM Prelabels

Linux/WSL example:

```bash
export DASHSCOPE_API_KEY="<dashscope-api-key>"
python -m tools.generate_vlm_prelabels \
  --session-dir tools/.captures/fa18c-coldstart-run-001 \
  --model-name qwen3.5-397b-a17b \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --lang en \
  --overwrite \
  --save-raw-response
```

Wrapper entrypoints:

```bash
python -m tools.generate_vlm_prelabels_en --session-dir tools/.captures/<session-id> --overwrite
python -m tools.generate_vlm_prelabels_zh --session-dir tools/.captures/<session-id> --overwrite
```

Outputs are written to:

```text
tools/.captures/<session-id>/prelabels/
```

Important output files:

- `vision_prelabels.jsonl`
- `label_studio_tasks.json`
- `raw_model_outputs.jsonl`
- `prelabels_failures.jsonl`

## Review with Label Studio

Linux/WSL setup:

```bash
python3 -m venv ~/venvs/label-studio-wsl
source ~/venvs/label-studio-wsl/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install label-studio
```

Start Label Studio from the repository:

```bash
cd <repo>
./tools/start_label_studio_wsl.sh
```

Open:

```text
http://localhost:8080
```

Use this labeling config:

```text
tools/label_studio_review_config.xml
```

Import:

```text
tools/.captures/<session-id>/prelabels/label_studio_tasks.json
```

## Export Reviewed SFT Data

Linux/WSL example:

```bash
python -m tools.export_vision_sft_dataset \
  --input tools/project-1-at-<timestamp>.json \
  --output-dir datasets/vision_sft \
  --lang both \
  --overwrite
```

This writes:

- `reviewed.jsonl`
- `sft_en.jsonl`
- `sft_zh.jsonl`
- `stats.json`

The final SFT samples keep the single composite-panel image as a data URL and do not expose `frame_id`, `session_id`, image paths, `source_frame_id`, or `confidence` to the model target.
