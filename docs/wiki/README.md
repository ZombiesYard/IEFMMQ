# SimTutor Wiki

This wiki collects operational notes that are too detailed for the front page.

## Pages

- [Quickstart](quickstart.md): local setup, tests, mock runs, replay.
- [DCS live operation](dcs-live-operation.md): simulator hook installation, monitor setup, live loop, sidecar capture.
- [VLM dataset pipeline](vlm-dataset-pipeline.md): screenshot capture, pre-labeling, Label Studio review, SFT export.
- [Qwen3.5 VLM fine-tuning](fine-tuning-qwen35-vlm.md): Unsloth + PEFT LoRA + TRL `SFTTrainer` training workflow.
- [Benchmarking](benchmarking.md): base-vs-LoRA evaluation and metric interpretation.
- [Replay and CLI reference](replay-and-cli-reference.md): frequently used replay and model-provider commands.
- [Artifacts and large files](artifacts-and-large-files.md): Git LFS, datasets, adapter, benchmark artifacts, ignored capture caches.

## Path Placeholders

Public documentation uses placeholders instead of machine-specific paths:

| Placeholder | Meaning |
|---|---|
| `<repo>` | Local checkout of this repository |
| `<saved-games-dir>` | DCS Saved Games directory |
| `<scratch-root>` | Remote/HPC scratch workspace |
| `<hf-cache-dir>` | Hugging Face cache directory |
| `<remote-host>` | Remote training host |
| `<hf-username>` | Hugging Face account or organization |

## Shell Conventions

Command blocks are labeled by shell:

- `bash`: Linux or WSL.
- `powershell`: Windows host commands.
- `tcsh`: remote training host examples when the login shell is tcsh.
