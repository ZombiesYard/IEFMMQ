"""
Train Qwen/Qwen3.5-9B-Base with Unsloth on the SimTutor VLM SFT dataset.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
from typing import Any, Iterable, Sequence


class ListDataset:
    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        self._rows = list(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._rows[index]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen/Qwen3.5-9B-Base as a VLM with Unsloth."
    )
    parser.add_argument(
        "--train-jsonl",
        nargs="+",
        required=True,
        help="One or more SFT JSONL files exported by tools/export_vision_sft_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for LoRA checkpoints and final adapter export.",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3.5-9B-Base",
        help="Vision model name to fine-tune.",
    )
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--num-train-epochs", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument(
        "--report-to",
        default="none",
        help="Trainer report target, for example 'none' or 'tensorboard'.",
    )
    return parser


def _load_rows(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                payload = line.strip()
                if not payload:
                    continue
                row = json.loads(payload)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} must be a JSON object")
                rows.append(row)
    return rows


def _normalize_message_content_item(item: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(item)
    if normalized.get("type") == "image_url":
        image_url = normalized.get("image_url")
        if isinstance(image_url, dict) and "url" in image_url:
            normalized = {"type": "image", "image": image_url["url"]}
        elif isinstance(image_url, str):
            normalized = {"type": "image", "image": image_url}
    return normalized


def _normalize_messages_for_unsloth(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_messages: list[dict[str, Any]] = []
    for message in messages:
        normalized_message = deepcopy(message)
        content = normalized_message.get("content")
        if isinstance(content, list):
            normalized_message["content"] = [
                _normalize_message_content_item(item) if isinstance(item, dict) else item
                for item in content
            ]
        normalized_messages.append(normalized_message)
    return normalized_messages


def _normalize_rows_for_unsloth(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized = deepcopy(row)
        messages = normalized.get("messages")
        if not isinstance(messages, list):
            raise ValueError("Each training row must contain a messages list")
        normalized_messages = _normalize_messages_for_unsloth(messages)
        normalized_rows.append({"messages": normalized_messages})
    return normalized_rows


def _split_rows(
    rows: list[dict[str, Any]],
    *,
    eval_ratio: float,
    seed: int,
):
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 2 or eval_ratio <= 0:
        return ListDataset(shuffled), None

    eval_count = max(1, int(round(len(shuffled) * eval_ratio)))
    if eval_count >= len(shuffled):
        eval_count = max(1, len(shuffled) - 1)

    eval_rows = shuffled[:eval_count]
    train_rows = shuffled[eval_count:]
    return ListDataset(train_rows), ListDataset(eval_rows)


def _select_train_rows(
    rows: Sequence[dict[str, Any]],
    *,
    max_train_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    selected = list(rows)
    if max_train_samples <= 0 or max_train_samples >= len(selected):
        return selected
    random.Random(seed).shuffle(selected)
    return selected[:max_train_samples]


def train_qwen35_vlm_unsloth(
    *,
    train_jsonl: Sequence[str | Path],
    output_dir: str | Path,
    model_name: str,
    max_seq_length: int,
    num_train_epochs: float,
    learning_rate: float,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int,
    warmup_ratio: float,
    weight_decay: float,
    logging_steps: int,
    save_total_limit: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    eval_ratio: float,
    seed: int,
    max_train_samples: int,
    report_to: str,
) -> dict[str, Any]:
    from unsloth import FastVisionModel, is_bf16_supported
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(train_jsonl)
    rows = _select_train_rows(rows, max_train_samples=max_train_samples, seed=seed)
    if not rows:
        raise ValueError("No training rows were loaded")

    rows = _normalize_rows_for_unsloth(rows)
    train_dataset, eval_dataset = _split_rows(rows, eval_ratio=eval_ratio, seed=seed)

    model, processor = FastVisionModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
        trust_remote_code=True,
        gpu_memory_utilization=0.6,
    )
    model = FastVisionModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        max_seq_length=max_seq_length,
    )
    model.for_training()

    bf16 = is_bf16_supported()
    data_collator = UnslothVisionDataCollator(
        model,
        processor,
        max_seq_length=max_seq_length,
    )

    trainer_args = SFTConfig(
        output_dir=str(output_path),
        max_seq_length=max_seq_length,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_total_limit=save_total_limit,
        seed=seed,
        bf16=bf16,
        fp16=not bf16,
        report_to=report_to,
        remove_unused_columns=False,
        dataset_num_proc=1,
    )

    trainer = SFTTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor,
    )
    train_result = trainer.train()

    trainer.save_model(str(output_path / "adapter"))
    processor.save_pretrained(str(output_path / "adapter"))

    metrics = dict(train_result.metrics)
    summary = {
        "model_name": model_name,
        "output_dir": str(output_path),
        "train_rows": len(train_dataset),
        "eval_rows": 0 if eval_dataset is None else len(eval_dataset),
        "input_files": [str(Path(path).expanduser().resolve()) for path in train_jsonl],
        "max_seq_length": max_seq_length,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "seed": seed,
        "metrics": metrics,
    }
    (output_path / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = train_qwen35_vlm_unsloth(
        train_jsonl=args.train_jsonl,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        report_to=args.report_to,
    )
    print(
        "[TRAIN_QWEN35_VLM_UNSLOTH] summary="
        + json.dumps(summary, ensure_ascii=False, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
