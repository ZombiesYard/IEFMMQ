"""Launch vLLM serve with Qwen3.5-9B-Base + SimTutor LoRA.

Run this helper with a working system Python, not the vLLM venv Python.
The vLLM venv under /scratch may survive while its original uv-managed base
interpreter under /home gets cleaned, which leaves venv/bin/python broken.

Example on remote host:
    /usr/bin/python3 scripts/dev_launch_vllm.py
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

VENV_DIR = Path("/scratch/yz50/vllm_qwen35/venv")
VLLM_BIN = VENV_DIR / "bin" / "vllm"
VENV_PY = VENV_DIR / "bin" / "python3"
SYSTEM_PYTHON = Path("/usr/bin/python3.12")
LOG_PATH = Path("/scratch/yz50/iefmmq_vlm_models/qwen35_vllm.log")
ADAPTER_NAME = "simtutor"
ADAPTER_PATH = Path(
    "/scratch/yz50/iefmmq_vlm_models/full_qwen35_9b_base_bilingual_run003_plus_run005x2_v1/adapter"
)
CACHE_DIRS = [
    Path("/scratch/yz50/tmp"),
    Path("/scratch/yz50/.cache"),
    Path("/scratch/yz50/.cache/vllm"),
    Path("/scratch/yz50/.config/vllm"),
    Path("/scratch/yz50/.cache/triton"),
    Path("/scratch/yz50/.cache/torchinductor"),
    Path("/scratch/yz50/.cache/huggingface"),
    Path("/scratch/yz50/.cache/huggingface/hub"),
    Path("/scratch/yz50/.cache/huggingface/transformers"),
]


def _repair_broken_venv_python() -> None:
    """Repair venv entrypoints when the old uv-managed home Python vanished."""
    if VENV_PY.exists():
        return
    if not SYSTEM_PYTHON.exists():
        raise FileNotFoundError(
            "vLLM venv python is broken and fallback system python3.12 is unavailable"
        )

    (VENV_DIR / "bin" / "python").unlink(missing_ok=True)
    (VENV_DIR / "bin" / "python3").unlink(missing_ok=True)
    (VENV_DIR / "bin" / "python3.12").unlink(missing_ok=True)
    (VENV_DIR / "bin" / "python").symlink_to(SYSTEM_PYTHON)
    (VENV_DIR / "bin" / "python3").symlink_to("python")
    (VENV_DIR / "bin" / "python3.12").symlink_to("python")

    cfg_path = VENV_DIR / "pyvenv.cfg"
    if not cfg_path.exists():
        return

    rewritten: list[str] = []
    for line in cfg_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("home = "):
            rewritten.append("home = /usr/bin")
        elif line.startswith("version_info = "):
            rewritten.append("version_info = 3.12.3")
        elif line.startswith("uv = "):
            continue
        else:
            rewritten.append(line)
    cfg_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


for directory in CACHE_DIRS:
    directory.mkdir(parents=True, exist_ok=True)

_repair_broken_venv_python()

CMD = [
    str(VLLM_BIN),
    "serve",
    "Qwen/Qwen3.5-9B-Base",
    "--enable-lora",
    "--enable-tower-connector-lora",
    "--lora-modules",
    f"{ADAPTER_NAME}={ADAPTER_PATH}",
    "--served-model-name",
    "simtutor-qwen35-9b-lora",
    "--download-dir",
    "/scratch/yz50/.cache/huggingface/hub",
    "--gpu-memory-utilization",
    "0.75",
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
    "--max-model-len",
    "16384",
]

env = os.environ.copy()
env["HOME"] = "/scratch/yz50"
env["TMPDIR"] = "/scratch/yz50/tmp"
env["XDG_CACHE_HOME"] = "/scratch/yz50/.cache"
env["HF_HOME"] = "/scratch/yz50/.cache/huggingface"
env["HUGGINGFACE_HUB_CACHE"] = "/scratch/yz50/.cache/huggingface/hub"
env["TRANSFORMERS_CACHE"] = "/scratch/yz50/.cache/huggingface/transformers"
env["VLLM_CACHE_ROOT"] = "/scratch/yz50/.cache/vllm"
env["VLLM_CONFIG_ROOT"] = "/scratch/yz50/.config/vllm"
env["VLLM_NO_USAGE_STATS"] = "1"
env["TRITON_CACHE_DIR"] = "/scratch/yz50/.cache/triton"
env["TORCHINDUCTOR_CACHE_DIR"] = "/scratch/yz50/.cache/torchinductor"
env["CUDA_VISIBLE_DEVICES"] = "0"
env["PATH"] = "/scratch/yz50/.local/bin:/scratch/yz50/vllm_qwen35/venv/bin:/usr/local/bin:/usr/bin:/bin"

print(f"Launching vLLM: {' '.join(CMD)}", flush=True)
print(f"Log: {LOG_PATH}", flush=True)

with LOG_PATH.open("a", encoding="utf-8") as log_f:
    log_f.write(f"\n{'=' * 60}\n")
    log_f.write(f"Launch at PID {os.getpid()}\n")
    log_f.write(f"VENV_PY={VENV_PY}\n")
    log_f.write(f"VLLM_CACHE_ROOT={env.get('VLLM_CACHE_ROOT')}\n")
    log_f.write(f"VLLM_CONFIG_ROOT={env.get('VLLM_CONFIG_ROOT')}\n")
    log_f.write(f"TRITON_CACHE_DIR={env.get('TRITON_CACHE_DIR')}\n")
    log_f.write(f"XDG_CACHE_HOME={env.get('XDG_CACHE_HOME')}\n")
    log_f.write(f"{'=' * 60}\n")
    log_f.flush()
    proc = subprocess.Popen(
        CMD,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
    )
    print(f"vLLM pid: {proc.pid}", flush=True)
    proc.wait()
