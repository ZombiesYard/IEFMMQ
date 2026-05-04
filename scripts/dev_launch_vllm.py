"""Launch vLLM serve with Qwen3.5-9B-Base + LoRA adapter.

Usage (on remote tcsh):
    setenv TMPDIR /scratch/yz50/tmp
    setenv HF_HOME /scratch/yz50/.cache/huggingface
    /scratch/yz50/vllm_qwen35/venv/bin/python3 /scratch/yz50/launch_vllm.py
"""

import os
import sys
import subprocess

# --- Cache redirects (must be set before vllm imports anything) ---
os.environ.setdefault("TMPDIR", "/scratch/yz50/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/scratch/yz50/.cache")
os.environ.setdefault("TRITON_CACHE_DIR", "/scratch/yz50/.cache/triton")
os.environ.setdefault("HF_HOME", "/scratch/yz50/.cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/scratch/yz50/.cache/huggingface/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/scratch/yz50/.cache/huggingface/transformers")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Ensure cache dirs exist
for d in [
    "/scratch/yz50/tmp",
    "/scratch/yz50/.cache",
    "/scratch/yz50/.cache/triton",
    "/scratch/yz50/.cache/vllm",
    "/scratch/yz50/.cache/huggingface",
    "/scratch/yz50/.cache/huggingface/hub",
    "/scratch/yz50/.cache/huggingface/transformers",
]:
    os.makedirs(d, exist_ok=True)

LOG_PATH = "/scratch/yz50/iefmmq_vlm_models/qwen35_vllm.log"

ADAPTER_NAME = "simtutor"
ADAPTER_PATH = "/scratch/yz50/iefmmq_vlm_models/full_qwen35_9b_base_bilingual_run003_plus_run005x2_v1/adapter"

CMD = [
    sys.executable, "-m", "vllm", "serve",
    "Qwen/Qwen3.5-9B-Base",
    "--enable-lora",
    "--lora-modules", f"{ADAPTER_NAME}={ADAPTER_PATH}",
    "--served-model-name", "simtutor-qwen35-9b-lora",
    "--download-dir", "/scratch/yz50/.cache/huggingface/hub",
    "--gpu-memory-utilization", "0.75",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--max-model-len", "16384",
]

print(f"Launching vLLM: {' '.join(CMD)}", flush=True)
print(f"Log: {LOG_PATH}", flush=True)

with open(LOG_PATH, "a") as log_f:
    log_f.write(f"\n{'='*60}\n")
    log_f.write(f"Launch at PID {os.getpid()}\n")
    log_f.write(f"TRITON_CACHE_DIR={os.environ.get('TRITON_CACHE_DIR')}\n")
    log_f.write(f"XDG_CACHE_HOME={os.environ.get('XDG_CACHE_HOME')}\n")
    log_f.write(f"{'='*60}\n")
    log_f.flush()
    proc = subprocess.Popen(
        CMD,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    print(f"vLLM pid: {proc.pid}", flush=True)
    proc.wait()
