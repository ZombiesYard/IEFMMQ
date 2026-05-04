#!/bin/tcsh -f

set VENV_DIR=/scratch/yz50/vllm_qwen35/venv
set VLLM_BIN=$VENV_DIR/bin/vllm
set VENV_PY=$VENV_DIR/bin/python3
set SYS_PY=/usr/bin/python3.12
set ADAPTER_PATH=/scratch/yz50/iefmmq_vlm_models/full_qwen35_9b_base_bilingual_run003_plus_run005x2_v1/adapter

mkdir -p /scratch/yz50/tmp
mkdir -p /scratch/yz50/.cache/huggingface/hub
mkdir -p /scratch/yz50/.cache/huggingface/transformers
mkdir -p /scratch/yz50/.cache/vllm
mkdir -p /scratch/yz50/.config/vllm
mkdir -p /scratch/yz50/.cache/triton
mkdir -p /scratch/yz50/.cache/torchinductor

setenv HOME /scratch/yz50
setenv TMPDIR /scratch/yz50/tmp
setenv XDG_CACHE_HOME /scratch/yz50/.cache
setenv HF_HOME /scratch/yz50/.cache/huggingface
setenv HUGGINGFACE_HUB_CACHE /scratch/yz50/.cache/huggingface/hub
setenv TRANSFORMERS_CACHE /scratch/yz50/.cache/huggingface/transformers
setenv VLLM_CACHE_ROOT /scratch/yz50/.cache/vllm
setenv VLLM_CONFIG_ROOT /scratch/yz50/.config/vllm
setenv VLLM_NO_USAGE_STATS 1
setenv TRITON_CACHE_DIR /scratch/yz50/.cache/triton
setenv TORCHINDUCTOR_CACHE_DIR /scratch/yz50/.cache/torchinductor
setenv CUDA_VISIBLE_DEVICES 0
setenv PATH /scratch/yz50/.local/bin:/scratch/yz50/vllm_qwen35/venv/bin:/usr/local/bin:/usr/bin:/bin
rehash

if (! -x $VENV_PY) then
  if (-x $SYS_PY) then
    echo "Repairing broken vLLM venv python entrypoints..."
    ln -sfn $SYS_PY $VENV_DIR/bin/python
    ln -sfn python $VENV_DIR/bin/python3
    ln -sfn python $VENV_DIR/bin/python3.12
    if (-f $VENV_DIR/pyvenv.cfg) then
      sed -i -e 's#^home = .*#home = /usr/bin#' \
             -e '/^uv = /d' \
             $VENV_DIR/pyvenv.cfg
    endif
    rehash
  else
    echo "ERROR: broken vLLM venv python and no fallback /usr/bin/python3.12"
    exit 1
  endif
endif

if (! -x $VLLM_BIN) then
  echo "ERROR: vLLM executable not found: $VLLM_BIN"
  exit 1
endif

echo "Starting vLLM serve with Qwen/Qwen3.5-9B-Base + SimTutor LoRA..."
exec $VLLM_BIN serve Qwen/Qwen3.5-9B-Base \
  --enable-lora \
  --enable-tower-connector-lora \
  --lora-modules "simtutor=$ADAPTER_PATH" \
  --served-model-name simtutor-qwen35-9b-lora \
  --download-dir /scratch/yz50/.cache/huggingface/hub \
  --gpu-memory-utilization 0.75 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 16384
