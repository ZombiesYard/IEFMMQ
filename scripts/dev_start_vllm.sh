#!/bin/tcsh
setenv TMPDIR /scratch/yz50/tmp
setenv XDG_CACHE_HOME /scratch/yz50/.cache
setenv TRITON_CACHE_DIR /scratch/yz50/.cache/triton
setenv HF_HOME /scratch/yz50/.cache/huggingface
setenv HUGGINGFACE_HUB_CACHE /scratch/yz50/.cache/huggingface/hub
setenv TRANSFORMERS_CACHE /scratch/yz50/.cache/huggingface/transformers
setenv CUDA_VISIBLE_DEVICES 0
setenv PATH /scratch/yz50/.local/bin:/scratch/yz50/vllm_qwen35/venv/bin:${PATH}
rehash

echo "Starting vLLM serve..."
vllm serve Qwen/Qwen3.5-9B-Base \
  --enable-lora \
  --lora-modules "simtutor=/scratch/yz50/iefmmq_vlm_models/full_qwen35_9b_base_bilingual_run003_plus_run005x2_v1/adapter" \
  --served-model-name simtutor-qwen35-9b-lora \
  --download-dir /scratch/yz50/.cache/huggingface/hub \
  --gpu-memory-utilization 0.75 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 16384
