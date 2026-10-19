#!/usr/bin/env bash
# SoT agent baseline on deepsearch with in-process vLLM inference (--model vllm).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASELINES_DIR="$(cd "${SOT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BASELINES_DIR}/.." && pwd)"
# API keys: export them or put them in ${REPO_ROOT}/.env (see baselines/README.md)
if [[ -f "${REPO_ROOT}/.env" ]]; then set -a; source "${REPO_ROOT}/.env"; set +a; fi
: "${SERPER_API_KEY:?set SERPER_API_KEY (see baselines/README.md)}"
: "${JINA_API_KEY:?set JINA_API_KEY (see baselines/README.md)}"
DATA_DIR="${DATA_DIR:-${BASELINES_DIR}/data}"
cd "${SOT_DIR}"
export PYTHONPATH="${SOT_DIR}:${PYTHONPATH:-}"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-32B}"
SELECTED_GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# Tensor parallel size should normally equal the number of selected GPUs.
TENSOR_PARALLEL="${TENSOR_PARALLEL:-4}"
GPU_UTIL="${GPU_UTIL:-0.9}"
export CUDA_VISIBLE_DEVICES="${SELECTED_GPUS}"

echo "Loading model with local vLLM: ${TARGET_MODEL}"
echo "GPUs: ${SELECTED_GPUS} (TP=${TENSOR_PARALLEL})"

python sot/main.py \
    --model vllm \
    --api-model "${TARGET_MODEL}" \
    --scheduler outline \
    --data-path "${DATA_DIR}/deepsearch/DeepSearch-2510_decrypted.json" \
    --output-folder results/DeepSearch/Qwen3-32B_vllm_local \
    --vllm-tensor-parallel-size "${TENSOR_PARALLEL}" \
    --vllm-gpu-memory-utilization "${GPU_UTIL}" \
    --vllm-trust-remote-code \
    --vllm-dtype "auto" \
    --temperature 0.7 \
    --max-tokens 4096 \
    --top-p 0.95 \
    --prompt-file prompts/new_agent.yaml \
    --limit 10 \
    --max-workers 5 \
    --max-turns 6 \
    --max-dataset-workers 5 \
    --timeout 1200
