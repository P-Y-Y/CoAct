#!/usr/bin/env bash
# SoT agent baseline on hle: starts a local vLLM OpenAI-compatible server, then runs the agent against it.
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

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-32B}"  # HF repo id or local path
SELECTED_GPUS="${CUDA_VISIBLE_DEVICES:-1,2}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
GPU_UTIL="${GPU_UTIL:-0.9}"
PORT="${PORT:-8008}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-30000}"

# The local vLLM server does not check the key, but the client requires one.
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export CUDA_VISIBLE_DEVICES="${SELECTED_GPUS}"

# Multimodal inspector tools: VisualInspectorTool calls a separate OpenAI-compatible endpoint
: "${VISION_API_KEY:?set VISION_API_KEY (see baselines/README.md)}"
export VISION_BASE_URL="${VISION_BASE_URL:-https://openrouter.ai/api/v1}"
export VISION_MODEL="${VISION_MODEL:-openai/chatgpt-4o-latest}"

# Shut down the vLLM server on exit or Ctrl+C.
SERVER_PID=""
cleanup() {
    echo ""
    echo ">>> [Cleanup] Shutting down vLLM server (PID: ${SERVER_PID})..."
    if [[ -n "${SERVER_PID}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT

echo ">>> [Phase 1] Starting vLLM OpenAI API server..."
echo "    GPUs: ${SELECTED_GPUS} (TP=${TENSOR_PARALLEL})"
echo "    Port: ${PORT}"

python -m vllm.entrypoints.openai.api_server \
    --model "${TARGET_MODEL}" \
    --served-model-name "${TARGET_MODEL}" \
    --tensor-parallel-size "${TENSOR_PARALLEL}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --trust-remote-code \
    --dtype auto \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --disable-log-requests &
SERVER_PID=$!
echo ">>> vLLM server PID: ${SERVER_PID}"

echo ">>> Waiting for server to start..."
while ! nc -z localhost "${PORT}"; do
    sleep 2
    echo -n "."
done
echo ""
echo ">>> Port ${PORT} is open; waiting 10s for model weights to finish loading..."
sleep 10

echo ">>> [Phase 2] Running SoT agent against the local server..."
python sot/main.py \
    --model openai \
    --api-model "${TARGET_MODEL}" \
    --api-base "http://localhost:${PORT}/v1" \
    --scheduler outline \
    --data-path "${DATA_DIR}/hle/hle_sampled_500.json" \
    --output-folder results/hle/Qwen3-32B_vllm_client_all_0207 \
    --temperature 0.7 \
    --max-tokens 8192 \
    --top-p 0.95 \
    --prompt-file prompts/new_agent.yaml \
    --limit 500 \
    --max-workers 3 \
    --max-turns 7 \
    --max-dataset-workers 5 \
    --max-worker-turns 40 \
    --enable-mm-tools \
    --timeout 1200
