#!/usr/bin/env bash
# ReAct baseline on xbench-DeepSearch with a local vLLM server (Qwen3-32B).
# Starts the vLLM OpenAI-compatible server, runs the agent against it, and shuts
# the server down on exit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="$(cd "${BASELINES_DIR}/.." && pwd)"
# API keys: export them or put them in ${REPO_ROOT}/.env (see baselines/README.md)
if [[ -f "${REPO_ROOT}/.env" ]]; then set -a; source "${REPO_ROOT}/.env"; set +a; fi
: "${SERPER_API_KEY:?set SERPER_API_KEY (see baselines/README.md)}"
: "${JINA_API_KEY:?set JINA_API_KEY (see baselines/README.md)}"
DATA_DIR="${DATA_DIR:-${BASELINES_DIR}/data}"

# ================= 1. Configuration =================

# Model (HF repo ID or local path) and the name it is served under
MODEL_PATH="Qwen/Qwen3-32B"
SERVED_MODEL_NAME="Qwen/Qwen3-32B"

# GPUs / vLLM server
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,7}"
TENSOR_PARALLEL=2      # usually the number of GPUs
GPU_UTIL=0.90
PORT=8009
MAX_MODEL_LEN=30000

# Task
INPUT_FILE="${DATA_DIR}/deepsearch/DeepSearch-2510_decrypted.json"
MAX_STEPS=40
CONCURRENCY=5
LIMIT=0        # 0 = all

# The local vLLM server ignores the key, but the OpenAI client needs a non-empty value.
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# ================= 2. Log / output paths =================

LOG_DIR="${BASELINES_DIR}/react/logs/deepsearch"
OUT_DIR="${BASELINES_DIR}/react/results/deepsearch"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

RUN_TS=$(date '+%Y%m%d_%H%M%S')
VLLM_LOG="${LOG_DIR}/vllm_server_${RUN_TS}.log"
AGENT_LOG="${LOG_DIR}/react_agent_${RUN_TS}.log"
OUTPUT_JSON="${OUT_DIR}/answer_vllm_${RUN_TS}.json"

# Run from the repo root so `python -m baselines.react...` resolves.
cd "${REPO_ROOT}"

echo ">>> [Init] Log files:"
echo "    vLLM Log:  $VLLM_LOG"
echo "    Agent Log: $AGENT_LOG"

# ================= 3. Cleanup on exit =================

SERVER_PID=""

cleanup() {
    echo ""
    echo "========================================================"
    echo ">>> [Cleanup] Shutting down..."
    if [ -n "$SERVER_PID" ]; then
        echo ">>> Killing vLLM Server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        echo ">>> vLLM Server stopped."
    fi
    echo ">>> Done."
    exit
}

trap cleanup EXIT SIGINT SIGTERM

if nc -z localhost "$PORT" 2>/dev/null; then
    echo ""
    echo "[Error] Port $PORT is already in use by another process!"
    echo "    Change PORT in this script or stop the existing process."
    exit 1
fi

# ================= 4. Start vLLM server =================

echo ""
echo ">>> [Phase 1] Starting vLLM Server..."
echo "    Model: $MODEL_PATH"
echo "    TP: $TENSOR_PARALLEL | GPU Util: $GPU_UTIL | Port: $PORT"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    --dtype auto \
    --port "$PORT" \
    > "$VLLM_LOG" 2>&1 &

SERVER_PID=$!
echo ">>> vLLM Server PID: $SERVER_PID"

# ================= 5. Wait until the server is up =================

echo ">>> Waiting for vLLM to load model weights (see $VLLM_LOG)..."

MAX_RETRIES=60  # 60 * 5s = 5 minutes
COUNT=0

while ! nc -z localhost "$PORT"; do
    sleep 5
    COUNT=$((COUNT+1))
    echo -n "."

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        echo "[Error] vLLM process died unexpectedly! Last log lines:"
        tail -n 20 "$VLLM_LOG"
        exit 1
    fi

    if [ "$COUNT" -ge "$MAX_RETRIES" ]; then
        echo ""
        echo "[Error] Timeout waiting for vLLM port $PORT."
        exit 1
    fi
done

echo ""
echo "Port $PORT is open. Sleeping 10s to ensure the API is responsive..."
sleep 10

# ================= 6. Run the ReAct agent =================

echo ""
echo ">>> [Phase 2] Running ReAct Agent Dataset..."

export OPENAI_BASE_URL="http://localhost:$PORT/v1"
export DEFAULT_MODEL="$SERVED_MODEL_NAME"

/usr/bin/time -p python -m baselines.react.run_dataset \
    --backend "vllm" \
    --dataset "${INPUT_FILE}" \
    --out "${OUTPUT_JSON}" \
    --max-steps "${MAX_STEPS}" \
    --limit "${LIMIT}" \
    --concurrency "${CONCURRENCY}" \
    2>&1 | tee -a "$AGENT_LOG"

echo ""
echo ">>> [Complete] Results saved to: $OUTPUT_JSON"
