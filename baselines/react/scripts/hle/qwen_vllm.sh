#!/usr/bin/env bash
# ReAct baseline on HLE with a local vLLM server (Qwen3-32B).
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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4}"
TENSOR_PARALLEL=2
GPU_UTIL=0.90
PORT=8009
MAX_MODEL_LEN=30000

# Task
INPUT_FILE="${DATA_DIR}/hle/test-00000-of-00001.json"
MAX_STEPS=40
CONCURRENCY=5
LIMIT=50

# Multimodal tools: HLE only needs the visual tool. Qwen3-32B is text-only, so the
# visual tool calls an external vision API and needs its own key.
: "${VISION_API_KEY:?set VISION_API_KEY (OpenAI/OpenRouter key for the visual tool)}"
export ENABLE_MM_TOOLS="true"
export ENABLE_VISUAL_TOOL="true"
export ENABLE_TEXT_TOOL="false"
export ENABLE_AUDIO_TOOL="false"
export VISION_MODEL="gpt-4o"
export VISION_BASE_URL="https://openrouter.ai/api/v1"

# The local vLLM server ignores the key, but the OpenAI client needs a non-empty value.
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="http://localhost:${PORT}/v1"

# ================= 2. Log / output paths =================

LOG_DIR="${BASELINES_DIR}/react/logs/hle"
OUT_DIR="${BASELINES_DIR}/react/results/hle"
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
echo "    Output:    $OUTPUT_JSON"

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

echo ">>> Starting vLLM server with model: $MODEL_PATH on GPUs: $CUDA_VISIBLE_DEVICES"
echo "    (Logs -> $VLLM_LOG)"

nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --port "$PORT" \
    --disable-log-requests \
    > "$VLLM_LOG" 2>&1 &

SERVER_PID=$!
echo ">>> vLLM Server PID: $SERVER_PID"

echo ">>> Waiting for vLLM server to be ready at localhost:$PORT..."
while ! curl -s "http://localhost:${PORT}/v1/models" > /dev/null; do
    sleep 3
    # Exit immediately if the server process crashed
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        echo "[Error] vLLM server process died unexpectedly! Last log lines:"
        tail -n 20 "$VLLM_LOG"
        exit 1
    fi
    echo -n "."
done
echo ""
echo "vLLM Server is ready."

# ================= 5. Run the ReAct agent =================

echo ">>> Running ReAct Agent on HLE dataset..."
echo "    Dataset: $INPUT_FILE"
echo "    Output:  $OUTPUT_JSON"

# backend=openai so the agent talks to the local vLLM endpoint via OPENAI_BASE_URL
EXIT_CODE=0
python -m baselines.react.run_dataset \
    --dataset "$INPUT_FILE" \
    --backend "openai" \
    --max-steps "$MAX_STEPS" \
    --limit "$LIMIT" \
    --out "$OUTPUT_JSON" \
    --concurrency "$CONCURRENCY" \
    > "$AGENT_LOG" 2>&1 || EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "ReAct Agent finished successfully."
    echo "    See results in: $OUTPUT_JSON"
else
    echo "ReAct Agent failed with exit code $EXIT_CODE."
    echo "    Check log: $AGENT_LOG"
fi
