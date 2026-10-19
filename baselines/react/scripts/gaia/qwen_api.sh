#!/usr/bin/env bash
# ReAct baseline: GAIA subset with Qwen3-32B via OpenRouter (multimodal tools on).
# Dataset batch mode; tools (web_search / wiki_search / crawl_page, ...) are registered
# by fs_tools_adapter. All settings below can be overridden from the environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="$(cd "${BASELINES_DIR}/.." && pwd)"
# API keys: export them or put them in ${REPO_ROOT}/.env (see baselines/README.md)
if [[ -f "${REPO_ROOT}/.env" ]]; then set -a; source "${REPO_ROOT}/.env"; set +a; fi
: "${SERPER_API_KEY:?set SERPER_API_KEY (see baselines/README.md)}"
: "${JINA_API_KEY:?set JINA_API_KEY (see baselines/README.md)}"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY to your OpenRouter key (see baselines/README.md)}"
DATA_DIR="${DATA_DIR:-${BASELINES_DIR}/data}"

# Model endpoint (OpenRouter)
export DEFAULT_MODEL="qwen/qwen3-32B"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_API_BASE="https://openrouter.ai/api/v1"

# Multimodal tools (GAIA attachments). The visual tool calls a separate vision API;
# VISION_API_KEY defaults to OPENAI_API_KEY (same OpenRouter key).
export ENABLE_MM_TOOLS="true"
export ENABLE_VISUAL_TOOL="true"
export ENABLE_TEXT_TOOL="true"
export ENABLE_AUDIO_TOOL="true"
export VISION_API_KEY="${VISION_API_KEY:-${OPENAI_API_KEY}}"
export VISION_MODEL="gpt-4o"
export VISION_BASE_URL="https://openrouter.ai/api/v1"
# Audio tool uses local faster-whisper (no API key needed).
export MTU_MODEL="large-v3"

# Run parameters
BACKEND=${BACKEND:-"nebius"}
MAX_STEPS=${MAX_STEPS:-40}
INPUT=${INPUT:-"${DATA_DIR}/gaia/find-missing-file-names.json"}
CONCURRENCY=${CONCURRENCY:-1}
LIMIT=${LIMIT:-5}  # 0 = all

LOG_DIR=${LOG_DIR:-"${BASELINES_DIR}/react/logs/gaia"}
OUT_DIR=${OUT_DIR:-"${BASELINES_DIR}/react/results/gaia"}
RUN_TS=$(date '+%Y%m%d_%H%M%S')
OUTPUT=${OUTPUT:-"${OUT_DIR}/answer_${RUN_TS}.json"}
LOG_FILE="${LOG_DIR}/react_${RUN_TS}.log"

# Run from the repo root so `python -m baselines.react...` resolves.
cd "${REPO_ROOT}"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"
exec > >(stdbuf -oL -eL tee -a "${LOG_FILE}") 2>&1

echo "[react] Logging to: ${LOG_FILE}"
echo "[react] BACKEND=${BACKEND}"
echo "[react] MAX_STEPS=${MAX_STEPS}"
echo "[react] INPUT=${INPUT}"
echo "[react] OUTPUT=${OUTPUT}"
echo "[react] CONCURRENCY=${CONCURRENCY} LIMIT=${LIMIT}"

start_ts=$(date +%s)

/usr/bin/time -p python -m baselines.react.run_dataset \
	--backend "${BACKEND}" \
	--dataset "${INPUT}" \
	--out "${OUTPUT}" \
	--max-steps "${MAX_STEPS}" \
	--limit "${LIMIT}" \
	--concurrency "${CONCURRENCY}"

end_ts=$(date +%s)
echo "[react] Elapsed: $((end_ts - start_ts)) seconds"
