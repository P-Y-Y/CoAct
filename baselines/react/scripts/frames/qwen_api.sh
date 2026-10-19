#!/usr/bin/env bash
# ReAct baseline: FRAMES with Qwen3-32B via OpenRouter.
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

# Run parameters
BACKEND=${BACKEND:-"nebius"}
MAX_STEPS=${MAX_STEPS:-40}
INPUT=${INPUT:-"${DATA_DIR}/frames/test.json"}
CONCURRENCY=${CONCURRENCY:-1}
LIMIT=${LIMIT:-2}  # 0 = all

LOG_DIR=${LOG_DIR:-"${BASELINES_DIR}/react/logs/frames/qwen"}
OUT_DIR=${OUT_DIR:-"${BASELINES_DIR}/react/results/frames/qwen"}
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
