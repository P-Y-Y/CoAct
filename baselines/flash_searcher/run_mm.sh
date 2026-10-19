#!/usr/bin/env bash
# Run Flash-Searcher with multimodal tools (GAIA / HLE) through an OpenAI-compatible API.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BASELINES_DIR}/.." && pwd)"
# API keys: export them or put them in ${REPO_ROOT}/.env (see baselines/README.md)
if [[ -f "${REPO_ROOT}/.env" ]]; then set -a; source "${REPO_ROOT}/.env"; set +a; fi
: "${SERPER_API_KEY:?set SERPER_API_KEY (see baselines/README.md)}"
: "${JINA_API_KEY:?set JINA_API_KEY (see baselines/README.md)}"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY (see baselines/README.md)}"
DATA_DIR="${DATA_DIR:-${BASELINES_DIR}/data}"
cd "${SCRIPT_DIR}"

# ---- Configuration (override via environment) ----
export DEFAULT_MODEL="${DEFAULT_MODEL:-Qwen/Qwen3-32B}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-${OPENAI_BASE_URL}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# GAIA multimodal input file (attachments are resolved relative to its directory)
INFILE="${INFILE:-${DATA_DIR}/gaia/gaia_validation.json}"
LOG_DIR="${LOG_DIR:-./logs}"
OUT_DIR="${OUT_DIR:-./outputs}"
SUMMARY_INTERVAL="${SUMMARY_INTERVAL:-8}"
CONCURRENCY="${CONCURRENCY:-5}"
MAX_STEPS="${MAX_STEPS:-40}"
LIMIT="${LIMIT:-50}"
TOOL_MAX_WORKERS="${TOOL_MAX_WORKERS:-3}"

RUN_TS=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/flash_searcher_mm_${RUN_TS}.log"
OUT_FILE="${OUT_DIR}/answer_mm_${RUN_TS}.json"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

# Line-buffered output, mirrored to the console and a timestamped log file.
export PYTHONUNBUFFERED=1
exec > >(stdbuf -oL -eL tee -a "${LOG_FILE}") 2>&1

echo "[run_mm.sh] Logging to: ${LOG_FILE}"
start_ts=$(date +%s)
echo "[run_mm.sh] Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[run_mm.sh] INFILE: ${INFILE}"
echo "[run_mm.sh] Answer outfile: ${OUT_FILE}"
echo "[run_mm.sh] DEFAULT_MODEL: ${DEFAULT_MODEL}"
echo "[run_mm.sh] Summary interval: ${SUMMARY_INTERVAL}, concurrency: ${CONCURRENCY}, max steps: ${MAX_STEPS}, limit: ${LIMIT}, tool workers: ${TOOL_MAX_WORKERS}"

# To run a local vLLM model instead of the API, add:
#   --use_vllm --vllm_model_name "${DEFAULT_MODEL}" --vllm_tensor_parallel_size 2 --vllm_max_model_len 30000
/usr/bin/time -p python run_flash_searcher_mm.py \
	--infile "${INFILE}" \
	--outfile "${OUT_FILE}" \
	--summary_interval "${SUMMARY_INTERVAL}" \
	--limit "${LIMIT}" \
	--concurrency "${CONCURRENCY}" \
	--max_steps "${MAX_STEPS}" \
	--tool_max_workers "${TOOL_MAX_WORKERS}"

end_ts=$(date +%s)
echo "[run_mm.sh] End:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "[run_mm.sh] Elapsed: $((end_ts - start_ts)) seconds"
