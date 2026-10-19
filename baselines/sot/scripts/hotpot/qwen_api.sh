#!/usr/bin/env bash
# SoT agent baseline on hotpot via a remote OpenAI-compatible API (qwen/qwen3-32b).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASELINES_DIR="$(cd "${SOT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BASELINES_DIR}/.." && pwd)"
# API keys: export them or put them in ${REPO_ROOT}/.env (see baselines/README.md)
if [[ -f "${REPO_ROOT}/.env" ]]; then set -a; source "${REPO_ROOT}/.env"; set +a; fi
: "${SERPER_API_KEY:?set SERPER_API_KEY (see baselines/README.md)}"
: "${JINA_API_KEY:?set JINA_API_KEY (see baselines/README.md)}"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY (see baselines/README.md)}"
DATA_DIR="${DATA_DIR:-${BASELINES_DIR}/data}"
cd "${SOT_DIR}"
export PYTHONPATH="${SOT_DIR}:${PYTHONPATH:-}"

TARGET_MODEL="${TARGET_MODEL:-qwen/qwen3-32b}"
API_URL="${API_URL:-https://openrouter.ai/api/v1}"
PROVIDER_NAME="${PROVIDER_NAME:-nebius/base}"  # OpenRouter provider routing

echo "Model: ${TARGET_MODEL}"
echo "API base: ${API_URL}"

# --model openai selects the OpenAI-compatible client; --api-model is the actual model name.
python sot/main.py \
    --model openai \
    --scheduler outline \
    --data-path "${DATA_DIR}/hotpot/hard-level-questions-500_validation.json" \
    --output-folder results/hotpot/qwen3-32b_api_worker8_0209 \
    --api-type open_ai \
    --api-base "$API_URL" \
    --api-model "$TARGET_MODEL" \
    --provider-name "$PROVIDER_NAME" \
    --temperature 0.7 \
    --max-tokens 8192 \
    --top-p 0.95 \
    --frequency-penalty 0 \
    --presence-penalty 0 \
    --prompt-file prompts/new_agent.yaml \
    --timeout 1200 \
    --max-workers 8 \
    --max-turns 7 \
    --max-dataset-workers 5 \
    --max-worker-turns 40 \
    --limit 500
