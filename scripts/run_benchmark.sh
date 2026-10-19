#!/usr/bin/env bash
# Run CoAct on a benchmark, then extract cost statistics and grade the answers.
#
#   scripts/run_benchmark.sh <benchmark> <dataset file> [extra `python -m coact` args...]
#   scripts/run_benchmark.sh gaia data/gaia/metadata.jsonl --backend vllm
#   scripts/run_benchmark.sh hotpot data/hotpot/hard-level-questions-500_validation.json --scheduler random
#
# Credentials are read from .env (see .env.example).
set -euo pipefail

benchmark=${1:?usage: $0 <benchmark> <dataset file> [args...]}
dataset=${2:?usage: $0 <benchmark> <dataset file> [args...]}
shift 2

run_id="${benchmark}_$(date +%Y%m%d_%H%M%S)"
log="logs/${run_id}.log"
stats="stats/${run_id}.json"

python -m coact --benchmark "$benchmark" --input "$dataset" \
    --log-file "$log" --output "results/${run_id}.json" "$@"
python -m coact.evaluation.parse_log "$log" -o "$stats"
python -m coact.evaluation.judge "$stats" -o "judge_results/${run_id}.json"
