# CoAct

CoAct answers complex, multi-hop questions with a **central planner** that coordinates a
pool of **parallel tool-calling sub-agents**.

Each round:

1. **Plan** – the planner checks whether the facts gathered so far already answer the
   question. If not, it decomposes the remaining information gaps into 3–10 *atomic*
   actions, each with explicit input/output dependencies.
2. **Schedule** – action objectives are embedded. Sub-agents start on the most mutually
   dissimilar actions, so they explore in different directions. Whenever a sub-agent
   finishes, it picks the unfinished action most related to its *own* completed work,
   which lets it reuse what it already found.
3. **Execute** – every sub-agent runs a fresh ReAct-style tool-calling agent (web search,
   page crawling, and optionally Wikipedia and file/image/audio inspectors) on its action.
4. **Merge** – successful results are appended to a shared history, and the next round
   plans from it.

A per-query step budget limits cost. Once the budget is spent, the planner must answer
with the evidence it has.

## Repository layout

```
coact/
├── __main__.py        # CLI: python -m coact ...
├── config.py          # run configuration and per-benchmark presets
├── workflow.py        # central planner loop
├── scheduler.py       # semantic (CoAct) and random (ablation) action scheduling
├── subagent.py        # sub-agent wrapper around the tool-calling agent
├── prompts.py         # planner / sub-agent prompt templates
├── models.py          # LLM, tool and embedding factories
├── data.py            # dataset loading across benchmark schemas
├── evaluation/
│   ├── parse_log.py       # run log -> per-query cost statistics
│   ├── judge.py           # LLM-as-judge accuracy + cost averages
│   └── plan_similarity.py # motivation study: dependency vs. embedding similarity
└── framework/         # tool-calling agent framework (adapted from Flash-Searcher)
baselines/             # ReAct, Flash-Searcher and Skeleton-of-Thought baselines
scripts/run_benchmark.sh
tests/
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                    # core: API backend, text benchmarks
pip install -e ".[multimodal]"      # file / image / audio inspectors (GAIA, HLE)
pip install -e ".[vllm]"            # local inference with vLLM
```

## Configuration

Copy `.env.example` to `.env` and fill in your credentials. `.env` is git-ignored.

| Variable | Used for |
| --- | --- |
| `OPENAI_API_KEY`, `OPENAI_URL`, `OPENAI_MODEL` | planner and sub-agent LLM (any OpenAI-compatible endpoint) when `--backend api`. The visual inspector also uses this endpoint, with `gpt-4o`. |
| `EMBEDDING_API_KEY`, `EMBEDDING_URL`, `EMBEDDING_MODEL` | action embeddings for the semantic scheduler |
| `SERPER_API_KEY` | Google search via [Serper](https://serper.dev) |
| `JINA_API_KEY` | page reading via [Jina Reader](https://jina.ai/reader) |
| `JUDGE_API_KEY` (or `OPENROUTER_API_KEY`), `JUDGE_MODEL`, `JUDGE_BASE_URL` | answer grading (defaults: OpenRouter, `openai/gpt-5-mini`) |

## Data

Datasets are not included in the repository. Place them under `data/`, which is
git-ignored:

| Benchmark | `--benchmark` | Expected file |
| --- | --- | --- |
| HotpotQA (hard) | `hotpot` | `data/hotpot/hard-level-questions-500_validation.json` |
| FRAMES | `frames` | `data/frames/frames_benchmark_test.json` (JSONL) |
| GAIA | `gaia` | `data/gaia/metadata.jsonl` (attachments next to it) |
| HLE | `hle` | `data/hle/test-00000-of-00001.json` (images in `test-00000-of-00001_images/`) |
| DeepSearch | `deepsearch` | `data/deepsearch/DeepSearch-2510_decrypted.json` |

Any `.json` / `.jsonl` file works if its records use one of the question fields
`Question` / `prompt` / `question` / `Prompt` and one of the answer fields
`Final answer` / `answer` / `Answer`. `baselines/common/` has scripts to convert and
decrypt the raw releases.

## Running

```bash
# Full pipeline: run -> parse log -> judge
scripts/run_benchmark.sh gaia data/gaia/metadata.jsonl

# Or step by step
python -m coact --benchmark hotpot --input data/hotpot/hard-level-questions-500_validation.json \
    --log-file logs/hotpot.log --output results/hotpot.json
python -m coact.evaluation.parse_log logs/hotpot.log -o stats/hotpot.json
python -m coact.evaluation.judge stats/hotpot.json
```

Useful options (see `python -m coact --help`):

| Option | Default | Meaning |
| --- | --- | --- |
| `--backend {api,vllm}` | `api` | OpenAI-compatible API or local vLLM (`Qwen/Qwen3-32B`) |
| `--scheduler {semantic,random}` | `semantic` | CoAct scheduling or the random-assignment ablation |
| `--num-agents` | 3 | parallel sub-agents per query |
| `--max-rounds` | 10 | planning rounds per query |
| `--step-budget` | 120 | total sub-agent steps per query |
| `--max-steps`, `--tools` | preset | per-sub-task step limit and toolset (presets in `coact/config.py`) |
| `--concurrency` | 5 | queries processed in parallel |
| `--ids 3,17,42` / `--start N`, `--limit N` | all | query selection (1-based) |

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

## Acknowledgements

`coact/framework` is adapted from [Flash-Searcher](https://github.com/OPPO-PersonalAI/Flash-Searcher),
which builds on HuggingFace [smolagents](https://github.com/huggingface/smolagents). Both are
licensed under Apache-2.0.
