# Baselines

Baseline agents that CoAct is compared against, plus shared evaluation helpers.
All code here is for reproducing the paper's comparisons. It is excluded from
the repo's ruff config so the third-party code stays easy to diff against its upstream.

| Baseline | Upstream | Entrypoint | Notes |
|---|---|---|---|
| [`react/`](react/) | Our own reimplementation of ReAct (Yao et al., 2023). Not a fork of any upstream repo | `python -m baselines.react.run_dataset` (run from the repo root); wrappers in `react/run*.sh` and `react/scripts/<benchmark>/*.sh` | A single-agent Thought → Action → Observation loop. It uses the same search/crawl tools as the other baselines (Serper + Jina) and optional multimodal inspectors for GAIA/HLE. |
| [`flash_searcher/`](flash_searcher/) | [OPPO-PersonalAI/Flash-Searcher](https://github.com/OPPO-PersonalAI/Flash-Searcher) (Apache-2.0) | `flash_searcher/run_flash_searcher.py` (text), `flash_searcher/run_flash_searcher_mm.py` (multimodal / GAIA); wrappers `run.sh`, `run_mm.sh` | DAG-based parallel tool-calling agent (`FlashOAgents`, derived from smolagents). We added token, tool and communication usage accounting and OpenRouter/vLLM model options. `lasj.py` is upstream's LLM-as-judge evaluator. |
| [`sot/`](sot/) | [imagination-research/sot](https://github.com/imagination-research/sot) (Skeleton-of-Thought, ICLR 2024) | `sot/sot/main.py --scheduler outline --prompt-file prompts/new_agent.yaml`; wrappers in `sot/scripts/<benchmark>/*.sh` | We adapted the SoT outline scheduler into a tool-using agent: a planner writes an outline, then parallel point workers solve the points with search tools. The upstream SoT files that are kept (FastChat/router/training code) are unchanged. Upstream `demo/`, `prompts/*.json` and `data/` are not included. Check the upstream repo for its license terms. |
| [`common/`](common/) | – | see below | Shared judge, statistics and dataset helpers. |

Each upstream-derived directory keeps its original README with a short "notes for this repo" section prepended. Keep the upstream license notices when you redistribute.

### `common/`

| File | Purpose |
|---|---|
| `judge.py` | LLM-as-judge over a results JSON/JSONL. Compares the agent answer with the gold answer through an OpenRouter-compatible API (default judge `openai/gpt-5-mini`), then writes `judge_result_<ts>.json` with accuracy and usage statistics. `react/judge.py` and `sot/judge.py` are variants of it that read each baseline's own output format. |
| `update_judge_stats.py` | Adds any missing aggregate statistics (tokens, steps, durations, tool and communication usage) to an existing `judge_result_*.json`. |
| `recompute_judge_subset.py` | Recomputes the judge aggregates on a top-N subset of questions, for example the first 200 HotpotQA questions. |
| `convert_parquet_to_json.py` | Converts Hugging Face Parquet snapshots (GAIA, HLE, …) to JSON and extracts embedded images. |
| `download_gaia.py` | Downloads the gated GAIA dataset snapshot. Requires `huggingface-cli login`. |
| `dataset/convert_and_decrypt_dataset.py` | Downloads `xbench/DeepSearch-2510` and decrypts it. |
| `dataset/extract_first_n.py` | Takes the first N items of a JSON dataset. |

## Data

Datasets are **not** in the repo. Put them under `baselines/data/<benchmark>/`, which is
gitignored (see `baselines/.gitignore`). The run scripts expect this layout by default:

```
baselines/data/
  gaia/        # GAIA JSON + attachments, e.g. gaia/2023/validation/<task files>
  hotpot/      # hard-level-questions-500_validation.json
  frames/      # FRAMES benchmark (test.json)
  hle/         # Humanity's Last Exam JSON (+ images/)
  deepsearch/  # DeepSearch-2510_decrypted.json (xbench)
  browsecomp/  # BrowseComp subsets
  multihop_rag/
```

To use another location, set `DATA_DIR` for the shell scripts and `GAIA_DATA_DIR` for the
GAIA attachment lookup in `react` and `sot`. WebWalkerQA (Alibaba-NLP/WebWalker,
[arXiv:2501.07572](https://arxiv.org/abs/2501.07572)) was also used as an external dataset. Download it from
its upstream source and place it under `data/webwalkerqa/`.

Run outputs go to `results/`, `logs/` and `outputs/` inside each baseline. All three are gitignored.

## API keys and environment variables

No credentials are committed. Each shell script loads `<repo root>/.env` if that file exists, and then
checks that the variables it needs are set, failing early with a clear message if one is missing. You can
also export the variables directly. A minimal `.env`:

```bash
OPENAI_API_KEY=...        # LLM endpoint key (OpenRouter key for the *_api.sh scripts; any value, e.g. EMPTY, for local vLLM)
OPENROUTER_API_KEY=...    # judge.py (LLM-as-judge)
SERPER_API_KEY=...        # Google search via serper.dev
JINA_API_KEY=...          # page reading via r.jina.ai
VISION_API_KEY=...        # multimodal inspector tools (GAIA/HLE runs), OpenAI-compatible
```

| Variable | Used by | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | all baselines | Key for the agent LLM (OpenAI-compatible endpoint). |
| `OPENAI_BASE_URL` / `OPENAI_API_BASE` | all baselines | Base URL of the LLM endpoint. Defaults to OpenRouter in the scripts, or `http://localhost:<port>/v1` for vLLM. |
| `OPENROUTER_API_KEY` | `common/judge.py`, `react/judge.py`, `sot/judge.py` | Judge model key. You can also pass it with `--api-key`. |
| `SERPER_API_KEY` | all baselines | Serper web search. |
| `JINA_API_KEY` | all baselines | Jina Reader page fetching. |
| `VISION_API_KEY`, `VISION_BASE_URL`, `VISION_MODEL` | `react`, `sot` | Visual inspector tool used for GAIA/HLE attachments. |
| `MTU_API_KEY`, `MTU_BASE_URL`, `MTU_MODEL` | `react`, `sot` | Text/audio inspector tool. `MTU_MODEL` also selects the local faster-whisper model. |
| `ENABLE_MM_TOOLS`, `ENABLE_VISUAL_TOOL`, `ENABLE_TEXT_TOOL`, `ENABLE_AUDIO_TOOL` | `react` | Turn the multimodal tools on or off. |
| `DEFAULT_MODEL`, `SUMMARY_MODEL`, `LOCAL_MODEL_NAME` | `flash_searcher`, `react` | Model names for the agent, the page summarizer and local vLLM. |
| `LLM_MAX_RETRIES`, `LLM_RETRY_DELAY`, `WIKI_USER_AGENT` | `react` | Retry policy and Wikipedia user agent. |
| `HF_HOME`, `HF_HUB_DISABLE_PROGRESS_BARS` | `flash_searcher` (vLLM) | Hugging Face cache settings. |
| `SLACK_USER_TOKEN` | `sot/sot/models/claude_slack_model.py` | Only for the upstream SoT Claude-over-Slack backend (see `sot/claude_setup_guide.md`). |
| `DATA_DIR`, `GAIA_DATA_DIR` | scripts, `react`, `sot` | Dataset locations (default `baselines/data`, `baselines/data/gaia`). |
| `CUDA_VISIBLE_DEVICES`, `LOG_DIR`, `OUT_DIR`, `INPUT` | scripts | Optional overrides for the vLLM and run scripts. |

## Running

Install the per-baseline requirements (`react/requirements.txt`,
`flash_searcher/requirements.txt`, `sot/pyproject.toml`), configure `.env`, put the data
under `baselines/data/`, and run a wrapper script:

```bash
bash baselines/react/scripts/hotpot/qwen_api.sh          # ReAct, Qwen3-32B via OpenRouter
bash baselines/flash_searcher/run.sh                     # Flash-Searcher, text benchmarks
bash baselines/sot/scripts/gaia/qwen_vllm_online.sh      # SoT agent on GAIA with a local vLLM server

python baselines/common/judge.py -f <results.json>       # judge a run
```

The `*_vllm*.sh` scripts start a local vLLM OpenAI-compatible server and stop it when the run finishes.
The `*_api.sh` scripts call a hosted endpoint, OpenRouter by default.
