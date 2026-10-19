# ReAct baseline

A custom, minimal reimplementation of the ReAct (Reasoning + Acting) agent used as a
sequential tool-calling baseline. It is not a port of any upstream repository.

Each step the LLM is prompted with the question, the tool list and the history so far
(prompt templates in `toolcalling_agent.yaml`). It returns a tool-call plan (JSON), the
agent executes the tool(s), appends the observations, and repeats until the model calls
`final_answer` or `--max-steps` is reached. On the step limit the agent issues one last
"answer now" prompt and records the forced answer.

## Tools

Registered in `fs_tools_adapter.py`:

- `web_search`: Google search through the Serper API (`SERPER_API_KEY`)
- `wiki_search`: Wikipedia lookup
- `crawl_page`: fetches a URL through Jina Reader (`JINA_API_KEY`) and summarizes it with the agent's LLM

Optional multimodal tools (`mm_tools.py`, enabled with `ENABLE_MM_TOOLS=true` when a record
has an attachment, e.g. GAIA files or HLE images):

- `VisualInspectorTool`: image questions through an external vision model (`VISION_*`)
- `TextInspectorTool`: reads documents (PDF, XLSX, PPTX, DOCX, ...)
- `AudioInspectorTool`: local transcription with faster-whisper (`MTU_MODEL`)

Per-thread token, call and latency usage for the LLM, Serper, Jina and multimodal tools is
tracked and written to the result JSON.

## Files

| File | Purpose |
| --- | --- |
| `react_agent.py` | ReAct loop (`ReactAgent`) |
| `llm.py` | OpenAI-compatible client (OpenRouter or local vLLM), retries, token accounting |
| `fs_tools_adapter.py` | Tool registry and the search / crawl tools |
| `mm_tools.py` | Multimodal inspector tools |
| `memory.py` | Step memory |
| `toolcalling_agent.yaml` | System and step prompt templates |
| `run_dataset.py` | Batch runner (concurrency, JSONL backups, aggregate stats) |
| `run.py` | Single-question / simple dataset runner |
| `judge.py` | LLM-as-judge scoring of a result file (OpenRouter) |
| `run.sh`, `run_qwen.sh`, `run_vllm.sh` | Example runs |
| `scripts/<benchmark>/*.sh` | Per-benchmark runs: `*_api.sh` use OpenRouter, `*_vllm.sh` start a local vLLM server |
| `scripts/check_missing_indices.py` | List record indices missing from a JSONL result file |
| `scripts/extract_top_n_results.py` | Keep results for the first N dataset questions |
| `scripts/filter_remaining_data.py` | Build a dataset of not-yet-processed records (resume a run) |

`prompts.py`, `tools.py` and the `tools/` package from the original code were removed: they
were not imported by anything (`tools.py` was shadowed by `tools/`).

## Running

Install dependencies (`pip install -r baselines/react/requirements.txt`), put the datasets
under `baselines/data/<benchmark>/` (see `baselines/README.md`), and set the API keys
(export them or put them in `<repo>/.env`).

Run from the repository root:

```bash
python -m baselines.react.run_dataset \
    --backend nebius \
    --dataset baselines/data/hotpot/hard-level-questions-500_validation.json \
    --out baselines/react/results/hotpot/answer.json \
    --max-steps 40 --limit 100 --concurrency 5
```

`--backend` is passed to the LLM client as the OpenRouter provider name (e.g. `nebius`,
`groq`); use `vllm` for a local vLLM server (`OPENAI_BASE_URL=http://localhost:<port>/v1`).
The model is taken from `DEFAULT_MODEL`.

Or use a script, e.g. `bash baselines/react/scripts/hotpot/qwen_api.sh`. Scripts write
logs and results under `baselines/react/logs/` and `baselines/react/results/`
(gitignored). Settings such as `INPUT`, `LIMIT`, `CONCURRENCY`, `DATA_DIR` and
`CUDA_VISIBLE_DEVICES` can be overridden from the environment.

Score a result file:

```bash
python baselines/react/judge.py --file baselines/react/results/hotpot/answer_<ts>.json
```

## Environment variables

| Variable | Used by | Notes |
| --- | --- | --- |
| `OPENAI_API_KEY` | `llm.py`, `mm_tools.py` | OpenRouter key for `*_api.sh`; `EMPTY` for local vLLM |
| `OPENAI_BASE_URL` | `llm.py`, `mm_tools.py` | e.g. `https://openrouter.ai/api/v1` or `http://localhost:8009/v1` |
| `OPENAI_API_BASE` | scripts | Set alongside `OPENAI_BASE_URL` for compatibility |
| `DEFAULT_MODEL` | `llm.py` | Agent model (default `gpt-5`) |
| `SERPER_API_KEY` | `fs_tools_adapter.py` | Serper web search |
| `JINA_API_KEY` | `fs_tools_adapter.py` | Jina Reader page fetching |
| `OPENROUTER_API_KEY` | `judge.py` | Judge model key (or pass `--api-key`) |
| `ENABLE_MM_TOOLS`, `ENABLE_VISUAL_TOOL`, `ENABLE_TEXT_TOOL`, `ENABLE_AUDIO_TOOL` | `run_dataset.py`, `fs_tools_adapter.py` | Multimodal tool switches |
| `VISION_API_KEY`, `VISION_BASE_URL`, `VISION_MODEL` | `mm_tools.py` | Vision model for the visual tool |
| `MTU_MODEL` | `mm_tools.py` | faster-whisper model size (e.g. `large-v3`) |
| `GAIA_DATA_DIR` | `run_dataset.py` | GAIA attachment root (default `baselines/data/gaia`) |
| `LLM_MAX_RETRIES`, `LLM_RETRY_DELAY` | `llm.py` | Retry policy (defaults 3 and 2s) |
| `WIKI_USER_AGENT` | `fs_tools_adapter.py` | User-Agent for Wikipedia requests |
| `DATA_DIR` | scripts | Dataset root (default `baselines/data`) |
