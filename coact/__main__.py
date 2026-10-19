"""Command-line entry point: ``python -m coact --benchmark gaia --input data/gaia/metadata.jsonl``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from coact.config import PRESETS, TOOL_NAMES, RunConfig
from coact.data import answer_of, load_records, question_of
from coact.framework.context import CURRENT_TASK_IDX, task_tag
from coact.framework.models import Model
from coact.models import build_model, build_tools
from coact.subagent import SubAgent
from coact.workflow import CentralWorkflow

logger = logging.getLogger("coact")


async def solve(idx: int, record: dict, config: RunConfig, model: Model) -> dict:
    """Run CoAct on a single query; failures are recorded rather than raised."""
    CURRENT_TASK_IDX.set(idx)  # Scoped to this asyncio task; tags every log line below.
    question = question_of(record, config.input_file)
    answer = answer_of(record)
    subagents = [
        SubAgent(
            name=f"Executor{idx}_{i}",
            query_idx=idx,
            model=model,
            tools=build_tools(config.tools, model),
            max_steps=config.max_steps,
            step_budget=config.step_budget,
        )
        for i in range(config.num_agents)
    ]
    workflow = CentralWorkflow(
        model,
        subagents,
        query_idx=idx,
        scheduler=config.scheduler,
        max_rounds=config.max_rounds,
        step_budget=config.step_budget,
        seed=config.seed,
    )

    logger.info(f"Start task {idx}: {question}")
    prediction, error = None, None
    try:
        prediction = await workflow.run(question)
    except Exception as e:
        error = str(e)
        logger.info(f"failed, skipped [{error}]")
    finally:
        workflow.close()
    logger.info(f"End task {idx} | Answer: {answer}")

    return {
        "task_id": idx,
        "question": question,
        "answer": answer,
        "final_answer": prediction,
        "error": error,
    }


async def run(config: RunConfig, records: list[dict], ids: list[int]) -> list[dict]:
    """Solve the records with the given 1-based ids (these are the task ids in the log)."""
    model = build_model(config.backend, config.model, config.tensor_parallel_size)
    logger.info(
        f"Central workflow start: input_file={config.input_file} queries={len(ids)} "
        f"backend={config.backend} scheduler={config.scheduler}"
    )

    semaphore = asyncio.Semaphore(config.concurrency)

    async def bounded(idx: int) -> dict:
        async with semaphore:
            return await solve(idx, records[idx - 1], config, model)

    tasks = [asyncio.create_task(bounded(idx)) for idx in ids]
    results = [await t for t in tqdm(asyncio.as_completed(tasks), total=len(tasks), unit="task")]
    logger.info("Central workflow finished")
    return sorted(results, key=lambda r: r["task_id"])


def setup_logging(log_file: Path) -> None:
    class TaskTagFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.task_tag = task_tag()
            return True

    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.addFilter(TaskTagFilter())
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(task_tag)s%(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="coact", description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=sorted(PRESETS))
    parser.add_argument("--input", required=True, type=Path, help=".json or .jsonl dataset")
    parser.add_argument("--output", type=Path, help="default: results/<benchmark>_<time>.json")
    parser.add_argument("--log-file", type=Path, help="default: logs/<benchmark>_<time>.log")
    parser.add_argument("--backend", choices=("api", "vllm"), default="api")
    parser.add_argument(
        "--model", help="model id (default: $OPENAI_MODEL for api, Qwen3-32B for vllm)"
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=2, help="vllm only")
    parser.add_argument("--scheduler", choices=("semantic", "random"), default="semantic")
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=10, help="planning rounds per query")
    parser.add_argument(
        "--max-steps", type=int, default=0, help="steps per sub-task (default: preset)"
    )
    parser.add_argument("--step-budget", type=int, default=120, help="sub-agent steps per query")
    parser.add_argument("--tools", help=f"comma-separated subset of {', '.join(TOOL_NAMES)}")
    parser.add_argument("--concurrency", type=int, default=5, help="queries run in parallel")
    parser.add_argument("--seed", type=int, default=0)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ids", help="comma-separated 1-based query ids to run")
    selection.add_argument("--start", type=int, default=1, help="first 1-based query id to run")
    parser.add_argument("--limit", type=int, help="run at most this many queries")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = RunConfig(
        benchmark=args.benchmark,
        input_file=args.input,
        output_file=args.output or Path("results") / f"{args.benchmark}_{stamp}.json",
        backend=args.backend,
        model=args.model,
        scheduler=args.scheduler,
        num_agents=args.num_agents,
        max_rounds=args.max_rounds,
        step_budget=args.step_budget,
        concurrency=args.concurrency,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        tools=tuple(args.tools.split(",")) if args.tools else (),
        max_steps=args.max_steps,
    )
    setup_logging(args.log_file or Path("logs") / f"{args.benchmark}_{stamp}.log")

    records = load_records(config.input_file)
    if args.ids:
        ids = [int(i) for i in args.ids.split(",")]
    else:
        ids = list(range(args.start, len(records) + 1))
    if args.limit is not None:
        ids = ids[: args.limit]

    results = asyncio.run(run(config, records, ids))

    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    config.output_file.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Results saved to {config.output_file}")


if __name__ == "__main__":
    main()
