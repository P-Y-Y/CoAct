"""Motivation study: are dependent actions semantically closer than independent ones?

Usage: python -m coact.evaluation.plan_similarity data/hotpot/hard-level-questions-500_validation.json

Plans each question once, embeds every action objective, and prints the CDF of pairwise
cosine similarity for action pairs linked by an input/output dependency vs. unlinked pairs.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from coact.data import load_records, question_of
from coact.models import build_model, embed
from coact.utils import extract_json

PLANNING_PROMPT = """
You are given a complex task: {task}.

Decompose the task into atomic actions.

Each action MUST follow:
- atomic factual unit
- clear input / output dependency
- input is ["none"] if no dependency

Output STRICT JSON:

{{
  "completed": false,
  "final_answer": "",
  "actions": [
    {{
      "action_id": "A1",
      "objective": "...",
      "action": "...",
      "input": ["..."],
      "output": ["..."]
    }}
  ]
}}
"""


def artifacts(names: list[str]) -> set[str]:
    return {name for name in names if name != "none"}


def is_dependent(a: dict, b: dict) -> bool:
    return bool(
        artifacts(a["output"]) & artifacts(b["input"])
        or artifacts(b["output"]) & artifacts(a["input"])
    )


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def print_cdf(title: str, values: list[float]) -> None:
    xs = np.sort(values)
    cdf = np.arange(1, len(xs) + 1) / len(xs)
    print(f"\n=== {title} pairs CDF ===\ncosine_similarity\tcdf")
    for x, y in zip(xs, cdf, strict=True):
        print(f"{x:.6f}\t{y:.6f}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--backend", choices=("api", "vllm"), default="vllm")
    parser.add_argument("--model")
    args = parser.parse_args()

    model = build_model(args.backend, args.model)
    similarities: dict[bool, list[float]] = {True: [], False: []}
    for query_id, record in enumerate(load_records(args.input)[: args.limit], start=1):
        print(f"Processing query {query_id}")
        prompt = PLANNING_PROMPT.format(task=question_of(record, args.input))
        actions = extract_json(model(messages=[{"role": "user", "content": prompt}]).content).get(
            "actions", []
        )
        if len(actions) < 2:
            continue
        embeddings = np.array(embed([a["objective"] for a in actions]))
        for i, j in itertools.combinations(range(len(actions)), 2):
            dependent = is_dependent(actions[i], actions[j])
            similarities[dependent].append(cosine(embeddings[i], embeddings[j]))

    print_cdf("Dependent", similarities[True])
    print_cdf("Independent", similarities[False])
    print(f"#dependent pairs: {len(similarities[True])}")
    print(f"#independent pairs: {len(similarities[False])}")


if __name__ == "__main__":
    main()
