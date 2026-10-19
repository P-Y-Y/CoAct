from __future__ import annotations
import argparse
import json
import os
from typing import List, Dict, Any
from .react_agent import ReactAgent


def run_single_question(agent: ReactAgent, question: str) -> None:
    answer = agent.run(question)
    print("\n=== Answer ===\n" + answer)


def run_dataset(agent: ReactAgent, input_path: str, output_path: str) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    results: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        problem = item.get("problem")
        gt_answer = item.get("answer")
        if not problem:
            results.append({"index": i, "error": "missing problem"})
            continue
        pred = agent.run(problem)
        results.append({
            "index": i,
            "problem": problem,
            "answer_gt": gt_answer,
            "answer_pred": pred,
        })

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ReAct agent")
    parser.add_argument("--backend", type=str, default="rules", help="LLM backend: rules|openai")
    parser.add_argument("--question", type=str, help="Question to answer")
    parser.add_argument("--input", type=str, help="Path to JSON dataset with 'problem'/'answer' fields")
    parser.add_argument("--output", type=str, default="baselines/react/results/predictions.json", help="Where to write results when using --input")
    parser.add_argument("--max-steps", type=int, default=6)
    args = parser.parse_args()

    agent = ReactAgent(backend=args.backend, max_steps=args.max_steps)
    if args.input:
        run_dataset(agent, args.input, args.output)
    else:
        if not args.question:
            parser.error("--question is required when --input is not provided")
        run_single_question(agent, args.question)


if __name__ == "__main__":
    main()
