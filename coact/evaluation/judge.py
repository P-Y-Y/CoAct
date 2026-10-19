"""LLM-as-judge accuracy, plus average cost statistics over correctly answered queries.

Usage: python -m coact.evaluation.judge stats/gaia.json [-o judge_results/gaia.json]

Input records need ``question``, ``answer`` (reference) and ``final_answer`` (prediction);
both the run output of ``python -m coact`` and the output of ``parse_log`` qualify. Numeric
cost fields produced by ``parse_log`` are averaged over the correct records.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from coact.evaluation.parse_log import COUNTERS

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-5-mini"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Cost fields averaged over correct records.
STAT_FIELDS = ("duration_seconds", *COUNTERS)

GRADER_INSTRUCTIONS = (
    "You are a grader. Given a question, a golden answer, and an agent's answer, "
    "decide if the agent's answer is CORRECT.\n\n"
    "FIRST, check if the agent's answer is SUBSTANTIALLY CORRECT:\n"
    "- If the golden answer has MULTIPLE items (e.g., 'Harbinger, Tidal'), the agent must include ALL of them.\n"
    "- Missing any key part of the answer = NO (e.g., golden='Harbinger, Tidal', agent='Harbinger' → NO)\n"
    "- Wrong answer or different meaning = NO\n"
    "- If asking for exact quotes/text, the quoted text must match.\n\n"
    "THEN, if the answer is substantially correct, be LENIENT on FORMAT differences:\n"
    "- Case differences (e.g., 'Apple' = 'apple') → OK\n"
    "- Punctuation and whitespace → OK\n"
    "- Singular vs plural (e.g., 'goat' = 'goats') → OK\n"
    "- Word form variants (e.g., 'egalitarian' = 'Egalitarianism') → OK\n"
    "- Agent provides MORE detail but includes the golden answer → OK\n"
    "  (e.g., golden='THE CASTLE', agent='INT. THE CASTLE - DAY 1' → YES)\n"
    "- Numeric formatting (e.g., '1,000' = '1000') → OK\n\n"
    "In summary: The agent must get the CONTENT right first. "
    "Only then be lenient on formatting/wording.\n"
    "Respond with exactly one token: YES or NO."
)


def build_messages(question: str, golden: str, prediction: str) -> list[dict[str, str]]:
    user = (
        f"Question:\n{question}\n\nGolden Answer:\n{golden}\n\nAgent Answer:\n{prediction}"
        "\n\nIs the Agent Answer consistent with the Golden Answer? Reply ONLY YES or NO."
    )
    return [{"role": "system", "content": GRADER_INSTRUCTIONS}, {"role": "user", "content": user}]


class Judge:
    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 60.0) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.model = model
        self.timeout = timeout

    def complete(
        self, messages: list[dict[str, str]], retries: int = 3, backoff: float = 2.0
    ) -> str:
        payload = {"model": self.model, "messages": messages, "temperature": 0.0}
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                resp = requests.post(
                    self.url, headers=self.headers, json=payload, timeout=self.timeout
                )
                if resp.status_code == 200:
                    return str(
                        resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                    )
                if resp.status_code not in RETRYABLE_STATUS:
                    resp.raise_for_status()
                last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            except Exception as e:
                last_error = e
            time.sleep(backoff**attempt)
        raise RuntimeError(f"Judge request failed: {last_error}")

    def grade(self, record: dict) -> dict:
        try:
            reply = self.complete(
                build_messages(record["question"], record["answer"], record["final_answer"])
            )
            verdict = "YES" if reply.strip().upper().startswith("YES") else "NO"
        except RuntimeError as e:
            print(f"[task {record.get('task_id')}] {e}", file=sys.stderr)
            verdict = "ERROR"
        return {**record, "correct": verdict == "YES", "judge_result": verdict}


def is_gradable(record: dict) -> bool:
    return all(key in record for key in ("question", "answer", "final_answer")) and bool(
        str(record["final_answer"] or "").strip()
    )


def summarize(graded: list[dict]) -> dict:
    correct = [r for r in graded if r["correct"]]
    summary = {
        "total": len(graded),
        "correct": len(correct),
        "accuracy": len(correct) / len(graded) if graded else 0.0,
        "judge_errors": sum(r["judge_result"] == "ERROR" for r in graded),
    }
    if correct:
        for field in STAT_FIELDS:
            values = [r.get(field) or 0 for r in correct]
            summary[f"avg_{field}"] = sum(values) / len(correct)
    return summary


def load_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = next(
            (data[k] for k in ("records", "results", "data") if isinstance(data.get(k), list)),
            [data],
        )
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path, help="records JSON (run output or parse_log output)")
    parser.add_argument("-o", "--output", type=Path, help="default: judge_results/<input name>")
    parser.add_argument("--model", default=os.getenv("JUDGE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.getenv("JUDGE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    api_key = os.getenv("JUDGE_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Set JUDGE_API_KEY (or OPENROUTER_API_KEY) to call the judge model.")

    records = [r for r in load_records(args.input) if is_gradable(r)]
    judge = Judge(api_key, args.base_url, args.model, args.timeout)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        graded = list(pool.map(judge.grade, records))
    summary = summarize(graded)

    output = args.output or Path("judge_results") / args.input.name
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "judge_model": args.model, "records": graded}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
