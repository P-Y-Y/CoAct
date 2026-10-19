#!/usr/bin/env python3
"""
Filter a judge_result JSON by a top-N question list and recompute aggregates.

Usage:
  python recompute_judge_subset.py --top data/hotpot/hard-level-questions-500_validation_top200.json \
    results/judge_result_A.json results/judge_result_B.json

Outputs new files alongside inputs, prefixed with 'new_'.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Set


def normalize_text(s: Any) -> str:
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    return s.strip()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_top_questions(top_path: Path) -> Set[str]:
    data = load_json(top_path)
    questions: Set[str] = set()
    if isinstance(data, list):
        for item in data:
            q = normalize_text(item.get("question"))
            if q:
                questions.add(q)
    elif isinstance(data, dict):
        # try common keys
        for key in ("data", "items", "questions", "results"):
            lst = data.get(key)
            if isinstance(lst, list):
                for item in lst:
                    q = normalize_text(item.get("question"))
                    if q:
                        questions.add(q)
                break
    return questions


def recompute_from_details(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(details)
    correct = sum(1 for d in details if normalize_text(d.get("gpt_result")) == "YES")
    # avg duration only over correct
    correct_durations = [float(d.get("duration_sec", 0) or 0) for d in details if normalize_text(d.get("gpt_result")) == "YES" and d.get("duration_sec") is not None]
    avg_correct_duration_sec = round(sum(correct_durations) / len(correct_durations), 3) if correct_durations else 0.0
    accuracy = (correct / total) if total else 0.0

    # correct sample stats
    cs = {
        "llm_tokens": 0,
        "llm_calls": 0,
        "llm_duration_sec": 0.0,
        "jina_tokens": 0,
        "jina_calls": 0,
        "jina_duration_sec": 0.0,
        "serper_credits": 0,
        "serper_calls": 0,
        "serper_duration_sec": 0.0,
        "comm_input_bytes": 0,
        "comm_output_bytes": 0,
        "comm_calls": 0,
        "steps": 0,
    }
    correct_details = [d for d in details if normalize_text(d.get("gpt_result")) == "YES"]
    for d in correct_details:
        tu = d.get("token_usage", {})
        llm = tu.get("llm", {})
        jina = tu.get("jina", {})
        serper = tu.get("serper", {})
        comm = tu.get("comm", {})
        cs["llm_tokens"] += int(llm.get("total_tokens", 0) or 0)
        cs["llm_calls"] += int(llm.get("total_calls", 0) or 0)
        cs["llm_duration_sec"] += float(llm.get("total_duration_sec", 0.0) or 0.0)
        cs["jina_tokens"] += int(jina.get("total_tokens", 0) or 0)
        cs["jina_calls"] += int(jina.get("total_calls", 0) or 0)
        cs["jina_duration_sec"] += float(jina.get("total_duration_sec", 0.0) or 0.0)
        cs["serper_credits"] += int(serper.get("total_credits", 0) or 0)
        cs["serper_calls"] += int(serper.get("total_calls", 0) or 0)
        cs["serper_duration_sec"] += float(serper.get("total_duration_sec", 0.0) or 0.0)
        cs["comm_input_bytes"] += int(comm.get("total_input_bytes", 0) or 0)
        cs["comm_output_bytes"] += int(comm.get("total_output_bytes", 0) or 0)
        cs["comm_calls"] += int(comm.get("total_calls", 0) or 0)
        cs["steps"] += int(d.get("steps", 0) or 0)

    correct_avg_stats: Dict[str, Any] = {}
    if correct_details:
        n = len(correct_details)
        correct_avg_stats = {
            "avg_llm_tokens": round(cs["llm_tokens"] / n, 2),
            "avg_llm_calls": round(cs["llm_calls"] / n, 2),
            "avg_llm_duration_sec": round(cs["llm_duration_sec"] / n, 3),
            "avg_jina_tokens": round(cs["jina_tokens"] / n, 2),
            "avg_jina_calls": round(cs["jina_calls"] / n, 2),
            "avg_jina_duration_sec": round(cs["jina_duration_sec"] / n, 3),
            "avg_serper_credits": round(cs["serper_credits"] / n, 2),
            "avg_serper_calls": round(cs["serper_calls"] / n, 2),
            "avg_serper_duration_sec": round(cs["serper_duration_sec"] / n, 3),
            "avg_comm_input_bytes": round(cs["comm_input_bytes"] / n, 2),
            "avg_comm_output_bytes": round(cs["comm_output_bytes"] / n, 2),
            "avg_comm_calls": round(cs["comm_calls"] / n, 2),
            "avg_steps": round(cs["steps"] / n, 2),
            "total_correct_samples": n,
            "total_llm_tokens": cs["llm_tokens"],
            "total_jina_tokens": cs["jina_tokens"],
            "total_serper_credits": cs["serper_credits"],
            "total_comm_bytes": cs["comm_input_bytes"] + cs["comm_output_bytes"],
            "avg_duration_sec": avg_correct_duration_sec,
        }

    # global sample stats over all filtered samples
    def sum_field(getter) -> float:
        total_val = 0.0
        for d in details:
            total_val += float(getter(d) or 0)
        return total_val

    # duration avg over all samples with duration
    durations = [float(d.get("duration_sec", 0) or 0) for d in details if d.get("duration_sec") is not None]
    avg_duration_sec = round(sum(durations) / len(durations), 3) if durations else 0.0

    # token/calls averages per sample
    def llm_get(d, key):
        return (d.get("token_usage", {}).get("llm", {}).get(key, 0))
    def jina_get(d, key):
        return (d.get("token_usage", {}).get("jina", {}).get(key, 0))
    def serper_get(d, key):
        return (d.get("token_usage", {}).get("serper", {}).get(key, 0))
    def comm_get(d, key):
        return (d.get("token_usage", {}).get("comm", {}).get(key, 0))

    def avg_over(key_fn) -> float:
        return round(sum_field(lambda d: key_fn(d)) / total, 2) if total else 0.0

    # For durations, use seconds (float) and round to 2-3 decimals
    global_sample_stats = {
        "avg_llm_calls": avg_over(lambda d: llm_get(d, "total_calls")),
        "avg_llm_input_tokens": avg_over(lambda d: llm_get(d, "total_input_tokens")),
        "avg_llm_output_tokens": avg_over(lambda d: llm_get(d, "total_output_tokens")),
        "avg_llm_tokens": avg_over(lambda d: llm_get(d, "total_tokens")),
        "avg_llm_duration_sec": round(sum_field(lambda d: llm_get(d, "total_duration_sec")) / total, 2) if total else 0.0,
        "avg_jina_calls": avg_over(lambda d: jina_get(d, "total_calls")),
        "avg_jina_tokens": avg_over(lambda d: jina_get(d, "total_tokens")),
        "avg_jina_duration_sec": round(sum_field(lambda d: jina_get(d, "total_duration_sec")) / total, 2) if total else 0.0,
        "avg_serper_calls": avg_over(lambda d: serper_get(d, "total_calls")),
        "avg_serper_credits": avg_over(lambda d: serper_get(d, "total_credits")),
        "avg_serper_duration_sec": round(sum_field(lambda d: serper_get(d, "total_duration_sec")) / total, 2) if total else 0.0,
        "avg_comm_calls": avg_over(lambda d: comm_get(d, "total_calls")),
        "avg_comm_input_bytes": avg_over(lambda d: comm_get(d, "total_input_bytes")),
        "avg_comm_output_bytes": avg_over(lambda d: comm_get(d, "total_output_bytes")),
        "avg_comm_bytes": avg_over(lambda d: (comm_get(d, "total_input_bytes") + comm_get(d, "total_output_bytes"))),
        "avg_duration_sec": avg_duration_sec,
    }

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_correct_duration_sec": avg_correct_duration_sec,
        "correct_sample_stats": correct_avg_stats,
        "global_sample_stats": global_sample_stats,
    }


def process_one(judge_path: Path, top_questions: Set[str]) -> Path:
    data = load_json(judge_path)
    details = data.get("details", [])
    # filter by question text
    filtered = [d for d in details if normalize_text(d.get("question")) in top_questions]

    # recompute aggregates
    recomputed = recompute_from_details(filtered)

    # update payload (keep meta and config fields as-is)
    data["details"] = filtered
    for k in ("total", "correct", "accuracy", "avg_correct_duration_sec", "correct_sample_stats", "global_sample_stats"):
        data[k] = recomputed.get(k)

    # write to new_ prefixed file in same dir
    out_path = judge_path.with_name("new_" + judge_path.name)
    save_json(out_path, data)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Recompute judge_result on subset of top questions")
    parser.add_argument("judge_files", nargs="+", help="Paths to judge_result_*.json files")
    parser.add_argument("--top", required=True, help="Path to top-N questions JSON (list of {question,...})")
    args = parser.parse_args()

    top_path = Path(args.top)
    top_questions = load_top_questions(top_path)
    if not top_questions:
        print("[WARN] No top questions loaded from:", top_path)

    for jf in args.judge_files:
        out = process_one(Path(jf), top_questions)
        print("✓ Recomputed:", out)


if __name__ == "__main__":
    main()
