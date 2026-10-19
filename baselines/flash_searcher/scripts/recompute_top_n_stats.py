import json
import argparse
from pathlib import Path


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return s.strip()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def recompute_meta_for_subset(original_meta: dict, subset_results: list) -> dict:
    # Copy meta to avoid mutating original
    meta = dict(original_meta)

    count = len(subset_results)

    # Aggregate durations
    total_duration_sec = sum(r.get("duration_sec", 0.0) for r in subset_results)
    avg_duration_sec = (total_duration_sec / count) if count else 0.0

    # Aggregate token usage
    llm_total_calls = 0
    llm_total_input_tokens = 0
    llm_total_output_tokens = 0
    llm_total_tokens = 0
    llm_total_call_duration_sec = 0.0
    llm_per_call_avgs = []  # for min/max/last

    jina_total_tokens = 0
    jina_total_calls = 0
    jina_total_duration_sec = 0.0

    serper_total_credits = 0
    serper_total_calls = 0
    serper_total_duration_sec = 0.0

    comm_total_input_bytes = 0
    comm_total_output_bytes = 0
    comm_total_calls = 0

    for r in subset_results:
        tu = r.get("token_usage", {})

        llm = tu.get("llm", {})
        calls = int(llm.get("total_calls", 0) or 0)
        dur = float(llm.get("total_duration_sec", 0.0) or 0.0)
        llm_total_calls += calls
        llm_total_input_tokens += int(llm.get("total_input_tokens", 0) or 0)
        llm_total_output_tokens += int(llm.get("total_output_tokens", 0) or 0)
        llm_total_tokens += int(llm.get("total_tokens", 0) or 0)
        llm_total_call_duration_sec += dur
        if calls > 0:
            llm_per_call_avgs.append(dur / calls)

        jina = tu.get("jina", {})
        jina_total_tokens += int(jina.get("total_tokens", 0) or 0)
        jina_total_calls += int(jina.get("total_calls", 0) or 0)
        jina_total_duration_sec += float(jina.get("total_duration_sec", 0.0) or 0.0)

        serper = tu.get("serper", {})
        serper_total_credits += int(serper.get("total_credits", 0) or 0)
        serper_total_calls += int(serper.get("total_calls", 0) or 0)
        serper_total_duration_sec += float(serper.get("total_duration_sec", 0.0) or 0.0)

        comm = tu.get("comm", {})
        comm_total_input_bytes += int(comm.get("total_input_bytes", 0) or 0)
        comm_total_output_bytes += int(comm.get("total_output_bytes", 0) or 0)
        comm_total_calls += int(comm.get("total_calls", 0) or 0)

    # Compute LLM average/min/max/last call duration based on per-call average across results
    llm_avg_call_duration_sec = (llm_total_call_duration_sec / llm_total_calls) if llm_total_calls else 0.0
    llm_min_call_duration_sec = min(llm_per_call_avgs) if llm_per_call_avgs else 0.0
    llm_max_call_duration_sec = max(llm_per_call_avgs) if llm_per_call_avgs else 0.0
    llm_last_call_duration_sec = llm_per_call_avgs[-1] if llm_per_call_avgs else 0.0

    # Update meta fields
    meta["total_duration_sec"] = round(total_duration_sec, 3)
    meta["avg_duration_sec"] = round(avg_duration_sec, 3)
    meta["count"] = count

    meta_llm = dict(meta.get("llm_usage", {}))
    meta_llm.update({
        "total_calls": llm_total_calls,
        "total_input_tokens": llm_total_input_tokens,
        "total_output_tokens": llm_total_output_tokens,
        "total_tokens": llm_total_tokens,
        "total_call_duration_sec": round(llm_total_call_duration_sec, 3),
        "avg_call_duration_sec": round(llm_avg_call_duration_sec, 3),
        "min_call_duration_sec": round(llm_min_call_duration_sec, 3),
        "max_call_duration_sec": round(llm_max_call_duration_sec, 3),
        "last_call_duration_sec": round(llm_last_call_duration_sec, 3),
    })
    meta["llm_usage"] = meta_llm

    meta_jina = dict(meta.get("jina_usage", {}))
    meta_jina.update({
        "total_tokens": jina_total_tokens,
        "total_calls": jina_total_calls,
        "total_duration_sec": round(jina_total_duration_sec, 3),
    })
    meta["jina_usage"] = meta_jina

    meta_serper = dict(meta.get("serper_usage", {}))
    meta_serper.update({
        "total_credits": serper_total_credits,
        "total_calls": serper_total_calls,
        "total_duration_sec": round(serper_total_duration_sec, 3),
    })
    meta["serper_usage"] = meta_serper

    meta_comm = dict(meta.get("comm_usage", {}))
    meta_comm.update({
        "total_input_bytes": comm_total_input_bytes,
        "total_output_bytes": comm_total_output_bytes,
        "total_calls": comm_total_calls,
    })
    meta["comm_usage"] = meta_comm

    return meta


def main():
    parser = argparse.ArgumentParser(description="Filter stats by top100 questions and recompute meta aggregates.")
    parser.add_argument(
        "--stats",
        required=True,
        help="Path to original stats JSON",
    )
    parser.add_argument(
        "--top100",
        required=True,
        help="Path to top100 questions JSON",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to write filtered stats JSON",
    )
    args = parser.parse_args()

    stats_path = Path(args.stats)
    top100_path = Path(args.top100)
    out_path = Path(args.out)

    stats = load_json(stats_path)
    top100 = load_json(top100_path)

    # Build lookup of problem -> result entries
    problem_to_result = {}
    for r in stats.get("results", []):
        problem = normalize_text(r.get("problem", ""))
        if problem:  # keep first occurrence
            problem_to_result.setdefault(problem, r)

    # Filter results by matching question text
    filtered_results = []
    missing = []
    for item in top100:
        q = normalize_text(item.get("question", ""))
        r = problem_to_result.get(q)
        if r:
            filtered_results.append(r)
        else:
            missing.append(q)

    # Recompute meta
    new_meta = recompute_meta_for_subset(stats.get("meta", {}), filtered_results)

    new_stats = {
        "meta": new_meta,
        "results": filtered_results,
    }

    # Save output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, new_stats)

    # Print brief report
    print(f"Matched: {len(filtered_results)} / {len(top100)}")
    if missing:
        print("Missing questions (not found in stats results):")
        for m in missing[:10]:  # show a few
            print(f" - {m}")


if __name__ == "__main__":
    main()
