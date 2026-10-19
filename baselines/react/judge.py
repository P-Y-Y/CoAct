#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
import concurrent.futures as _futures
from typing import Dict, Any, List, Optional, Tuple

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-5-mini" 
DEFAULT_API_KEY = os.environ.get("OPENROUTER_API_KEY")


def extract_timestamp(path: str) -> Optional[str]:
    import re
    name = os.path.basename(path)
    m = re.search(r"(\d{8}_\d{6})", name)
    if m:
        return m.group(1)
    return None

def load_records(path: str) -> List[Dict[str, Any]]:
    # First try: standard JSON (list or dict)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ("data", "items", "records", "results"):
                if isinstance(data.get(key), list):
                    return data[key]
            return [data]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        # Fall through to NDJSON/multi-JSON parsing
        pass

    # Fallback: parse as multiple top-level JSON values (concatenated or NDJSON)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    objs: List[Any] = []
    while idx < n:
        # Skip whitespace
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError as e:
            # Give a helpful error
            raise ValueError(f"Unsupported JSON format near position {idx}: {e}") from e
        objs.append(obj)
        idx = end

    # Flatten lists or dict containers into a single record list
    records: List[Dict[str, Any]] = []
    for obj in objs:
        if isinstance(obj, list):
            for it in obj:
                if isinstance(it, dict):
                    records.append(it)
        elif isinstance(obj, dict):
            appended = False
            for key in ("data", "items", "records", "results"):
                val = obj.get(key)
                if isinstance(val, list):
                    for it in val:
                        if isinstance(it, dict):
                            records.append(it)
                    appended = True
                    break
            if not appended:
                records.append(obj)
        # ignore other primitives

    if not records:
        raise ValueError("No records found after parsing JSON content")
    return records


def is_valid_record(rec: Dict[str, Any]) -> bool:
    # Accept records as long as they have the question and golden answer reference
    # We count missing/empty answers as valid attempts (that will fail judgment)
    has_standard = "question" in rec and "golden_answer" in rec
    has_flash = "problem" in rec and "answer" in rec
    return has_standard or has_flash


def build_prompt(question: Any, golden: Any, agent: Any) -> List[Dict[str, str]]:
    system = {
        "role": "system",
        "content": (
            "You are a lenient but principled grader. Given a question, a golden answer, and an agent's answer, "
            "decide if the agent's answer is semantically consistent with the golden answer. "
            "Treat the following as EQUIVALENT: case differences; punctuation and whitespace; leading articles (a/an/the); "
            "singular vs plural forms (e.g., goat vs goats); obvious morphological variants; numeric formatting (e.g., commas). "
            "Focus on meaning. If the golden answer has multiple key points, ensure the agent covers them. "
            "Respond with exactly one token: YES or NO."
        ),
    }
    user = {
        "role": "user",
        "content": (
            "Question:\n" + str(question) +
            "\n\nGolden Answer:\n" + str(golden) +
            "\n\nAgent Answer:\n" + str(agent) +
            "\n\nIs the Agent Answer consistent with the Golden Answer? Reply ONLY YES or NO."
        ),
    }
    return [system, user]


def call_openrouter(
    api_key: str,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    timeout: float = 60.0,
    max_retries: int = 3,
    retry_backoff: float = 2.0,
) -> str:
    if requests is None:
        raise RuntimeError(
            "The 'requests' package is required. Install via: pip install requests"
        )

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional best practices (won't break if omitted)
        # "HTTP-Referer": "http://localhost",
        # "X-Title": "Judge Script",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if not isinstance(content, str):
                    content = str(content)
                return content
            # Retry on rate limit or server error
            if resp.status_code in (429, 500, 502, 503, 504):
                last_error = RuntimeError(
                    f"HTTP {resp.status_code}: {resp.text[:200]}"
                )
                time.sleep(retry_backoff ** (attempt - 1))
                continue
            # Other errors: raise directly
            resp.raise_for_status()
        except Exception as e:  # network or parse error
            last_error = e
            time.sleep(retry_backoff ** (attempt - 1))
    # If we get here, all retries failed
    raise RuntimeError(f"OpenRouter request failed after {max_retries} attempts: {last_error}")


def parse_yes_no(text: str) -> Optional[bool]:
    if not text:
        return None
    s = text.strip().upper()
    if s.startswith("YES"):
        return True
    if s.startswith("NO"):
        return False
    # Try to find a standalone YES/NO token
    for token in (" YES ", "\nYES\n", " NO ", "\nNO\n"):
        if token in f" {s} ":
            return token.strip() == "YES"
    return None


# Offline judge removed; always use model calls


def evaluate(
    records: List[Dict[str, Any]],
    api_key: Optional[str],
    base_url: str,
    model: str,
    max_items: Optional[int] = None,
    per_request_sleep: float = 0.0,
    timeout: float = 60.0,
    workers: int = 1,
    judge_retries: int = 2,
) -> Dict[str, Any]:
    valid_records: List[Dict[str, Any]] = [r for r in records if is_valid_record(r)]
    if max_items is not None:
        valid_records = valid_records[:max_items]
    
    total = len(valid_records)
    if total == 0:
        return {
            "total": 0, "correct": 0, "accuracy": 0.0, "avg_correct_duration_sec": 0.0,"avg_total_duration_sec":0.0,
            "metrics": {}, "metrics_correct": {}, "details": []
        }

    def _judge_one(rec: Dict[str, Any]) -> Dict[str, Any]:
        # Support both field sets: standard and Flash-Searcher format
        q = rec.get("question") if rec.get("question") is not None else rec.get("problem")
        g = rec.get("golden_answer") if rec.get("golden_answer") is not None else rec.get("answer")
        a = rec.get("agent_result") if rec.get("agent_result") is not None else rec.get("pred")
        dur = rec.get("duration_sec")

        # Helper to safely get number
        def get_num(r, keys, default=0):
            for k in keys:
                if k in r:
                    val = r[k]
                    if isinstance(val, (int, float)):
                        return val
            return default

        stats = {
            "steps": get_num(rec, ["steps"]),
            
            # LLM Stats
            "llm_calls": get_num(rec, ["llm_calls"]),
            "llm_input_tokens": get_num(rec, ["llm_input_tokens", "llm_in"]),
            "llm_output_tokens": get_num(rec, ["llm_output_tokens", "llm_out"]),
            "llm_total_time_sec": get_num(rec, ["llm_total_time_sec", "llm_time"]),
            "tool_total_llm_time": get_num(rec, ["tool_total_llm_time", "tool_llm_time"]),
            "tool_total_time_sec": get_num(rec, ["tool_total_time_sec", "tool_time"]), # [Add] Tool total time
            
            # Web Search Stats
            "jina_calls": get_num(rec, ["jina_calls"]),
            "jina_tokens": get_num(rec, ["jina_tokens"]),
            "jina_time": get_num(rec, ["jina_time"]),
            "serper_calls": get_num(rec, ["serper_calls"]),
            "serper_credits": get_num(rec, ["serper_credits"]),
            "serper_time": get_num(rec, ["serper_time"]),

            # Multi-Modal Stats
            "visual_calls": get_num(rec, ["visual_calls"]),
            "visual_tokens": get_num(rec, ["visual_tokens"]),
            "visual_time": get_num(rec, ["visual_time"]),
            
            "text_tool_calls": get_num(rec, ["text_tool_calls"]),
            "text_tool_tokens": get_num(rec, ["text_tool_tokens"]),
            "text_tool_time": get_num(rec, ["text_tool_time"]),
            
            "whisper_calls": get_num(rec, ["whisper_calls"]),
            "whisper_tokens": get_num(rec, ["whisper_tokens"]),
            "whisper_time": get_num(rec, ["whisper_time"]),
        }

        # [Check for empty/missing answer] -> Auto Fail
        is_missing = False
        if a is None:
            is_missing = True
        elif isinstance(a, str) and not a.strip():
            is_missing = True
        elif isinstance(a, (list, dict)) and not a:
            is_missing = True
        
        # Treat error records as failures (if they passed validity check)
        if rec.get("error"):
            is_missing = True

        if is_missing:
            detail = {
                "question": q,
                "golden_answer": g,
                "agent_result": a,
                "gpt_result": "NO_OUTPUT",
                "duration_sec": dur,
                "stats": stats
            }
            return {"ok": False, "dur": dur, "detail": detail, "stats": stats}

        try:
            if not api_key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY not set. Export it or pass --api-key."
                )
            result_token = "UNKNOWN"
            for attempt in range(1, max(1, judge_retries) + 1):
                messages = build_prompt(q, g, a)
                content = call_openrouter(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    timeout=timeout,
                )
                # Print GPT raw result for each call
                try:
                    print(f"[GPT RESULT] {content.strip()}")
                except Exception:
                    pass
                parsed = parse_yes_no(content)
                if parsed is True:
                    result_token = "YES"
                    break
                elif parsed is False:
                    result_token = "NO"
                    break
                else:
                    result_token = "UNKNOWN"
                    # retry if allowed
                    continue
            ok = result_token == "YES"
            detail = {
                "question": q,
                "golden_answer": g,
                "agent_result": a,
                "gpt_result": result_token,
                "duration_sec": dur,
                "stats": stats
            }
            return {"ok": ok, "dur": dur, "detail": detail, "stats": stats}
        finally:
            if per_request_sleep > 0:
                time.sleep(per_request_sleep)

    results = []
    if workers and workers > 1:
        with _futures.ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_judge_one, valid_records))
    else:
        for rec in valid_records:
            results.append(_judge_one(rec))

    correct = sum(1 for r in results if r["ok"])
    durations_correct = [r["dur"] for r in results if r["ok"] and isinstance(r["dur"], (int, float))]
    
    avg_correct_time = (sum(durations_correct) / len(durations_correct)) if durations_correct else 0.0
    acc = correct / total if total else 0.0

    durations_all = [r["dur"] for r in results if isinstance(r["dur"], (int, float))]
    avg_total_time = (sum(durations_all) / len(durations_all)) if durations_all else 0.0

    # Aggregate metrics
    # Initialize all possible metrics keys for aggregation
    metrics_keys = [
        "steps", 
        "llm_calls", "llm_input_tokens", "llm_output_tokens", 
        "llm_total_time_sec", "tool_total_llm_time", "tool_total_time_sec",
        "jina_calls", "jina_tokens", "jina_time",
        "serper_calls", "serper_credits", "serper_time",
        "visual_calls", "visual_tokens", "visual_time",
        "text_tool_calls", "text_tool_tokens", "text_tool_time",
        "whisper_calls", "whisper_tokens", "whisper_time"
    ]
    
    metrics_all = {}
    metrics_correct = {} # metrics over correct samples

    # Init
    for key in metrics_keys:
        metrics_all[f"total_{key}"] = 0
        metrics_all[f"avg_{key}"] = 0.0
        metrics_correct[f"total_{key}"] = 0
        metrics_correct[f"avg_{key}"] = 0.0

    # Calc All
    for k in metrics_keys:
        total_val = sum(r["stats"].get(k, 0) for r in results)
        metrics_all[f"total_{k}"] = total_val
        metrics_all[f"avg_{k}"] = total_val / total if total else 0.0

    # Calc Correct Only
    correct_results = [r for r in results if r["ok"]]
    count_correct = len(correct_results)
    for k in metrics_keys:
        total_val = sum(r["stats"].get(k, 0) for r in correct_results)
        metrics_correct[f"total_{k}"] = total_val
        metrics_correct[f"avg_{k}"] = total_val / count_correct if count_correct else 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "avg_total_duration_sec": avg_total_time,
        "avg_correct_duration_sec": avg_correct_time,
        "metrics": metrics_all,
        "metrics_correct": metrics_correct,
        "details": [r["detail"] for r in results]
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Judge agent results vs golden answers via LLM")
    parser.add_argument("--file", "-f", required=True, help="Path to JSON file")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenRouter API base URL")
    parser.add_argument("--judge-model", default=DEFAULT_MODEL, help="Model identifier")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="OpenRouter API key")
    parser.add_argument("--max", type=int, default=None, help="Max number of valid items to evaluate")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (threads)")
    parser.add_argument("--judge-retries", type=int, default=2, help="Retries per record when judgment is UNKNOWN")
    parser.add_argument("--model-use", type=str, help="Model use for produce the results")
    args = parser.parse_args(argv)
    if not args.api_key:
        parser.error("OPENROUTER_API_KEY is not set; export it (see baselines/README.md) or pass --api-key")

    try:
        records = load_records(args.file)
    except Exception as e:
        print(f"Failed to load JSON: {e}", file=sys.stderr)
        return 2

    eval_res = evaluate(
        records=records,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.judge_model,
        max_items=args.max,
        per_request_sleep=args.sleep,
        timeout=args.timeout,
        workers=args.workers,
        judge_retries=args.judge_retries,
    )

    print("\n================ EVALUATION SUMMARY ================")
    print(f"Total Records: {eval_res['total']}")
    print(f"Correct:       {eval_res['correct']}")
    print(f"Accuracy:      {eval_res['accuracy']:.4f}")
    print(f"Avg Time (All):     {eval_res['avg_total_duration_sec']:.3f} s")
    print(f"Avg Time (Correct): {eval_res['avg_correct_duration_sec']:.3f} s")

    # Print metrics
    print("\n---------------- Detailed Metrics (Average) ----------------")
    print(f"{'Metric':<25} | {'Avg (All)':<12} | {'Avg (Correct)':<12}")
    print("-" * 55)
    
    m_all = eval_res["metrics"]
    m_cor = eval_res["metrics_correct"]
    
    def p(label, key, unit=""):
        val_all = m_all.get(f"avg_{key}", 0)
        val_cor = m_cor.get(f"avg_{key}", 0)
        
        # Only show non-zero or core metrics
        is_core = key in ["steps", "llm_calls", "llm_total_time_sec"]
        if is_core or val_all > 0.001 or val_cor > 0.001:
            # Format
            s_all = f"{val_all:.2f}{unit}"
            s_cor = f"{val_cor:.2f}{unit}"
            print(f"{label:<25} | {s_all:<12} | {s_cor:<12}")

    p("Steps", "steps")
    p("LLM Calls", "llm_calls")
    p("LLM Input Tokens", "llm_input_tokens")
    p("LLM Output Tokens", "llm_output_tokens")
    p("LLM Total Time", "llm_total_time_sec", "s")
    p("Tool LLM Time", "tool_total_llm_time", "s")
    p("Tool Total Time", "tool_total_time_sec", "s")
    
    print("-" * 55)
    p("Jina Calls", "jina_calls")
    p("Jina Tokens", "jina_tokens")
    p("Jina Time", "jina_time", "s")
    p("Serper Calls", "serper_calls")
    p("Serper Credits", "serper_credits")
    p("Serper Time", "serper_time", "s")
    
    print("-" * 55)
    p("Visual Calls", "visual_calls")
    p("Visual Tokens", "visual_tokens")
    p("Visual Time", "visual_time", "s")
    
    p("Text Tool Calls", "text_tool_calls")
    p("Text Tool Tokens", "text_tool_tokens")
    p("Text Tool Time", "text_tool_time", "s")
    
    p("Whisper Calls", "whisper_calls")
    p("Whisper Tokens", "whisper_tokens")
    p("Whisper Time", "whisper_time", "s")

    # Write results to judge_result_{timestamp}.json in the same directory as input
    ts = extract_timestamp(args.file)
    if ts is None:
        # fallback to current time if not found in filename
        ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(os.path.join(".", "judge_results"))
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass
    out_path = os.path.join(out_dir, f"judge_result_{ts}.json")
    result_payload = {
        "input_file": os.path.abspath(args.file),
        "timestamp": ts,
        "model_use": args.model_use,
        "judge_config": {
            "model": args.judge_model,
            "base_url": args.base_url
        },
        "summary": {
            "total": eval_res["total"],
            "correct": eval_res["correct"],
            "accuracy": eval_res["accuracy"],
            "avg_total_duration_sec": eval_res["avg_total_duration_sec"],
            "avg_correct_duration_sec": eval_res["avg_correct_duration_sec"],
            "metrics_all": eval_res["metrics"],       # [Modified] renamed key for clarity
            "metrics_correct": eval_res["metrics_correct"] # [New]
        },
        "details": eval_res["details"],
    }
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, ensure_ascii=False, indent=2)
        print("\n[Save] Results saved to:", out_path)
    except Exception as e:
        print(f"[Error] Failed to write results file: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())