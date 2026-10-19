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
        pass

    # Fallback: parse as multiple top-level JSON values (concatenated or NDJSON)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    objs: List[Any] = []
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError as e:
            raise ValueError(f"Unsupported JSON format near position {idx}: {e}") from e
        objs.append(obj)
        idx = end

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

    if not records:
        raise ValueError("No records found after parsing JSON content")
    return records


def is_valid_record(rec: Dict[str, Any]) -> bool:
    # Only check for essential fields to identify the test case.
    # Errors or missing answers are now considered valid (but incorrect) records.
    if "question" not in rec:
        return False
    if "golden_answer" not in rec and "ground_truth" not in rec:
        return False
    return True


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
        raise RuntimeError("The 'requests' package is required.")

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
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
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return str(content)
            if resp.status_code in (429, 500, 502, 503, 504):
                last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(retry_backoff ** (attempt - 1))
                continue
            resp.raise_for_status()
        except Exception as e:
            last_error = e
            time.sleep(retry_backoff ** (attempt - 1))
    raise RuntimeError(f"OpenRouter request failed: {last_error}")


def parse_yes_no(text: str) -> Optional[bool]:
    if not text:
        return None
    s = text.strip().upper()
    if s.startswith("YES"):
        return True
    if s.startswith("NO"):
        return False
    for token in (" YES ", "\nYES\n", " NO ", "\nNO\n"):
        if token in f" {s} ":
            return token.strip() == "YES"
    return None


def _init_stats_container():
    """Create a fresh container for execution stats."""
    return {
        "subtask_count": 0,
        "total_llm_calls": 0,
        "total_turns": 0,
        "total_tokens_input": 0,
        "total_tokens_output": 0,
        "total_tool_calls_serper": 0,
        "total_tool_calls_jina": 0,
        "total_tool_usage_serper_credits": 0.0,
        "total_tool_usage_serper_time": 0.0,
        "total_tool_usage_jina_tokens": 0,
        "total_tool_usage_jina_time": 0.0,
        "total_mm_usage_visual_calls": 0,
        "total_mm_usage_visual_time": 0.0,
        "total_mm_usage_visual_input_tokens": 0,
        "total_mm_usage_visual_output_tokens": 0,
        "total_mm_usage_audio_calls": 0,
        "total_mm_usage_audio_time": 0.0,
        "total_mm_usage_audio_input_tokens": 0,
        "total_mm_usage_audio_output_tokens": 0,
        "total_mm_usage_text_calls": 0,
        "total_mm_usage_text_time": 0.0,
        "total_mm_usage_text_input_tokens": 0,
        "total_mm_usage_text_output_tokens": 0,
        "total_llm_time": 0.0,
        "manager_llm_time": 0.0,
        "worker_llm_time": 0.0,
    }

def _accumulate_stats(target_stats: Dict, record: Dict):
    """Accumulate stats from a single record into target_stats."""
    es = record.get("execution_stats")
    if es and isinstance(es, dict):
        try:
            target_stats["total_llm_calls"] += es.get("total_llm_calls", 0)
            target_stats["total_turns"] += es.get("total_turns", 0)
            target_stats["total_llm_time"] += float(es.get("total_llm_time", 0.0))
            target_stats["manager_llm_time"] += float(es.get("manager_llm_time", 0.0))
            target_stats["worker_llm_time"] += float(es.get("worker_llm_time", 0.0))

            tokens = es.get("total_tokens", {})
            target_stats["total_tokens_input"] += tokens.get("input", 0)
            target_stats["total_tokens_output"] += tokens.get("output", 0)

            tool_calls = es.get("total_tool_calls", {})
            target_stats["total_tool_calls_serper"] += tool_calls.get("serper", 0)
            target_stats["total_tool_calls_jina"] += tool_calls.get("jina", 0)

            tool_usage = es.get("total_tool_usage", {})
            target_stats["total_tool_usage_serper_credits"] += tool_usage.get("serper_credits", 0)
            target_stats["total_tool_usage_serper_time"] += float(tool_usage.get("serper_time", 0.0))
            target_stats["total_tool_usage_jina_tokens"] += tool_usage.get("jina_tokens", 0)
            target_stats["total_tool_usage_jina_time"] += float(tool_usage.get("jina_time", 0.0))

            mm = es.get("total_mm_usage", {})
            target_stats["total_mm_usage_visual_calls"] += mm.get("visual_calls", 0)
            target_stats["total_mm_usage_visual_time"] += float(mm.get("visual_time", 0.0))
            target_stats["total_mm_usage_visual_input_tokens"] += mm.get("visual_input_tokens", 0)
            target_stats["total_mm_usage_visual_output_tokens"] += mm.get("visual_output_tokens", 0)

            target_stats["total_mm_usage_audio_calls"] += mm.get("whisper_calls", 0)
            target_stats["total_mm_usage_audio_time"] += float(mm.get("whisper_time", 0.0))
            target_stats["total_mm_usage_audio_input_tokens"] += mm.get("whisper_input_tokens", 0)
            target_stats["total_mm_usage_audio_output_tokens"] += mm.get("whisper_output_tokens", 0)

            target_stats["total_mm_usage_text_calls"] += mm.get("text_calls", 0)
            target_stats["total_mm_usage_text_time"] += float(mm.get("text_time", 0.0))
            target_stats["total_mm_usage_text_input_tokens"] += mm.get("text_input_tokens", 0)
            target_stats["total_mm_usage_text_output_tokens"] += mm.get("text_output_tokens", 0)

            target_stats["subtask_count"] += es.get("subtask_count", 0)
        except (ValueError, TypeError):
            pass

def _accumulate_comm(target_comm: Dict, record: Dict):
    """Accumulate communication stats."""
    c_stats = record.get("communication_stats")
    if c_stats and isinstance(c_stats, dict):
        try:
            target_comm["downlink"] += float(c_stats.get("downlink_bytes", 0))
            target_comm["uplink"] += float(c_stats.get("uplink_bytes", 0))
            target_comm["total"] += float(c_stats.get("total_bytes", 0))
            target_comm["count"] += 1
        except (ValueError, TypeError):
            pass

def _accumulate_manager(target_manager: Dict, record: Dict):
    """Accumulate manager stats."""
    ms = record.get("manager_stats")
    if ms and isinstance(ms, dict):
        try:
            target_manager["llm_calls"] += ms.get("llm_calls", 0)
            target_manager["turns"] += ms.get("turns", 0)
            tokens = ms.get("tokens", {})
            target_manager["tokens_input"] += tokens.get("input", 0)
            target_manager["tokens_output"] += tokens.get("output", 0)
            target_manager["count"] += 1
        except: pass


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
) -> Tuple[Any]:  # simplified signature
    valid_records: List[Dict[str, Any]] = [r for r in records if is_valid_record(r)]
    if max_items is not None:
        valid_records = valid_records[:max_items]
    
    total = len(valid_records)
    
    # === Stats Containers ===
    # 1. Total (All Valid Records)
    total_time_sum = 0.0
    time_count = 0
    comm_sum = {"downlink": 0.0, "uplink": 0.0, "total": 0.0, "count": 0}
    exec_stats_sum = _init_stats_container()
    exec_stats_count = 0
    worker_stats_sum = {}
    manager_stats_sum = {"llm_calls": 0, "turns": 0, "tokens_input": 0, "tokens_output": 0, "count": 0}

    # 2. Correct (Only Correct Records)
    correct_comm_sum = {"downlink": 0.0, "uplink": 0.0, "total": 0.0, "count": 0}
    correct_exec_stats_sum = _init_stats_container()
    correct_exec_stats_count = 0
    correct_worker_stats_sum = {}  # Optional: breakdown for correct ones
    correct_manager_stats_sum = {"llm_calls": 0, "turns": 0, "tokens_input": 0, "tokens_output": 0, "count": 0}

    # Helper for LLM Judge
    def _judge_one(rec: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        q = rec.get("question")
        g = rec.get("golden_answer") or rec.get("ground_truth")
        a = rec.get("agent_result") or rec.get("answer")
        t = rec.get("total_time")

        # Check for execution failure or missing answer
        has_error = rec.get("error")
        is_empty_answer = (a is None) or (isinstance(a, str) and not a.strip())
        
        if has_error or is_empty_answer:
            return False, {
                "question": q, "golden_answer": g, "agent_result": a,
                "time": t, "gpt_result": "EXEC_FAIL"
            }

        try:
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY not set.")
            result_token = "UNKNOWN"
            for attempt in range(1, max(1, judge_retries) + 1):
                messages = build_prompt(q, g, a)
                content = call_openrouter(api_key, base_url, model, messages, 0.0, timeout)
                try:
                    print(f"[GPT RESULT] {content.strip()}")
                except: pass
                
                parsed = parse_yes_no(content)
                if parsed is True:
                    result_token = "YES"; break
                elif parsed is False:
                    result_token = "NO"; break
                else:
                    continue # retry
            
            ok = (result_token == "YES")
            detail = {
                "question": q, "golden_answer": g, "agent_result": a,
                "time": t, "gpt_result": result_token
            }
            return ok, detail
        finally:
            if per_request_sleep > 0:
                time.sleep(per_request_sleep)

    # === Evaluation Loop ===
    correct_count = 0
    details = []
    correct_time_sum = 0.0
    correct_time_count = 0

    # Choose executor or sequential
    if workers and workers > 1:
        executor = _futures.ThreadPoolExecutor(max_workers=workers)
        iterator = zip(valid_records, executor.map(_judge_one, valid_records))
    else:
        executor = None
        iterator = ((r, _judge_one(r)) for r in valid_records)

    try:
        for r, (ok, info) in iterator:
            # --- Total Stats Accumulation ---
            t_val = r.get("total_time")
            if t_val is not None:
                try:
                    total_time_sum += float(t_val)
                    time_count += 1
                except: pass
            
            _accumulate_comm(comm_sum, r)
            _accumulate_stats(exec_stats_sum, r)
            exec_stats_count += 1
            _accumulate_manager(manager_stats_sum, r)
            
            # Worker breakdown (Total)
            es = r.get("execution_stats", {})
            wb = es.get("worker_breakdown", {})
            if isinstance(wb, dict):
                for wid, wdata in wb.items():
                    if str(wid) == "manager": continue
                    w_key = str(wid)
                    if w_key not in worker_stats_sum:
                        worker_stats_sum[w_key] = {"llm_calls": 0, "turns": 0}
                    worker_stats_sum[w_key]["llm_calls"] += wdata.get("llm_calls", 0)
                    worker_stats_sum[w_key]["turns"] += wdata.get("turns", 0)

            # --- Correct Stats Accumulation ---
            details.append(info)
            if ok:
                correct_count += 1
                
                # Time
                if t_val is not None:
                    try:
                        correct_time_sum += float(t_val)
                        correct_time_count += 1
                    except: pass
                
                # Comm & Exec
                _accumulate_comm(correct_comm_sum, r)
                _accumulate_stats(correct_exec_stats_sum, r)
                correct_exec_stats_count += 1
                _accumulate_manager(correct_manager_stats_sum, r)
                
                # Worker breakdown (Correct)
                if isinstance(wb, dict):
                    for wid, wdata in wb.items():
                        if str(wid) == "manager": continue
                        w_key = str(wid)
                        if w_key not in correct_worker_stats_sum:
                            correct_worker_stats_sum[w_key] = {"llm_calls": 0, "turns": 0}
                        correct_worker_stats_sum[w_key]["llm_calls"] += wdata.get("llm_calls", 0)
                        correct_worker_stats_sum[w_key]["turns"] += wdata.get("turns", 0)

    finally:
        if executor:
            executor.shutdown()

    # === Averages Calculation Helper ===
    def calc_avg_comm(c_sum):
        cnt = c_sum["count"]
        return {
            "downlink_bytes": c_sum["downlink"] / cnt if cnt > 0 else 0.0,
            "uplink_bytes": c_sum["uplink"] / cnt if cnt > 0 else 0.0,
            "total_bytes": c_sum["total"] / cnt if cnt > 0 else 0.0
        }

    def calc_avg_exec(e_sum, count, w_sum):
        res = {}
        if count > 0:
            for k, v in e_sum.items():
                res[k] = v / count
            
            res["worker_avg_stats"] = {}
            for wid, stats in w_sum.items():
                res["worker_avg_stats"][wid] = {
                    "llm_calls": stats["llm_calls"] / count,
                    "turns": stats["turns"] / count
                }
        else:
            for k in e_sum.keys(): res[k] = 0.0
            res["worker_avg_stats"] = {}
        return res

    def calc_avg_mgr(m_sum):
        cnt = m_sum["count"]
        if cnt > 0:
            return {
                "llm_calls": m_sum["llm_calls"] / cnt,
                "turns": m_sum["turns"] / cnt,
                "tokens_input": m_sum["tokens_input"] / cnt,
                "tokens_output": m_sum["tokens_output"] / cnt
            }
        return {k: 0.0 for k in m_sum if k != "count"}

    # 1. Total Averages
    acc = correct_count / total if total else 0.0
    avg_time = total_time_sum / time_count if time_count > 0 else 0.0
    correct_avg_time = correct_time_sum / correct_time_count if correct_time_count > 0 else 0.0
    
    avg_comm_total = calc_avg_comm(comm_sum)
    avg_exec_total = calc_avg_exec(exec_stats_sum, exec_stats_count, worker_stats_sum)
    avg_mgr_total = calc_avg_mgr(manager_stats_sum)

    # 2. Correct Averages
    avg_comm_correct = calc_avg_comm(correct_comm_sum)
    avg_exec_correct = calc_avg_exec(correct_exec_stats_sum, correct_exec_stats_count, correct_worker_stats_sum)
    avg_mgr_correct = calc_avg_mgr(correct_manager_stats_sum)

    return (
        total, correct_count, acc, avg_time, correct_avg_time, details,
        # Total stats
        avg_comm_total, avg_exec_total, avg_mgr_total,
        # Correct stats
        avg_comm_correct, avg_exec_correct, avg_mgr_correct
    )


def print_stats(title, comm, exc, mgr):
    """Helper to print a block of statistics."""
    def format_bytes(size):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024: return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} TB"

    # 1. Average number of subtasks
    avg_subtasks = exc.get('subtask_count', 0)
    
    # 2. Average communication count (times)
    avg_comm_times = avg_subtasks

    print("=" * 50)
    print(f"{title}")
    print("=" * 50)
    print(f"Communication (Avg):")
    print(f"  Rounds:    {avg_comm_times:.2f}")
    print(f"  Total:    {(comm['total_bytes'])}")
    print(f"  Downlink: {(comm['downlink_bytes'])}")
    print(f"  Uplink:   {(comm['uplink_bytes'])}")
    
    print("-" * 50)
    print(f"Execution (Avg):")
    print(f"  Subtasks:        {exc.get('subtask_count', 0):.2f}")
    print(f"  LLM Calls:       {exc['total_llm_calls']:.2f}")
    print(f"  Turns:           {exc['total_turns']:.2f}")
    print(f"  Tokens (In/Out): {exc['total_tokens_input']:.2f} / {exc['total_tokens_output']:.2f}")
    print(f"  LLM Time (Total):    {exc['total_llm_time']:.2f}s (Mgr: {exc['manager_llm_time']:.2f}s, Wkr: {exc['worker_llm_time']:.2f}s)")
    
    print(f"  Serper:          {exc['total_tool_calls_serper']:.2f} calls ({exc['total_tool_usage_serper_time']:.2f}s)")
    print(f"  Jina:            {exc['total_tool_calls_jina']:.2f} calls ({exc['total_tool_usage_jina_time']:.2f}s)")
    
    # MM Tools (Optional print)
    vis = exc.get('total_mm_usage_visual_calls', 0)
    if vis > 0:
        print(f"  Visual Calls:    {vis:.2f}")

    print("-" * 50)
    print(f"Manager (Avg):")
    print(f"  LLM Calls:       {mgr['llm_calls']:.2f}")
    print(f"  Tokens (In/Out): {mgr['tokens_input']:.2f} / {mgr['tokens_output']:.2f}")
    print("\n")

    # Manually summed totals for verification clarity
    total_tokens_in = exc['total_tokens_input'] # Manager tokens already included in the source JSON logic? 
    # Actually, based on previous analysis of outline_scheduler.py, execution_stats['total_tokens'] includes manager's tokens.
    # But judge.py _accumulate_stats reads 'total_tokens_input' from JSON 'execution_stats'. 
    # Let's double check if judge.py accidentally double counts or misses manager.
    
    # In outline_scheduler.py: 
    # "total_tokens": { "input": sum(workers) + manager } -> stored in execution_stats
    
    # In judge.py:
    # _accumulate_stats reads from execution_stats["total_tokens"] -> So exc['total_tokens_input'] ALREADY HAS MANAGER.
    # _accumulate_manager reads from manager_stats -> So mgr['tokens_input'] is JUST MANAGER.
    
    # Therefore:
    # (exc['total_tokens_input']) IS THE GRAND TOTAL.
    # Adding them together (exc + mgr) would be DOUBLE COUNTING.
    
    print("-" * 50)
    print(f"GRAND TOTALS (Query Level):")
    print(f"  Total LLM Time:  {exc['total_llm_time']:.2f}s")
    print(f"  Total LLM Calls: {exc['total_llm_calls']:.2f}")
    print(f"  Total Input Tokens:  {exc['total_tokens_input']:.2f}")
    print(f"  Total Output Tokens: {exc['total_tokens_output']:.2f}")
    print(f"  Total steps:  {exc.get('total_turns', 0):.2f}")
	

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Judge agent results vs golden answers via LLM")
    parser.add_argument("--file", "-f", required=True, help="Path to JSON file")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenRouter API base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model identifier")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="OpenRouter API key")
    parser.add_argument("--max", type=int, default=None, help="Max number of valid items to evaluate")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (threads)")
    parser.add_argument("--judge-retries", type=int, default=2, help="Retries per record when judgment is UNKNOWN")
    parser.add_argument("--model_use",type=str,help="the model used for produce the data")
    args = parser.parse_args(argv)
    if not args.api_key:
        parser.error("OPENROUTER_API_KEY is not set; export it (see baselines/README.md) or pass --api-key")

    try:
        records = load_records(args.file)
    except Exception as e:
        print(f"Failed to load JSON: {e}", file=sys.stderr)
        return 2

    # Unpack updated return values
    (
        total, correct, acc, avg_time, correct_avg_time, details,
        avg_comm_total, avg_exec_total, avg_mgr_total,
        avg_comm_correct, avg_exec_correct, avg_mgr_correct
    ) = evaluate(
        records=records,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        max_items=args.max,
        per_request_sleep=args.sleep,
        timeout=args.timeout,
        workers=args.workers,
        judge_retries=args.judge_retries,
    )

    print(f"\n>>> Evaluation Summary for {os.path.basename(args.file)} <<<")
    print(f"Total Samples: {total}")
    print(f"Correct:       {correct}")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Avg Time (All): {avg_time:.4f}s")
    print(f"Avg Time (Cor): {correct_avg_time:.4f}s")
    print("\n")

    # Print Total Stats
    print_stats("STATISTICS (ALL SAMPLES)", avg_comm_total, avg_exec_total, avg_mgr_total)
    
    # Print Correct Stats
    if correct > 0:
        print_stats("STATISTICS (CORRECT SAMPLES ONLY)", avg_comm_correct, avg_exec_correct, avg_mgr_correct)
    else:
        print("No correct samples to calculate specific stats.")

		

    # Output File
    ts = extract_timestamp(args.file)
    if ts is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
    
    abs_input_path = os.path.abspath(args.file)
    try:
        parts = abs_input_path.split(os.sep)
        if "results" in parts:
            res_idx = parts.index("results")
            if res_idx + 1 < len(parts) - 1:
                subdir = parts[res_idx + 1]
            else:
                subdir = "root" 
        else:
            subdir = "others"
    except Exception:
        subdir = ""

    out_dir = os.path.abspath(os.path.join(".", "judge_results", subdir))
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception: pass
    
    out_path = os.path.join(out_dir, f"judge_result_{ts}.json")
    
    result_payload = {
        "input_file": os.path.abspath(args.file),
        "timestamp": ts,
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "average_time": avg_time,
        "correct_average_time": correct_avg_time,
        
        "stats_all": {
            "communication": avg_comm_total,
            "execution": avg_exec_total,
            "manager": avg_mgr_total
        },
        "stats_correct": {
            "communication": avg_comm_correct,
            "execution": avg_exec_correct,
            "manager": avg_mgr_correct
        },
        
        # Legacy fields for backward compatibility if needed
        "average_communication": avg_comm_total,
        "average_execution_stats": avg_exec_total,
        "average_manager_stats": avg_mgr_total,

        "model_use": args.model_use,
        "judge_model": args.model,
        "base_url": args.base_url,
        "details": details,
    }
    
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, ensure_ascii=False, indent=2)
        print(f"Detailed results saved to: {out_path}")
    except Exception as e:
        print(f"Failed to write output file: {e}", file=sys.stderr)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())