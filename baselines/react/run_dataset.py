from __future__ import annotations
import argparse
import json
import os
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from .react_agent import ReactAgent


def load_records(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError("Dataset JSON must be a list of records")


def _process_record(idx: int, rec: Dict[str, Any], backend: str, max_steps: int, dataset_path: str = None) -> Dict[str, Any]:
    start = time.time()
    
    question = rec.get("problem") or rec.get("question") or rec.get("query") or rec.get("Question")
    gold = rec.get("answer") or rec.get("Answer") or rec.get("ground_truth") or rec.get("final_answer") or rec.get("Final answer")
    
    # GAIA attachment path
    file_name = rec.get("file_name")
    gaia_file_path = rec.get("file_path")
    
    # HLE image path
    hle_preview = rec.get("image_preview")
    
    # GAIA attachments live under $GAIA_DATA_DIR (default: baselines/data/gaia)
    here = os.path.dirname(os.path.abspath(__file__))  # .../baselines/react
    baselines_dir = os.path.dirname(here)              # .../baselines
    gaia_base = os.environ.get("GAIA_DATA_DIR") or os.path.join(baselines_dir, "data", "gaia")
    
    target_file = None
    if gaia_file_path:
         target_file = os.path.join(gaia_base, gaia_file_path)
    elif file_name:
         # Fallback to default validation path
         target_file = os.path.join(gaia_base, "2023/validation", file_name)
    elif hle_preview and isinstance(hle_preview, dict) and hle_preview.get("path") and dataset_path:
         # HLE dataset image handling
         # Image directory is typically "images"
         dir_name = os.path.dirname(os.path.abspath(dataset_path))
         image_dir = os.path.join(dir_name, "images")
         target_file = os.path.join(image_dir, hle_preview.get("path"))

    enable_mm = False
    if target_file:
        if os.path.exists(target_file):
            # [Modified] Check selective tool enablement from env
            use_visual = os.getenv("ENABLE_VISUAL_TOOL", "true").lower() == "true"
            use_text   = os.getenv("ENABLE_TEXT_TOOL", "true").lower() == "true"
            use_audio  = os.getenv("ENABLE_AUDIO_TOOL", "true").lower() == "true"
            
            valid_tools = []
            if use_visual: valid_tools.append("inspect_file_as_image")
            if use_text: valid_tools.append("inspect_file_as_text")
            if use_audio: valid_tools.append("inspect_file_as_audio")
            
            hints_str = ", ".join(valid_tools)

            question += f"\n\n[System Notification]: The attached file to this task. File Path: {target_file}\nPlease use the appropriate inspection tool (e.g., {hints_str}) to read the content of this file if needed to answer the question."
            enable_mm = True
        else:
            print(f"[Warn] Attached file not found: {target_file}")

    if not question:
        return {"index": idx, "error": "missing problem", "duration_sec": 0.0}
    agent = None
    try:
        agent = ReactAgent(backend=backend, max_steps=max_steps, enable_mm_tools=enable_mm)
        pred = agent.run(question)
        duration = time.time() - start
        result = {
            "index": idx,
            "problem": question,
            "pred": pred,
            "answer": gold,
            "duration_sec": round(duration, 3),
            "llm_total_time_sec": getattr(agent, "stats", {}).get("llm_total_time_sec", 0.0),
            "tool_total_llm_time": getattr(agent, "stats", {}).get("tool_total_llm_time", 0.0),
            "steps": getattr(agent, "stats", {}).get("steps", 0),
            "llm_calls": getattr(agent, "stats", {}).get("llm_calls", 0),
            "llm_input_tokens": getattr(agent, "stats", {}).get("llm_input_tokens", 0),
            "llm_output_tokens": getattr(agent, "stats", {}).get("llm_output_tokens", 0),
            "jina_calls": getattr(agent, "stats", {}).get("jina_calls", 0),
            "jina_tokens": getattr(agent, "stats", {}).get("jina_tokens", 0),
            "jina_time": getattr(agent, "stats", {}).get("jina_time", 0.0),
            "serper_calls": getattr(agent, "stats", {}).get("serper_calls", 0),
            "serper_credits": getattr(agent, "stats", {}).get("serper_credits", 0),
            "serper_time": getattr(agent, "stats", {}).get("serper_time", 0.0),
            "visual_calls": getattr(agent, "stats", {}).get("visual_calls", 0),
            "visual_tokens": getattr(agent, "stats", {}).get("visual_tokens", 0),
            "visual_time": getattr(agent, "stats", {}).get("visual_time", 0.0),
            "text_tool_calls": getattr(agent, "stats", {}).get("text_tool_calls", 0),
            "text_tool_tokens": getattr(agent, "stats", {}).get("text_tool_tokens", 0),
            "text_tool_time": getattr(agent, "stats", {}).get("text_tool_time", 0.0),
            "whisper_calls": getattr(agent, "stats", {}).get("whisper_calls", 0),
            "whisper_tokens": getattr(agent, "stats", {}).get("whisper_tokens", 0),
            "whisper_time": getattr(agent, "stats", {}).get("whisper_time", 0.0),
            # include trace for logging only; will be stripped from final answers file
            "_model_trace": getattr(agent, "trace", []),
        }
        # copy any per-tool time fields from agent.stats (keys start with 'tool_time_')
        try:
            stats = getattr(agent, "stats", {}) or {}
            for k, v in stats.items():
                if isinstance(k, str) and k.startswith("tool_time_"):
                    result[k] = v
        except Exception:
            pass
        return result
    except Exception as e:
        duration = time.time() - start
        stats = getattr(agent, "stats", {}) if agent else {}
        result = {
            "index": idx,
            "problem": question,
            "error": str(e),
            "answer": gold,
            "duration_sec": round(duration, 3),
            "llm_total_time_sec": stats.get("llm_total_time_sec", 0.0),
            "tool_total_llm_time": stats.get("tool_total_llm_time", 0.0),
            "steps": stats.get("steps", 0),
            "llm_calls": stats.get("llm_calls", 0),
            "llm_input_tokens": stats.get("llm_input_tokens", 0),
            "llm_output_tokens": stats.get("llm_output_tokens", 0),
            "jina_calls": stats.get("jina_calls", 0),
            "jina_tokens": stats.get("jina_tokens", 0),
            "jina_time": stats.get("jina_time", 0.0),
            "serper_calls": stats.get("serper_calls", 0),
            "serper_credits": stats.get("serper_credits", 0),
            "serper_time": stats.get("serper_time", 0.0),
            "visual_calls": stats.get("visual_calls", 0),
            "visual_tokens": stats.get("visual_tokens", 0),
            "visual_time": stats.get("visual_time", 0.0),
            "text_tool_calls": stats.get("text_tool_calls", 0),
            "text_tool_tokens": stats.get("text_tool_tokens", 0),
            "text_tool_time": stats.get("text_tool_time", 0.0),
            "whisper_calls": stats.get("whisper_calls", 0),
            "whisper_tokens": stats.get("whisper_tokens", 0),
            "whisper_time": stats.get("whisper_time", 0.0),
            "_model_trace": getattr(agent, "trace", []) if agent else [],
        }
        try:
            for k, v in (stats or {}).items():
                if isinstance(k, str) and k.startswith("tool_time_"):
                    result[k] = v
        except Exception:
            pass
        return result


def main():
    parser = argparse.ArgumentParser(description="Run ReAct agent on dataset JSON (problem/answer)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSON dataset (list of {problem, answer, ...})")
    parser.add_argument("--backend", type=str, default="rules", help="LLM backend: rules|openai")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--limit", type=int, default=5, help="Max records to run (0 for all)")
    parser.add_argument("--out", type=str, default="outputs/react_answers.json", help="Output JSON path")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of parallel workers (>=1)")
    args = parser.parse_args()

    # 1. Load data
    records = load_records(args.dataset)
    if args.limit and args.limit > 0:
        records = records[:args.limit]

    # 2. Prepare output paths
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Timestamped final file name
    if out_path.endswith(".json"):
        base = os.path.splitext(out_path)[0]
        final_out_path = f"{base}_{ts}.json"
    else:
        final_out_path = f"{out_path}_{ts}.json"

    # Backup file (.jsonl) and trace file (.jsonl)
    # JSONL allows cheap appends and is robust to crashes
    backup_path = final_out_path.replace(".json", "_backup.jsonl")
    trace_log_path = final_out_path.replace(".json", "_trace.jsonl")

    print(f"[Init] Final results will be: {final_out_path}")
    print(f"[Init] Real-time backup:      {backup_path} (Save this if crash happens!)")

    start_all = time.time()
    outputs: List[Dict[str, Any]] = []

    # --- Helper: append a single result to disk ---
    def save_single_result(res_item: Dict[str, Any]):
        # 1. Write result backup (without trace, to keep it small)
        try:
            clean_item = {k: v for k, v in res_item.items() if k != "_model_trace"}
            with open(backup_path, 'a', encoding='utf-8') as f_bk:
                f_bk.write(json.dumps(clean_item, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[Warn] Failed to write backup: {e}")

        # 2. Write trace log (if present)
        trace_data = res_item.get("_model_trace", [])
        if trace_data:
            try:
                entry = {
                    "index": res_item.get("index"),
                    "problem": res_item.get("problem"),
                    "model_trace": trace_data
                }
                with open(trace_log_path, 'a', encoding='utf-8') as f_tr:
                    f_tr.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:
                pass

    # 3. Process records (parallel or sequential)
    if args.concurrency and args.concurrency > 1:
        print(f"[run_dataset] Parallel processing with concurrency={args.concurrency}")
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(_process_record, i, rec, args.backend, args.max_steps, args.dataset): i for i, rec in enumerate(records, 1)}
            
            for fut in as_completed(futures):
                res = fut.result()
                idx = res.get("index")
                q = res.get("problem", "")
                print(f"\n=== Record {idx} ===\nQ: {q}\nDuration: {res.get('duration_sec')}s\n")
                
                # Keep in memory for the final summary
                outputs.append(res)
                # Persist backup immediately
                save_single_result(res)
    else:
        print("[run_dataset] Sequential processing")
        for i, rec in enumerate(records, 1):
            print(f"\n=== Record {i} ===")
            res = _process_record(i, rec, args.backend, args.max_steps, args.dataset)
            print(f"Q: {res.get('problem','')}\nDuration: {res.get('duration_sec')}s\n")
            
            outputs.append(res)
            save_single_result(res)

    # 4. All done: compute aggregate stats and build the final JSON
    total_duration = round(time.time() - start_all, 3)
    count = len(outputs)
    avg_duration = round(total_duration / max(1, count), 3)
    print(f"[run_dataset] Total elapsed: {total_duration}s | Avg per record: {avg_duration}s")

    # --- Aggregate stats ---
    def safe_sum(key):
        return sum(float(r.get(key, 0) or 0) for r in outputs)

    total_llm_calls = safe_sum("llm_calls")
    total_llm_input_tokens = safe_sum("llm_input_tokens")
    total_llm_output_tokens = safe_sum("llm_output_tokens")
    
    total_steps = safe_sum("steps")
    
    total_jina_calls = safe_sum("jina_calls")
    total_jina_tokens = safe_sum("jina_tokens")
    total_jina_time = safe_sum("jina_time")

    total_serper_calls = safe_sum("serper_calls")
    total_serper_credits = safe_sum("serper_credits")
    total_serper_time = safe_sum("serper_time")
    
    total_visual_calls = safe_sum("visual_calls")
    total_visual_tokens = safe_sum("visual_tokens")
    total_visual_time = safe_sum("visual_time")
    
    total_text_tool_calls = safe_sum("text_tool_calls")
    total_text_tool_tokens = safe_sum("text_tool_tokens")
    total_text_tool_time = safe_sum("text_tool_time")
    
    total_whisper_calls = safe_sum("whisper_calls")
    total_whisper_tokens = safe_sum("whisper_tokens")
    total_whisper_time = safe_sum("whisper_time")
    
    total_llm_time = safe_sum("llm_total_time_sec")
    total_tool_llm_time = safe_sum("tool_total_llm_time")

    avg_llm_calls = round(total_llm_calls / max(1, count), 2)
    avg_steps = round(total_steps / max(1, count), 2)
    avg_llm_time = round(total_llm_time / max(1, count), 3)
    avg_tool_llm_time = round(total_tool_llm_time / max(1, count), 3)

    # Strip traces from the final results
    sanitized_results = []
    for r in outputs:
        sr = {k: v for k, v in r.items() if k != "_model_trace"}
        sanitized_results.append(sr)

    payload = {
        "meta": {
            "dataset_path": args.dataset,
            "backend": args.backend,
            "max_steps": args.max_steps,
            "concurrency": args.concurrency,
            "total_duration_sec": total_duration,
            "avg_duration_sec": avg_duration,
            "count": count,
            
            # --- Aggregated Metrics ---
            "total_steps": total_steps,
            "avg_steps": avg_steps,
            
            "total_llm_calls": total_llm_calls,
            "avg_llm_calls": avg_llm_calls,
            "total_llm_input_tokens": total_llm_input_tokens,
            "total_llm_output_tokens": total_llm_output_tokens,
            "total_llm_time_sec": total_llm_time,
            "avg_llm_time_sec": avg_llm_time,
            "total_tool_llm_time": total_tool_llm_time,
            "avg_tool_llm_time": avg_tool_llm_time,
            
            "total_jina_calls": total_jina_calls,
            "total_jina_tokens": total_jina_tokens,
            "total_jina_time": total_jina_time,
            
            "total_serper_calls": total_serper_calls,
            "total_serper_credits": total_serper_credits,
            "total_serper_time": total_serper_time,
            
            "total_visual_calls": total_visual_calls,
            "total_visual_tokens": total_visual_tokens,
            "total_visual_time": total_visual_time,
            
            "total_text_tool_calls": total_text_tool_calls,
            "total_text_tool_tokens": total_text_tool_tokens,
            "total_text_tool_time": total_text_tool_time,
            
            "total_whisper_calls": total_whisper_calls,
            "total_whisper_tokens": total_whisper_tokens,
            "total_whisper_time": total_whisper_time,
        },
        "results": sanitized_results,
    }

    # 5. Write final results
    with open(final_out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    
    print(f"Saved FINAL aggregated results to {final_out_path}")


if __name__ == "__main__":
    main()