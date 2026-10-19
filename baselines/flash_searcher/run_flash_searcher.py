#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. PersonalAI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import argparse
import json
import logging
import time
from tqdm import tqdm
import threading
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from FlashOAgents import OpenAIServerModel, LocalQwenModel, VLLMModel
from FlashOAgents.search_tools import (
    get_total_jina_usage,
    get_total_serper_usage,
    get_total_comm_usage,
    get_query_usage,
)
from base_agent import SearchAgent
from utils import read_jsonl, write_jsonl

# Default to INFO, but can be overridden by --debug flag
_log_level = logging.INFO
logging.basicConfig(
    level=_log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv(override=True)

def _extract_question(item: dict) -> str | None:
    # Prefer 'problem', else 'Question', else 'question'
    for key in ("problem", "Question", "question", "prompt"):
        v = item.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _extract_answer(item: dict) -> str | None:
    # Prefer 'answer', else 'Final answer', else 'Final Answer', else 'golden_answer'
    for key in ("answer", "Final answer", "Final Answer", "golden_answer"):
        if key not in item:
            continue

        v = item[key]

        # Accept str / int / float
        if isinstance(v, (str, int, float)):
            s = str(v).strip()
            if s:
                return s

    return None


def _compute_delta(after: dict | None, before: dict | None, keys: list[str], float_keys: list[str] = None) -> dict:
    """Utility: compute non-negative delta for specified numeric keys."""
    after = after or {}
    before = before or {}
    float_keys = float_keys or []
    delta: dict = {}
    for k in keys:
        a = after.get(k, 0) or 0
        b = before.get(k, 0) or 0
        try:
            if k in float_keys:
                a_val = float(a)
                b_val = float(b)
            else:
                a_val = int(a)
                b_val = int(b)
        except Exception:
            a_val, b_val = 0, 0
        diff = a_val - b_val
        if diff < 0:
            diff = 0
        # Round floats to 3 decimal places
        if k in float_keys:
            diff = round(diff, 3)
        delta[k] = diff
    return delta


def process_item(
    item,
    model,
    summary_interval,
    prompts_type,
    max_steps,
    index: int,
    debug: bool = False,
    max_workers: int = None,
):
    # Pass query_id to SearchAgent; execute_tool_call sets the contextvars
    query_id = f"query_{index}"
    logger.debug(f"[process_item] Creating SearchAgent with query_id={query_id}")

    search_agent = SearchAgent(
        model, 
        summary_interval=summary_interval, 
        prompts_type=prompts_type, 
        max_steps=max_steps,
        debug=debug,
        max_workers=max_workers,
        query_id=query_id,
    )

    question = _extract_question(item)
    golden_answer = _extract_answer(item)
    if not question:
        logger.error(f"Skip index={index}: missing question in item keys {list(item.keys())}")
        return None
    if golden_answer is None:
        logger.warning(f"Index={index}: missing golden answer; proceeding without it")

    # Snapshot this thread's LLM token usage before the call
    thread_id = threading.get_ident()
    try:
        llm_before = model.get_thread_usage(thread_id) if hasattr(model, "get_thread_usage") else {}
    except Exception:
        llm_before = {}

    start_t = time.perf_counter()
    try:
        result = search_agent(question)
    except Exception as e:
        logger.error(f"Exception occurred while calling multi_agent: {str(e)}")
        return None
    duration = time.perf_counter() - start_t

    # Snapshot LLM usage after the call and compute this query's delta
    thread_id = threading.get_ident()
    try:
        llm_after = model.get_thread_usage(thread_id) if hasattr(model, "get_thread_usage") else {}
    except Exception:
        llm_after = {}

    llm_delta = _compute_delta(
        llm_after,
        llm_before,
        ["total_input_tokens", "total_output_tokens", "total_tokens", "total_calls", "total_duration_sec"],
        float_keys=["total_duration_sec"],
    )

    # Fetch this query's jina/serper usage
    query_usage = get_query_usage(query_id)

    token_usage = {
        "llm": llm_delta,
        "jina": {
            "total_tokens": query_usage.get("jina_tokens", 0),
            "total_calls": query_usage.get("jina_calls", 0),
            "total_duration_sec": round(query_usage.get("jina_duration_sec", 0.0), 3),
        },
        "serper": {
            "total_credits": query_usage.get("serper_credits", 0),
            "total_calls": query_usage.get("serper_calls", 0),
            "total_duration_sec": round(query_usage.get("serper_duration_sec", 0.0), 3),
        },
        "comm": {
            "total_input_bytes": query_usage.get("comm_input_bytes", 0),
            "total_output_bytes": query_usage.get("comm_output_bytes", 0),
            "total_calls": query_usage.get("comm_calls", 0),
        },
    }

    steps = 0
    try:
        traj = result.get("agent_trajectory", []) if isinstance(result, dict) else []
        steps = len(traj)
    except Exception:
        steps = 0

    return {
        "index": index,
        "question": question,
        "golden_answer": golden_answer,
        "duration_sec": duration,
        "steps": steps,
        "token_usage": token_usage,
        **result,
    }


def main(args):
    # Set logging level based on debug flag
    global _log_level
    if args.debug:
        _log_level = logging.DEBUG
        # Update root logger and all handlers to DEBUG level
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        for handler in root_logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled - DEBUG level logging activated")
    
    custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
    
    # Choose local model / vLLM / API backend from args
    if args.use_vllm:
        vllm_model_name = args.vllm_model_name or args.local_model_name or os.environ.get("LOCAL_MODEL_NAME", "Qwen/Qwen3-32B")
        logger.info(f"Using vLLM model: {vllm_model_name}")
        model = VLLMModel(
            model_id=vllm_model_name,
            tokenizer_name=args.vllm_tokenizer_name or vllm_model_name,
            custom_role_conversions=custom_role_conversions,
            max_completion_tokens=16384,
            enable_thinking=not args.disable_thinking,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            dtype=args.vllm_dtype,
            max_model_len=args.vllm_max_model_len,
            trust_remote_code=args.vllm_trust_remote_code,
            enforce_eager=args.vllm_enforce_eager,
        )
    elif args.use_local_model:
        local_model_name = args.local_model_name or os.environ.get("LOCAL_MODEL_NAME", "Qwen/Qwen3-32B")
        logger.info(f"Using local model: {local_model_name}")
        model = LocalQwenModel(
            model_id=local_model_name,
            custom_role_conversions=custom_role_conversions,
            max_completion_tokens=16384,
            enable_thinking=not args.disable_thinking,
            torch_dtype=args.torch_dtype if hasattr(args, 'torch_dtype') else "auto",
            device_map=args.device_map if hasattr(args, 'device_map') else "auto",
        )
    else:
        logger.info("Using API model")
        model = OpenAIServerModel(
            os.environ.get("DEFAULT_MODEL"),
            custom_role_conversions=custom_role_conversions,
            max_completion_tokens=16384,
            api_key=os.environ.get("OPENAI_API_KEY"),
            api_base=os.environ.get("OPENAI_API_BASE"),
        )

    if args.infile.lower().endswith('.json'):
        with open(args.infile, 'r') as f:
            data = json.load(f)
    else:
        data = read_jsonl(args.infile)

    # Apply limit first if provided; otherwise fallback to sample_num
    if getattr(args, "limit", None) is not None:
        try:
            n = int(args.limit)
            if n > 0:
                data = data[:n]
        except Exception:
            logger.warning("invalid --limit value, ignoring")
    elif args.sample_num is not None:
        data = data[:args.sample_num]
    
    try:
        out_data = read_jsonl(args.outfile)
    except Exception:
        out_data = []
    # Normalize questions for dedup
    done_questions = set()
    for item in out_data:
        q = _extract_question(item) if isinstance(item, dict) else None
        if q:
            done_questions.add(q)
    data_to_run = []
    for item in data:
        q = _extract_question(item) if isinstance(item, dict) else None
        if q not in done_questions:
            data_to_run.append(item)
    logger.info(f"Total data: {len(data)}, Completed: {len(done_questions)}, Remaining: {len(data_to_run)}")

    results = []
    file_lock = threading.Lock()

    def safe_write(result):
        with file_lock:
            write_jsonl(args.outfile, [result], "a")

    # Backend type, written to meta and to every result
    if args.use_vllm:
        backend = "vllm"
    elif args.use_local_model:
        backend = "local"
    else:
        backend = "openai"

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        # Ensure summary_interval never becomes 0 due to random jitter
        lower = max(1, args.summary_interval - 1)
        upper = max(lower, args.summary_interval + 1)
        summary_interval = max(1, random.randint(lower, upper))

        futures = []
        for idx, item in enumerate(data_to_run, start=1):
            futures.append(
                executor.submit(
                    process_item,
                    item,
                    model,
                    summary_interval,
                    args.prompts_type,
                    args.max_steps,
                    idx,
                    args.debug,
                    args.tool_max_workers,
                )
            )
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result:
                results.append(result)
                safe_write(result)

    logger.info(f"Processing completed. Newly added: {len(results)}, Total completed: {len(done_questions) + len(results)}")

    # Emit total token usage and timing summary if the model tracked it
    try:
        totals = model.get_total_usage()
        logger.info(
            "[tokens-llm] total_calls=%s total_input=%s total_output=%s total=%s",
            str(totals.get("total_calls")),
            str(totals.get("total_input_tokens")),
            str(totals.get("total_output_tokens")),
            str(totals.get("total_tokens")),
        )
        jina_totals = get_total_jina_usage()
        serper_totals = get_total_serper_usage()
        logger.info(
            "[jina] total_calls=%s total_tokens=%s",
            str(jina_totals.get("total_calls")),
            str(jina_totals.get("total_tokens")),
        )
        logger.info(
            "[serper] total_calls=%s total_credits=%s",
            str(serper_totals.get("total_calls")),
            str(serper_totals.get("total_credits")),
        )
        # Timing stats
        if "total_call_duration_sec" in totals:
            logger.info(
                "[timing] total_duration_sec=%.3f avg_duration_sec=%.3f min_duration_sec=%.3f max_duration_sec=%.3f",
                totals.get("total_call_duration_sec", 0),
                totals.get("avg_call_duration_sec", 0),
                totals.get("min_call_duration_sec", 0),
                totals.get("max_call_duration_sec", 0),
            )
    except Exception:
        pass

    # Write summary stats JSON (meta + results), following ReAct format
    total_duration = sum(r.get("duration_sec", 0.0) for r in results)
    count = len(results) if results else 0
    avg_duration = (total_duration / count) if count else 0.0
    
    # Get model_id
    model_id = getattr(model, "model_id", None) or os.environ.get("DEFAULT_MODEL", "unknown")
    
    # Global LLM / Jina / Serper usage
    try:
        llm_usage = model.get_total_usage()
    except Exception:
        llm_usage = {}
    jina_usage = get_total_jina_usage()
    serper_usage = get_total_serper_usage()
    comm_usage = get_total_comm_usage()
    
    meta = {
        "backend": backend,
        "model": model_id,
        "infile": args.infile,
        "max_steps": args.max_steps,
        "concurrency": args.concurrency,
        "tool_max_workers": args.tool_max_workers,
        "total_duration_sec": round(total_duration, 3),
        "avg_duration_sec": round(avg_duration, 3),
        "count": count,
        "llm_usage": llm_usage,
        "jina_usage": jina_usage,
        "serper_usage": serper_usage,
        "comm_usage": comm_usage,
    }
    summary_results = []
    for r in results:
        summary_results.append({
            "index": r.get("index"),
            "problem": r.get("question"),
            "pred": r.get("agent_result"),
            "answer": r.get("golden_answer"),
            "duration_sec": round(r.get("duration_sec", 0.0), 3),
            "steps": int(r.get("steps", 0)),
            "token_usage": r.get("token_usage"),
        })
    summary = {"meta": meta, "results": summary_results}
    base, ext = os.path.splitext(args.outfile)
    summary_path = base + "_stats.json"
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Wrote stats summary: {summary_path}")
    except Exception as e:
        logger.error(f"Failed to write stats summary: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data generation script')

    parser.add_argument('--infile', type=str, default="./data/<example.json>", help='input path')
    parser.add_argument('--outfile', type=str, default="./output/<example.jsonl>", help='output path')
    parser.add_argument('--sample_num', type=int, default=None, help='Number of samples to process')
    parser.add_argument('--limit', type=int, default=None, help='Process only the first N samples')
    parser.add_argument('--summary_interval', type=int, default=8, help='Summary interval')
    parser.add_argument('--prompts_type', type=str, default="default", help='Type of prompts to use')
    parser.add_argument('--concurrency', type=int, default=15, help='Number of concurrency')
    parser.add_argument('--max_steps', type=int, default=40, help='Maximum number of steps')
    parser.add_argument('--tool_max_workers', type=int, default=None, help='Maximum number of parallel tool executions per query (default: unlimited)')
    
    # Local vLLM inference options
    parser.add_argument('--use_vllm', action='store_true', help='Use vLLM for local inference')
    parser.add_argument('--vllm_model_name', type=str, default=None, help='vLLM model name or path (default: LOCAL_MODEL_NAME env or local_model_name)')
    parser.add_argument('--vllm_tokenizer_name', type=str, default=None, help='Optional tokenizer name/path for vLLM (default: same as model)')
    parser.add_argument('--vllm_tensor_parallel_size', type=int, default=1, help='Tensor parallel size for vLLM')
    parser.add_argument('--vllm_gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization for vLLM')
    parser.add_argument('--vllm_dtype', type=str, default="auto", help='dtype for vLLM engine (auto/float16/bfloat16)')
    parser.add_argument('--vllm_max_model_len', type=int, default=None, help='Optional max model length override for vLLM')
    parser.add_argument('--vllm_trust_remote_code', action=argparse.BooleanOptionalAction, default=True, help='Allow trust_remote_code when loading with vLLM')
    parser.add_argument('--vllm_enforce_eager', action='store_true', help='Force vLLM eager mode (debugging)')

    # Local transformers model options
    parser.add_argument('--use_local_model', action='store_true', help='Use local Qwen model instead of API')
    parser.add_argument('--local_model_name', type=str, default=None, help='Local model name or path (e.g., "Qwen/Qwen3-32B")')
    parser.add_argument('--disable_thinking', action='store_true', help='Disable thinking mode for local model (default: enabled)')
    parser.add_argument('--torch_dtype', type=str, default="auto", help='Torch dtype for local model (default: auto)')
    parser.add_argument('--device_map', type=str, default="auto", help='Device map for local model (default: auto)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with DEBUG level logging')

    args = parser.parse_args()
    
    main(args)
    