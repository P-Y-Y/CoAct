import re
import sys
import json
import logging
import time
import concurrent.futures
import queue
import os
import copy
from typing import Dict, Any

# Silence the noisy "DEBUG:openai._base_client..." logs
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

try:
    from termcolor import colored
except ImportError:
    def colored(text, *args, **kwargs):
        return text
import threading

from .scheduler import Scheduler
from sot.utils import _print_to_streams
# Tool classes (modified for usage tracking)
from sot.tools.search_tools import WebSearchTool, WikiSearchTool, CrawlPageTool 
from sot.tools.mm_tools import VisualInspectorTool, TextInspectorTool, AudioInspectorTool
from sot.utils.prompt_manager import PromptManager

# === Model call and token-usage tracking wrapper ===
class ModelUsageTracker:
    """
    Intercepts get_response calls and tracks:
    1. Number of calls (llm_calls)
    2. Input Tokens (prompt_tokens)
    3. Output Tokens (completion_tokens)
    4. Execution Time (llm_time/tool_time)
    """
    def __init__(self, model, worker_id, stats_container, usage_key="main", log_to_main=False):
        self._model = model
        self.worker_id = worker_id
        self.stats_container = stats_container
        self.usage_key = usage_key  # "main", "visual" (gpt-4o), "whisper" (audio), "text"
        self.log_to_main = log_to_main # If True, also log to "main" stats even if usage_key represents a tool

    def _add_to_main(self, stats, calls, p_tokens, c_tokens, time_cost):
        stats["llm_calls"] += calls
        stats["tokens"]["input"] += p_tokens
        stats["tokens"]["output"] += c_tokens
        if "llm_time" not in stats: stats["llm_time"] = 0.0
        stats["llm_time"] += time_cost

    def _add_to_mm(self, stats, prefix, calls, p_tokens, c_tokens, time_cost):
        mm_stats = stats.setdefault("mm_tools", {})
        call_key = f"{prefix}_calls"
        token_key = f"{prefix}_tokens"
        time_key = f"{prefix}_time"
        
        if call_key not in mm_stats: mm_stats[call_key] = 0
        if token_key not in mm_stats: mm_stats[token_key] = {"input": 0, "output": 0}
        if time_key not in mm_stats: mm_stats[time_key] = 0.0
        
        mm_stats[call_key] += calls
        mm_stats[token_key]["input"] += p_tokens
        mm_stats[token_key]["output"] += c_tokens
        mm_stats[time_key] += time_cost

    def _log_usage(self, calls=0, p_tokens=0, c_tokens=0, time_cost=0.0):
        stats = self.stats_container[self.worker_id]
        
        # 1. Log to the specific key bucket
        if self.usage_key == "main":
            self._add_to_main(stats, calls, p_tokens, c_tokens, time_cost)
        else:
            self._add_to_mm(stats, self.usage_key, calls, p_tokens, c_tokens, time_cost)
            
            # 2. Conditional: Also log to main (Double accounting for holistic view)
            if self.log_to_main:
                self._add_to_main(stats, calls, p_tokens, c_tokens, time_cost)

    def get_response(self, *args, **kwargs):
        # 1. Increment call counter
        self._log_usage(calls=1)
        
        start_t = time.time()
        
        # 2. Get the underlying generator
        response_generator = self._model.get_response(*args, **kwargs)
        
        # 3. Wrap the generator to capture usage info
        for chunk_generator in response_generator:
            yield self._intercept_chunk_generator(chunk_generator, start_t)
            start_t = time.time() 

    def _intercept_chunk_generator(self, chunk_generator, start_time):
        """
        Iterate over the chunk generator, accumulating usage info
        and elapsed time.
        """
        for chunk in chunk_generator:
            # Record the time at which output tokens are produced
            now = time.time()
            duration = now - start_time
            self._log_usage(time_cost=duration)
            start_time = now
            
            # Check for usage info (usually in the last chunk)
            usage = None
            if isinstance(chunk, dict) and "usage" in chunk:
                usage = chunk["usage"]
            elif hasattr(chunk, "usage"):
                usage = chunk.usage

            if usage:
                # Support both attribute and dict access
                p_tokens = getattr(usage, "prompt_tokens", usage.get("prompt_tokens", 0))
                c_tokens = getattr(usage, "completion_tokens", usage.get("completion_tokens", 0))
                
                # Accumulate tokens
                self._log_usage(p_tokens=p_tokens, c_tokens=c_tokens)
                
            yield chunk

    def record_usage(self, prompt_tokens, completion_tokens, total_tokens, time_cost=0.0):
        """Allow tools such as VisualInspectorTool to report usage explicitly."""
        self._log_usage(calls=1, p_tokens=prompt_tokens, c_tokens=completion_tokens, time_cost=time_cost)

    def __getattr__(self, name):
        return getattr(self._model, name)

# === Composite model proxy, used to split AudioInspector accounting ===
class CompositeModel:
    """
    Routes get_response (QA) and record_usage (tool) statistics to different targets.
    main_model: used for get_response (counted towards the main model)
    tool_tracker: used for record_usage (counted towards tool stats)
    """
    def __init__(self, main_model, tool_tracker):
        self.main = main_model
        self.tool_tracker = tool_tracker
    
    def get_response(self, *args, **kwargs):
        return self.main.get_response(*args, **kwargs)
    
    def record_usage(self, *args, **kwargs):
        return self.tool_tracker.record_usage(*args, **kwargs)
        
    def __getattr__(self, name):
        try:
            return getattr(self.main, name)
        except AttributeError:
            return getattr(self.tool_tracker, name)
            
# =================================

    def reconfigure_mm_tools(self, allowed_tools):
        """
        Dynamically reconfigure MM tools.
        Useful when dataset requirements differ from initial config.
        """
        logging.info(f"Reconfiguring MM tools to: {allowed_tools}")
        self.allowed_mm_tools = allowed_tools
        self.enable_mm_tools = True
        
        # Clear existing
        self.visual_inspector = None
        self.text_inspector = None
        self.audio_inspector = None
        for k in ["inspect_file_as_image", "inspect_file_as_text", "inspect_file_as_audio"]:
            self.tool_metadata.pop(k, None)
            
        # Register requested
        if "visual" in self.allowed_mm_tools:
            self.visual_inspector = VisualInspectorTool(self._model, 20000)
            self.tool_metadata["inspect_file_as_image"] = {
                "name": "inspect_file_as_image",
                "description": self.visual_inspector.description,
                "inputs": "{'file_path': 'string', 'question': 'string'}",
                "output_type": "string"
            }

        if "text" in self.allowed_mm_tools:
            self.text_inspector = TextInspectorTool(self._model, 20000)
            self.tool_metadata["inspect_file_as_text"] = {
                "name": "inspect_file_as_text",
                "description": self.text_inspector.description,
                "inputs": "{'file_path': 'string', 'question': 'string'}",
                "output_type": "string"
            }
            
        if "audio" in self.allowed_mm_tools:
            self.audio_inspector = AudioInspectorTool(self._model, 20000)
            self.tool_metadata["inspect_file_as_audio"] = {
                "name": "inspect_file_as_audio",
                "description": self.audio_inspector.description,
                "inputs": "{'file_path': 'string', 'question': 'string'}",
                "output_type": "string"
            }
# =================================


# === Robust tool-call parsing ===
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\n", "", s)
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()

def _json_repair(s: str) -> str:
    # 1. Handle "arguments": { "content" } -> "arguments": "content"
    # Matches : { "string" } style where it should be a key-value pair but isn't
    s = re.sub(r'(:\s*)\{\s*("(?:[^"\\]|\\.)*")\s*\}', r'\1\2', s)
    # 2. Remove trailing commas
    s = re.sub(r",\s*(\]|\})", r"\1", s)
    return s

def _parse_tool_plan_robust(text: str) -> Dict[str, Any]:
    """
    Robustly parse the LLM output to extract 'think' and 'tools'.
    Handles JSON embedded in text, markdown code blocks, and malformed JSON.
    """
    raw = _strip_code_fences(text)
    
    # Attempt 1: Full text is valid JSON
    try:
        fixed = _json_repair(raw)
        parsed = json.loads(fixed)
        if isinstance(parsed, dict): return parsed
        if isinstance(parsed, list): return {"think": "", "tools": parsed}
    except: pass

    # Attempt 2: Regex find outermost JSON object
    try:
        m = re.search(r"\{[\s\S]*\}\s*$", raw)
        if m:
            fixed = _json_repair(m.group(0))
            parsed = json.loads(fixed)
            if isinstance(parsed, dict): return parsed
    except: pass
    
    # Attempt 3: Regex find tool list directly (sometimes models just output the list)
    try:
        m = re.search(r"\[\s*\{[\s\S]*\}\s*\]", raw)
        if m:
            fixed = _json_repair(m.group(0))
            parsed = json.loads(fixed)
            if isinstance(parsed, list): return {"think": text, "tools": parsed}
    except: pass

    # Fallback
    return {"think": text, "tools": []}

def _normalize_tool_args(tool_name: str, tool_args) -> dict:
    """Helper to normalize tool arguments from various malformed states."""
    # 1. Handle String input (e.g. "arguments": "some query")
    if isinstance(tool_args, str):
        # Try loading as JSON first
        try:
            temp = json.loads(tool_args)
            tool_args = temp
        except: pass
    
    # Arg Mapping Configuration
    target_arg_map = {
        "web_search": ["query"],
        "wiki_search": ["query"],
        "crawl_page": ["url", "query"],
        "inspect_file_as_image": ["file_path", "question"],
        "inspect_file_as_text": ["file_path", "question"],
        "inspect_file_as_audio": ["file_path", "question"],
        "final_answer": ["answer"]
    }
    
    # 2. If still string, map to first expected argument
    if isinstance(tool_args, str):
        if tool_name in target_arg_map:
            expected_keys = target_arg_map[tool_name]
            if expected_keys:
                return {expected_keys[0]: tool_args}
        return {} # Should be dict

    # 3. Handle List input
    elif isinstance(tool_args, list):
        if tool_name in target_arg_map:
            expected_keys = target_arg_map[tool_name]
            new_args = {}
            for i, val in enumerate(tool_args):
                if i < len(expected_keys):
                    new_args[expected_keys[i]] = val
            return new_args
        return {}

    # 4. Handle Dict input - fuzzy matching for keys if missing
    if isinstance(tool_args, dict):
        if tool_name in target_arg_map:
            expected_keys = target_arg_map[tool_name]
            primary_key = expected_keys[0]
            
            # Check if primary key exists
            if primary_key not in tool_args:
                # Heuristics: Look for 'answer', 'content', 'value', 'input'
                candidates = [v for k,v in tool_args.items() if isinstance(v, str) and len(v) > 0]
                
                fallback_keys = ["answer", "content", "value", "input", "topic", "keyword", "url", "query", "result", "response", "output"]
                found = False
                for fk in fallback_keys:
                    if fk in tool_args:
                         # Use raw value if found (could be list/dict too? usually str)
                         tool_args[primary_key] = tool_args[fk]
                         found = True
                         break
                
                if not found and candidates:
                    # Just take the first string value found? Risky but better than nothing?
                    tool_args[primary_key] = candidates[0]
        return tool_args

    return {}

# =========================================


class OutlineScheduler(Scheduler):
    PROMPT_ROLE_SWITCH_STR = "[ROLESWITCHING assistant:]"

    def __init__(
        self, 
        prompt_file=None, 
        config_file=None,  
        web_search=None,
        wiki_search=None,
        crawl_page=None,
        max_workers=3, 
        max_turns=40,
        enable_mm_tools=False,
        max_worker_turns=None,
        allowed_mm_tools=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_workers = max_workers
        self.max_turns = max_turns
        self.enable_mm_tools = enable_mm_tools
        self.max_worker_turns = max_worker_turns
        self.allowed_mm_tools = allowed_mm_tools or ["visual", "text", "audio"]
        self.init_lock = threading.Lock()
        
        logging.info(f"OutlineScheduler initialized with max_turns={self.max_turns}")

        if not prompt_file or not os.path.exists(prompt_file):
            raise ValueError(f"Prompt file not found: {prompt_file}")
        
        self.pm = PromptManager(prompt_file)

        # Placeholder tools
        self.web_search = web_search if web_search else WebSearchTool()
        self.wiki_search = wiki_search if wiki_search else WikiSearchTool()
        self.crawl_page = crawl_page if crawl_page else CrawlPageTool(model=self._model)

        self.tool_metadata = {
            "web_search": {
                "name": "web_search",
                "description": "Search the internet for factual information.",
                "inputs": "{'query': 'string'}",
                "output_type": "string"
            },
            "wiki_search": {
                "name": "wiki_search",
                "description": "Search Wikipedia for detailed encyclopedia entries.",
                "inputs": "{'query': 'string'}",
                "output_type": "string"
            },
            "crawl_page": {
                "name": "crawl_page",
                "description": "Crawl a specific URL to get its content.",
                "inputs": "{'url': 'string', 'query': 'string'}",
                "output_type": "string"
            }
        }

        if self.enable_mm_tools:
            # Register MM tools according to allowed_mm_tools
            
            if "visual" in self.allowed_mm_tools:
                self.visual_inspector = VisualInspectorTool(self._model, 20000)
                self.tool_metadata.update({
                    "inspect_file_as_image": {
                        "name": "inspect_file_as_image",
                        "description": self.visual_inspector.description,
                        "inputs": "{'file_path': 'string', 'question': 'string'}",
                        "output_type": "string"
                    }
                })

            if "text" in self.allowed_mm_tools:
                self.text_inspector = TextInspectorTool(self._model, 20000)
                self.tool_metadata.update({
                    "inspect_file_as_text": {
                        "name": "inspect_file_as_text",
                        "description": self.text_inspector.description,
                        "inputs": "{'file_path': 'string', 'question': 'string'}",
                        "output_type": "string"
                    }
                })
            
            if "audio" in self.allowed_mm_tools:
                self.audio_inspector = AudioInspectorTool(self._model, 20000)
                self.tool_metadata.update({
                    "inspect_file_as_audio": {
                        "name": "inspect_file_as_audio",
                        "description": self.audio_inspector.description,
                        "inputs": "{'file_path': 'string', 'question': 'string'}",
                        "output_type": "string"
                    }
                })

        self.worker_models = []

    @staticmethod
    def command_line_parser():
        parser = super(OutlineScheduler, OutlineScheduler).command_line_parser()
        parser.add_argument("--prompt-file", type=str, default=None)
        parser.add_argument("--config-file", type=str, default=None) 
        parser.add_argument("--max-workers", type=int, default=3)
        parser.add_argument("--max-turns", type=int, default=8)
        parser.add_argument("--enable-mm-tools", action="store_true")
        parser.add_argument("--max-worker-turns", type=int, default=None)
        return parser

    def stream_output(self, output_generator, streams):
        if streams is None: streams = [sys.stderr]
        pre = 0
        output_text = ""
        cur_stage = "outline"
        cur_point = 0
        for outputs in output_generator:
            if outputs["stage"] == "summarize":
                _print_to_streams(streams, " ".join(output_text[pre:]), flush=True)
                _print_to_streams(streams, "\n\n", flush=True)
                return outputs
            if not outputs["stage"] == cur_stage:
                _print_to_streams(streams, " ".join(output_text[pre:]), flush=True)
                _print_to_streams(streams, "\n\n", flush=True)
                cur_stage = outputs["stage"]
                pre = 0
                cur_point = 0
            if outputs["stage"] == "expand" and outputs["point_index"] != cur_point:
                _print_to_streams(streams, " ".join(output_text[pre:]), flush=True)
                _print_to_streams(streams, "\n\n", flush=True)
                pre = 0
                cur_point = outputs["point_index"]
            
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                _print_to_streams(streams, " ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now

    def _get_response_stream(self, request, history=""):
        # Stage 0: Init Model Pool
        if not self.worker_models:
            with self.init_lock:
                if not self.worker_models:
                    logging.info(colored(f"Initializing {self.max_workers} Worker Models...", "cyan"))
                    is_vllm = self._model.__class__.__name__ == "VLLMModel"
                    for i in range(self.max_workers):
                        if is_vllm:
                            self.worker_models.append(self._model)
                        else:
                            new_model = copy.deepcopy(self._model)
                            if hasattr(new_model, "_init_client"): new_model._init_client()
                            self.worker_models.append(new_model)

        # === Per-worker statistics container ===
        # Layout: { worker_id: { llm_calls, llm_time, turns, tokens: {input, output}, tools: {serper_calls, serper_credits, serper_time, jina_calls, jina_tokens, jina_time} } }
        current_query_stats = {
            i: {
                "llm_calls": 0, 
                "llm_time": 0.0,
                "turns": 0,
                "tokens": {"input": 0, "output": 0},
                "tools": {
                    "serper_calls": 0, "serper_credits": 0, "serper_time": 0.0,
                    "jina_calls": 0, "jina_tokens": 0, "jina_time": 0.0
                }
            } for i in range(self.max_workers)
        }
        
        # Statistics for the main model (manager)
        current_query_stats["manager"] = {
            "llm_calls": 0,
            "llm_time": 0.0,
            "turns": 0,
            "tokens": {"input": 0, "output": 0},
            "tools": {
                "serper_calls": 0, "serper_credits": 0, "serper_time": 0.0,
                "jina_calls": 0, "jina_tokens": 0, "jina_time": 0.0
            },
            "mm_tools": {}  # manager may also have MM tool stats (rarely used)
        }
        
        wrapped_main_model = ModelUsageTracker(self._model, "manager", current_query_stats, usage_key="main")
        # ====================================

        if isinstance(request, list): original_question = request[-1]
        else: original_question = request

        # Stage 1: Generate Outline
        logging.info(colored(f"=== Process Started. Query: {original_question} ===", "green", attrs=["bold"]))
        logging.info(colored("Stage 1: Generating Outline...", "magenta", attrs=["bold"]))

        global_start_ts = time.time()
        outline_messages = self.pm.get_messages('planning', tools=self.tool_metadata, task=original_question)
        
        outline = ""
        # Use wrapped_main_model for outline generation so tokens and calls are counted
        response_iterator = wrapped_main_model.get_response([outline_messages], stream=True)
        for response_generator in response_iterator:
            for chunk_data in response_generator:
                delta = chunk_data.get("text", "")
                outline += delta 
                chunk_data["text"] = outline 
                chunk_data["stage"] = "outline"
                yield chunk_data
        
        logging.info(colored(f"Stage 1 Complete. Outline generated ({len(outline)} chars).", "magenta"))

        # [CLEANUP] Remove <think> tags from outline if present
        outline = re.sub(r"<think>.*?</think>", "", outline, flags=re.DOTALL | re.IGNORECASE).strip()

        if not outline:
            logging.warning(colored("Warning: Outline is empty after removing <think> tags. Check model output.", "red"))

        # Stage 2: Extract Points
        re_result = re.findall(r"(?m)^\s*(?:##\s*Goal\s*)?(\d+)[.:]\s*(.*?)(?=\n(?:##|\d+[.:])|$)", outline, re.DOTALL)
        if len(re_result) > 0:
            points, point_outlines = zip(*re_result)
        else:
            points, point_outlines = [], []
            logging.warning(colored(f"Warning: No subtasks extracted from outline. Raw outline start: {outline[:200]}...", "red"))
        
        if len(points) > 0:
            points_filtered, point_outlines_filtered = [], []
            points_set = set([])
            for i in range(len(points)):
                if points[i] not in points_set:
                    points_set.add(points[i])
                    points_filtered.append(points[i])
                    point_outlines_filtered.append(point_outlines[i])
            points = points_filtered
            point_outlines = point_outlines_filtered

        subtasks_data = []
        for i in range(len(points)):
             subtasks_data.append({"point": points[i], "point_outline": point_outlines[i]})

        contents = [""] * len(subtasks_data)
        contents_time_wall = [0.0] * len(subtasks_data)
        
        # Stage 3: Parallel Expansion
        logging.info(colored(f"Stage 2 & 3: Assigning {len(subtasks_data)} subtasks to {self.max_workers} workers...", "magenta", attrs=["bold"]))

        out_q = queue.Queue()
        task_queue = queue.Queue()
        total_downlink_bytes = 0
        total_uplink_bytes = 0

        for i, data in enumerate(subtasks_data):
            # [Downlink Calculation]
            # Simulate the socket/RPC payload structure including headers/metadata
            downlink_msg = {
                "type": "task_assignment",
                "task_id": i + 1,
                "payload": {
                    "instruction": data['point_outline'],
                    "context": {
                        "original_question": original_question,
                        "global_outline": outline
                    }
                }

            }
            msg_payload = json.dumps(downlink_msg, ensure_ascii=False).encode('utf-8')

            total_downlink_bytes += len(msg_payload)
            task_queue.put((i, data, downlink_msg))
        
        logging.info(colored(f"Manager -> Workers: Assigned {len(subtasks_data)} tasks. Total Downlink: {total_downlink_bytes} bytes.", "magenta"))

        # --- Worker Loop ---
        def machine_worker_loop(worker_id):
            raw_model = self.worker_models[worker_id % len(self.worker_models)]
            
            # 1. Wrap the model (with the stats container) - main usage
            wrapped_model = ModelUsageTracker(raw_model, worker_id, current_query_stats, usage_key="main")

            # === Tool callback ===
            def tool_usage_callback(tool_name: str, usage: dict):
                """
                Callback that updates tool usage statistics.
                Example usage: {'calls': 1, 'credits': 5, 'time': 0.5} or {'calls': 1, 'tokens': 2000, 'time': 1.2}
                """
                stats = current_query_stats[worker_id]["tools"]
                
                if tool_name == "serper":
                    stats["serper_calls"] += usage.get("calls", 0)
                    stats["serper_credits"] += usage.get("credits", 0)
                    stats["serper_time"] += usage.get("time", 0.0)
                elif tool_name == "jina":
                    stats["jina_calls"] += usage.get("calls", 0)
                    stats["jina_tokens"] += usage.get("tokens", 0)
                    stats["jina_time"] += usage.get("time", 0.0)
            # ==================================

            # 2. Initialize tools with the callback
            local_tools_map = {
                "web_search": WebSearchTool(stats_callback=tool_usage_callback),
                "wiki_search": self.wiki_search,
                "crawl_page": CrawlPageTool(model=wrapped_model, stats_callback=tool_usage_callback)
            }

            if self.enable_mm_tools:
                # Wrap the model with a specific usage_key so tokens are counted under mm_tools rather than the main LLM
                visual_tracker = ModelUsageTracker(raw_model, worker_id, current_query_stats, usage_key="visual")
                
                # Text: recorded both separately and in main
                text_tracker = ModelUsageTracker(raw_model, worker_id, current_query_stats, usage_key="text", log_to_main=True)
                
                # Audio: split accounting (Whisper -> audio_tracker, QA -> main/wrapped_model)
                audio_tracker = ModelUsageTracker(raw_model, worker_id, current_query_stats, usage_key="whisper")

                local_tools_map["inspect_file_as_image"] = VisualInspectorTool(visual_tracker, 20000)
                local_tools_map["inspect_file_as_text"] = TextInspectorTool(text_tracker, 20000)
                
                # CompositeModel: get_response goes to wrapped_model (main stats), record_usage to audio_tracker (Whisper stats)
                local_tools_map["inspect_file_as_audio"] = AudioInspectorTool(CompositeModel(wrapped_model, audio_tracker), 20000)

            worker_cumulative_turns = 0

            while True:
                if self.max_worker_turns is not None and worker_cumulative_turns >= self.max_worker_turns: return
                try:
                    idx, item_data, downlink_msg_recv = task_queue.get(block=False)
                except queue.Empty: return
                
                logging.info(colored(f"\n[Downlink JSON] Manager -> Worker {worker_id}:\n{json.dumps(downlink_msg_recv, ensure_ascii=False, indent=2)}", "blue"))

                point_start_ts = time.time()
                curr_time_str = time.strftime("%H:%M:%S", time.localtime())
                
                full_task_log = f"\n[{curr_time_str}] [Agent {worker_id}] === STARTED Goal {item_data['point']} ===\n"
                full_task_log += f"    Target: {item_data['point_outline'].strip()}\n"
                logging.info(colored(full_task_log, "yellow", attrs=["bold"]))

                recv_payload = downlink_msg_recv["payload"]
                ctx = recv_payload.get("context", {})
                current_question = ctx.get("original_question", original_question)
                current_outline = ctx.get("global_outline", outline)
                task_id = downlink_msg_recv["task_id"]
                sub_task_desc = recv_payload.get("instruction", item_data["point_outline"])
                step_messages = self.pm.get_messages('step', tools=self.tool_metadata, tool_functions_json=json.dumps(self.tool_metadata, indent=2), task=sub_task_desc, global_request=current_question, global_outline=current_outline, point=str(idx+1))
                history_messages = copy.deepcopy(step_messages)
                full_trace = []
                last_time = 0.0
                
                try:
                    for turn in range(self.max_turns):
                        # Update turn count
                        current_query_stats[worker_id]["turns"] += 1
                        worker_cumulative_turns += 1
                        
                        response_iterator = wrapped_model.get_response([history_messages], stream=True)
                        full_response_text = ""
                        
                        # Process the LLM response stream
                        for response_gen in response_iterator:
                            for chunk in response_gen:
                                full_response_text += chunk.get("text", "")
                                last_time += chunk.get("time", 0.0)
                        
                        log_msg = f"[Agent {worker_id}] Goal {idx+1} Turn {turn+1} | Time: {last_time:.2f}s"
                        logging.info(colored(log_msg, "cyan"))

                        full_trace.append(f"--- Turn {turn+1} ---\n[Agent Thought]:\n{full_response_text}\n")
                        history_messages.append({"role": "assistant", "content": full_response_text})

                        # Tool Extraction & Execution
                        tool_calls = []
                        tool_match_legacy = re.search(r'\[TOOL_CALL:\s*({.*?})\]', full_response_text, re.DOTALL)
                        if tool_match_legacy:
                            try:
                                d = json.loads(tool_match_legacy.group(1))
                                if isinstance(d, dict):
                                    if "tools" in d and isinstance(d["tools"], list): tool_calls = d["tools"]
                                    elif "name" in d: tool_calls = [d]
                            except: pass
                        
                        if not tool_calls:
                            candidates = re.findall(r'\{[\s\S]*\}', full_response_text)
                            candidates.sort(key=len, reverse=True)
                            for c in candidates:
                                try:
                                    cleaned_c = c.replace("```json", "").replace("```", "").strip()
                                    # [Robustness] Use json_repair logic
                                    try:
                                        cleaned_c = _json_repair(cleaned_c)
                                    except: pass
                                    
                                    d = json.loads(cleaned_c)
                                    if isinstance(d, dict) and "tools" in d and isinstance(d["tools"], list):
                                        tool_calls = d["tools"]
                                        break
                                    if isinstance(d, dict) and "tools" in d and isinstance(d["tools"], dict):
                                        tool_calls = [d["tools"]]
                                        break
                                except Exception as e:
                                    if "tools" in c and len(c) < 500:
                                        logging.warning(colored(f"[JSON Parse Error] Agent {worker_id} failed to parse candidate: {c[:100]}... Error: {e}", "red"))
                                    pass

                        if tool_calls:
                            full_output_str = ""
                            final_answer_found = False
                            for tc in tool_calls:
                                tool_name = tc.get("name")
                                if not tool_name:
                                    logging.error(colored(f"[Tool Error] Missing 'name' in tool call: {json.dumps(tc)}", "red"))
                                    full_output_str += f"\n> Error: Tool name missing in call.\n"
                                    continue

                                tool_args = tc.get("arguments", {})
                                
                                if tool_name == "final_answer":
                                    # [Robustness] Normalize final_answer too
                                    tool_args = _normalize_tool_args(tool_name, tool_args)
                                    full_trace.append(f"[Final Answer]: {json.dumps(tool_args, ensure_ascii=False)}")
                                    final_answer_found = True
                                    continue

                                logging.info(colored(f"\n[{curr_time_str}] [Agent {worker_id}] >>> CALLING TOOL: {tool_name}\n    Args: {json.dumps(tool_args, ensure_ascii=False)}", "blue", attrs=["bold"]))
                                
                                tool_output = "Tool not found."
                                if tool_name in local_tools_map:
                                    try:
                                        # [Robustness] Normalize args
                                        tool_args = _normalize_tool_args(tool_name, tool_args)
                                        if not isinstance(tool_args, dict): tool_args = {}

                                        tool_output = local_tools_map[tool_name].forward(**tool_args)
                                        
                                        content_len = len(tool_output)
                                        if "❌" in tool_output or "Error:" in tool_output:
                                            logging.error(colored(f"Tool Execution Failed: {tool_output[:200]}...", "red", attrs=["bold"]))
                                        else:
                                            logging.info(f"✅ Tool Output Success (Len={content_len}): \n{tool_output[:500]}...\n[...truncated...]")
                                    except Exception as e:
                                        tool_output = f"Error: {e}"
                                        logging.error(colored(f"[Tool Execution Exception] {e}", "red"))
                                else:
                                    logging.error(colored(f"[Tool Error] Unknown tool '{tool_name}' called by Agent {worker_id}.", "red"))
                                    tool_output = f"Error: Tool '{tool_name}' not found. Available tools: {list(local_tools_map.keys())}"
                                
                                full_output_str += f"\n> Tool {tool_name} output:\n{tool_output}\n"

                            if final_answer_found: break
                            full_trace.append(f"[Tool Output]:\n{full_output_str}\n")
                            history_messages.append({"role": "user", "content": f"[TOOL_OUTPUT]:\n{full_output_str}"})
                            out_q.put((idx, {"text": f"[Thinking] Turn {turn}...", "time": last_time, "stage": "expand", "point_index": idx}))
                        else:
                            if "final_answer" in full_response_text.lower(): full_trace.append("[Final Answer Found]")
                            break
                    
                    final_report = "\n".join(full_trace)
                    out_q.put((idx, {"text": final_report, "time": last_time, "stage": "expand", "point_index": idx}))

                except Exception as e:
                    logging.error(f"Worker Error: {e}")
                    final_report = f"Error: {e}"
                    # Error message is handled in finally block to prevent double submission
                    
                finally:
                    point_end_ts = time.time()
                    res_text = final_report if 'final_report' in locals() else "Error"

                    # [Worker Side] Extract Result & Calculate Traffic
                    # 1. Attempt Extract Final Answer
                    clean_result = None
                    final_answer_matches = re.findall(r"\[Final Answer\]:\s*(.*)", res_text)
                    if final_answer_matches:
                        raw_answer = final_answer_matches[-1].strip()
                        clean_result = raw_answer
                        try:
                            if raw_answer.startswith("{") and raw_answer.endswith("}"):
                                answer_json = json.loads(raw_answer)
                                if isinstance(answer_json, dict) and "answer" in answer_json:
                                    clean_result = answer_json["answer"]
                        except: pass
                    
                    # 2. Construct Payload
                    uplink_msg = {
                        "type": "task_result",
                        "task_id": idx + 1,
                        "payload": {}
                    }
                    
                    extracted_content = ""
                    if clean_result:
                        uplink_msg["payload"]["answer"] = clean_result
                        extracted_content = clean_result
                    else:
                        # Fallback: Truncated Trace
                        max_chars = 10000
                        if len(res_text) > max_chars:
                            truncated = "... [Previous steps truncated] ...\n" + res_text[-max_chars:]
                        else:
                            truncated = res_text
                        uplink_msg["payload"]["trace"] = truncated
                        extracted_content = truncated

                    # 3. Calculate Bytes
                    uplink_json = json.dumps(uplink_msg, ensure_ascii=False)
                    uplink_sz = len(uplink_json.encode('utf-8'))
                    
                    # [LOGGING] Print Uplink JSON
                    logging.info(colored(f"\n[Uplink JSON] Worker {worker_id} -> Manager (Task {idx+1}):\n{json.dumps(uplink_msg, ensure_ascii=False, indent=2)}", "green"))
                    
                    # Log Completion (Worker -> Manager)
                    logging.info(colored(f"[Agent {worker_id}] Finished Goal {idx+1}. Uplink: {uplink_sz} bytes. Result status: {'Success' if clean_result else 'Fallback'}", "green"))

                    # NOTE: 'text' field is what gets printed to console by stream_output. 
                    # We avoid sending res_text (full huge trace) to keep logs clean.
                    display_text = f"\n[Agent {worker_id}] Goal {idx+1} Completed.\nResult: {extracted_content}\n"
                    
                    out_q.put((idx, {
                        "_final": True, 
                        "text": display_text, 
                        "extracted_payload": extracted_content,
                        "time": last_time, 
                        "wall_time": point_end_ts - point_start_ts, 
                        "end_ts": point_end_ts, 
                        "uplink_bytes": uplink_sz
                    }))
                    task_queue.task_done()

        # Start Workers & Collect Results
        worker_threads = []
        for mid in range(min(self.max_workers, len(subtasks_data))):
            t = threading.Thread(target=machine_worker_loop, args=(mid,))
            t.daemon = True; t.start(); worker_threads.append(t)

        remaining = len(subtasks_data)
        while remaining > 0:
            try:
                idx, item = out_q.get(timeout=10)
                if isinstance(item, dict) and item.get("_final"):
                    # Use extracted payload for summary context
                    contents[idx] = item.get("extracted_payload", item.get("text", "")) 
                    contents_time_wall[idx] = item.get("wall_time", 0.0)
                    total_uplink_bytes += item.get("uplink_bytes", 0)
                    remaining -= 1
                    
                    logging.info(colored(f"Workers -> Manager: Received result for Goal {idx+1}. Uplink: {item.get('uplink_bytes',0)} bytes.", "cyan"))
                    continue
                if isinstance(item, dict) and item.get("_error"):
                    contents[idx] = "[ERROR]"; remaining -= 1; continue
                yield item
            except queue.Empty:
                if all(not t.is_alive() for t in worker_threads) and remaining > 0: break
        for t in worker_threads: t.join(timeout=1)

        # Stage 4: Summarize
        logging.info(colored(f"Stage 4: Summarizing results from {len(contents)} subtasks...", "magenta", attrs=["bold"]))
        summary_start_ts = time.time()
        
        evidence_chain = ""
        for i, (content, subtask_item) in enumerate(zip(contents, subtasks_data)):
            subtask_question = subtask_item['point_outline'].strip()
            # Content is already clean answer or truncated trace from worker
            evidence_chain += f"Goal {i+1} Question: {subtask_question}\nGoal {i+1} Result: {content}\n{'-'*30}\n"


        final_messages = self.pm.get_messages('final_answer', task=original_question, evidence=evidence_chain)
        final_answer_text = ""
        try:
            # Use wrapped_main_model for summary generation so tokens and calls are counted
            summary_response_iterator = wrapped_main_model.get_response([final_messages], stream=True)
            for response_generator in summary_response_iterator:
                for chunk_data in response_generator:
                    delta = chunk_data.get("text", "")
                    final_answer_text += delta
                    chunk_data["text"] = final_answer_text
        except Exception as e:
            logging.error(f"Summary Error: {e}")
            final_answer_text = "Summary failed."

        summary_end_ts = time.time()
        logging.info(colored("=== Process Complete. Final Answer Generated. ===", "green", attrs=["bold"]))
        
        # === Aggregate statistics ===
        # 1. Extract and remove manager stats so they are not mixed into worker stats
        manager_data = current_query_stats.pop("manager", {})
        
        # 2. Compute worker (execution) stats
        worker_stats_values = current_query_stats.values()

        # Helper to sum nested mm stats
        def sum_mm_stat(key, subkey):
            total = 0
            for w in worker_stats_values:
                if "mm_tools" in w and key in w["mm_tools"]:
                    val = w["mm_tools"][key]
                    if isinstance(val, dict) and subkey in val:
                        total += val[subkey]
                    elif isinstance(val, (int, float)) and subkey is None:
                        total += val
            return total

        total_stats = {
            "subtask_count": len(subtasks_data),
            "total_llm_time": sum(w["llm_time"] for w in worker_stats_values) + manager_data.get("llm_time", 0.0),
            "worker_llm_time": sum(w["llm_time"] for w in worker_stats_values),
            "manager_llm_time": manager_data.get("llm_time", 0.0),
            "total_llm_calls": sum(w["llm_calls"] for w in worker_stats_values) + manager_data.get("llm_calls", 0),
            "total_turns": sum(w["turns"] for w in worker_stats_values),
            "total_tokens": {
                "input": sum(w["tokens"]["input"] for w in worker_stats_values) + manager_data.get("tokens", {}).get("input", 0),
                "output": sum(w["tokens"]["output"] for w in worker_stats_values) + manager_data.get("tokens", {}).get("output", 0)
            },
            "total_tool_calls": {
                "serper": sum(w["tools"]["serper_calls"] for w in worker_stats_values) + manager_data.get("tools").get("serper_calls", 0),
                "jina": sum(w["tools"]["jina_calls"] for w in worker_stats_values) + manager_data.get("tools").get("jina_calls", 0)
            },
            "total_tool_usage": {
                "serper_calls": sum(w["tools"]["serper_calls"] for w in worker_stats_values) + manager_data.get("tools").get("serper_calls", 0),
                "serper_credits": sum(w["tools"]["serper_credits"] for w in worker_stats_values) + manager_data.get("tools").get("serper_credits", 0),
                "serper_time": sum(w["tools"]["serper_time"] for w in worker_stats_values) + manager_data.get("tools").get("serper_time", 0.0),
                
                "jina_calls": sum(w["tools"]["jina_calls"] for w in worker_stats_values) + manager_data.get("tools").get("jina_calls", 0),
                "jina_tokens": sum(w["tools"]["jina_tokens"] for w in worker_stats_values) + manager_data.get("tools").get("jina_tokens", 0),
                "jina_time": sum(w["tools"]["jina_time"] for w in worker_stats_values) + manager_data.get("tools").get("jina_time", 0.0)
            },
            "total_mm_usage": {
                "visual_calls": sum_mm_stat("visual_calls", None),
                "visual_time": sum_mm_stat("visual_time", None),
                "visual_input_tokens": sum_mm_stat("visual_tokens", "input"),
                "visual_output_tokens": sum_mm_stat("visual_tokens", "output"),
                
                "whisper_calls": sum_mm_stat("whisper_calls", None),
                "whisper_time": sum_mm_stat("whisper_time", None),
                # Whisper usually doesn't have "tokens" in the LLM sense, but might have if ModelUsageTracker used
                "whisper_input_tokens": sum_mm_stat("whisper_tokens", "input"), 
                "whisper_output_tokens": sum_mm_stat("whisper_tokens", "output"),
                
                "text_calls": sum_mm_stat("text_calls", None),
                "text_time": sum_mm_stat("text_time", None),
                "text_input_tokens": sum_mm_stat("text_tokens", "input"),
                "text_output_tokens": sum_mm_stat("text_tokens", "output"),
            },
            "worker_breakdown": current_query_stats
        }

        # 3. Build manager stats
        manager_stats = {
            "llm_calls": manager_data.get("llm_calls", 0),
            "llm_time": manager_data.get("llm_time", 0.0),
            "turns": manager_data.get("turns", 0),
            "tokens": manager_data.get("tokens", {"input": 0, "output": 0}),
            "tools": manager_data.get("tools", {})
        }
        # ==================

        yield {
            "stage": "summarize",
            "request": request,
            "response": final_answer_text,
            "evidence": evidence_chain,
            "outline": outline,
            "contents": contents,
            "point_times": contents_time_wall,
            "total_time": summary_end_ts - global_start_ts,
            "communication_stats": {
                "downlink_bytes": total_downlink_bytes,
                "uplink_bytes": total_uplink_bytes,
                "total_bytes": total_downlink_bytes + total_uplink_bytes
            },
            "execution_stats": total_stats,
            "manager_stats": manager_stats
        }

    def get_response(self, request, stream=False):
        if isinstance(request, str): request = [request]
        if len(request) % 2 != 1: raise ValueError("The length of the request messages should be odd.")
        if stream: return self._get_response_stream(request)
        outputs = {"text": "No response generated", "error": True}
        for outputs in self._get_response_stream(request): pass
        return outputs