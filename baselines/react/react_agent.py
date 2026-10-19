from __future__ import annotations
import json
import re
import os
import yaml
import logging
from typing import Optional, Any, Dict, List
from rich.console import Console
from .memory import ConversationMemory
from .fs_tools_adapter import (
    default_registry, 
    ToolRegistry, 
    reset_thread_usage, 
    get_thread_jina_usage, 
    get_thread_serper_usage,
    get_thread_tool_time
)
from .mm_tools import (
    get_thread_visual_usage,
    get_thread_text_usage,
    get_thread_audio_usage
)
from .llm import make_llm

logger = logging.getLogger(__name__)
console = Console()


def _load_system_prompt_from_yaml() -> str:
    """Load system_prompt from baselines/react/toolcalling_agent.yaml.
    Strips the Jinja 'Available Tools' template block to avoid leaking template syntax.
    Falls back to a minimal default system prompt if file is missing.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(here, "toolcalling_agent.yaml")
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        sp = data.get("system_prompt", "") if isinstance(data, dict) else ""
        if not isinstance(sp, str) or not sp.strip():
            raise ValueError("system_prompt missing or empty in YAML")
        # Remove the templated Available Tools section to avoid Jinja artifacts.
        # This removes everything between a line starting with '# Available Tools' and the next line starting with '# Rules'.
        sp = re.sub(r"(?ms)^\s*# Available Tools[\s\S]*?^\s*# Rules", "\n# Rules", sp)
        return sp
    except Exception as e:
        console.print(f"[yellow]Warn:[/yellow] failed to load system_prompt from YAML: {e}")
        return (
            "You are an expert assistant who solves tasks through structured tool calls, step by step.\n"
            "Return ONLY one JSON object per step with 'think' and 'tools'. When ready, use 'final_answer'."
        )

def _available_tools_doc(reg: ToolRegistry) -> str:
    names = reg.names()
    lines = ["Available tools:"]
    for n in names:
        tool = reg.get(n)
        desc = tool.description if tool else ""
        # Format: "- tool_name: tool_description"
        lines.append(f"- {n}: {desc}")
    lines.append("- final_answer (virtual): return final answer when fully verified")
    return "\n".join(lines)


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\n", "", s)
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def _json_repair(s: str) -> str:
    # 1. Handle the case where arguments are wrapped in braces like a set: "arguments": { "content" }
    # regex matches: : { "string" } -> : "string"
    # It ensures there's no colon after the string (which would make it a valid key-value pair)
    s_new, count = re.subn(r'(:\s*)\{\s*("(?:[^"\\]|\\.)*")\s*\}', r'\1\2', s)
    if count > 0:
        logger.warning(f"⚠️ Repaired {count} occurrence(s) of set-like argument syntax (invalid JSON corrected).")
    s = s_new
    
    # 2. Remove trailing commas before ] or }
    s = re.sub(r",\s*(\]|\})", r"\1", s)
    return s


def _parse_tool_plan(text: str) -> Dict[str, Any]:
    raw = _strip_code_fences(text)
    fixed = _json_repair(raw)
    try:
        parsed = json.loads(fixed)
        if isinstance(parsed, list):
            return {"think": "", "tools": parsed}
        if isinstance(parsed, dict):
            return parsed
    except Exception as e:
        logger.warning(f"⚠️ JSON parsing failed in _parse_tool_plan: {e}")
        # Fallback: try to extract JSON object
        m = re.search(r"\{[\s\S]*\}\s*$", fixed)
        if m:
            try:
                return json.loads(_json_repair(m.group(0)))
            except Exception as e2:
                logger.warning(f"⚠️ JSON fallback parsing failed: {e2}")
                pass
    return {"think": text, "tools": []}


def _load_prompt_templates() -> Dict[str, Any]:
    """Load the full prompt templates YAML for planning/summary/step reuse."""
    here = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(here, "toolcalling_agent.yaml")
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        console.print(f"[yellow]Warn:[/yellow] failed to load prompt templates from YAML: {e}")
    return {}


class ReactAgent:
    def __init__(self, backend: str = "openai", tools: Optional[ToolRegistry] = None, max_steps: int = 6, debug: bool = True, enable_mm_tools: bool = False):
        self.llm = make_llm(backend)
        self.tools = tools or default_registry(backend=backend, shared_llm=self.llm, enable_mm_tools=enable_mm_tools)
        self.max_steps = max_steps
        self.prompt_templates: Dict[str, Any] = _load_prompt_templates()
        self.initial_plan_text: Optional[str] = None
        self.memory = ConversationMemory()
        self.debug = debug
        self.trace: List[Dict[str, Any]] = []
        self.system_prompt: str = _load_system_prompt_from_yaml()
        self.stats: Dict[str, Any] = {}
        self.stats["tool_total_time_sec"] = 0.0  # Initialize tool total time

    def _generate_initial_plan(self, question: str) -> None:
        planning_cfg = self.prompt_templates.get("planning", {}) if isinstance(self.prompt_templates, dict) else {}
        initial_plan_tpl = planning_cfg.get("initial_plan") or ""
        task_input_tpl = planning_cfg.get("task_input") or ""
        # Strip the templated Available Tools block if present in the planning template
        initial_plan_clean = re.sub(r"(?ms)^\s*### Available Tools:[\s\S]*?^\s*### Key Execution Notes:", "\n### Key Execution Notes:", initial_plan_tpl)
        # Fill in minimal placeholders in task_input
        task_input_filled = task_input_tpl.replace("{{task}}", question)
        planning_prompt = f"{initial_plan_clean}\n\n{task_input_filled}".strip()
        if not planning_prompt:
            self.initial_plan_text = None
            return
        try:
            plan_out = self.llm.generate(planning_prompt)
            self.initial_plan_text = str(plan_out).strip()
        except Exception as e:
            console.print(f"[yellow]Warn:[/yellow] failed to generate initial plan: {e}")
            self.initial_plan_text = None

    def _build_step_prompt(self, question: str, history: List[Dict[str, Any]]) -> str:
        # NOTE: system_prompt has "Available Tools" stripped by default loader.
        # We must append the actual tool list here so the LLM knows what to use.
        tool_doc = _available_tools_doc(self.tools)
        
        parts = [self.system_prompt, "", tool_doc, "", f"Question: {question}"]
        if self.initial_plan_text:
            parts.append(f"Initial Plan:\n{self.initial_plan_text}")

        if history:
            parts.append("Context so far (previous steps):")
            for i, h in enumerate(history, 1):
                parts.append(
                    f"Step {i}:\nthink: {h.get('think','')}\n"
                    f"tools: {json.dumps(h.get('tools', []), ensure_ascii=False)}\n"
                    f"observation: {h.get('observation','')}"
                )
        parts.append("Respond with only the JSON object described above. The output must be valid JSON (no code fences, no extra text).")
        return "\n\n".join(parts)

    def _fix_to_json(self, bad_output: str) -> str:
        """Ask the model to convert previous output to strictly valid JSON per the schema."""
        fix_prompt = (
            "You previously returned an output that is not valid JSON. "
            "Convert the following content into a strictly valid JSON object that matches the schema described earlier. "
            "Return ONLY the JSON. No code fences, no extra text.\n\n"
            f"Content to convert:\n{bad_output}"
        )
        return self.llm.generate(fix_prompt)

    def run(self, question: str) -> str:
        # Reset tool usage stats for this thread before running
        reset_thread_usage()
        
        # Generate initial plan once before starting steps
        self._generate_initial_plan(question)
        # Log task start
        logger.info(f"🚀 [Start Task] Question: {question}")
        if self.initial_plan_text:
             logger.info(f"📅 [Initial Plan]: {self.initial_plan_text}")
        history: List[Dict[str, Any]] = []
        for step in range(self.max_steps - 1):
            # Log step separator
            logger.info(f"──────────────── Step {step + 1} ────────────────")

            prompt = self._build_step_prompt(question, history)
            model_out = self.llm.generate(prompt)

            try:
                # Try to parse JSON for pretty printing (indent=2)
                # Reuses _strip_code_fences and _json_repair
                cleaned_json = _strip_code_fences(model_out)
                cleaned_json = _json_repair(cleaned_json)
                parsed_obj = json.loads(cleaned_json)
                
                # Pretty print
                pretty_out = json.dumps(parsed_obj, ensure_ascii=False, indent=2)
                logger.info(f"🤖 LLM Output:\n{pretty_out}")
                
            except Exception as e:
                # Not JSON (or parse failed): print the raw string
                logger.error(f"❌ JSON Parse Error for logging: {e}")
                logger.info(f"🤖 LLM Output (Raw):\n{model_out}")

            parsed = _parse_tool_plan(model_out)
            # If parsing yields no tools and think is raw text, try one JSON-fix retry
            if not isinstance(parsed, dict) or not parsed.get("tools"):
                logger.warning(f"⚠️ Failed to parse valid tools from output. Attempting self-correction...")
                try:
                    repaired = self._fix_to_json(model_out)
                    parsed_fixed = _parse_tool_plan(repaired)
                    if isinstance(parsed_fixed, dict) and parsed_fixed.get("tools") is not None:
                        parsed = parsed_fixed
                        model_out = repaired
                        logger.info(f"✅ Self-correction successful.")
                    else:
                        logger.warning(f"❌ Self-correction failed.")
                except Exception as e:
                    logger.error(f"❌ Self-correction exception: {e}")
                    pass
            think = parsed.get("think", "")
            tool_calls = parsed.get("tools", []) or []

            observations: List[str] = []
            final_answer: Optional[str] = None

            for tc in tool_calls:
                if not isinstance(tc, dict):
                    observations.append(f"Invalid tool call item: {tc} (expected dict)")
                    continue

                name = tc.get("name", "")
                args = tc.get("arguments", {})
                if name == "final_answer":
                    # Case 1: arguments is a plain string
                    if isinstance(args, str):
                        final_answer = args
                    # Case 2: arguments is a dict
                    elif isinstance(args, dict):
                        # 2.1: try standard keys
                        final_answer = args.get("answer") or args.get("content") or args.get("result")
                        
                        # 2.2: no standard key, fall back to heuristics
                        if not final_answer:
                            # e.g. {"The fictional defendant...": null}
                            # i.e. a single long key with an empty value
                            if len(args) == 1:
                                k, v = list(args.items())[0]
                                # value is None / empty string / empty list or dict
                                if not v: 
                                    final_answer = k
                                else:
                                    # value is non-empty: prefer it (e.g. {"unknown_key": "Answer is..."})
                                    final_answer = str(v) if isinstance(v, (str, int, float)) else str(args)
                            
                            # 2.3: still nothing, or multiple keys: last-resort fallback
                            if not final_answer:
                                if args: # has content but unexpected keys
                                    # join all values, or dump the dict
                                    # e.g. {"part1": "A", "part2": "B"}
                                    try:
                                        vals = [str(v) for v in args.values() if v]
                                        if vals:
                                            final_answer = " ".join(vals)
                                        else:
                                            # keys only, no values
                                            final_answer = " ".join([str(k) for k in args.keys()])
                                    except:
                                        final_answer = json.dumps(args, ensure_ascii=False)
                                else:
                                    final_answer = "" # empty dict

                    else:
                        # arguments is neither str nor dict (e.g. list)
                        final_answer = str(args)
                    continue

                tool = self.tools.get(name)
                if not tool:
                    observations.append(f"Unknown tool: {name}")
                    continue
                try:
                    if isinstance(args, dict):
                        obs = tool.func(args)
                    else:
                        obs = tool.func(str(args))
                except Exception as e:
                    obs = f"Tool '{name}' error: {e}"
                observations.append(f"{name}: {obs}")

            obs_joined = "\n\n".join(observations) if observations else "(no observations)"

            
            history.append({"think": think, "tools": tool_calls, "observation": obs_joined})
            action_str = json.dumps(tool_calls, ensure_ascii=False)
            self.memory.add_step(
                thought=think, 
                action=action_str, 
                action_input="",  # arguments are already included in action_str
                observation=obs_joined
            )

            if self.debug:
                # Record raw model output and parsed content for debugging
                try:
                    self.trace.append({
                        "step": step + 1,
                        "model_output_raw": model_out,
                        "parsed_think": think,
                        "parsed_tools": tool_calls,
                        "observation": obs_joined,
                    })
                except Exception:
                    pass

            logger.info(f"[Stats]LLM call count: {self.llm.call_count}")

            if final_answer is not None:
                self.memory.set_final(final_answer)
                
                # Gather stats
                jina_stats = get_thread_jina_usage()
                serper_stats = get_thread_serper_usage()
                
                self.stats = {
                    "steps": step + 1,
                    "llm_calls": self.llm.call_count,
                    "llm_total_time_sec": round(getattr(self.llm, 'total_call_time', 0.0), 3),
                    "llm_input_tokens": getattr(self.llm, 'total_input_tokens', 0),
                    "llm_output_tokens": getattr(self.llm, 'total_output_tokens', 0),
                    "jina_calls": jina_stats.get("total_calls", 0),
                    "jina_tokens": jina_stats.get("total_tokens", 0),
                    "jina_time": round(float(jina_stats.get("total_time", 0.0) or 0.0), 3),
                    "serper_calls": serper_stats.get("total_calls", 0),
                    "serper_credits": serper_stats.get("total_tokens", 0),
                    "serper_time": round(float(serper_stats.get("total_time", 0.0) or 0.0), 3),
                    "visual_calls": get_thread_visual_usage().get("calls", 0),
                    "visual_tokens": get_thread_visual_usage().get("tokens", 0),
                    "visual_time": round(get_thread_visual_usage().get("time", 0.0) or 0.0, 3),
                    "text_tool_calls": get_thread_text_usage().get("calls", 0),
                    "text_tool_tokens": get_thread_text_usage().get("tokens", 0),
                    "text_tool_time": round(get_thread_text_usage().get("time", 0.0) or 0.0, 3),
                    "whisper_calls": get_thread_audio_usage().get("calls", 0),
                    "whisper_tokens": get_thread_audio_usage().get("tokens", 0),
                    "whisper_time": round(get_thread_audio_usage().get("time", 0.0) or 0.0, 3),
                    "tool_total_llm_time": round(getattr(self.llm, "total_tool_call_time", 0.0) or 0.0, 3),
                }
                # Add per-tool fields like tool_time_<name>_sec
                try:
                    per = get_thread_tool_time().get("per_tool", {})
                    for tname, tval in per.items():
                        key = re.sub(r'[^0-9a-zA-Z]+', '_', tname).lower()
                        self.stats[f"tool_time_{key}_sec"] = round(float(tval or 0.0), 3)
                except Exception:
                    pass
                return final_answer

        # ============================================================
        # Max steps reached: force the model to give a final answer
        # ============================================================
        logger.warning(f"🛑 [Max Steps Reached] Forcing a final answer...")
        
        # Build the forcing prompt
        force_prompt = self._build_step_prompt(question, history)
        force_prompt += (
            "\n\nSYSTEM URGENT: You have reached the maximum number of steps. "
            "You MUST now provide a final answer based on the information gathered so far. "
            "Do NOT try to use search tools anymore. "
            "Use the 'final_answer' tool to output your conclusion immediately."
        )

        try:
            # One last LLM call
            final_out = self.llm.generate(force_prompt)
            parsed = _parse_tool_plan(final_out)
            
            forced_ans = None
            # Try to parse final_answer
            if isinstance(parsed, dict):
                tools = parsed.get("tools", [])
                if isinstance(tools, list):
                    for tc in tools:
                        if isinstance(tc, dict) and tc.get("name") == "final_answer":
                            args = tc.get("arguments", {})
                            if isinstance(args, str): forced_ans = args
                            elif isinstance(args, dict): forced_ans = args.get("answer") or args.get("content") or json.dumps(args, ensure_ascii=False)
                            else: forced_ans = str(args)
                            break
            
            # No tool call: fall back to think or raw output
            if not forced_ans:
                # Prefer think; fall back to raw output if missing/empty
                forced_ans = parsed.get("think") if isinstance(parsed, dict) else final_out
                if not forced_ans:
                    forced_ans = final_out

        except Exception as e:
            logger.error(f"Force answer failed: {e}")
            forced_ans = history[-1]["observation"] if history else "(no steps)"

        # Record stats
        if hasattr(self.llm, 'call_count'):
            logger.info(f"📊 [Stats] Total LLM Calls (incl. force): {self.llm.call_count}")

        self.memory.set_final(forced_ans)
        
        # Gather stats
        jina_stats = get_thread_jina_usage()
        serper_stats = get_thread_serper_usage()

        self.stats = {
            "steps": self.max_steps,
            "llm_calls": self.llm.call_count,
            "llm_total_time_sec": round(getattr(self.llm, 'total_call_time', 0.0), 3),
            "llm_input_tokens": getattr(self.llm, 'total_input_tokens', 0),
            "llm_output_tokens": getattr(self.llm, 'total_output_tokens', 0),
            "jina_calls": jina_stats.get("total_calls", 0),
            "jina_tokens": jina_stats.get("total_tokens", 0),
            "jina_time": round(float(jina_stats.get("total_time", 0.0) or 0.0), 3),
            "serper_calls": serper_stats.get("total_calls", 0),
            "serper_credits": serper_stats.get("total_tokens", 0),
            "serper_time": round(float(serper_stats.get("total_time", 0.0) or 0.0), 3),
            "visual_calls": get_thread_visual_usage().get("calls", 0),
            "visual_tokens": get_thread_visual_usage().get("tokens", 0),
            "visual_time": round(get_thread_visual_usage().get("time", 0.0) or 0.0, 3),
            "text_tool_calls": get_thread_text_usage().get("calls", 0),
            "text_tool_tokens": get_thread_text_usage().get("tokens", 0),
            "text_tool_time": round(get_thread_text_usage().get("time", 0.0) or 0.0, 3),
            "whisper_calls": get_thread_audio_usage().get("calls", 0),
            "whisper_tokens": get_thread_audio_usage().get("tokens", 0),
            "whisper_time": round(get_thread_audio_usage().get("time", 0.0) or 0.0, 3),
            "tool_total_llm_time": round(getattr(self.llm, "total_tool_call_time", 0.0) or 0.0, 3),
        }
        # add per-tool breakdown for forced path as well
        try:
            per = get_thread_tool_time().get("per_tool", {})
            for tname, tval in per.items():
                key = re.sub(r'[^0-9a-zA-Z]+', '_', tname).lower()
                self.stats[f"tool_time_{key}_sec"] = round(float(tval or 0.0), 3)
        except Exception:
            pass
        logger.info(f"⚠️ [Forced Answer]: {forced_ans}")
        console.print(f"[yellow]Forced Answer[/yellow]: {forced_ans}")
        return forced_ans
