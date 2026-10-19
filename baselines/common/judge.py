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
	if "error" in rec and rec["error"]:
		return False
	# Accept either (question/golden_answer/agent_result) or (problem/answer/pred)
	has_standard = all(k in rec for k in ("question", "golden_answer", "agent_result"))
	has_flash = all(k in rec for k in ("problem", "answer", "pred"))
	if not (has_standard or has_flash):
		return False
	# agent_result/pred must be a non-empty string-like content
	ar = rec.get("agent_result")
	if ar is None:
		ar = rec.get("pred")
	if ar is None:
		return False
	if isinstance(ar, str) and not ar.strip():
		return False
	return True


def build_prompt(question: Any, golden: Any, agent: Any) -> List[Dict[str, str]]:
	system = {
		"role": "system",
		"content": (
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
) -> Tuple[int, int, float, float, List[Dict[str, Any]], Dict[str, Any]]:
	# Use the total number of samples as the denominator (errored / empty answers are not excluded)
	all_records: List[Dict[str, Any]] = records[:]
	if max_items is not None:
		all_records = all_records[:max_items]

	# Split samples into judgeable (valid answer) and non-judgeable (error or no answer)
	valid_records: List[Dict[str, Any]] = []
	invalid_records: List[Dict[str, Any]] = []
	for r in all_records:
		if is_valid_record(r):
			valid_records.append(r)
		else:
			invalid_records.append(r)

	total = len(all_records)
	if total == 0:
		return 0, 0, 0.0, 0.0, [], {}

	def _judge_one(rec: Dict[str, Any]) -> Tuple[bool, Optional[float], Dict[str, Any], Dict[str, Any]]:
		# Support both field sets: standard and Flash-Searcher format
		q = rec.get("question") if rec.get("question") is not None else rec.get("problem")
		g = rec.get("golden_answer") if rec.get("golden_answer") is not None else rec.get("answer")
		a = rec.get("agent_result") if rec.get("agent_result") is not None else rec.get("pred")
		dur = rec.get("duration_sec")
		token_usage = rec.get("token_usage", {})
		steps = rec.get("steps", 0)
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
				"token_usage": token_usage,
				"steps": steps,
			}
			return ok, (float(dur) if isinstance(dur, (int, float)) else None), detail, token_usage
		finally:
			if per_request_sleep > 0:
				time.sleep(per_request_sleep)

	durations_correct: List[float] = []
	# Accumulators for statistics over correct samples
	correct_stats = {
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
	
	def _accumulate_correct_stats(token_usage: Dict[str, Any], steps: int):
		"""Accumulate statistics of a correct sample."""
		llm = token_usage.get("llm", {})
		jina = token_usage.get("jina", {})
		serper = token_usage.get("serper", {})
		comm = token_usage.get("comm", {})
		
		correct_stats["llm_tokens"] += llm.get("total_tokens", 0) or 0
		correct_stats["llm_calls"] += llm.get("total_calls", 0) or 0
		correct_stats["llm_duration_sec"] += llm.get("total_duration_sec", 0) or 0
		
		correct_stats["jina_tokens"] += jina.get("total_tokens", 0) or 0
		correct_stats["jina_calls"] += jina.get("total_calls", 0) or 0
		correct_stats["jina_duration_sec"] += jina.get("total_duration_sec", 0) or 0
		
		correct_stats["serper_credits"] += serper.get("total_credits", 0) or 0
		correct_stats["serper_calls"] += serper.get("total_calls", 0) or 0
		correct_stats["serper_duration_sec"] += serper.get("total_duration_sec", 0) or 0
		
		correct_stats["comm_input_bytes"] += comm.get("total_input_bytes", 0) or 0
		correct_stats["comm_output_bytes"] += comm.get("total_output_bytes", 0) or 0
		correct_stats["comm_calls"] += comm.get("total_calls", 0) or 0
		
		correct_stats["steps"] += steps or 0

	# Non-judgeable samples are recorded as incorrect without calling the judge model
	details: List[Dict[str, Any]] = []
	correct = 0
	for rec in invalid_records:
		q = rec.get("question") if rec.get("question") is not None else rec.get("problem")
		g = rec.get("golden_answer") if rec.get("golden_answer") is not None else rec.get("answer")
		a = rec.get("agent_result") if rec.get("agent_result") is not None else rec.get("pred")
		dur = rec.get("duration_sec")
		token_usage = rec.get("token_usage", {})
		steps = rec.get("steps", 0)
		# Mark as incorrect
		details.append({
			"question": q,
			"golden_answer": g,
			"agent_result": a,
			"gpt_result": "NO",
			"duration_sec": dur,
			"token_usage": token_usage,
			"steps": steps,
			"judge_reason": "error_or_missing_answer",
		})

	# Call the judge model on judgeable samples
	if workers and workers > 1:
		with _futures.ThreadPoolExecutor(max_workers=workers) as ex:
			for ok, dur, info, token_usage in ex.map(_judge_one, valid_records):
				if ok:
					correct += 1
					if isinstance(dur, float):
						durations_correct.append(dur)
					_accumulate_correct_stats(token_usage, info.get("steps", 0))
				details.append(info)
	else:
		for rec in valid_records:
			ok, dur, info, token_usage = _judge_one(rec)
			if ok:
				correct += 1
				if isinstance(dur, float):
					durations_correct.append(dur)
				_accumulate_correct_stats(token_usage, info.get("steps", 0))
			details.append(info)

	acc = correct / total if total else 0.0
	avg_correct_time = (sum(durations_correct) / len(durations_correct)) if durations_correct else 0.0
	
	# Average statistics over correct samples
	correct_avg_stats = {}
	if correct > 0:
		correct_avg_stats = {
			"avg_llm_tokens": round(correct_stats["llm_tokens"] / correct, 2),
			"avg_llm_calls": round(correct_stats["llm_calls"] / correct, 2),
			"avg_llm_duration_sec": round(correct_stats["llm_duration_sec"] / correct, 3),
			"avg_jina_tokens": round(correct_stats["jina_tokens"] / correct, 2),
			"avg_jina_calls": round(correct_stats["jina_calls"] / correct, 2),
			"avg_jina_duration_sec": round(correct_stats["jina_duration_sec"] / correct, 3),
			"avg_serper_credits": round(correct_stats["serper_credits"] / correct, 2),
			"avg_serper_calls": round(correct_stats["serper_calls"] / correct, 2),
			"avg_serper_duration_sec": round(correct_stats["serper_duration_sec"] / correct, 3),
			"avg_comm_input_bytes": round(correct_stats["comm_input_bytes"] / correct, 2),
			"avg_comm_output_bytes": round(correct_stats["comm_output_bytes"] / correct, 2),
			"avg_comm_calls": round(correct_stats["comm_calls"] / correct, 2),
			"avg_steps": round(correct_stats["steps"] / correct, 2),
			"total_correct_samples": correct,
			"total_llm_tokens": correct_stats["llm_tokens"],
			"total_jina_tokens": correct_stats["jina_tokens"],
			"total_serper_credits": correct_stats["serper_credits"],
			"total_comm_bytes": correct_stats["comm_input_bytes"] + correct_stats["comm_output_bytes"],
		}
	
	return total, correct, acc, avg_correct_time, details, correct_avg_stats


def load_meta_from_stats_file(input_file: str) -> Dict[str, Any]:
	"""Try to load run meta information from the matching _stats.json file."""
	base_name = os.path.basename(input_file)
	dir_name = os.path.dirname(input_file)
	
	# Try several naming patterns
	candidates = []
	
	# 1. If the input file itself is a _stats.json, use it directly
	if base_name.endswith("_stats.json"):
		candidates.append(input_file)
	
	# 2. answer_xxx.json -> answer_xxx_stats.json
	if base_name.endswith(".json") and not base_name.endswith("_stats.json"):
		stats_name = base_name.replace(".json", "_stats.json")
		candidates.append(os.path.join(dir_name, stats_name))
	
	for stats_path in candidates:
		if os.path.exists(stats_path):
			try:
				with open(stats_path, "r", encoding="utf-8") as f:
					data = json.load(f)
				if isinstance(data, dict) and "meta" in data:
					return data["meta"]
			except Exception:
				pass
	return {}


# ========== Statistics enrichment ==========

def calculate_all_global_averages(result_data: Dict[str, Any]) -> Dict[str, Any]:
	"""Compute global per-sample averages for all tools (over all samples)."""
	total_samples = result_data.get("total", 0)
	
	if total_samples == 0:
		print("Warning: total is 0, cannot compute global averages")
		return {}
	
	stats = {}
	details = result_data.get("details", [])
	
	# LLM statistics
	if "global_llm_usage" in result_data:
		llm = result_data["global_llm_usage"]
		# Prefer total_duration_sec, fall back to total_call_duration_sec
		total_call_duration = llm.get("total_duration_sec", 0)
		if total_call_duration == 0:
			total_call_duration = llm.get("total_call_duration_sec", 0)
		
		# If global_llm_usage has no duration field, compute it from details
		if total_call_duration == 0 and details:
			for detail in details:
				token_usage = detail.get("token_usage", {})
				llm_usage = token_usage.get("llm", {})
				total_call_duration += llm_usage.get("total_duration_sec", 0)
			
			if total_call_duration > 0:
				print(f"  LLM total_duration_sec computed from details: {total_call_duration:.2f}")
		
		stats["avg_llm_calls"] = round(llm.get("total_calls", 0) / total_samples, 2)
		stats["avg_llm_input_tokens"] = round(llm.get("total_input_tokens", 0) / total_samples, 2)
		stats["avg_llm_output_tokens"] = round(llm.get("total_output_tokens", 0) / total_samples, 2)
		stats["avg_llm_tokens"] = round(llm.get("total_tokens", 0) / total_samples, 2)
		stats["avg_llm_duration_sec"] = round(total_call_duration / total_samples, 2)
	
	# Jina statistics
	if "global_jina_usage" in result_data:
		jina = result_data["global_jina_usage"]
		total_jina_duration = jina.get("total_duration_sec", 0)
		
		if total_jina_duration == 0 and details:
			for detail in details:
				token_usage = detail.get("token_usage", {})
				jina_usage = token_usage.get("jina", {})
				total_jina_duration += jina_usage.get("total_duration_sec", 0)
			
			if total_jina_duration > 0:
				print(f"  Jina total_duration_sec computed from details: {total_jina_duration:.2f}")
		
		stats["avg_jina_calls"] = round(jina.get("total_calls", 0) / total_samples, 2)
		stats["avg_jina_tokens"] = round(jina.get("total_tokens", 0) / total_samples, 2)
		stats["avg_jina_duration_sec"] = round(total_jina_duration / total_samples, 2)
	
	# Serper statistics
	if "global_serper_usage" in result_data:
		serper = result_data["global_serper_usage"]
		total_serper_duration = serper.get("total_duration_sec", 0)
		
		if total_serper_duration == 0 and details:
			for detail in details:
				token_usage = detail.get("token_usage", {})
				serper_usage = token_usage.get("serper", {})
				total_serper_duration += serper_usage.get("total_duration_sec", 0)
			
			if total_serper_duration > 0:
				print(f"  Serper total_duration_sec computed from details: {total_serper_duration:.2f}")
		
		stats["avg_serper_calls"] = round(serper.get("total_calls", 0) / total_samples, 2)
		stats["avg_serper_credits"] = round(serper.get("total_credits", 0) / total_samples, 2)
		stats["avg_serper_duration_sec"] = round(total_serper_duration / total_samples, 2)
	
	# Communication statistics
	if "global_comm_usage" in result_data:
		comm = result_data["global_comm_usage"]
		total_input_bytes = comm.get("total_input_bytes", 0)
		total_output_bytes = comm.get("total_output_bytes", 0)
		stats["avg_comm_calls"] = round(comm.get("total_calls", 0) / total_samples, 2)
		stats["avg_comm_input_bytes"] = round(total_input_bytes / total_samples, 2)
		stats["avg_comm_output_bytes"] = round(total_output_bytes / total_samples, 2)
		stats["avg_comm_bytes"] = round((total_input_bytes + total_output_bytes) / total_samples, 2)
	
	# Visual/Text/Audio statistics (if present)
	run_meta = result_data.get("run_meta", {})
	
	for tool_name in ["visual", "text", "audio"]:
		tool_key = f"{tool_name}_usage"
		if tool_key in run_meta:
			tool_data = run_meta[tool_key]
			total_duration = tool_data.get("total_duration_sec", 0)
			
			if total_duration == 0 and details:
				for detail in details:
					token_usage = detail.get("token_usage", {})
					usage = token_usage.get(tool_name, {})
					total_duration += usage.get("total_duration_sec", 0)
			
			stats[f"avg_{tool_name}_calls"] = round(tool_data.get("total_calls", 0) / total_samples, 2)
			stats[f"avg_{tool_name}_tokens"] = round(tool_data.get("total_tokens", 0) / total_samples, 2)
			stats[f"avg_{tool_name}_duration_sec"] = round(total_duration / total_samples, 2)
	
	# Add global average run time
	avg_run_duration = result_data.get("avg_run_duration_sec", 0)
	if avg_run_duration > 0:
		stats["avg_duration_sec"] = avg_run_duration
	elif details:
		# Compute from details
		total_duration = sum(d.get("duration_sec", 0) for d in details if "duration_sec" in d)
		valid_samples = sum(1 for d in details if "duration_sec" in d)
		if valid_samples > 0:
			stats["avg_duration_sec"] = round(total_duration / valid_samples, 2)
	
	return stats


def calculate_global_avg_steps(result_data: Dict[str, Any]) -> Optional[float]:
	"""Compute the global average number of steps."""
	details = result_data.get("details", [])
	
	if not details:
		return None
	
	total_steps = 0
	valid_samples = 0
	
	for detail in details:
		if "steps" in detail:
			total_steps += detail["steps"]
			valid_samples += 1
	
	if valid_samples == 0:
		return None
	
	return round(total_steps / valid_samples, 2)


def add_token_stats_to_correct_sample_stats(result_data: Dict[str, Any]) -> None:
	"""Add token statistics to correct_sample_stats."""
	if "correct_sample_stats" not in result_data:
		return
	
	details = result_data.get("details", [])
	correct_samples = [d for d in details if d.get("gpt_result") == "YES"]
	
	if not correct_samples:
		return
	
	total_input_tokens = 0
	total_output_tokens = 0
	
	for sample in correct_samples:
		token_usage = sample.get("token_usage", {})
		llm_tokens = token_usage.get("llm", {})
		total_input_tokens += llm_tokens.get("total_input_tokens", 0)
		total_output_tokens += llm_tokens.get("total_output_tokens", 0)
	
	num_correct = len(correct_samples)
	
	result_data["correct_sample_stats"]["avg_input_tokens"] = round(total_input_tokens / num_correct, 2)
	result_data["correct_sample_stats"]["avg_output_tokens"] = round(total_output_tokens / num_correct, 2)
	result_data["correct_sample_stats"]["total_input_tokens"] = total_input_tokens
	result_data["correct_sample_stats"]["total_output_tokens"] = total_output_tokens


def enrich_result_data(result_data: Dict[str, Any]) -> None:
	"""Enrich the result data with global statistics and missing fields."""
	print("\n" + "=" * 60)
	print("Computing additional statistics...")
	print("=" * 60)
	
	# 1. Global average statistics
	if "global_sample_stats" not in result_data:
		print("\nComputing global average statistics...")
		global_stats = calculate_all_global_averages(result_data)
		if global_stats:
			result_data["global_sample_stats"] = global_stats
			print(f"Added global_sample_stats ({len(global_stats)} metrics)")
	
	# 2. Global average steps
	if "avg_steps" not in result_data:
		print("\nComputing global average steps...")
		avg_steps = calculate_global_avg_steps(result_data)
		if avg_steps is not None:
			result_data["avg_steps"] = avg_steps
			print(f"Added avg_steps: {avg_steps}")
	
	# 3. Token statistics in correct_sample_stats
	if "correct_sample_stats" in result_data:
		if "avg_input_tokens" not in result_data["correct_sample_stats"]:
			print("\nAdding token statistics to correct_sample_stats...")
			add_token_stats_to_correct_sample_stats(result_data)
			print("Added token statistics")
		
		# 4. avg_comm_bytes
		if "avg_comm_bytes" not in result_data["correct_sample_stats"]:
			avg_input = result_data["correct_sample_stats"].get("avg_comm_input_bytes", 0)
			avg_output = result_data["correct_sample_stats"].get("avg_comm_output_bytes", 0)
			result_data["correct_sample_stats"]["avg_comm_bytes"] = round(avg_input + avg_output, 2)
			print("Added avg_comm_bytes")
		
		# 5. avg_duration_sec in correct_sample_stats
		if "avg_duration_sec" not in result_data["correct_sample_stats"]:
			avg_correct_dur = result_data.get("avg_correct_duration_sec", 0)
			if avg_correct_dur > 0:
				result_data["correct_sample_stats"]["avg_duration_sec"] = avg_correct_dur
				print("Added avg_duration_sec to correct_sample_stats")
	
	print("\nStatistics enrichment done.")


def main(argv: Optional[List[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="Judge agent results vs golden answers via LLM")
	parser.add_argument("--file", "-f", required=True, help="Path to JSON file")
	parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenRouter API base URL")
	parser.add_argument("--model", default=DEFAULT_MODEL, help="Model identifier")
	parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="OpenRouter API key")
	parser.add_argument("--max", type=int, default=None, help="Max number of valid items to evaluate")
	parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests")
	parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds")
	parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers (threads)")
	parser.add_argument("--judge-retries", type=int, default=2, help="Retries per record when judgment is UNKNOWN")
	args = parser.parse_args(argv)
	if not args.api_key:
		parser.error("OPENROUTER_API_KEY is not set; export it (see baselines/README.md) or pass --api-key")

	try:
		records = load_records(args.file)
	except Exception as e:
		print(f"Failed to load JSON: {e}", file=sys.stderr)
		return 2

	# Try to load run meta information
	meta_info = load_meta_from_stats_file(args.file)

	total, correct, acc, avg_correct_time, details, correct_avg_stats = evaluate(
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

	print("\n" + "=" * 60)
	print("Basic statistics")
	print("=" * 60)
	print("Total samples:", total)
	print("Correct:", correct)
	print("Accuracy:", f"{acc:.4f}")
	print("Avg time of correct queries (s):", f"{avg_correct_time:.3f}")
	
	if correct_avg_stats:
		print("\n" + "=" * 60)
		print("Averages over correct samples")
		print("=" * 60)
		print(f"Avg LLM tokens: {correct_avg_stats.get('avg_llm_tokens', 0)}")
		print(f"Avg LLM calls: {correct_avg_stats.get('avg_llm_calls', 0)}")
		print(f"Avg LLM time (s): {correct_avg_stats.get('avg_llm_duration_sec', 0)}")
		print(f"Avg Jina tokens: {correct_avg_stats.get('avg_jina_tokens', 0)}")
		print(f"Avg Jina calls: {correct_avg_stats.get('avg_jina_calls', 0)}")
		print(f"Avg Jina time (s): {correct_avg_stats.get('avg_jina_duration_sec', 0)}")
		print(f"Avg Serper credits: {correct_avg_stats.get('avg_serper_credits', 0)}")
		print(f"Avg Serper calls: {correct_avg_stats.get('avg_serper_calls', 0)}")
		print(f"Avg Serper time (s): {correct_avg_stats.get('avg_serper_duration_sec', 0)}")
		print(f"Avg comm input (bytes): {correct_avg_stats.get('avg_comm_input_bytes', 0)}")
		print(f"Avg comm output (bytes): {correct_avg_stats.get('avg_comm_output_bytes', 0)}")
		print(f"Avg comm calls: {correct_avg_stats.get('avg_comm_calls', 0)}")
		print(f"Avg steps: {correct_avg_stats.get('avg_steps', 0)}")
	
	if meta_info:
		print("\n" + "=" * 60)
		print("Run meta information")
		print("=" * 60)
		print(f"Backend: {meta_info.get('backend', 'N/A')}")
		print(f"Model: {meta_info.get('model', 'N/A')}")
		print(f"Input file: {meta_info.get('infile', 'N/A')}")
		print(f"Max steps: {meta_info.get('max_steps', 'N/A')}")
		print(f"Concurrency: {meta_info.get('concurrency', 'N/A')}")
		print(f"Tool concurrency: {meta_info.get('tool_max_workers', 'N/A')}")
		print(f"Total time (s): {meta_info.get('total_duration_sec', 'N/A')}")
		print(f"Avg time (s): {meta_info.get('avg_duration_sec', 'N/A')}")
		print(f"Samples: {meta_info.get('count', 'N/A')}")

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
		# Basic statistics
		"total": total,
		"correct": correct,
		"accuracy": acc,
		"avg_correct_duration_sec": avg_correct_time,
		# Run configuration (from meta)
		"agent_model": meta_info.get("model", "N/A"),
		"agent_backend": meta_info.get("backend", "N/A"),
		"max_steps": meta_info.get("max_steps", "N/A"),
		"concurrency": meta_info.get("concurrency", "N/A"),
		"tool_max_workers": meta_info.get("tool_max_workers", "N/A"),
		"total_run_duration_sec": meta_info.get("total_duration_sec", "N/A"),
		"avg_run_duration_sec": meta_info.get("avg_duration_sec", "N/A"),
		"sample_count": meta_info.get("count", "N/A"),
		# Correct-sample statistics
		"correct_sample_stats": correct_avg_stats,
		# Global usage statistics (from meta)
		"global_llm_usage": meta_info.get("llm_usage", {}),
		"global_jina_usage": meta_info.get("jina_usage", {}),
		"global_serper_usage": meta_info.get("serper_usage", {}),
		"global_comm_usage": meta_info.get("comm_usage", {}),
		# Judge configuration
		"judge_model": args.model,
		"judge_base_url": args.base_url,
		# Raw meta information
		"run_meta": meta_info,
		# Per-sample details
		"details": details,
	}
	
	# Enrich the result data (global statistics etc.)
	enrich_result_data(result_payload)
	
	try:
		with open(out_path, "w", encoding="utf-8") as f:
			json.dump(result_payload, f, ensure_ascii=False, indent=2)
		print("\nResults saved:", out_path)
	except Exception as e:
		print(f"Failed to write result file: {e}", file=sys.stderr)
	return 0


if __name__ == "__main__":
	sys.exit(main())