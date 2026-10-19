"""Turn a CoAct run log into per-query records with cost statistics.

Usage: python -m coact.evaluation.parse_log logs/gaia_20260101_120000.log -o stats/gaia.json

Each record holds the question, reference answer, the planner's final answer, wall-clock
time, LLM token usage, sub-agent steps, search/crawl API usage, tool-specific usage and
centre<->sub-agent communication volume. The output can be fed directly to the judge.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from coact.utils import extract_json

TIME_FORMAT = "%Y-%m-%d %H:%M:%S,%f"

# "<time> - <LEVEL> - [task N] <message>"; some framework lines repeat the task tag.
LINE = re.compile(
    r"^(?P<time>\d{4}-\d{2}-\d{2} [\d:,]+) - \w+ - \[task (?P<id>\d+)\] (?:\[task \d+\] )?(?P<msg>.*)$"
)

COUNTERS = {
    "llm_calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "llm_duration_seconds": 0.0,
    "steps": 0,
    "center_to_sub_bytes": 0,
    "center_to_sub_count": 0,
    "sub_to_center_bytes": 0,
    "sub_to_center_count": 0,
    "jina_tokens": 0,
    "jina_calls": 0,
    "serper_credits": 0,
    "serper_calls": 0,
    "text_inspector_input_tokens": 0,
    "text_inspector_output_tokens": 0,
    "text_inspector_total_tokens": 0,
    "visual_inspector_input_tokens": 0,
    "visual_inspector_output_tokens": 0,
    "visual_inspector_total_tokens": 0,
    "visual_inspector_duration": 0.0,
    "whisper_tokens": 0,
    "whisper_duration": 0.0,
    "audio_inspector_tokens": 0,
    "audio_inspector_duration": 0.0,
}

# message pattern -> {record field: regex group} to accumulate (plus an implicit call count).
_INT, _FLOAT = r"(\d+)", r"([\d.]+)"
USAGE_PATTERNS: list[tuple[re.Pattern, dict[str, int], str | None]] = [
    (
        re.compile(
            rf"\[LLM\] model=\S+ input_tokens={_INT} output_tokens={_INT} "
            rf"total_tokens={_INT} duration_sec={_FLOAT}"
        ),
        {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3, "llm_duration_seconds": 4},
        "llm_calls",
    ),
    (re.compile(r"step\+\+"), {}, "steps"),
    (
        re.compile(rf"center to sub communication size:{_INT} bytes"),
        {"center_to_sub_bytes": 1},
        "center_to_sub_count",
    ),
    (
        re.compile(rf"sub to center communication size:{_INT} bytes"),
        {"sub_to_center_bytes": 1},
        "sub_to_center_count",
    ),
    (re.compile(rf"\[Jina Usage\] tokens = {_INT}"), {"jina_tokens": 1}, "jina_calls"),
    (re.compile(rf"\[Serper credits\]: {_INT}"), {"serper_credits": 1}, "serper_calls"),
    (
        re.compile(
            rf"\[TextInspector Usage\] total_tokens={_INT} input_tokens={_INT} output_tokens={_INT}"
        ),
        {
            "text_inspector_total_tokens": 1,
            "text_inspector_input_tokens": 2,
            "text_inspector_output_tokens": 3,
        },
        None,
    ),
    (
        re.compile(
            rf"\[Tool:visual_inspector\] input_tokens={_INT} output_tokens={_INT} total_tokens={_INT}"
        ),
        {
            "visual_inspector_input_tokens": 1,
            "visual_inspector_output_tokens": 2,
            "visual_inspector_total_tokens": 3,
        },
        None,
    ),
    (
        re.compile(rf"\[Tool:visual_inspector\] duration={_FLOAT}s, tokens=\d+"),
        {"visual_inspector_duration": 1},
        None,
    ),
    (
        re.compile(rf"\[Whisper Usage\] duration={_FLOAT}s estimated_transcript_tokens={_INT}"),
        {"whisper_duration": 1, "whisper_tokens": 2},
        None,
    ),
    (
        re.compile(rf"\[Tool:audio_inspector\] transcription only, duration={_FLOAT}s"),
        {"audio_inspector_duration": 1},
        None,
    ),
    (
        re.compile(rf"\[Tool:audio_inspector\] duration={_FLOAT}s, tokens={_INT}"),
        {"audio_inspector_duration": 1, "audio_inspector_tokens": 2},
        None,
    ),
]

START = re.compile(r"Start task \d+: (?P<question>.*)")
END = re.compile(r"End task \d+ \| Answer: (?P<answer>.*)")
FAILED = re.compile(r"failed, skipped \[(?P<reason>.*)\]")
PLANNER_OUTPUT = "llm call end,result:"


def new_record(task_id: int) -> dict:
    return {
        "task_id": task_id,
        "question": None,
        "answer": None,
        "final_answer": None,
        "status": "unknown",
        "fail_reason": None,
        "start_time": None,
        "end_time": None,
        "duration_seconds": None,
        **COUNTERS,
    }


def parse_log(path: Path, step_budget: int = 120) -> list[dict]:
    records: dict[int, dict] = {}
    # The planner's raw output can span several lines; buffer it until the next log line.
    pending: tuple[int, list[str]] | None = None

    def flush_planner_output() -> None:
        nonlocal pending
        if pending is None:
            return
        task_id, lines = pending
        pending = None
        try:
            plan = extract_json("\n".join(lines))
        except ValueError:
            return
        if isinstance(plan, dict) and "final_answer" in plan:
            records[task_id]["final_answer"] = plan["final_answer"]

    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = LINE.match(line)
            if not m:
                if pending is not None:
                    pending[1].append(line)
                continue
            flush_planner_output()

            task_id, time, msg = int(m["id"]), m["time"], m["msg"]
            rec = records.setdefault(task_id, new_record(task_id))
            for handler in (_lifecycle, _usage):
                if handler(rec, time, msg):
                    break
            else:
                if msg.startswith(PLANNER_OUTPUT):
                    pending = (task_id, [msg[len(PLANNER_OUTPUT) :]])
        flush_planner_output()

    for rec in records.values():
        rec["step_full"] = rec["steps"] >= step_budget
    return sorted(
        (r for r in records.values() if r["question"] is not None), key=lambda r: r["task_id"]
    )


def _lifecycle(rec: dict, time: str, msg: str) -> bool:
    if m := START.match(msg):
        rec.update(question=m["question"], start_time=time, status="running")
    elif m := END.match(msg):
        rec.update(answer=m["answer"], end_time=time, status="completed")
        if rec["start_time"]:
            elapsed = _parse_time(time) - _parse_time(rec["start_time"])
            rec["duration_seconds"] = round(elapsed.total_seconds(), 3)
    elif m := FAILED.match(msg):
        rec.update(status="failed", fail_reason=m["reason"])
    else:
        return False
    return True


def _usage(rec: dict, time: str, msg: str) -> bool:
    for pattern, fields, count_field in USAGE_PATTERNS:
        if m := pattern.match(msg):
            for field, group in fields.items():
                rec[field] += _number(m[group])
            if count_field:
                rec[count_field] += 1
            return True
    return False


def _number(text: str) -> int | float:
    return float(text) if "." in text else int(text)


def _parse_time(text: str) -> datetime:
    return datetime.strptime(text.strip(), TIME_FORMAT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log_file", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--step-budget", type=int, default=120, help="flag queries that hit it")
    args = parser.parse_args()

    records = parse_log(args.log_file, args.step_budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Parsed {len(records)} tasks -> {args.output}")


if __name__ == "__main__":
    main()
