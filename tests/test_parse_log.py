from pathlib import Path

from coact.evaluation.parse_log import parse_log

FIXTURE = Path(__file__).parent / "fixtures" / "sample_run.log"


def test_completed_task_statistics():
    task = parse_log(FIXTURE)[0]
    assert task["question"] == "Who wrote X?"
    assert task["answer"] == "Alice B."
    assert task["final_answer"] == "Alice"  # last planner output wins
    assert task["status"] == "completed"
    assert task["duration_seconds"] == 6.0
    assert (task["llm_calls"], task["total_tokens"], task["llm_duration_seconds"]) == (2, 350, 3.5)
    assert (task["steps"], task["serper_credits"], task["jina_tokens"]) == (2, 1, 555)
    assert (task["center_to_sub_bytes"], task["sub_to_center_bytes"]) == (300, 900)


def test_failed_task_and_tool_usage():
    task = parse_log(FIXTURE)[1]
    assert task["status"] == "completed"  # "End task" follows the failure line
    assert task["fail_reason"] == "Max rounds reached"
    assert task["text_inspector_total_tokens"] == 30
    assert task["visual_inspector_total_tokens"] == 11
    assert task["visual_inspector_duration"] == 2.5
    assert (task["whisper_tokens"], task["whisper_duration"]) == (40, 1.25)
    assert (task["audio_inspector_tokens"], task["audio_inspector_duration"]) == (70, 3.5)


def test_step_budget_flag():
    assert parse_log(FIXTURE, step_budget=2)[0]["step_full"] is True
