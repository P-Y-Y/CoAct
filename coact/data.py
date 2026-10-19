"""Loading benchmark files and reading questions/answers across their schemas."""

from __future__ import annotations

import json
from pathlib import Path

# Field names used by the supported benchmarks, in lookup order:
# GAIA, DeepSearch, HotpotQA / HLE, FRAMES.
_QUESTION_KEYS = ("Question", "prompt", "question", "Prompt")
_ANSWER_KEYS = ("Final answer", "answer", "Answer")


def load_records(path: Path) -> list[dict]:
    """Load a ``.json`` list (or single object) or a ``.jsonl`` file."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() != ".jsonl":
        try:
            data = json.loads(text)
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            pass  # Fall back to one JSON object per line.
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def question_of(record: dict, input_file: Path) -> str:
    """Build the task text, appending paths of attached files and images.

    Attachments are resolved relative to the dataset file: GAIA's ``file_name`` sits next
    to it, HLE images live in ``<dataset stem>_images/``.
    """
    question = next((record[k] for k in _QUESTION_KEYS if record.get(k)), "")
    if not question:
        return question

    data_dir = input_file.resolve().parent
    if record.get("file_name"):
        question = f"{question.strip()} {data_dir / record['file_name']}"
    image = (record.get("image_preview") or {}).get("path")
    if image:
        question += f"\nimage path：{data_dir / f'{input_file.stem}_images' / image}"
    return question


def answer_of(record: dict) -> str:
    return next((record[k] for k in _ANSWER_KEYS if record.get(k)), "")
