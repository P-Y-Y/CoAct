"""Small helpers shared across the package."""

from __future__ import annotations

import json
import re


def extract_json(text: str) -> dict:
    """Parse the first JSON object in ``text``, tolerating stray LaTeX backslashes."""
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start found")
    candidate = text[start:]
    decoder = json.JSONDecoder()
    try:
        return decoder.raw_decode(candidate)[0]
    except json.JSONDecodeError:
        fixed = re.sub(r"\\([^\"\\/bfnrtu])", r"\\\\\1", candidate)
        return decoder.raw_decode(fixed)[0]
