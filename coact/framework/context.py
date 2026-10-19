"""Per-query context shared across the framework.

Holds the index of the query currently being processed (used to tag log lines)
and a thread-safe step counter used to enforce a per-query step budget across
all sub-agents working on the same query.
"""

import contextvars
import threading
from collections import defaultdict

# Index of the outer task/query being processed. Set by the orchestrator and
# propagated to worker threads via `contextvars.copy_context()`.
CURRENT_TASK_IDX: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "CURRENT_TASK_IDX", default=None
)

_lock = threading.Lock()
_step_counts: defaultdict[int | None, int] = defaultdict(int)


def task_tag() -> str:
    """Return a log prefix like ``"[task 3] "`` when a task index is set, else ``""``."""
    idx = CURRENT_TASK_IDX.get()
    return f"[task {idx}] " if idx is not None else ""


def _resolve(idx: int | None) -> int | None:
    return CURRENT_TASK_IDX.get(None) if idx is None else idx


def report_step(idx: int | None = None, n: int = 1) -> None:
    """Add ``n`` steps to the counter of query ``idx`` (defaults to ``CURRENT_TASK_IDX``)."""
    key = _resolve(idx)
    with _lock:
        _step_counts[key] += int(n)


def get_steps(idx: int | None = None) -> int:
    """Return the step count of query ``idx`` (defaults to ``CURRENT_TASK_IDX``).

    If no index is given and none is set in the context, return the total across all queries.
    """
    key = _resolve(idx)
    with _lock:
        if key is None:
            return sum(_step_counts.values())
        return _step_counts.get(key, 0)


def reset_steps(idx: int | None = None) -> None:
    """Reset the counter of query ``idx`` (defaults to ``CURRENT_TASK_IDX``), or all counters."""
    key = _resolve(idx)
    with _lock:
        if key is None:
            _step_counts.clear()
        else:
            _step_counts.pop(key, None)


__all__ = ["CURRENT_TASK_IDX", "task_tag", "report_step", "get_steps", "reset_steps"]
