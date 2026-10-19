"""Run configuration and per-benchmark presets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Canonical tool order; sub-agent tool lists are always rendered in this order.
TOOL_NAMES = (
    "wiki_search",
    "web_search",
    "crawl_page",
    "final_answer",
    "visual_inspector",
    "text_inspector",
    "audio_inspector",
)


@dataclass(frozen=True)
class BenchmarkPreset:
    tools: tuple[str, ...]
    max_steps: int = 7


_TEXT_TOOLS = ("web_search", "crawl_page")

PRESETS: dict[str, BenchmarkPreset] = {
    "hotpot": BenchmarkPreset(_TEXT_TOOLS),
    "frames": BenchmarkPreset(_TEXT_TOOLS),
    "deepsearch": BenchmarkPreset((*_TEXT_TOOLS, "visual_inspector"), max_steps=12),
    "hle": BenchmarkPreset(("wiki_search", *_TEXT_TOOLS, "visual_inspector")),
    "gaia": BenchmarkPreset(
        ("wiki_search", *_TEXT_TOOLS, "visual_inspector", "text_inspector", "audio_inspector")
    ),
}


@dataclass
class RunConfig:
    """Everything needed to run CoAct over one dataset file."""

    benchmark: str
    input_file: Path
    output_file: Path
    backend: str = "api"  # "api" (OpenAI-compatible endpoint) or "vllm" (local inference)
    model: str | None = None
    scheduler: str = "semantic"  # "semantic" (embedding dissimilarity) or "random" (ablation)
    num_agents: int = 3
    max_rounds: int = 10  # planning rounds per query
    step_budget: int = 120  # total sub-agent steps per query before forcing an answer
    concurrency: int = 5  # queries processed in parallel
    tensor_parallel_size: int = 2
    seed: int = 0
    tools: tuple[str, ...] = field(default=())
    max_steps: int = 0  # tool-calling steps per sub-task

    def __post_init__(self) -> None:
        preset = PRESETS[self.benchmark]
        if not self.tools:
            self.tools = preset.tools
        if not self.max_steps:
            self.max_steps = preset.max_steps
        unknown = set(self.tools) - set(TOOL_NAMES)
        if unknown:
            raise ValueError(f"Unknown tools: {sorted(unknown)}; choose from {TOOL_NAMES}")
        self.tools = tuple(name for name in TOOL_NAMES if name in {*self.tools, "final_answer"})
