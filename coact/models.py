"""Factories for the LLM, the sub-agent toolset and the embedding client."""

from __future__ import annotations

import os
from collections.abc import Callable

from openai import OpenAI

from coact import framework
from coact.framework.models import Model, OpenAIServerModel, VLLMModel
from coact.framework.tools import Tool

DEFAULT_VLLM_MODEL = "Qwen/Qwen3-32B"
MAX_COMPLETION_TOKENS = 16384
INSPECTOR_TEXT_LIMIT = 50000


def build_model(backend: str, model_id: str | None = None, tensor_parallel_size: int = 2) -> Model:
    """Create the model shared by the planner and all sub-agents of a run."""
    if backend == "api":
        return OpenAIServerModel(
            model_id=model_id or os.environ["OPENAI_MODEL"],
            api_base=os.environ["OPENAI_URL"],
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
    if backend == "vllm":
        model_id = model_id or DEFAULT_VLLM_MODEL
        return VLLMModel(
            model_id=model_id,
            tokenizer_name=model_id,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            enable_thinking=False,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.9,
            dtype="auto",
            max_model_len=30000,
            trust_remote_code=True,
            enforce_eager=False,
        )
    raise ValueError(f"Unknown backend: {backend!r}")


# Inspector tools are resolved through `framework` lazily, so text-only benchmarks run
# without the multimodal extras installed.
_TOOL_FACTORIES: dict[str, Callable[[Model], Tool]] = {
    "wiki_search": lambda model: framework.WikiSearchTool(),
    "web_search": lambda model: framework.WebSearchTool(),
    "crawl_page": lambda model: framework.CrawlPageTool(model=model),
    "final_answer": lambda model: framework.FinalAnswerTool(),
    "visual_inspector": lambda model: framework.VisualInspectorTool(model, INSPECTOR_TEXT_LIMIT),
    "text_inspector": lambda model: framework.TextInspectorTool(model, INSPECTOR_TEXT_LIMIT),
    "audio_inspector": lambda model: framework.AudioInspectorTool(model, INSPECTOR_TEXT_LIMIT),
}


def build_tools(names: tuple[str, ...], model: Model) -> list[Tool]:
    return [_TOOL_FACTORIES[name](model) for name in names]


def embed(texts: list[str]) -> list[list[float]]:
    """Embed each text with the configured OpenAI-compatible embedding endpoint."""
    client = OpenAI(api_key=os.environ["EMBEDDING_API_KEY"], base_url=os.environ["EMBEDDING_URL"])
    model = os.environ["EMBEDDING_MODEL"]
    return [client.embeddings.create(input=text, model=model).data[0].embedding for text in texts]
