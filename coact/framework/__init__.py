"""Tool-calling agent framework used by CoAct.

Adapted from FlashOAgents (Flash-Searcher, OPPO PersonalAI), itself derived from
HuggingFace smolagents. Both are licensed under the Apache License, Version 2.0.

Heavy optional dependencies are imported lazily: `vllm`/`transformers` only when a
`VLLMModel` is constructed, and the multimodal stack (`mm_tools`) only when one of the
inspector tools is first accessed.
"""

from typing import TYPE_CHECKING

from .agents import ToolCallingAgent
from .models import OpenAIServerModel, VLLMModel
from .monitoring import LogLevel
from .search_tools import CrawlPageTool, WebSearchTool, WikiSearchTool
from .tools import FinalAnswerTool, Tool

if TYPE_CHECKING:
    from .mm_tools import AudioInspectorTool, TextInspectorTool, VisualInspectorTool

_LAZY_MM_TOOLS = {"VisualInspectorTool", "TextInspectorTool", "AudioInspectorTool"}


def __getattr__(name: str):
    # Defer importing `mm_tools`, which requires the optional multimodal dependencies.
    if name in _LAZY_MM_TOOLS:
        from . import mm_tools

        return getattr(mm_tools, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ToolCallingAgent",
    "Tool",
    "FinalAnswerTool",
    "OpenAIServerModel",
    "VLLMModel",
    "LogLevel",
    "WebSearchTool",
    "CrawlPageTool",
    "WikiSearchTool",
    "VisualInspectorTool",
    "TextInspectorTool",
    "AudioInspectorTool",
]
