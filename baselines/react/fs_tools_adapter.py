from __future__ import annotations
import os
import json
import time
import logging
import threading
from typing import Callable, Dict, Any, Optional, Tuple, List
import requests

from .llm import make_llm, OpenAILLM
from .mm_tools import (
    VisualInspectorTool, 
    TextInspectorTool, 
    AudioInspectorTool,
    reset_mm_thread_usage  # [Modified]
)

# Logger
logger = logging.getLogger(__name__)

# =============================================================================
# Tracker Logic (thread-safe usage tracking)
# =============================================================================

# Per-thread (approximate) Jina / Serper token usage, so concurrent queries are tracked separately
_JINA_USAGE_BY_THREAD: Dict[int, Dict[str, Any]] = {}
_SERPER_USAGE_BY_THREAD: Dict[int, Dict[str, Any]] = {}
_TOOL_TIME_BY_THREAD: Dict[int, Dict[str, Any]] = {}

def _update_usage(store: Dict[int, Dict[str, Any]], tokens: Optional[int], time_sec: float = 0.0) -> None:
    """
    Accumulate tokens, call count and time_sec for the current thread.
    """
    try:
        tid = threading.get_ident()
    except Exception:
        return
    
    # Needs locking if dict modification isn't atomic, but in CPython GIL protects dict ops usually.
    # Safe init
    if tid not in store:
        store[tid] = {"total_tokens": 0, "total_calls": 0, "total_time": 0.0}
    
    usage = store[tid]
    if isinstance(tokens, int) and tokens > 0:
        usage["total_tokens"] += tokens
    usage["total_calls"] += 1
    # Accumulate elapsed time
    if time_sec > 0:
        usage["total_time"] = usage.get("total_time", 0.0) + time_sec

def _record_tool_time(tool_name: str, duration: float):
    try:
        tid = threading.get_ident()
        if tid not in _TOOL_TIME_BY_THREAD:
             _TOOL_TIME_BY_THREAD[tid] = {"total_time": 0.0, "per_tool": {}}
        
        entry = _TOOL_TIME_BY_THREAD[tid]
        entry["total_time"] += duration
        
        per = entry["per_tool"]
        if tool_name not in per:
            per[tool_name] = 0.0
        per[tool_name] += duration
    except Exception:
        pass

def get_thread_jina_usage(thread_id: Optional[int] = None) -> Dict[str, Any]:
    """Approximate Jina token usage for the given thread."""
    try:
        if thread_id is None:
            thread_id = threading.get_ident()
    except Exception:
        return {}
    usage = _JINA_USAGE_BY_THREAD.get(thread_id)
    return dict(usage) if usage is not None else {}

def get_thread_serper_usage(thread_id: Optional[int] = None) -> Dict[str, Any]:
    """Approximate Serper token usage for the given thread."""
    try:
        if thread_id is None:
            thread_id = threading.get_ident()
    except Exception:
        return {}
    usage = _SERPER_USAGE_BY_THREAD.get(thread_id)
    return dict(usage) if usage is not None else {}

def get_thread_tool_time(thread_id: Optional[int] = None) -> Dict[str, Any]:
    try:
        if thread_id is None:
            thread_id = threading.get_ident()
    except Exception:
        return {}
    return dict(_TOOL_TIME_BY_THREAD.get(thread_id, {}))

def reset_thread_usage():
    """Clear usage for current thread (useful before starting a new agent run)."""
    try:
        tid = threading.get_ident()
        if tid in _JINA_USAGE_BY_THREAD:
             del _JINA_USAGE_BY_THREAD[tid]
        if tid in _SERPER_USAGE_BY_THREAD:
             del _SERPER_USAGE_BY_THREAD[tid]
        if tid in _TOOL_TIME_BY_THREAD:
             del _TOOL_TIME_BY_THREAD[tid]
        
        # Reset Multi-modal tools logic
        reset_mm_thread_usage()
        
    except Exception:
        pass


class ToolWrapper:
    def __init__(self, name: str, description: str, call: Callable[[Any], str]):
        self.name = name
        self.description = description
        self._call = call

    def func(self, action_input: Any) -> str:
        # Catch-all so a crashing tool does not kill the agent
        t0 = time.time()
        # Enable tool mode for LLM tracking within this thread
        OpenAILLM.set_tool_mode(True)
        try:
            return self._call(action_input)
        except Exception as e:
            logger.error(f"🚨 Tool '{self.name}' UNHANDLED EXCEPTION: {e}", exc_info=True)
            return f"Tool execution failed with internal error: {e}"
        finally:
            OpenAILLM.set_tool_mode(False)
            duration = time.time() - t0
            _record_tool_time(self.name, duration)

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolWrapper] = {}

    def register(self, tool: ToolWrapper):
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolWrapper | None:
        return self._tools.get(name)

    def names(self):
        return list(self._tools.keys())


def _serper_search(
    query: str,
    serp_num: int = 5,
    max_retries: int = 3,
) -> Tuple[List[Dict[str, Any]], str]:
    if not query or not query.strip():
        return [], "Query is empty. Please provide a valid search query."
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "location": "United States", "num": serp_num})
    headers = {
        "X-API-KEY": os.getenv("SERPER_API_KEY", ""),
        "Content-Type": "application/json",
    }
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            resp = requests.post(url, headers=headers, data=payload, timeout=10)
            resp.raise_for_status()
            t1 = time.time()
            
            # --- Usage Tracking ---
            raw_text = resp.text
            data = json.loads(raw_text)
            
            credits_val = data.get("credits", 0)
            try:
                credits = int(credits_val) if credits_val is not None else 0
            except Exception:
                credits = 0
            _update_usage(_SERPER_USAGE_BY_THREAD, credits, time_sec=(t1 - t0))
            logger.info("[serper] query=%s credits=%s time=%.2fs", query, credits, (t1 - t0))
            # ----------------------

            organic = data.get("organic", [])
            if not organic:
                logger.warning(f"Serper returned no organic results for query: {query}")
                return [], f"No results found for '{query}'. Try a more general query."
            results = []
            for idx, page in enumerate(organic, 1):
                results.append(
                    {
                        "idx": idx,
                        "title": page.get("title", "No title"),
                        "date": f"\nDate published: {page['date']}" if "date" in page else "",
                        "snippet": f"\n{page.get('snippet', 'No snippet')}",
                        "source": f"\nSource: {page.get('source', 'Unknown source')}",
                        "link": page.get("link", "#"),
                    }
                )
            return results, ""
        except (requests.RequestException, json.JSONDecodeError) as e:
            # On failure still count the call, with 0 tokens
            _update_usage(_SERPER_USAGE_BY_THREAD, 0)
            
            # Log search failure
            logger.warning(f"Serper API failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logger.error(f"🚨 Serper API gave up after {max_retries} attempts. Query: {query}")
                return [], f"Search failed after {max_retries} attempts: {e}"
            time.sleep(1)
    return [], "Unexpected error in web search"


def _web_search_call(action_input: Any) -> str:
    query = action_input if isinstance(action_input, str) else action_input.get("query", "")
    results, err = _serper_search(query, serp_num=5)
    if err:
        return err
    lines = []
    for r in results:
        lines.append(
            f"{r['idx']}. [{r['title']}]({r['link']}){r['date']}{r['source']}\n   {r['snippet'].strip()}"
        )
    return "\n\n".join(lines) if lines else "No search results found"


def _wiki_search_call(action_input: Any) -> str:
    query = action_input if isinstance(action_input, str) else action_input.get("query", "")
    if not query.strip():
        return "Empty query"
    base_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|info",
        "exintro": True,
        "explaintext": True,
        "titles": query,
        "redirects": 1,
        "inprop": "url",
    }
    try:
        # Add a polite User-Agent to avoid 403 from some endpoints
        headers = {
            "User-Agent": os.getenv("WIKI_USER_AGENT", "ParaReAct/1.0 (+https://example.com)")
        }
        t0 = time.time()
        resp = requests.get(base_url, params=params, headers=headers, timeout=10)
        t1 = time.time()
        # record wiki call elapsed into tool usage store for diagnostics
        try:
            _record_tool_time("wiki_search", t1 - t0)
        except Exception:
            pass
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            err = data["error"]
            err_msg = f"{err.get('code','unknown')} - {err.get('info','unknown')}"
            logger.error(f"🚨 Wikipedia API Error: {err_msg}")
            return f"Wikipedia API error: {err_msg}"
        pages = data.get("query", {}).get("pages", {})
        out = []
        for pid, page in pages.items():
            try:
                if int(pid) < 0:
                    continue
            except Exception:
                pass
            title = page.get("title", "Unknown Title")
            extract = page.get("extract", "No extract available")
            url = page.get("fullurl", "No URL available")
            out.append(f"[{title}]({url})\nSummary: {extract[:500]}{'...' if len(extract)>500 else ''}")
        return "\n\n".join(out) if out else f"No relevant information found for: {query}"
    except requests.Timeout:
        logger.error("🚨 Wikipedia Timeout")
        return "Request to Wikipedia API timed out. Please try again later."
    except requests.RequestException as e:
        logger.error(f"🚨 Wikipedia Network Error: {e}")
        return f"Network error occurred: {e}"
    except Exception as e:
        logger.error(f"🚨 Wikipedia Unexpected Error: {e}")
        return f"Unexpected error: {e}"


def _read_page(url: str) -> str:
    jina_url = f"https://r.jina.ai/{url}"
    headers = {
        "Authorization": f"Bearer {os.getenv('JINA_API_KEY','')}",
        "X-Engine": "browser",
        "X-Return-Format": "markdown",
        "X-Remove-Selector": "header, .class, #id",
        "X-Retain-Images": "none",
        "X-Timeout": "60", # Increased timeout
        "X-Token-Budget": "80000",
        "Accept": "application/json", # Request JSON to get token usage
    }
    
    # Retry strategy for Jina
    max_retries = 3
    base_delay = 5
    
    for attempt in range(max_retries):
        start = time.time()
        try:
            # Increase read timeout to accompany X-Timeout
            resp = requests.get(jina_url, headers=headers, timeout=65)
            
            # Special handling for 409 Conflict (Jina overload)
            if resp.status_code == 409:
                logger.warning(f"⚠️ Jina 409 Conflict (Resource Overload) for {url}, attempt {attempt+1}/{max_retries}")
                time.sleep(base_delay * (attempt + 1))
                continue
                
            resp.raise_for_status()

            # --- Usage Extraction & Content Parsing ---
            approx_tokens = 0
            content_text = ""
            try:
                data = resp.json()
                # Try to parse structure: {"data": {"content": "...", "usage": {"tokens": ...}}}
                usage_dict = None
                
                # Check different Jina response shapes
                if isinstance(data, dict):
                    data_node = data.get("data")
                    # Shape 1: data is dict with content inside
                    if isinstance(data_node, dict):
                        content_text = data_node.get("content", "")
                        usage_dict = data_node.get("usage")
                    # Shape 2: data is string (content)
                    elif isinstance(data_node, str):
                        content_text = data_node
                    
                    # Top level usage
                    if usage_dict is None:
                        usage_dict = data.get("usage")
                        
                    # Also check meta.usage
                    if usage_dict is None:
                        meta = data.get("meta")
                        if isinstance(meta, dict):
                            usage_dict = meta.get("usage")
                    
                    # Extract tokens from usage dict
                    if isinstance(usage_dict, dict):
                        for k in ("tokens", "output_tokens", "total_tokens"):
                            v = usage_dict.get(k)
                            if isinstance(v, int) and v > 0:
                                approx_tokens = v
                                break
                    
                    # If content_text wasn't set from data.content, try top level or just resp.text fallback
                    if not content_text and isinstance(data.get("content"), str):
                         content_text = data.get("content")

                if not content_text:
                    content_text = resp.text

            except Exception:
                # Not JSON or parsing failed -> fallback to text and naive count
                content_text = resp.text

            # Fallback estimation if no token count from API
            if approx_tokens <= 0:
                try:
                    approx_tokens = max(1, len(content_text.split()))
                except Exception:
                    approx_tokens = 0
            
            # Record correct usage
            duration = time.time() - start
            _update_usage(_JINA_USAGE_BY_THREAD, approx_tokens, time_sec=duration)
            logger.info("[jina] url=%s tokens=%s time=%.2fs", url, approx_tokens, duration)
            
            return content_text
            # ------------------------------------------
            
        except requests.exceptions.Timeout as e:
            logger.warning(f"⚠️ Jina Request Timeout for {url}, attempt {attempt+1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (attempt + 1))
                continue
            # Tracking failed attempt
            _update_usage(_JINA_USAGE_BY_THREAD, 0)
            return f"Error reading page: Request timed out after {max_retries} attempts."
            
        except requests.RequestException as e:
            # Tracking failed attempt
            _update_usage(_JINA_USAGE_BY_THREAD, 0)
            
            # For 409/5xx we might want to retry, for others (404, 400) maybe not? 
            # But raise_for_status raises HTTPError which is a RequestException
            # If it's a 4xx (except 409/429) usually no point retrying. 
            if isinstance(e, requests.exceptions.HTTPError):
                code = e.response.status_code if e.response else 0
                if code not in [409, 429, 500, 502, 503, 504]:
                    logger.error(f"🚨 Jina Reader Failed for {url}: {e} (Non-retriable)")
                    return f"Error reading page: {e}"
            
            logger.warning(f"⚠️ Jina Reader Error for {url}: {e}, attempt {attempt+1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (attempt + 1))
                continue
            
            logger.error(f"🚨 Jina Reader Failed for {url}: {e}")
            return f"Error reading page: {e}"

    return "Error reading page: Unknown error after retries"


def _truncate(text: str, max_len: int = 20000) -> str:
    return text if len(text) <= max_len else text[:max_len] + "...(truncated)"


def _crawl_page_call_factory(backend: str = "openai", shared_llm=None):
    llm = shared_llm if shared_llm else make_llm(backend)

    def _summarize(url: str, query: str, content: str, max_retries: int = 3) -> str:
        prompt = (
            "Task: Extract all content from the web page that matches the search query.\n"
            f"Search Query: {query}\n\n"
            f"Web Page Content [url:{url}]:\n{content}\n\n"
            "Instructions:\n"
            "- Summarize all relevant content for the query (text, tables, lists) into concise points\n"
            "- If no relevant information exists, please straightly output 'No relevant information'\n"
            "- Keep the summary under 500 words"
        )
        for attempt in range(max_retries):
            try:
                return (llm.generate(prompt) or "").strip()
            except Exception as e:
                # Log summarization failure details (usually an LLM error such as model-not-found or 400)
                logger.warning(f"Crawl summarization failed (attempt {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"🚨 Crawl summarization gave up for {url}. Error: {e}")
                    return f"Content extraction failed: {e}"
                time.sleep(2 ** attempt)

    def _call(action_input: Any) -> str:
        if isinstance(action_input, dict):
            url = action_input.get("url", "")
            query = action_input.get("query", "")
        else:
            url = action_input if isinstance(action_input, str) else ""
            query = ""
        
        if not url.startswith(("http://", "https://")):
            return "Invalid URL format. Must start with http:// or https://"
        
        # 1. Fetch
        page = _read_page(url)
        if page.startswith("Error reading page"):
            return page
            
        # 2. Summarize
        return _summarize(url, query, _truncate(page))

    return _call


def default_registry(backend: str = "openai", shared_llm=None, enable_mm_tools: bool = False) -> ToolRegistry:
    
    reg = ToolRegistry()

    # Hardcoded tool set aligned with Flash-Searcher names and behavior
    reg.register(
        ToolWrapper(
            name="web_search",
            description="Perform a web search query and return the search results (Serper).",
            call=_web_search_call,
        )
    )

    reg.register(
        ToolWrapper(
            name="wiki_search",
            description="Retrieve relevant knowledge from Wikipedia and return summaries.",
            call=_wiki_search_call,
        )
    )

    reg.register(
        ToolWrapper(
            name="crawl_page",
            description="Access webpage via Jina reader and extract relevant content, summarized by LLM.",
            call=_crawl_page_call_factory(backend, shared_llm=shared_llm),
        )
    )

    # Multimodal tool switch
    env_enable = os.getenv("ENABLE_MM_TOOLS", "false").lower() == "true"
    if enable_mm_tools or env_enable:
        mm_llm = shared_llm if shared_llm else make_llm(backend)
        logger.info("Enabling Multi-Modal Tools...")
        
        # Feature Flags for individual tools (default True if MM enabled)
        use_visual = os.getenv("ENABLE_VISUAL_TOOL", "true").lower() == "true"
        use_text   = os.getenv("ENABLE_TEXT_TOOL", "true").lower() == "true"
        use_audio  = os.getenv("ENABLE_AUDIO_TOOL", "true").lower() == "true"
        
        # Visual
        if use_visual:
            vis_tool = VisualInspectorTool(mm_llm)
            reg.register(ToolWrapper(
                name=vis_tool.name,
                description=vis_tool.description,
                call=vis_tool.wrapped_call
            ))
        
        # Text
        if use_text:
            text_tool = TextInspectorTool(mm_llm)
            reg.register(ToolWrapper(
                name=text_tool.name,
                description=text_tool.description,
                call=text_tool.wrapped_call
            ))
        
        # Audio
        if use_audio:
            audio_tool = AudioInspectorTool(mm_llm)
            reg.register(ToolWrapper(
                name=audio_tool.name,
                description=audio_tool.description,
                call=audio_tool.wrapped_call
            ))

    return reg