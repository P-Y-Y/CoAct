#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. PersonalAI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Dict, Any, Optional, Tuple
import os
import requests
import json
import time
import threading
import logging
import contextvars
from .tools import Tool
from .models import OpenAIServerModel

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

logger = logging.getLogger(__name__)


# ============================================================================
# Per-query Jina / Serper stats via contextvars.
# contextvars propagate into ThreadPoolExecutor workers, so even when tools run in a
# worker thread the usage is attributed to the originating query.
# ============================================================================

# Context variable holding the current query ID
_current_query_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_current_query_id", default=None
)

# Per-query stats:{query_id: {"jina_tokens": int, "jina_calls": int, "jina_duration_sec": float, ...}}
_query_usage: Dict[str, Dict[str, Any]] = {}
_query_usage_lock = threading.Lock()

# Global stats (thread-safe)
_jina_usage_lock = threading.Lock()
_jina_total_tokens = 0
_jina_total_calls = 0
_jina_total_duration_sec = 0.0

_serper_usage_lock = threading.Lock()
_serper_total_credits = 0
_serper_total_calls = 0
_serper_total_duration_sec = 0.0

# Communication stats (global + per-query)
_comm_usage_lock = threading.Lock()
_comm_total_input_bytes = 0  # central agent -> tool bytes
_comm_total_output_bytes = 0  # tool -> central agent bytes
_comm_total_calls = 0


def set_current_query_id(query_id: str) -> contextvars.Token:
    """
    Set the current query ID; returns a token for restoring it later.
    Call at the start of process_item.
    """
    return _current_query_id.set(query_id)


def reset_current_query_id(token: contextvars.Token) -> None:
    """Restore the previous query ID."""
    _current_query_id.reset(token)


def get_query_usage(query_id: str) -> Dict[str, Any]:
    """Get Jina / Serper / mm_tools / communication stats (incl. timing) for a query."""
    with _query_usage_lock:
        usage = _query_usage.get(query_id)
        if usage is None:
            return {
                # search_tools stats
                "jina_tokens": 0,
                "jina_calls": 0,
                "jina_duration_sec": 0.0,
                "serper_credits": 0,
                "serper_calls": 0,
                "serper_duration_sec": 0.0,
                "comm_input_bytes": 0,
                "comm_output_bytes": 0,
                "comm_calls": 0,
                # mm_tools stats
                "visual_tokens": 0,
                "visual_calls": 0,
                "visual_duration_sec": 0.0,
                "text_tokens": 0,
                "text_calls": 0,
                "text_duration_sec": 0.0,
                "audio_tokens": 0,
                "audio_calls": 0,
                "audio_duration_sec": 0.0,
            }
        return dict(usage)


def update_comm_usage(input_bytes: int, output_bytes: int) -> None:
    """Update communication stats (global + per-query). Public helper."""
    global _comm_total_input_bytes, _comm_total_output_bytes, _comm_total_calls
    
    input_val = input_bytes if isinstance(input_bytes, int) and input_bytes > 0 else 0
    output_val = output_bytes if isinstance(output_bytes, int) and output_bytes > 0 else 0
    
    # Update global stats
    with _comm_usage_lock:
        _comm_total_calls += 1
        _comm_total_input_bytes += input_val
        _comm_total_output_bytes += output_val
    
    # Update per-query stats
    query_id = _current_query_id.get()
    if query_id is not None:
        with _query_usage_lock:
            _init_query_usage(query_id)
            _query_usage[query_id]["comm_calls"] += 1
            _query_usage[query_id]["comm_input_bytes"] += input_val
            _query_usage[query_id]["comm_output_bytes"] += output_val


def get_total_comm_usage() -> Dict[str, Any]:
    """Return global communication stats."""
    with _comm_usage_lock:
        return {
            "total_input_bytes": _comm_total_input_bytes,
            "total_output_bytes": _comm_total_output_bytes,
            "total_calls": _comm_total_calls,
        }


def _init_query_usage(query_id: str) -> None:
    """Initialize the per-query stats dict (internal helper).
    Uses setdefault so all fields exist even if another module created the entry first.
    """
    usage = _query_usage.setdefault(query_id, {})
    # search_tools stats
    usage.setdefault("jina_tokens", 0)
    usage.setdefault("jina_calls", 0)
    usage.setdefault("jina_duration_sec", 0.0)
    usage.setdefault("serper_credits", 0)
    usage.setdefault("serper_calls", 0)
    usage.setdefault("serper_duration_sec", 0.0)
    usage.setdefault("comm_input_bytes", 0)
    usage.setdefault("comm_output_bytes", 0)
    usage.setdefault("comm_calls", 0)
    # mm_tools stats (ensure present; updated by mm_tools)
    usage.setdefault("visual_tokens", 0)
    usage.setdefault("visual_calls", 0)
    usage.setdefault("visual_duration_sec", 0.0)
    usage.setdefault("text_tokens", 0)
    usage.setdefault("text_calls", 0)
    usage.setdefault("text_duration_sec", 0.0)
    usage.setdefault("audio_tokens", 0)
    usage.setdefault("audio_calls", 0)
    usage.setdefault("audio_duration_sec", 0.0)


def _update_jina_usage(tokens: Optional[int], duration_sec: float = 0.0) -> None:
    """Update Jina usage stats (global + per-query), incl. call time."""
    global _jina_total_tokens, _jina_total_calls, _jina_total_duration_sec
    
    tokens_val = tokens if isinstance(tokens, int) and tokens > 0 else 0
    duration_val = duration_sec if isinstance(duration_sec, (int, float)) and duration_sec > 0 else 0.0
    
    # Update global stats
    with _jina_usage_lock:
        _jina_total_calls += 1
        _jina_total_tokens += tokens_val
        _jina_total_duration_sec += duration_val
    
    # Update per-query stats
    query_id = _current_query_id.get()
    logger.debug(f"[_update_jina_usage] query_id={query_id}, tokens={tokens_val}, thread={threading.get_ident()}")
    if query_id is not None:
        with _query_usage_lock:
            _init_query_usage(query_id)
            _query_usage[query_id]["jina_calls"] += 1
            _query_usage[query_id]["jina_tokens"] += tokens_val
            _query_usage[query_id]["jina_duration_sec"] += duration_val
    else:
        logger.warning(f"[_update_jina_usage] query_id is None! tokens={tokens_val} not attributed to any query")


def _update_serper_usage(credits: Optional[int], duration_sec: float = 0.0) -> None:
    """Update Serper usage stats (global + per-query), incl. call time."""
    global _serper_total_credits, _serper_total_calls, _serper_total_duration_sec
    
    credits_val = credits if isinstance(credits, int) and credits > 0 else 0
    duration_val = duration_sec if isinstance(duration_sec, (int, float)) and duration_sec > 0 else 0.0
    
    # Update global stats
    with _serper_usage_lock:
        _serper_total_calls += 1
        _serper_total_credits += credits_val
        _serper_total_duration_sec += duration_val
    
    # Update per-query stats
    query_id = _current_query_id.get()
    logger.debug(f"[_update_serper_usage] query_id={query_id}, credits={credits_val}, thread={threading.get_ident()}")
    if query_id is not None:
        with _query_usage_lock:
            _init_query_usage(query_id)
            _query_usage[query_id]["serper_calls"] += 1
            _query_usage[query_id]["serper_credits"] += credits_val
            _query_usage[query_id]["serper_duration_sec"] += duration_val
    else:
        logger.warning(f"[_update_serper_usage] query_id is None! credits={credits_val} not attributed to any query")


def get_total_jina_usage() -> Dict[str, Any]:
    """Return global Jina usage stats (incl. timing)."""
    with _jina_usage_lock:
        return {
            "total_tokens": _jina_total_tokens,
            "total_calls": _jina_total_calls,
            "total_duration_sec": round(_jina_total_duration_sec, 3),
        }


def get_total_serper_usage() -> Dict[str, Any]:
    """Return global Serper usage stats (credits + timing)."""
    with _serper_usage_lock:
        return {
            "total_credits": _serper_total_credits,
            "total_calls": _serper_total_calls,
            "total_duration_sec": round(_serper_total_duration_sec, 3),
        }


def read_page(url: str) -> str:
    """Read and return the content of a webpage using Jina reader."""
    jina_url = f'https://r.jina.ai/{url}'
    # Mirror the plain curl call: only Authorization,
    # plus Accept: application/json so usage.tokens can be parsed
    headers = {
        'Authorization': f'Bearer {os.getenv("JINA_API_KEY")}',
        'Accept': 'application/json',
    }

    start_time = time.perf_counter()
    try:
        response = requests.get(jina_url, headers=headers, timeout=15)
        duration_sec = time.perf_counter() - start_time
        response.raise_for_status()

        approx_tokens = 0
        content_text = ""

        # Prefer JSON and read tokens from the usage field
        try:
            data = response.json()
            # Two response shapes are handled:
            # 1) {"data": { "content": "...", "usage": {"tokens": ...}}, "meta": {"usage": {"tokens": ...}}}
            # 2) {"data": "...", "usage": {...}} or {"content": "...", "usage": {...}}
            if isinstance(data, dict):
                usage_dict: Optional[Dict[str, Any]] = None

                # data may be a dict or the content string itself
                data_node = data.get("data")
                if isinstance(data_node, dict):
                    if isinstance(data_node.get("content"), str):
                        content_text = data_node["content"]
                    usage_dict = data_node.get("usage") or usage_dict
                elif isinstance(data_node, str):
                    content_text = data_node

                # top-level content / usage
                if not content_text and isinstance(data.get("content"), str):
                    content_text = data["content"]
                if usage_dict is None:
                    usage_dict = data.get("usage") or usage_dict

                # fall back to meta.usage
                meta = data.get("meta")
                if isinstance(meta, dict) and usage_dict is None:
                    usage_dict = meta.get("usage") or usage_dict

                # tokens from usage (Jina example: {"usage": {"tokens": 29}})
                if isinstance(usage_dict, dict):
                    for k in ("tokens", "output_tokens", "total_tokens"):
                        v = usage_dict.get(k)
                        if isinstance(v, int) and v > 0:
                            approx_tokens = v
                            break
            # Unexpected JSON shape: fall back to text-based estimate
            if not content_text:
                content_text = response.text
        except Exception:
            # JSON parse failed: use raw text
            content_text = response.text

        # No tokens in usage: estimate from content
        if approx_tokens <= 0:
            try:
                approx_tokens = max(1, len(content_text.split()))
            except Exception:
                approx_tokens = 0

        _update_jina_usage(approx_tokens, duration_sec)
        logger.info("[jina] url=%s tokens=%s duration_sec=%.3f", url, approx_tokens, duration_sec)
        return content_text
    except requests.RequestException as e:
        duration_sec = time.perf_counter() - start_time
        # Count the call even on failure, with 0 tokens
        _update_jina_usage(0, duration_sec)
        logger.info("[jina] url=%s tokens=%s duration_sec=%.3f error=%s", url, 0, duration_sec, str(e))
        return f"Error reading page: {str(e)}"

def web_search_google_serper(
    query: str, 
    filter_year: Optional[int] = None, 
    serp_num: int = 3, 
    max_retries: int = 3
) -> Tuple[List[Dict[str, Any]], str]:
    """Perform web search using Google Serper API."""
    if not query.strip():
        return [], "Query is empty. Please provide a valid search query."
    
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query,
        "location": "United States",
        "num": serp_num
    })
    headers = {
        'X-API-KEY': os.getenv("SERPER_API_KEY"),
        'Content-Type': 'application/json'
    }

    for attempt in range(max_retries):
        start_time = time.perf_counter()
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=10)
            duration_sec = time.perf_counter() - start_time
            response.raise_for_status()
            raw_text = response.text
            # Serper responses include "credits"; use them as the usage metric
            results = json.loads(raw_text)
            credits_val = results.get("credits", 0)
            try:
                credits = int(credits_val) if credits_val is not None else 0
            except Exception:
                credits = 0
            _update_serper_usage(credits, duration_sec)
            logger.info("[serper] query=%s credits=%s duration_sec=%.3f", query, credits, duration_sec)

            if "organic" not in results or not results["organic"]:
                year_filter_msg = f" with year filter={filter_year}" if filter_year else ""
                return [], f"No results found for '{query}'{year_filter_msg}. Try a more general query."
            
            search_results = []
            for idx, page in enumerate(results["organic"], 1):
                search_results.append({
                    "idx": idx,
                    "title": page.get("title", "No title"),
                    "date": f"\nDate published: {page['date']}" if "date" in page else "",
                    "snippet": f"\n{page.get('snippet', 'No snippet')}",
                    "source": f"\nSource: {page.get('source', 'Unknown source')}",
                    "link": page.get('link', '#')
                })
            
            return search_results, ""
        
        except (requests.RequestException, json.JSONDecodeError) as e:
            duration_sec = time.perf_counter() - start_time
            # Count the call even on failure, with 0 credits
            _update_serper_usage(0, duration_sec)
            logger.info("[serper] query=%s credits=%s duration_sec=%.3f error=%s", query, 0, duration_sec, str(e))
            if attempt == max_retries - 1:
                return [], f"Search failed after {max_retries} attempts: {str(e)}"
            time.sleep(1)
    
    return [], "Unexpected error in web search"

class WikiSearchTool(Tool):
    name = "wiki_search"
    description = "Retrieve relevant knowledge from Wikipedia and return the search results."
    inputs = {
        "query": {
            "type": "string", 
            "description": "Provide a query string for the information you want to retrieve from Wikipedia."
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.tool_name = "wiki_search"

    def forward(self, query: str) -> str:
        """Execute Wikipedia search and return formatted results."""
        base_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts|info",
            "exintro": True,
            "explaintext": True,
            "titles": query,
            "redirects": 1,
            "inprop": "url"
        }

        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'error' in data:
                error_info = data['error']
                return f"Wikipedia API error: {error_info.get('code', 'unknown')} - {error_info.get('info', 'unknown')}"

            pages = data.get("query", {}).get("pages", {})
            results = []
            
            for page_id, page_info in pages.items():
                if int(page_id) < 0:  # Skip invalid pages
                    continue
                    
                title = page_info.get("title", "Unknown Title")
                extract = page_info.get("extract", "No extract available")
                page_url = page_info.get("fullurl", "No URL available")
                
                results.append(
                    f"[{title}]({page_url})\n"
                    f"Summary: {extract[:500]}{'...' if len(extract) > 500 else ''}"
                )

            return "\n\n".join(results) if results else f"No relevant information found for: {query}"
        
        except requests.Timeout:
            return "Request to Wikipedia API timed out. Please try again later."
        except requests.RequestException as e:
            return f"Network error occurred: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

class WebSearchTool(Tool):
    name = "web_search"
    description = "Perform a web search query and return the search results."
    inputs = {
        "query": {
            "type": "string", 
            "description": "The web search query to perform."
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.tool_name = "web_search"

    def forward(self, query: str) -> str:
        """Execute web search and return formatted results."""
        search_results, error_msg = web_search_google_serper(query, serp_num=5)
        
        if error_msg:
            return error_msg
        
        formatted_results = []
        for result in search_results:
            formatted_results.append(
                f"{result['idx']}. [{result['title']}]({result['link']})"
                f"{result['date']}{result['source']}\n"
                f"   {result['snippet'].strip()}"
            )
        
        return "\n\n".join(formatted_results) if formatted_results else "No search results found"

class CrawlPageTool(Tool):
    name = "crawl_page"
    description = "Access webpage using the provided URL and extract relevant content.  Please make full use of this tool to verify the accuracy of the searched content."
    inputs = {
        "url": {
            "type": "string",
            "description": "The URL of the webpage to visit."
        },
        "query": {
            "type": "string",
            "description": "The specific information to extract from the webpage."
        }
    }
    output_type = "string"
    
    def __init__(self, model: OpenAIServerModel):
        super().__init__()
        self.tool_name = "crawl_page"
        self.model = model

    @staticmethod
    def truncate_text(text: str, max_length: int = 60000) -> str:
        """Truncate text to specified length."""
        return text if len(text) <= max_length else text[:max_length] + "...(truncated)"

    def get_summary_prompt(self, query: str, url: str, content: str) -> str:
        """Generate prompt for content summarization."""
        return (
            f"Task: Extract all content from the web page that matches the search query.\n"
            f"Search Query: {query}\n\n"
            f"Web Page Content [url:{url}]:\n{content}\n\n"
            "Instructions:\n"
            "- Summarize all relevant content for the query (text, tables, lists) into concise points\n"
            "- If no relevant information exists, please straightly output 'No relevant information'\n"
            "- Keep the summary under 500 words"
        )

    def retry_predict(self, prompt: str, max_retries: int = 3) -> str:
        """Retry model prediction with exponential backoff."""
        messages = [{"role": "user", "content": prompt}]
        
        for attempt in range(max_retries):
            try:
                response = self.model(messages)
                if hasattr(response, 'content'):
                    content = response.content
                    return content.strip() if isinstance(content, str) else str(content)
                return str(response)
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Content extraction failed: {str(e)}"
                wait_time = 2 ** attempt
                time.sleep(wait_time)
        
        return "Content extraction failed after multiple attempts"

    def forward(self, url: str, query: str) -> str:
        """Crawl webpage and extract relevant content."""
        # Validate URL
        if not url.startswith(('http://', 'https://')):
            return "Invalid URL format. Must start with http:// or https://"
        
        page_content = read_page(url)
        if page_content.startswith("Error"):
            return page_content
        
        truncated_content = self.truncate_text(page_content)
        prompt = self.get_summary_prompt(query, url, truncated_content)
        
        return self.retry_predict(prompt)
    
__all__ = [
    "WikiSearchTool",
    "WebSearchTool",
    "CrawlPageTool",
    "get_total_jina_usage",
    "get_total_serper_usage",
    "set_current_query_id",
    "reset_current_query_id",
    "get_query_usage",
    "update_comm_usage",
    "get_total_comm_usage",
]
