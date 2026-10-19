#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. PersonalAI team. All rights reserved.

from typing import List, Dict, Any, Optional, Tuple, Callable
import os
import requests
import json
import time
import logging
from .tools import Tool

# Import path adaptation
try:
    from ..models.openai_model import OpenAIModel
except ImportError:
    # Fallback if specific model class import fails
    from ..models import Model as OpenAIModel

logger = logging.getLogger(__name__)

def read_page(url: str, stats_callback: Optional[Callable[[str, Dict[str, int]], None]] = None) -> str:
    """
    Read and return the content of a webpage using Jina reader.
    Supports detailed token usage tracking via callback.
    """
    jina_key = os.getenv("JINA_API_KEY")
    if not jina_key:
        return "❌ [TOOL ERROR] Jina API Key missing."
        
    jina_url = f'https://r.jina.ai/{url}'
    headers = {
        'Authorization': f'Bearer {jina_key}',
        'Accept': 'application/json', # request JSON explicitly to get usage
        'X-Timeout': '20',
        'X-Token-Budget': '50000',
    }

    max_retries = 3
    backoff_factor = 2

    for attempt in range(max_retries):
        start_time = time.time()
        try:
            response = requests.get(jina_url, headers=headers, timeout=30)
            end_time = time.time()
            duration = end_time - start_time
            
            # Retry on 409/429
            if response.status_code in [409, 429]:
                sleep_time = backoff_factor ** (attempt + 1)
                logger.warning(f"[Jina] Hit {response.status_code}, retrying in {sleep_time}s...")
                time.sleep(sleep_time)
                continue
            
            response.raise_for_status()

            # === Jina usage accounting ===
            approx_tokens = 0
            content_text = ""
            
            try:
                data = response.json()
                # Handle different Jina response formats
                if isinstance(data, dict):
                    usage_dict = None
                    
                    # 1. Try data.usage
                    data_node = data.get("data")
                    if isinstance(data_node, dict):
                        content_text = data_node.get("content", "")
                        usage_dict = data_node.get("usage")
                    elif isinstance(data_node, str):
                        content_text = data_node
                    
                    # 2. Try top-level usage
                    if not content_text:
                        content_text = data.get("content", "")
                    if usage_dict is None:
                        usage_dict = data.get("usage")

                    # 3. Parse tokens
                    if isinstance(usage_dict, dict):
                        approx_tokens = usage_dict.get("tokens") or usage_dict.get("total_tokens") or 0
                
                if not content_text:
                    content_text = response.text
            except:
                content_text = response.text
            
            # Fallback: estimate tokens from characters (1 token ~= 4 chars)
            if not approx_tokens:
                approx_tokens = max(1, len(content_text) // 4)

            # === Report usage ===
            if stats_callback:
                stats_callback("jina", {"calls": 1, "tokens": approx_tokens, "time": duration})
            
            # Success log
            logger.info(f"[Jina] Success | URL: {url[:50]}... | Tokens: {approx_tokens} | Time: {duration:.2f}s")

            return content_text

        except requests.RequestException as e:
            if attempt == max_retries - 1:
                end_time = time.time()
                # Record failed calls too, with 0 tokens
                if stats_callback:
                    stats_callback("jina", {"calls": 1, "tokens": 0, "time": end_time - start_time})
                return f"❌ [TOOL ERROR] Jina Reader Failed: {str(e)}"
            time.sleep(backoff_factor ** (attempt + 1))

    return "❌ [TOOL ERROR] Jina Reader Failed: Max retries exceeded"

def web_search_google_serper(
    query: str, 
    filter_year: Optional[int] = None, 
    serp_num: int = 3, 
    max_retries: int = 3,
    stats_callback: Optional[Callable[[str, Dict[str, int]], None]] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """Perform web search using Google Serper API with stats tracking."""
    if not query.strip():
        return [], "Query is empty."
    
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
        start_time = time.time()
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=10)
            end_time = time.time()
            duration = end_time - start_time

            response.raise_for_status()
            results = response.json()
            
            # === Serper usage accounting ===
            credits_used = int(results.get("credits", 1)) # defaults to 1
            if stats_callback:
                stats_callback("serper", {"calls": 1, "credits": credits_used, "time": duration})
            # =====================

            if "organic" not in results or not results["organic"]:
                return [], f"No results found for '{query}'."
            
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
        
        except Exception as e:
            if attempt == max_retries - 1:
                end_time = time.time()
                if stats_callback: stats_callback("serper", {"calls": 1, "credits": 0, "time": end_time - start_time}) # record the failed call
                return [], f"❌ [TOOL ERROR] Search Failed: {str(e)}"
            time.sleep(1)
    
    return [], "Unexpected error"

class WikiSearchTool(Tool):
    name = "wiki_search"
    description = "Retrieve relevant knowledge from Wikipedia."
    inputs = {"query": {"type": "string", "description": "Query string."}}
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.tool_name = "wiki_search"

    def forward(self, query: str) -> str:
        # Wikipedia implementation (standard, free)
        base_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query", "format": "json", "prop": "extracts|info",
            "exintro": True, "explaintext": True, "titles": query, "redirects": 1, "inprop": "url"
        }
        try:
            # Wikipedia API requires a User-Agent
            headers = {
                "User-Agent": "ParaReAct/1.0 (Research Project; contact@example.com)"
            }
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            results = []
            for k, v in pages.items():
                if int(k) < 0: continue
                results.append(f"[{v.get('title')}]({v.get('fullurl')})\nSummary: {v.get('extract', '')[:500]}...")
            return "\n\n".join(results) if results else "No results."
        except Exception as e:
            return f"Wiki Error: {e}"

class WebSearchTool(Tool):
    name = "web_search"
    description = "Perform a web search query and return the search results."
    inputs = {"query": {"type": "string", "description": "The web search query."}}
    output_type = "string"

    # Usage-stats callback
    def __init__(self, stats_callback: Optional[Callable] = None):
        super().__init__()
        self.tool_name = "web_search"
        self.stats_callback = stats_callback

    def forward(self, query: str) -> str:
        search_results, error_msg = web_search_google_serper(query, serp_num=5, stats_callback=self.stats_callback)
        
        if error_msg: return error_msg
        
        formatted_results = []
        for result in search_results:
            formatted_results.append(
                f"{result['idx']}. [{result['title']}]({result['link']})"
                f"{result['date']}{result['source']}\n"
                f"   {result['snippet'].strip()}"
            )
        return "\n\n".join(formatted_results)

class CrawlPageTool(Tool):
    name = "crawl_page"
    description = "Access webpage using the provided URL and extract relevant content."
    inputs = {
        "url": {"type": "string", "description": "The URL to visit."},
        "query": {"type": "string", "description": "Information to extract."}
    }
    output_type = "string"
    
    # Usage-stats callback
    def __init__(self, model: OpenAIModel, stats_callback: Optional[Callable] = None):
        super().__init__()
        self.tool_name = "crawl_page"
        self.model = model
        self.stats_callback = stats_callback

    @staticmethod
    def truncate_text(text: str, max_length: int = 15000) -> str:
        return text if len(text) <= max_length else text[:max_length] + "...(truncated)"

    def get_summary_prompt(self, query: str, url: str, content: str) -> str:
        return (
            f"Task: Extract content matching query.\nQuery: {query}\n\nContent [url:{url}]:\n{content}\n\n"
            "Summarize relevant content under 500 words. If none, say 'No relevant information'."
        )

    def retry_predict(self, prompt: str, max_retries: int = 3) -> str:
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(max_retries):
            try:
                # model is already the wrapped model, so input/output tokens are counted automatically
                responses = self.model.get_response([messages], stream=False)
                response_gen = next(responses)
                content = ""
                for chunk in response_gen:
                    content += chunk.get("text", "")
                return content.strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Model failed: {e}")
                time.sleep(1)
        return "Error"

    def forward(self, url: str, query: str) -> str:
        if not url.startswith(('http://', 'https://')):
            return "Invalid URL"
        
        # Pass the callback through to read_page
        page_content = read_page(url, self.stats_callback)
        
        if "❌" in page_content: return page_content
        
        truncated_content = self.truncate_text(page_content)
        prompt = self.get_summary_prompt(query, url, truncated_content)
        return self.retry_predict(prompt)
    
__all__ = ["WikiSearchTool", "WebSearchTool", "CrawlPageTool"]