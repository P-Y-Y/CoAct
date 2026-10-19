from __future__ import annotations
import os
import time
import logging
from typing import Optional, Dict, Any
from openai import OpenAI, BadRequestError, APIStatusError, APIConnectionError, OpenAIError

# Logger
logger = logging.getLogger("LLM_Tracker")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import threading

class LLM:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

class OpenAILLM(LLM):
    """LLM that calls OpenAI-compatible API.
    Supports OpenRouter-style 'provider' routing via extra_body.
    """
    
    # Context manager for tracking tool usage
    _tool_mode = threading.local()

    @classmethod
    def set_tool_mode(cls, enabled: bool):
        cls._tool_mode.enabled = enabled

    @classmethod
    def is_tool_mode(cls) -> bool:
        return getattr(cls._tool_mode, "enabled", False)

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None, api_key: Optional[str] = None, provider: Optional[str] = None):
        # Prefer the explicit api_key, then environment variables
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
            
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("DEFAULT_MODEL", "gpt-5")
        self.provider = provider  # OpenRouter provider name, e.g. "nebius", "deepinfra"

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        self.call_count=0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        # Cumulative time (s) of all calls made by this LLM object
        self.total_call_time = 0.0
        # Cumulative time (s) of calls made from inside tools
        self.total_tool_call_time = 0.0

    def record_usage(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.total_input_tokens += prompt_tokens
        self.total_output_tokens += completion_tokens

    def generate(self, prompt: str) -> str:
        max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
        base_delay = float(os.getenv("LLM_RETRY_DELAY", "2"))

        last_err: Optional[Exception] = None
        self.call_count += 1
        
        for attempt in range(max_retries):
            start_time = time.time()
            try:
                # Build API request arguments
                api_kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                }

                # OpenRouter provider routing: if a provider is given (and it is not
                # the local "vllm" backend), pin requests to that provider.
                if self.provider and self.provider.lower() not in ["vllm"]:
                    api_kwargs["extra_body"] = {
                        "provider": {
                            "only": [self.provider]
                        }
                    }
                elif not self.provider :
                    # No provider given: use OpenRouter's default routing.
                    pass

                resp = self.client.chat.completions.create(**api_kwargs)
                
                # --- Stats and logging ---
                end_time = time.time()
                latency = end_time - start_time
                # Accumulate time spent in this call
                try:
                    self.total_call_time += latency
                    if self.is_tool_mode():
                        self.total_tool_call_time += latency
                except Exception:
                    pass
                content = resp.choices[0].message.content or ""
                
                # [Debug Log] Print Output
                logger.info(f"===== [LLM Output] =====\nContent Length: {len(content)}\nContent Head: {content[:200]}...\n========================")

                usage = resp.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0
                
                # Update totals
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                
                # Log the provider used (helps verify routing)

                provider_tag = self.provider if self.provider else "auto(price)"
                
                log_msg = (
                    f"[LLM] Provider: {provider_tag} | Model: {self.model} | "
                    f"Time: {latency:.2f}s | In: {input_tokens} / Out: {output_tokens}"
                )
                logger.info(log_msg)

                return content

            except APIConnectionError as e:
                last_err = e
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                raise
            except APIStatusError as e:
                last_err = e
                if attempt < max_retries - 1:
                    time.sleep(max(5.0, base_delay * (2 ** attempt)))
                    continue
                raise
            except BadRequestError as e:
                last_err = e
                raise
            except OpenAIError as e:
                last_err = e
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                raise
        
        if last_err:
            raise last_err
        return ""


def make_llm(backend: str) -> LLM:
    # `backend` is interpreted as the OpenRouter provider name, e.g.
    # `--backend nebius` pins OpenRouter to the nebius provider.
    # "openai" / "rules" (legacy defaults) are passed through unchanged.
    provider_name = backend
    
    return OpenAILLM(provider=provider_name)