import os
import time
import logging
import copy
from typing import List, Union, Dict, Any
from tqdm import tqdm

import openai
from openai import OpenAI, AzureOpenAI, APIError, AuthenticationError, BadRequestError, RateLimitError, APITimeoutError
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
)

from .model import Model
from sot.utils.logging import logger

class OpenAIModel(Model):
    def __init__(
        self,
        api_type,
        api_base,
        api_version,
        temperature,
        max_tokens,
        top_p,
        frequency_penalty,
        presence_penalty,
        engine,
        api_model,
        system_message,
        timeout,
        provider_name=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._api_type = api_type
        self._api_base = api_base
        self._api_version = api_version
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._frequency_penalty = frequency_penalty
        self._presence_penalty = presence_penalty
        self._engine = engine
        self._api_model = api_model
        self._system_message = system_message
        self._timeout = timeout
        self._provider_name = provider_name

        self._init_client()

    def _init_client(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = "EMPTY" 
        
        if hasattr(self, "_api_key") and self._api_key:
            api_key = self._api_key

        base_url = self._api_base
        
        if self._api_type == "azure":
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=self._api_version,
                azure_endpoint=base_url
            )
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self._timeout
            )

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj

        for k, v in self.__dict__.items():
            if k == 'client':
                setattr(new_obj, k, None)
            else:
                setattr(new_obj, k, copy.deepcopy(v, memo))
        
        new_obj._init_client()
        return new_obj

    @staticmethod
    def command_line_parser():
        parser = super(OpenAIModel, OpenAIModel).command_line_parser()
        parser.add_argument("--api-type", type=str, default="open_ai")
        parser.add_argument("--api-base", type=str, default=None)
        parser.add_argument("--api-version", type=str, default=None)
        parser.add_argument("--temperature", type=float, default=0.7)
        parser.add_argument("--max-tokens", type=int, default=2048)
        parser.add_argument("--top-p", type=float, default=1.0)
        parser.add_argument("--frequency-penalty", type=float, default=0.0)
        parser.add_argument("--presence-penalty", type=float, default=0.0)
        parser.add_argument("--engine", type=str, default=None)
        parser.add_argument("--api-model", type=str, required=True)
        parser.add_argument("--system-message", type=str, default="You are a helpful assistant.")
        parser.add_argument("--timeout", type=float, default=60)
        parser.add_argument("--provider-name", type=str, default=None, help="Specific OpenRouter provider name (e.g. 'Nebius', 'DeepInfra')")
        return parser

    def get_response(self, requests, stream=True):  
        # tqdm removed to avoid progress bars clashing with logs under multithreading
        for request in requests:
            if stream:
                yield self._generate_response(request, stream=True)
            else:
                yield self._generate_response(request, stream=False)

    @retry(
        retry=retry_if_not_exception_type((BadRequestError, AuthenticationError)),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _generate_response(self, request, stream=True):
        messages = []
        if isinstance(request, list) and len(request) > 0 and isinstance(request[0], dict):
            messages = request
        elif isinstance(request, str):
            messages = [
                {"role": "system", "content": self._system_message},
                {"role": "user", "content": request}
            ]
        elif isinstance(request, list) and len(request) > 0 and isinstance(request[0], str):
            messages = [{"role": "system", "content": self._system_message}]
            roles = ["user", "assistant"]
            for i, msg in enumerate(request):
                if msg:
                    messages.append({"role": roles[i % 2], "content": msg})
        else:
            messages = [{"role": "user", "content": " "}]

        start_time = time.time()
        
        try:
            api_kwargs = {
                "model": self._api_model,
                "messages": messages,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "top_p": self._top_p,
                "frequency_penalty": self._frequency_penalty,
                "presence_penalty": self._presence_penalty,
                "timeout": self._timeout,
                "stream": stream
            }

            if self._provider_name:
                api_kwargs["extra_body"] = {"provider": {"only":[self._provider_name]}}
            else:
                api_kwargs["extra_body"] = {"provider": {'by':'price', 'partition':'none'}}
            
            if stream:
                api_kwargs["stream_options"] = {"include_usage": True}

            response = self.client.chat.completions.create(**api_kwargs)

            if stream:
                collected_text = ""
                final_usage = None
                
                for chunk in response:
                    # 1. Capture usage info (OpenAI / vLLM return it in the last or an intermediate chunk)
                    if hasattr(chunk, 'usage') and chunk.usage:
                        final_usage = {
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                            "total_tokens": chunk.usage.total_tokens
                        }
                        # yield usage chunk
                        yield {
                            "text": "",
                            "time": time.time() - start_time,
                            "finish_reason": None,
                            "usage": final_usage
                        }

                    # 2. Handle content chunks
                    if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if delta.content is not None:
                            collected_text += delta.content
                            yield {
                                "text": delta.content,
                                "time": time.time() - start_time,
                                "finish_reason": None
                            }
                        
                        if chunk.choices[0].finish_reason is not None:
                            yield {
                                "text": "",
                                "time": time.time() - start_time,
                                "finish_reason": chunk.choices[0].finish_reason
                            }
                
                # Usage logging after the stream finishes
                elapsed = time.time() - start_time
                if final_usage:
                    usage_str = f"In: {final_usage['prompt_tokens']} | Out: {final_usage['completion_tokens']}"
                else:
                    usage_str = "Usage info missing (vLLM ver?)"
                
                logger.info(f"[LLM Call] Stream finished | Time: {elapsed:.2f}s | {usage_str}")

            else:
                # Non-streaming call
                full_content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                
                usage_info = {}
                if response.usage:
                    usage_info = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                
                # Usage logging for non-streaming calls
                elapsed = time.time() - start_time
                usage_str = f"In: {usage_info.get('prompt_tokens',0)} | Out: {usage_info.get('completion_tokens',0)}"
                logger.info(f"[LLM Call] Request finished | Time: {elapsed:.2f}s | {usage_str}")

                yield {
                    "text": full_content,
                    "time": elapsed,
                    "finish_reason": finish_reason,
                    "usage": usage_info
                }

        except Exception as e:
            logger.error(f"OpenAI API Error: {e}")
            raise e