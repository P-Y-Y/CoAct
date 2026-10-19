# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

# Portions of this file are modifications by OPPO PersonalAI Team and the CoAct authors.
# Licensed under the Apache License, Version 2.0.

import contextlib
import json
import logging
import queue
import threading
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import openai
from openai import APIConnectionError, APIStatusError, BadRequestError, OpenAIError

from .context import CURRENT_TASK_IDX, task_tag  # noqa: F401 (CURRENT_TASK_IDX re-exported)
from .tools import Tool
from .utils import encode_image_base64, make_image_url

logger = logging.getLogger(__name__)


class EmptyContentError(Exception):
    def __init__(self, response):
        self.response = response
        super().__init__(f"Empty content in response: {response}")


def get_dict_from_nested_dataclasses(obj, ignore_key=None):
    def convert(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert(v) for k, v in asdict(obj).items() if k != ignore_key}
        return obj

    return convert(obj)


@dataclass
class ChatMessageToolCallDefinition:
    arguments: Any
    name: str
    description: str | None = None

    @classmethod
    def from_hf_api(cls, tool_call_definition) -> "ChatMessageToolCallDefinition":
        return cls(
            arguments=tool_call_definition.arguments,
            name=tool_call_definition.name,
            description=tool_call_definition.description,
        )


@dataclass
class ChatMessageToolCall:
    function: ChatMessageToolCallDefinition
    id: str
    type: str

    @classmethod
    def from_hf_api(cls, tool_call) -> "ChatMessageToolCall":
        return cls(
            function=ChatMessageToolCallDefinition.from_hf_api(tool_call.function),
            id=tool_call.id,
            type=tool_call.type,
        )


@dataclass
class ChatMessage:
    role: str
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ChatMessageToolCall] | None = None
    raw: Any | None = None  # Stores the raw output from the API

    def model_dump_json(self):
        return json.dumps(get_dict_from_nested_dataclasses(self, ignore_key="raw"))

    @classmethod
    def from_hf_api(cls, message, raw) -> "ChatMessage":
        tool_calls = None
        if getattr(message, "tool_calls", None) is not None:
            tool_calls = [
                ChatMessageToolCall.from_hf_api(tool_call) for tool_call in message.tool_calls
            ]
        return cls(
            role=message.role,
            content=message.content,
            tool_calls=tool_calls,
            reasoning_content=message.reasoning_content,
            raw=raw,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        if data.get("tool_calls"):
            tool_calls = [
                ChatMessageToolCall(
                    function=ChatMessageToolCallDefinition(
                        **{k: v for k, v in tc["function"].items() if k != "parameters"}
                    ),
                    id=tc["id"],
                    type=tc["type"],
                )
                for tc in data["tool_calls"]
            ]
            data["tool_calls"] = tool_calls
        return cls(**data)

    def dict(self):
        return json.dumps(get_dict_from_nested_dataclasses(self))


def parse_json_if_needed(arguments: str | dict) -> str | dict:
    if isinstance(arguments, dict):
        return arguments
    else:
        try:
            return json.loads(arguments)
        except Exception:
            return arguments


def parse_tool_args_if_needed(message: ChatMessage) -> ChatMessage:
    if message.tool_calls is not None:
        for tool_call in message.tool_calls:
            tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
    return message


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool-call"
    TOOL_RESPONSE = "tool-response"

    @classmethod
    def roles(cls):
        return [r.value for r in cls]


tool_role_conversions = {
    MessageRole.TOOL_CALL: MessageRole.ASSISTANT,
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}


def get_tool_json_schema(tool: Tool) -> dict:
    properties = deepcopy(tool.inputs)
    required = []
    for key, value in properties.items():
        if value["type"] == "any":
            value["type"] = "string"
        if not ("nullable" in value and value["nullable"]):
            required.append(key)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def get_clean_message_list(
    message_list: list[dict[str, str]],
    role_conversions: dict[MessageRole, MessageRole] | None = None,
    convert_images_to_image_urls: bool = False,
    flatten_messages_as_text: bool = False,
) -> list[dict[str, str]]:
    """
    Subsequent messages with the same role will be concatenated to a single message.
    output_message_list is a list of messages that will be used to generate the final message that is chat template compatible with transformers LLM chat template.

    Args:
        message_list (`list[dict[str, str]]`): List of chat messages.
        role_conversions (`dict[MessageRole, MessageRole]`, *optional* ): Mapping to convert roles.
        convert_images_to_image_urls (`bool`, default `False`): Whether to convert images to image URLs.
        flatten_messages_as_text (`bool`, default `False`): Whether to flatten messages as text.
    """
    role_conversions = role_conversions or {}
    output_message_list = []
    message_list = deepcopy(message_list)  # Avoid modifying the original list
    for message in message_list:
        role = message["role"]
        if role not in MessageRole.roles():
            raise ValueError(
                f"Incorrect role {role}, only {MessageRole.roles()} are supported for now."
            )

        if role in role_conversions:
            message["role"] = role_conversions[role]
        # encode images if needed
        if isinstance(message["content"], list):
            for element in message["content"]:
                if element["type"] == "image":
                    assert not flatten_messages_as_text, (
                        f"Cannot use images with {flatten_messages_as_text=}"
                    )
                    if convert_images_to_image_urls:
                        element.update(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": make_image_url(encode_image_base64(element.pop("image")))
                                },
                            }
                        )
                    else:
                        element["image"] = encode_image_base64(element["image"])

        if len(output_message_list) > 0 and message["role"] == output_message_list[-1]["role"]:
            # Merge consecutive messages that share the same role
            if isinstance(message["content"], list):
                if flatten_messages_as_text:
                    # Extract the text parts from the content list
                    text_content = ""
                    for element in message["content"]:
                        if isinstance(element, dict) and element.get("type") == "text":
                            text_content += element.get("text", "")
                        elif isinstance(element, str):
                            text_content += element
                    output_message_list[-1]["content"] += text_content
                else:
                    output_message_list[-1]["content"] += message["content"]
            else:
                # Content is a string
                if flatten_messages_as_text:
                    output_message_list[-1]["content"] += str(message["content"])
                else:
                    output_message_list[-1]["content"] += message["content"]
        else:
            # Start a new message
            if flatten_messages_as_text:
                if isinstance(message["content"], list):
                    # Extract the text parts from the content list
                    text_content = ""
                    for element in message["content"]:
                        if isinstance(element, dict) and element.get("type") == "text":
                            text_content += element.get("text", "")
                        elif isinstance(element, str):
                            text_content += element
                    content = text_content
                else:
                    # Content is already a string
                    content = str(message["content"])
            else:
                content = message["content"]
            output_message_list.append({"role": message["role"], "content": content})
    return output_message_list


class Model:
    def __init__(self, **kwargs):
        self.last_input_token_count = None
        self.last_output_token_count = None
        self.last_call_duration = None  # Duration of the last call, in seconds
        self.kwargs = kwargs
        # cumulative usage across the lifetime of the model instance
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_tokens = 0
        self._total_calls = 0
        self._total_call_duration = 0.0  # Seconds, summed over all calls
        self._call_durations = []  # Per-call durations, in seconds
        self._usage_lock = threading.Lock()

    def _prepare_completion_kwargs(
        self,
        messages: list[dict[str, str]],
        stop_sequences: list[str] | None = None,
        grammar: str | None = None,
        tools_to_call_from: list[Tool] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        convert_images_to_image_urls: bool = False,
        flatten_messages_as_text: bool = False,
        **kwargs,
    ) -> dict:
        """
        Prepare parameters required for model invocation, handling parameter priorities.

        Parameter priority from high to low:
        1. Explicitly passed kwargs
        2. Specific parameters (stop_sequences, grammar, etc.)
        3. Default values in self.kwargs
        """
        # Clean and standardize the message list
        messages = get_clean_message_list(
            messages,
            role_conversions=custom_role_conversions or tool_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            flatten_messages_as_text=flatten_messages_as_text,
        )

        # Use self.kwargs as the base configuration
        completion_kwargs = {
            **self.kwargs,
            "messages": messages,
        }

        # Handle specific parameters
        if stop_sequences is not None:
            completion_kwargs["stop"] = stop_sequences
        if grammar is not None:
            completion_kwargs["grammar"] = grammar

        # Handle tools parameter
        if tools_to_call_from:
            completion_kwargs.update(
                {
                    "tools": [get_tool_json_schema(tool) for tool in tools_to_call_from],
                    "tool_choice": "required",
                }
            )

        # Finally, use the passed-in kwargs to override all settings
        completion_kwargs.update(kwargs)

        return completion_kwargs

    def get_token_counts(self) -> dict[str, int]:
        return {
            "input_token_count": self.last_input_token_count,
            "output_token_count": self.last_output_token_count,
        }

    def record_usage(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None = None,
        call_duration: float | None = None,
    ):
        """Accumulate token usage and call duration over the lifetime of this model instance."""
        with self._usage_lock:
            if isinstance(prompt_tokens, int):
                self._total_input_tokens += prompt_tokens
            if isinstance(completion_tokens, int):
                self._total_output_tokens += completion_tokens
            if isinstance(total_tokens, int):
                self._total_tokens += total_tokens
            elif isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                # Derive the total when it is not reported but both parts are.
                self._total_tokens += prompt_tokens + completion_tokens
            self._total_calls += 1
            if isinstance(call_duration, (int, float)) and call_duration > 0:
                self.last_call_duration = call_duration
                self._total_call_duration += call_duration
                self._call_durations.append(call_duration)

    def get_total_usage(self) -> dict[str, Any]:
        result = {
            "total_calls": self._total_calls,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_tokens,
        }
        # Timing statistics
        if self._total_calls > 0:
            result["total_call_duration_sec"] = round(self._total_call_duration, 3)
            result["avg_call_duration_sec"] = round(
                self._total_call_duration / self._total_calls, 3
            )
            if self._call_durations:
                result["min_call_duration_sec"] = round(min(self._call_durations), 3)
                result["max_call_duration_sec"] = round(max(self._call_durations), 3)
            if self.last_call_duration is not None:
                result["last_call_duration_sec"] = round(self.last_call_duration, 3)
        return result

    def __call__(
        self,
        messages: list[dict[str, str]],
        stop_sequences: list[str] | None = None,
        grammar: str | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """Process the input messages and return the model's response.

        Parameters:
            messages (`List[Dict[str, str]]`):
                A list of message dictionaries to be processed. Each dictionary should have the structure `{"role": "user/system", "content": "message content"}`.
            stop_sequences (`List[str]`, *optional*):
                A list of strings that will stop the generation if encountered in the model's output.
            grammar (`str`, *optional*):
                The grammar or formatting structure to use in the model's response.
            tools_to_call_from (`List[Tool]`, *optional*):
                A list of tools that the model can use to generate responses.
            **kwargs:
                Additional keyword arguments to be passed to the underlying model.

        Returns:
            `ChatMessage`: A chat message object containing the model's response.

        To be implemented in child classes.
        """

    def to_dict(self) -> dict:
        """
        Converts the model into a JSON-compatible dictionary.
        """
        model_dictionary = {
            **self.kwargs,
            "last_input_token_count": self.last_input_token_count,
            "last_output_token_count": self.last_output_token_count,
            "model_id": self.model_id,
        }
        for attribute in [
            "custom_role_conversion",
            "temperature",
            "max_tokens",
            "provider",
            "timeout",
            "api_base",
            "torch_dtype",
            "device_map",
            "organization",
            "project",
            "azure_endpoint",
        ]:
            if hasattr(self, attribute):
                model_dictionary[attribute] = getattr(self, attribute)

        dangerous_attributes = ["token", "api_key"]
        for attribute_name in dangerous_attributes:
            if hasattr(self, attribute_name):
                print(
                    f"For security reasons, we do not export the `{attribute_name}` attribute of your model. Please export it manually."
                )
        return model_dictionary

    @classmethod
    def from_dict(cls, model_dictionary: dict[str, Any]) -> "Model":
        model_instance = cls(
            **{
                k: v
                for k, v in model_dictionary.items()
                if k not in ["last_input_token_count", "last_output_token_count"]
            }
        )
        model_instance.last_input_token_count = model_dictionary.pop("last_input_token_count", None)
        model_instance.last_output_token_count = model_dictionary.pop(
            "last_output_token_count", None
        )
        return model_instance


class OpenAIServerModel(Model):
    """This model connects to an OpenAI-compatible API server.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-3.5-turbo").
        api_base (`str`, *optional*):
            The base URL of the OpenAI-compatible API server.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        organization (`str`, *optional*):
            The organization to use for the API request.
        project (`str`, *optional*):
            The project to use for the API request.
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        **kwargs:
            Additional keyword arguments to pass to the OpenAI API.
    """

    def __init__(
        self,
        model_id: str,
        api_base: str | None = None,
        api_key: str | None = None,
        organization: str | None | None = None,
        project: str | None | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.client = openai.OpenAI(
            base_url=api_base,
            api_key=api_key,
            organization=organization,
            project=project,
        )
        self.custom_role_conversions = custom_role_conversions

    @staticmethod
    def truncate_content_based_on_stop_sequences(content: str, stop_sequences: list[str]) -> str:
        if not stop_sequences:
            return content
        # Truncate the content right after the first stop sequence found
        for stop_seq in stop_sequences:
            index = content.find(stop_seq)
            if index != -1:
                content = content[: index + len(stop_seq)]
                break  # Only keep the first match
        return content

    def __call__(
        self,
        messages: list[dict[str, str]],
        stop_sequences: list[str] | None = None,
        grammar: str | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )

        # o3/o4 models do not accept `stop`; stop sequences are applied manually below.
        is_o_series = "o3" in self.model_id.lower() or "o4" in self.model_id.lower()
        if is_o_series:
            completion_kwargs.pop("stop", None)

        # Some OpenAI-compatible servers do not support `max_completion_tokens`.
        # Normalize to `max_tokens` for broader compatibility.
        if "max_completion_tokens" in completion_kwargs and "max_tokens" not in completion_kwargs:
            completion_kwargs["max_tokens"] = completion_kwargs.pop("max_completion_tokens")

        max_retries = 5
        retry_delay = 5  # seconds
        for attempt in range(max_retries):
            try:
                call_start_time = time.time()
                response = self.client.chat.completions.create(**completion_kwargs)
                call_duration = time.time() - call_start_time

                # Record token usage
                try:
                    self.last_input_token_count = getattr(response.usage, "prompt_tokens", None)
                    self.last_output_token_count = getattr(
                        response.usage, "completion_tokens", None
                    )
                    total_tokens = getattr(response.usage, "total_tokens", None)
                except Exception:
                    self.last_input_token_count = None
                    self.last_output_token_count = None
                    total_tokens = None

                # Emit per-call token usage and timing to logs for tracking
                logger.info(
                    "[LLM] model=%s input_tokens=%s output_tokens=%s total_tokens=%s duration_sec=%.3f",
                    self.model_id,
                    str(self.last_input_token_count),
                    str(self.last_output_token_count),
                    str(total_tokens),
                    call_duration,
                )
                # Accumulate totals on the model instance
                with contextlib.suppress(Exception):
                    self.record_usage(
                        self.last_input_token_count,
                        self.last_output_token_count,
                        total_tokens,
                        call_duration,
                    )

                # o1 / o3-mini may return neither content nor tool calls
                if not response.choices[0].message.content and not getattr(
                    response.choices[0].message, "tool_calls", None
                ):
                    raise EmptyContentError(response)

                message = ChatMessage.from_dict(
                    response.choices[0].message.model_dump(
                        include={"role", "content", "tool_calls", "reasoning_content"}
                    )
                )
                message.raw = response

                if is_o_series:
                    message.content = self.truncate_content_based_on_stop_sequences(
                        message.content, stop_sequences
                    )

                if tools_to_call_from is not None:
                    return parse_tool_args_if_needed(message)
                return message
            except BadRequestError as e:
                # Use %-style formatting to avoid logging formatting errors.
                logger.error("Bad Request Error: %s", e)
                raise
            except APIConnectionError as e:
                if attempt < max_retries - 1:
                    logging.warning(
                        f"Network error occurred: {e}. Retrying in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                else:
                    logging.error(f"Failed to complete request after {max_retries} retries.")
                    raise
            except (APIStatusError, EmptyContentError) as e:
                if attempt < max_retries - 1:
                    logging.warning(f"API status error occurred: {e}. Retrying in 60 seconds...")
                    time.sleep(60)
                else:
                    logging.error(f"Failed to complete request after {max_retries} retries.")
                    raise
            except OpenAIError as e:
                logging.error(f"API error occurred: {e}.")
                raise
            except Exception as e:
                logging.error(f"An unexpected error occurred: {e}.")
                raise


@dataclass
class _VLLMRequest:
    """A pending generation request queued for the VLLMModel batching worker."""

    prompt: str
    sampling_params: Any
    stop_sequences: list[str] | None
    event: threading.Event
    prompt_tokens: int
    result: Any = None
    error: Exception | None = None


class VLLMModel(Model):
    """
    vLLM-backed chat model using the vLLM Python API.

    Thread-safe: concurrent calls are queued and grouped into batches by a background worker,
    splitting batches so that they fit within the model's context length.
    `vllm` and `transformers` are imported lazily, so they are only required when this class is used.
    """

    def __init__(
        self,
        model_id: str,
        tokenizer_name: str | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        dtype: str | None = "auto",
        max_model_len: int | None = None,
        enforce_eager: bool = False,
        trust_remote_code: bool = True,
        enable_thinking: bool = True,
        max_batch_size: int = 12,
        batch_timeout: float = 0.01,
        **kwargs,
    ):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        super().__init__(**kwargs)

        self.model_id = model_id
        self.tokenizer_name = tokenizer_name or model_id
        self.custom_role_conversions = custom_role_conversions
        self.enable_thinking = enable_thinking

        self._SamplingParams = SamplingParams

        logger.info(f"[VLLMModel] Loading tokenizer: {self.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            trust_remote_code=trust_remote_code,
        )

        llm_kwargs = dict(
            model=model_id,
            tokenizer=self.tokenizer_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            enforce_eager=enforce_eager,
        )

        if dtype and dtype != "auto":
            llm_kwargs["dtype"] = dtype
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        self.llm = LLM(**llm_kwargs)

        # Authoritative maximum context length of the loaded model
        self._max_model_len = self.llm.llm_engine.model_config.max_model_len

        # Request batching
        self._request_queue: queue.Queue[_VLLMRequest] = queue.Queue()
        self._max_batch_size = max_batch_size
        self._batch_timeout = batch_timeout

        self._worker_thread = threading.Thread(
            target=self._batch_worker_loop,
            daemon=True,
        )
        self._worker_thread.start()

        logger.info(
            "[VLLMModel] Worker started (max_batch_size=%d, timeout=%.3fs, max_model_len=%d)",
            self._max_batch_size,
            self._batch_timeout,
            self._max_model_len,
        )

    def _batch_worker_loop(self):
        """Group queued requests into token-aware batches and run them through vLLM."""
        while True:
            reqs: list[_VLLMRequest] = []

            # Block until the first request arrives
            first_req = self._request_queue.get()
            reqs.append(first_req)

            total_prompt_tokens = first_req.prompt_tokens
            max_new_tokens = first_req.sampling_params.max_tokens

            start = time.time()

            while len(reqs) < self._max_batch_size:
                remaining = self._batch_timeout - (time.time() - start)
                if remaining <= 0:
                    break

                try:
                    next_req = self._request_queue.get(timeout=remaining)
                except queue.Empty:
                    break

                projected = total_prompt_tokens + next_req.prompt_tokens + max_new_tokens

                if projected > self._max_model_len:
                    # Too big for this batch: push it back for the next one
                    self._request_queue.put(next_req)
                    break

                reqs.append(next_req)
                total_prompt_tokens += next_req.prompt_tokens

            try:
                prompts = [r.prompt for r in reqs]
                sampling_params = reqs[0].sampling_params

                outputs = self.llm.generate(
                    prompts,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )

                for r, out in zip(reqs, outputs, strict=False):
                    r.result = out
                    r.event.set()

            except Exception as e:
                for r in reqs:
                    r.error = e
                    r.event.set()

    def __call__(
        self,
        messages: list[dict[str, str]],
        stop_sequences: list[str] | None = None,
        grammar: str | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            custom_role_conversions=self.custom_role_conversions,
            flatten_messages_as_text=True,
            **kwargs,
        )

        prompt_text = self.tokenizer.apply_chat_template(
            completion_kwargs["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        max_new_tokens = kwargs.get(
            "max_completion_tokens",
            kwargs.get("max_tokens", self.kwargs.get("max_completion_tokens", 16384)),
        )

        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        repetition_penalty = kwargs.get("repetition_penalty")

        sp_kwargs = dict(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=stop_sequences,
        )
        if repetition_penalty is not None:
            sp_kwargs["repetition_penalty"] = repetition_penalty

        sampling_params = self._SamplingParams(**sp_kwargs)

        # Reject a single request that cannot fit in the context window
        prompt_token_len = len(
            self.tokenizer(
                prompt_text,
                add_special_tokens=False,
            ).input_ids
        )

        if prompt_token_len + max_new_tokens > self._max_model_len:
            raise ValueError(
                f"[VLLMModel] Single prompt too long: "
                f"{prompt_token_len} + {max_new_tokens} "
                f"> {self._max_model_len}"
            )

        # Enqueue and wait for the batching worker
        ev = threading.Event()

        req = _VLLMRequest(
            prompt=prompt_text,
            sampling_params=sampling_params,
            stop_sequences=stop_sequences,
            event=ev,
            prompt_tokens=prompt_token_len,
        )
        call_start_time = time.time()

        self._request_queue.put(req)
        ev.wait()

        call_duration = time.time() - call_start_time

        if req.error:
            raise req.error

        output = req.result
        if not output or not output.outputs:
            raise EmptyContentError(output)

        first_choice = output.outputs[0]
        output_ids = first_choice.token_ids

        # Token usage
        input_token_count = (
            len(output.prompt_token_ids) if hasattr(output, "prompt_token_ids") else None
        )

        output_token_count = len(output_ids) if output_ids else None
        total_tokens = (input_token_count or 0) + (output_token_count or 0)

        self.last_input_token_count = input_token_count
        self.last_output_token_count = output_token_count

        with contextlib.suppress(Exception):
            self.record_usage(
                input_token_count,
                output_token_count,
                total_tokens,
                call_duration,
            )
        logger.info(
            "%s[LLM] model=%s input_tokens=%s output_tokens=%s total_tokens=%s duration_sec=%.3f",
            task_tag(),
            self.model_id,
            str(input_token_count),
            str(output_token_count),
            str(total_tokens),
            call_duration,
        )
        # Split reasoning from the answer at the last `</think>` token (id 151668 for Qwen3)
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(
            output_ids[:index],
            skip_special_tokens=True,
        ).strip("\n")

        content = self.tokenizer.decode(
            output_ids[index:],
            skip_special_tokens=True,
        ).strip("\n")

        if stop_sequences and content:
            for stop_seq in stop_sequences:
                if stop_seq in content:
                    content = content[: content.index(stop_seq)]
                    break

        return ChatMessage(
            role="assistant",
            content=content,
            reasoning_content=thinking_content or None,
        )


__all__ = [
    "MessageRole",
    "tool_role_conversions",
    "get_clean_message_list",
    "Model",
    "OpenAIServerModel",
    "VLLMModel",
    "ChatMessage",
]
