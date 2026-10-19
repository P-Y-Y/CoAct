import argparse
import time
import logging
import threading
import json
from typing import List, Dict, Any, Optional, Union, Iterator

# Use the module logger (falls back to default config)
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

class VLLMModel:
    """
    vLLM model wrapper for the SoT framework.
    Uses vLLM's offline inference engine (LLM) behind a pseudo-streaming interface for the scheduler.
    """
    def __init__(
        self,
        api_model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        # vLLM params
        vllm_tensor_parallel_size: int = 1,
        vllm_gpu_memory_utilization: float = 0.9,
        vllm_trust_remote_code: bool = True,
        vllm_enforce_eager: bool = False,
        vllm_dtype: str = "auto",
        vllm_max_model_len: Optional[int] = None,
        # Other
        tokenizer_name: str = None,
        # Thinking
        enable_thinking: bool = True, 
        **kwargs,
    ):
        self.model_path = api_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.enable_thinking = enable_thinking
        
        # --- Thread Safety ---
        # vLLM LLM class is not thread-safe for concurrent generation.
        # We use a lock to serialize requests from multiple worker threads.
        self.lock = threading.Lock()

        # --- Imports ---
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("Please install vllm and transformers: pip install vllm transformers")

        self.SamplingParams = SamplingParams

        logger.info(f"[VLLMModel] Initializing vLLM engine with model: {self.model_path}")
        logger.info(f"[VLLMModel] Params: TP={vllm_tensor_parallel_size}, GPU={vllm_gpu_memory_utilization}, Dtype={vllm_dtype}")

        self.tokenizer_name = tokenizer_name or self.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            trust_remote_code=vllm_trust_remote_code
        )

        llm_kwargs = {
            "model": self.model_path,
            "tokenizer": self.tokenizer_name,
            "tensor_parallel_size": vllm_tensor_parallel_size,
            "gpu_memory_utilization": vllm_gpu_memory_utilization,
            "trust_remote_code": vllm_trust_remote_code,
            "enforce_eager": vllm_enforce_eager,
            "dtype": vllm_dtype,
        }
        if vllm_max_model_len:
            llm_kwargs["max_model_len"] = vllm_max_model_len

        # Initialize LLM
        # Note: this is slow and memory-hungry; make sure it runs only once
        self.llm = LLM(**llm_kwargs)

    @classmethod
    def from_command_line_args(cls, args):
        """
        Parse command-line arguments and instantiate the model.
        Called from main.py: model, other_args = VLLMModel.from_command_line_args(other_args)
        """
        parser = argparse.ArgumentParser(description="VLLM Model Args", add_help=False)
        
        # Arguments passed in from the run scripts
        group = parser.add_argument_group("VLLMModel")
        group.add_argument("--api-model", type=str, required=True, help="Local model path")
        group.add_argument("--temperature", type=float, default=0.7)
        group.add_argument("--max-tokens", type=int, default=2048)
        group.add_argument("--top-p", type=float, default=0.9)
        
        # vLLM specific
        group.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
        group.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
        group.add_argument("--vllm-trust-remote-code", action="store_true", default=True)
        group.add_argument("--vllm-dtype", type=str, default="auto")
        group.add_argument("--vllm-enforce-eager", action="store_true")
        group.add_argument("--vllm-max-model-len", type=int, default=None)
        
        # Parse known args; leave the rest for the scheduler
        parsed_args, unknown_args = parser.parse_known_args(args)
        
        # Instantiate
        model = cls(
            api_model=parsed_args.api_model,
            temperature=parsed_args.temperature,
            max_tokens=parsed_args.max_tokens,
            top_p=parsed_args.top_p,
            vllm_tensor_parallel_size=parsed_args.vllm_tensor_parallel_size,
            vllm_gpu_memory_utilization=parsed_args.vllm_gpu_memory_utilization,
            vllm_trust_remote_code=parsed_args.vllm_trust_remote_code,
            vllm_dtype=parsed_args.vllm_dtype,
            vllm_enforce_eager=parsed_args.vllm_enforce_eager,
            vllm_max_model_len=parsed_args.vllm_max_model_len
        )
        
        return model, unknown_args

    def _prepare_prompt_text(self, req):
        messages = []
        if isinstance(req, str):
            messages = [{"role": "user", "content": req}]
        elif isinstance(req, list):
            messages = req
        
        # Apply chat template
        try:
            # Try passing enable_thinking (for tokenizers that support it)
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking
            )
        except TypeError:
            # Fall back to the standard call if the tokenizer does not support enable_thinking
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        return prompt_text

    def get_response(self, requests, stream=False):
        """
        Run vLLM generation.
        Returns Iterator[Iterator[Dict]] as expected by OutlineScheduler.
        """
        # Normalize input: wrap a single string into a list
        if isinstance(requests, str):
            requests = [requests]
            
        start_time = time.time()
        
        # 1. Prepare prompts
        prompts = [self._prepare_prompt_text(req) for req in requests]
        
        # 2. Sampling parameters
        sampling_params = self.SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        
        # 3. Run inference (under a lock)
        # main.py uses a ThreadPoolExecutor, so each requests list usually holds
        # a single element; the lock prevents concurrent self.llm.generate calls.
        with self.lock:
             outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)

        # 4. Wrap results as generators
        # Outer loop: one per request in the batch
        for output in outputs:
            # Inner generator that simulates a stream
            def response_generator():
                generated_text = output.outputs[0].text
                token_ids = output.outputs[0].token_ids
                finish_reason = output.outputs[0].finish_reason
                
                # --- Thinking Logic ---
                content = generated_text
                thinking_content = None
                
                # Try to split out <think> content (DeepSeek/Qwen reasoning pattern)
                # by looking for the special token 151668 (Qwen3 </think>)
                try:
                    if 151668 in token_ids:
                         rev_ids = token_ids[::-1]
                         idx_from_end = rev_ids.index(151668)
                         split_idx = len(token_ids) - idx_from_end
                         
                         thinking_text = self.tokenizer.decode(token_ids[:split_idx], skip_special_tokens=True)
                         real_content = self.tokenizer.decode(token_ids[split_idx:], skip_special_tokens=True)
                         
                         content = real_content.strip()
                         thinking_content = thinking_text.strip()
                except ValueError:
                    pass
                
                # Usage Stats
                prompt_tokens = len(output.prompt_token_ids)
                completion_tokens = len(token_ids)
                usage_info = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
                
                # Build the response dict
                response_dict = {
                    "text": content,
                    "time": (time.time() - start_time), 
                    "finish_reason": finish_reason,
                    "usage": usage_info
                }
                
                if thinking_content:
                    # Keep thinking content separately; OutlineScheduler only reads "text".
                    response_dict["reasoning_content"] = thinking_content

                # Yield a single chunk (pseudo-streaming)
                yield response_dict

            # Yield the inner generator
            yield response_generator()

