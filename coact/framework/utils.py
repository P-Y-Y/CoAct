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

# Portions of this file are modifications by OPPO PersonalAI Team.
# Licensed under the Apache License, Version 2.0.

import base64
import json
from io import BytesIO
from typing import Any

from .monitoring import AgentLogger

__all__ = ["AgentError"]


class AgentError(Exception):
    """Base class for other agent-related exceptions."""

    def __init__(self, message, logger: AgentLogger):
        super().__init__(message)
        self.message = message
        logger.log(f"[bold red]{message}[/bold red]", level="ERROR")

    def dict(self) -> dict[str, str]:
        return {"type": self.__class__.__name__, "message": str(self.message)}


class AgentExecutionError(AgentError):
    """Exception raised for errors in execution in the agent."""


class AgentMaxStepsError(AgentError):
    """Exception raised when the agent reaches its step limit."""


class AgentGenerationError(AgentError):
    """Exception raised for errors in generation in the agent."""


def make_json_serializable(obj: Any) -> Any:
    """Recursive function to make objects JSON serializable."""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        # Try to parse string as JSON if it looks like a JSON object/array
        if isinstance(obj, str):
            try:
                if (obj.startswith("{") and obj.endswith("}")) or (
                    obj.startswith("[") and obj.endswith("]")
                ):
                    parsed = json.loads(obj)
                    return make_json_serializable(parsed)
            except json.JSONDecodeError:
                pass
        return obj
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        return {
            "_type": obj.__class__.__name__,
            **{k: make_json_serializable(v) for k, v in obj.__dict__.items()},
        }
    else:
        return str(obj)


def encode_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def make_image_url(base64_image):
    return f"data:image/png;base64,{base64_image}"
