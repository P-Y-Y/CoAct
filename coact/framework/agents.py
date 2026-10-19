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
import contextvars
import json
import logging
import re
import textwrap
import time
from collections import deque
from collections.abc import Callable, Generator, Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, TypedDict

import json_repair
import yaml
from jinja2 import StrictUndefined, Template
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from .agent_types import AgentType, handle_agent_output_types
from .context import get_steps, report_step
from .memory import (
    ActionStep,
    AgentMemory,
    PlanningStep,
    SummaryStep,
    SystemPromptStep,
    TaskStep,
    ToolCall,
)
from .models import ChatMessage, MessageRole
from .monitoring import YELLOW_HEX, AgentLogger, LogLevel
from .tools import FinalAnswerTool, Tool
from .utils import AgentError, AgentExecutionError, AgentGenerationError, AgentMaxStepsError

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"

# Step budget shared by all sub-agents working on the same query (see `context.report_step`).
MAX_QUERY_STEPS = 120

NO_THINK = "No 'think' field in response"


def populate_template(template: str, variables: dict[str, Any]) -> str:
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}") from e


class PlanningPromptTemplate(TypedDict):
    """
    Prompt templates for the planning step.

    Args:
        initial_plan (`str`): Initial plan prompt.
        task_input (`str`): Prompt presenting the task to plan for.
    """

    initial_plan: str
    task_input: str


class SummaryPromptTemplate(TypedDict):
    """
    Prompt templates for the summary step.

    Args:
        update_pre_messages (`str`): Progress execution prompt.
        update_post_messages (`str`): Progress execution prompt.
    """

    update_pre_messages: str
    update_post_messages: str


class FinalAnswerPromptTemplate(TypedDict):
    """
    Prompt templates for the final answer.

    Args:
        pre_messages (`str`): Pre-messages prompt.
        post_messages (`str`): Post-messages prompt.
    """

    pre_messages: str
    post_messages: str


class StepPromptTemplate(TypedDict):
    """
    Prompt templates for an action step.

    Args:
        pre_messages (`str`): Instruction appended after the memory at each step.
    """

    pre_messages: str


class PromptTemplates(TypedDict):
    """
    Prompt templates for the agent.

    Args:
        system_prompt (`str`): System prompt.
        planning ([`~agents.PlanningPromptTemplate`]): Planning prompt templates.
        summary ([`~agents.SummaryPromptTemplate`]): Summary prompt templates.
        final_answer ([`~agents.FinalAnswerPromptTemplate`]): Final answer prompt templates.
        step ([`~agents.StepPromptTemplate`]): Action step prompt templates.
    """

    system_prompt: str
    planning: PlanningPromptTemplate
    summary: SummaryPromptTemplate
    final_answer: FinalAnswerPromptTemplate
    step: StepPromptTemplate


EMPTY_PROMPT_TEMPLATES = PromptTemplates(
    system_prompt="",
    planning=PlanningPromptTemplate(initial_plan=""),
    summary=SummaryPromptTemplate(),
    final_answer=FinalAnswerPromptTemplate(pre_messages="", post_messages=""),
)


class MultiStepAgent:
    """
    Agent class that solves the given task step by step, using the ReAct framework:
    While the objective is not reached, the agent will perform a cycle of action (given by the LLM)
    and observation (obtained from the environment).

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        query_idx (`int`): Index of the query this agent works on; used for the shared step budget.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        max_steps (`int`, default `6`): Maximum number of steps the agent can take to solve the task.
        verbosity_level (`LogLevel`, default `LogLevel.INFO`): Level of verbosity of the agent's logs.
        grammar (`dict[str, str]`, *optional*): Grammar used to parse the LLM output.
        managed_agents (`list`, *optional*): Managed agents that the agent can call.
        summary_interval (`int`, *optional*): If set, run a summary step every `summary_interval` steps.
        name (`str`, *optional*): Necessary for a managed agent only - the name by which this agent can be called.
        description (`str`, *optional*): Necessary for a managed agent only - the description of this agent.
        provide_run_summary (`bool`, *optional*): Whether to provide a run summary when called as a managed agent.
        prompts_type (`str`, default `"default"`): Name of the prompt set under `prompts/`.
    """

    def __init__(
        self,
        tools: list[Tool],
        query_idx: int,
        model: Callable[[list[dict[str, str]]], ChatMessage],
        prompt_templates: PromptTemplates | None = None,
        max_steps: int = 6,
        verbosity_level: LogLevel = LogLevel.INFO,
        grammar: dict[str, str] | None = None,
        managed_agents: list | None = None,
        summary_interval: int | None = None,
        name: str | None = None,
        description: str | None = None,
        provide_run_summary: bool = False,
        debug: bool = False,
        prompts_type: str | None = "default",
    ):
        self.agent_name = self.__class__.__name__
        self.query_idx = query_idx
        self.model = model
        self.prompt_templates = prompt_templates or EMPTY_PROMPT_TEMPLATES
        self.max_steps = max_steps
        self.step_number: int = 0
        self.grammar = grammar
        self.summary_interval = summary_interval
        self.state = {}
        self.name = name
        self.description = description
        self.provide_run_summary = provide_run_summary
        self.debug = debug
        self.action_trajectory = []
        self.managed_agents = {}

        for tool in tools:
            assert isinstance(tool, Tool), f"This element is not of class Tool: {tool!s}"
        self.tools = {tool.name: tool for tool in tools}
        self.tools["final_answer"] = FinalAnswerTool()
        self.system_prompt = self.initialize_system_prompt()
        self.input_messages = None
        self.task = None
        self.memory = AgentMemory(self.system_prompt)
        self.logger = AgentLogger(level=verbosity_level)
        self.prompts_type = prompts_type

    @property
    def logs(self):
        logger.warning(
            "The 'logs' attribute is deprecated and will soon be removed. "
            "Please use 'self.memory.steps' instead."
        )
        return [self.memory.system_prompt] + self.memory.steps

    def initialize_system_prompt(self):
        """To be implemented in child classes."""

    def write_memory_to_messages(
        self,
        memory_steps: list[ActionStep] | None = None,
        summary_mode: bool | None = False,
    ) -> list[dict[str, str]]:
        """
        Reads past llm_outputs, actions, and observations or errors from the memory into a series of messages
        that can be used as input to the LLM. Adds a number of keywords (such as PLAN, error, etc) to help
        the LLM.
        """
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        for memory_step in memory_steps if memory_steps else self.memory.steps:
            messages.extend(memory_step.to_messages(summary_mode=summary_mode))
        return messages

    def visualize(self):
        """Creates a rich tree visualization of the agent's structure."""
        self.logger.visualize_agent_tree(self)

    def provide_final_answer(self, task: str) -> tuple[str | None, str | None, str]:
        """
        Provide the final answer to the task, based on the logs of the agent's interactions.

        Args:
            task (`str`): Task to perform.

        Returns:
            `tuple`: (model reasoning content, "think" field, final answer).
        """
        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt_templates["final_answer"]["pre_messages"],
                    }
                ],
            }
        ]
        messages += self.write_memory_to_messages()[1:]
        messages += [
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["final_answer"]["post_messages"],
                            variables={"task": task},
                        ),
                    }
                ],
            }
        ]
        try:
            chat_message: ChatMessage = self.model(messages)
            final_answer = chat_message.content
            final_cot_think = chat_message.reasoning_content
            final_answer_json = json_repair.loads(final_answer)

            final_answer_think = ""
            if isinstance(final_answer_json, dict):
                final_answer_think = final_answer_json.get("think", "")
                final_answer_res = final_answer_json.get("answer", "")
            else:
                final_answer_res = final_answer  # Fall back to the plain-text output
            return final_cot_think, final_answer_think, final_answer_res
        except Exception as e:
            return None, None, f"Error in generating final LLM output:\n{e}"

    def execute_tool_call(self, tool_name: str, arguments: dict[str, str] | str) -> Any:
        """
        Execute tool with the provided input and returns the result.
        This method replaces arguments with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the Tool to execute (should be one from self.tools).
            arguments (`dict[str, str]` or `str`): Arguments passed to the Tool.
        """
        available_tools = {**self.tools, **self.managed_agents}
        if tool_name not in available_tools:
            error_msg = f"Unknown tool {tool_name}, should be instead one of {list(available_tools.keys())}."
            raise AgentExecutionError(error_msg, self.logger)

        tool = available_tools[tool_name]
        is_managed_agent = tool_name in self.managed_agents
        try:
            logging.info(f"Executing tool '{tool_name}' with arguments: {arguments} started.")
            if isinstance(arguments, str):
                if is_managed_agent:
                    observation = tool(arguments)
                else:
                    observation = tool(arguments, sanitize_inputs_outputs=True)
            elif isinstance(arguments, dict):
                for key, value in arguments.items():
                    if isinstance(value, str) and value in self.state:
                        arguments[key] = self.state[value]
                if is_managed_agent:
                    observation = tool(**arguments)
                else:
                    observation = tool(**arguments, sanitize_inputs_outputs=True)
            else:
                error_msg = (
                    f"Arguments passed to tool should be a dict or string: got a {type(arguments)}."
                )
                raise AgentExecutionError(error_msg, self.logger)
            logging.info(f"Executing tool '{tool_name}' with arguments: {arguments} ended.")
            return observation
        except Exception as e:
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                error_msg = (
                    f"Error when executing tool {tool_name} with arguments {arguments}: "
                    f"{type(e).__name__}: {e}\nYou should only use this tool with a correct input.\n"
                    f"As a reminder, this tool's description is the following: '{tool.description}'.\n"
                    f"It takes inputs: {tool.inputs} and returns output type {tool.output_type}"
                )
                raise AgentExecutionError(error_msg, self.logger) from e
            elif is_managed_agent:
                error_msg = (
                    f"Error in calling team member: {e}\n"
                    "You should only ask this team member with a correct request.\n"
                    f"As a reminder, this team member's description is the following:\n{tool}"
                )
                raise AgentExecutionError(error_msg, self.logger) from e

    def step(self, memory_step: ActionStep) -> Any | None:
        """To be implemented in children classes. Should return either None if the step is not final."""

    def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        answer: str | None = None,
        images: list[str] | None = None,
        additional_args: dict | None = None,
    ):
        self.task = task
        self.answer = answer

        self.system_prompt = self.initialize_system_prompt()
        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)

        self.logger.log_task(
            content=self.task.strip(),
            subtitle=f"{type(self.model).__name__} - {getattr(self.model, 'model_id', '')}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else None,
        )

        self.memory.steps.append(TaskStep(task=self.task, task_images=images))

        if stream:
            # The steps are returned as they are executed through a generator to iterate on.
            return self._run(task=self.task, images=images)
        # Outputs are returned only at the end as a string. We only look at the last step.
        return deque(self._run(task=self.task, images=images), maxlen=1)[0]

    def _run(
        self, task: str, images: list[str] | None = None
    ) -> Generator[ActionStep | AgentType, None, None]:
        """
        Run the agent in streaming mode and returns a generator of all the steps.

        Args:
            task (`str`): Task to perform.
            images (`list[str]`): Paths to image(s).
        """

    def planning_step(self, task) -> PlanningStep:
        """
        Ask the model for an initial plan to reach the objective.

        Args:
            task (`str`): Task to perform.
        """
        input_messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["initial_plan"],
                            variables={"tools": self.tools},
                        ),
                    }
                ],
            },
        ]
        task_messages = [
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["task_input"],
                            variables={"task": task},
                        ),
                    }
                ],
            }
        ]
        chat_message_plan: ChatMessage = self.model(input_messages + task_messages)
        plan = chat_message_plan.content

        final_plan_redaction = textwrap.dedent(
            f"""Here is the plan of action that I will follow to solve the task:\n```\n{plan}\n```\n"""
        )
        self.logger.log(
            Rule("[bold]Initial plan", style="orange"),
            Text(final_plan_redaction),
            level=LogLevel.INFO,
        )

        planning_step = PlanningStep(
            model_input_messages=input_messages,
            plan=plan,
            plan_think="",
            plan_reasoning=chat_message_plan.reasoning_content,
        )
        self.memory.steps.append(planning_step)
        return planning_step

    def summary_step(self, task, step: int) -> SummaryStep:
        """
        Used periodically by the agent to summarize the steps taken so far.

        Args:
            task (`str`): Task to perform.
            step (`int`): The number of the current step, used as an indication for the LLM.
        """
        memory_messages = self.write_memory_to_messages()[1:]

        update_pre_messages = {
            "role": MessageRole.SYSTEM,
            "content": [
                {"type": "text", "text": self.prompt_templates["summary"]["update_pre_messages"]}
            ],
        }
        update_post_messages = {
            "role": MessageRole.USER,
            "content": [
                {"type": "text", "text": self.prompt_templates["summary"]["update_post_messages"]}
            ],
        }
        input_messages = [update_pre_messages] + memory_messages + [update_post_messages]
        chat_message_summary: ChatMessage = self.model(input_messages)
        summary = chat_message_summary.content

        final_summary_redaction = textwrap.dedent(
            f"""
            Here is my summary of action to solve the task:
            ```
            {summary}
            ```"""
        )
        summary_step = SummaryStep(
            model_input_messages=input_messages,
            summary=summary,
            summary_reasoning=chat_message_summary.reasoning_content,
        )
        self.memory.steps.append(summary_step)
        self.logger.log(
            Rule("[bold]Summary", style="orange"),
            Text(final_summary_redaction),
            level=LogLevel.INFO,
        )
        return summary_step

    def to_dict(self) -> dict[str, Any]:
        """Converts agent into a dictionary."""
        tool_dicts = [tool.to_dict() for tool in self.tools.values()]
        tool_requirements = {
            req for tool in self.tools.values() for req in tool.to_dict()["requirements"]
        }
        managed_agents_requirements = {
            req
            for managed_agent in self.managed_agents.values()
            for req in managed_agent.to_dict()["requirements"]
        }
        requirements = tool_requirements | managed_agents_requirements
        if hasattr(self, "authorized_imports"):
            base_builtin_modules = [
                "collections",
                "datetime",
                "itertools",
                "math",
                "queue",
                "random",
                "re",
                "stat",
                "statistics",
                "time",
                "unicodedata",
            ]
            requirements.update(
                {
                    package.split(".")[0]
                    for package in self.authorized_imports
                    if package not in base_builtin_modules
                }
            )

        return {
            "tools": tool_dicts,
            "model": {
                "class": self.model.__class__.__name__,
                "data": self.model.to_dict(),
            },
            "managed_agents": {
                managed_agent.name: managed_agent.__class__.__name__
                for managed_agent in self.managed_agents.values()
            },
            "prompt_templates": self.prompt_templates,
            "max_steps": self.max_steps,
            "verbosity_level": int(self.logger.level),
            "grammar": self.grammar,
            "name": self.name,
            "description": self.description,
            "requirements": list(requirements),
        }


def _extract_structured_block(raw: str) -> str | None:
    """Extract the start of a YAML/JSON structure from model output, skipping any leading text."""
    # A YAML `tools:` block takes precedence.
    tools_match = re.search(r"(?m)^tools\s*:\s*", raw)
    if tools_match:
        return raw[tools_match.start() :]

    obj_match = re.search(r"\{[\s\S]*\}", raw)
    if obj_match:
        return obj_match.group(0)

    arr_match = re.search(r"\[[\s\S]*\]", raw)
    if arr_match:
        return arr_match.group(0)

    return None


def _extract_think(raw: str) -> str:
    match = re.search(r"think\s*:\s*(.*?)(?:\n\n|tools\s*:)", raw, re.S)
    return match.group(1).strip() if match else NO_THINK


class ToolCallingAgent(MultiStepAgent):
    """Agent that emits a batch of tool calls per step and executes them in parallel."""

    def __init__(
        self,
        tools: list[Tool],
        query_idx: int,
        model: Callable[[list[dict[str, str]]], ChatMessage],
        prompt_templates: PromptTemplates | None = None,
        summary_interval: int | None = None,
        prompts_type: str | None = "default",
        step_budget: int = MAX_QUERY_STEPS,
        **kwargs,
    ):
        self.step_budget = step_budget  # sub-agent steps allowed per query, shared across agents
        super().__init__(
            tools=tools,
            query_idx=query_idx,
            model=model,
            prompt_templates=prompt_templates,
            summary_interval=summary_interval,
            prompts_type=prompts_type,
            **kwargs,
        )
        prompt_path = PROMPTS_DIR / prompts_type / "toolcalling_agent.yaml"
        try:
            self.prompt_templates = prompt_templates or yaml.safe_load(
                prompt_path.read_text(encoding="utf-8")
            )
        except FileNotFoundError as e:
            raise AgentError(f"No prompt file: {prompts_type}/toolcalling_agent.yaml") from e
        except yaml.YAMLError as e:
            raise AgentError(f"Yaml parse error: {e}") from e
        self.summary_interval = summary_interval

    def initialize_system_prompt(self) -> str:
        return populate_template(
            self.prompt_templates["system_prompt"],
            variables={"tools": self.tools},
        )

    def _count_step(self) -> int:
        """Advance the local step number and the shared per-query step budget."""
        self.step_number += 1
        logging.info("step++")
        report_step(self.query_idx)
        return get_steps(self.query_idx)

    def _run(
        self, task: str, images: list[str] | None = None
    ) -> Generator[ActionStep | AgentType, None, None]:
        """
        Run the agent in streaming mode and returns a generator of all the steps.

        Args:
            task (`str`): Task to perform.
            images (`list[str]`): Paths to image(s).
        """
        final_answer = None
        self.step_number = 0
        cnt = 0
        while (
            final_answer is None and self.step_number <= self.max_steps and cnt <= self.step_budget
        ):
            cnt = get_steps(self.query_idx)
            step_start_time = time.time()
            memory_step = ActionStep(
                step_number=self.step_number,
                start_time=step_start_time,
                observations_images=images,
            )
            try:
                if self.step_number == 0:
                    self.planning_step(task)
                    cnt = self._count_step()
                elif (
                    self.summary_interval is not None
                    and self.step_number % self.summary_interval == 0
                ):
                    self.summary_step(task, step=self.step_number)
                    cnt = self._count_step()
                self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)
                final_answer = self.step(memory_step)
            except AgentError as e:
                memory_step.error = e
                raise
            finally:
                memory_step.end_time = time.time()
                memory_step.duration = memory_step.end_time - step_start_time
                self.memory.steps.append(memory_step)
                cnt = self._count_step()
                yield memory_step

        cnt = get_steps(self.query_idx)
        if final_answer is None and self.step_number > self.max_steps or cnt > self.step_budget:
            error_message = "Reached max steps."
            step_start_time = time.time()
            cot_think, final_think, final_answer = self.provide_final_answer(task)

            final_memory_step = ActionStep(
                step_number=self.step_number, error=AgentMaxStepsError(error_message, self.logger)
            )
            final_memory_step.action_reasoning = cot_think
            final_memory_step.action_think = final_think
            final_memory_step.action_output = final_answer
            final_memory_step.end_time = time.time()
            final_memory_step.duration = memory_step.end_time - step_start_time
            self.memory.steps.append(final_memory_step)

            yield final_memory_step

        yield handle_agent_output_types(final_answer)

    def reformulate_tool_fuctions(self, tool_list: list[Tool]) -> str:
        """Render the tools' signatures as a JSON list for the step prompt."""
        json_schema_list = []
        for tool in tool_list:
            required = []
            properties = deepcopy(tool.inputs)
            for key, value in properties.items():
                if value["type"] == "any":
                    value["type"] = "string"
                if not value.get("nullable"):
                    required.append(key)
            json_schema_list.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "properties": properties,
                        "required": required,
                    },
                }
            )
        return json.dumps(json_schema_list, indent=2, ensure_ascii=False)

    @staticmethod
    def _parse_structured_block(structured_block: str, raw_content: str) -> Any:
        """Parse a structured block as JSON, then YAML, then a bare `tools: [...]` list. Returns None on failure."""
        content_dict = None
        with contextlib.suppress(Exception):
            content_dict = json_repair.loads(structured_block)

        if content_dict is None:
            with contextlib.suppress(Exception):
                content_dict = yaml.safe_load(structured_block)

        if content_dict is None:
            tools_match = re.search(r"tools\s*:\s*(\[[\s\S]*?\])", structured_block)
            if tools_match:
                content_dict = {
                    "tools": json_repair.loads(tools_match.group(1)),
                    "think": _extract_think(raw_content),
                }
        return content_dict

    def step(self, memory_step: ActionStep, memory_messages=None) -> Any | None:
        memory_messages = (
            self.write_memory_to_messages() if memory_messages is None else memory_messages
        )
        self.input_messages = memory_messages
        memory_step.model_input_messages = memory_messages.copy()

        instruction_message = [
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["step"]["pre_messages"],
                            variables={
                                "tool_functions_json": self.reformulate_tool_fuctions(
                                    list(self.tools.values())
                                ),
                                "task": self.task,
                            },
                        ),
                    }
                ],
            }
        ]

        try:
            # 1. Call the model.
            model_message: ChatMessage = self.model(memory_messages + instruction_message)
            memory_step.model_output_messages = model_message
            raw_content = model_message.content

            # 2. Parse the model output (leading free text is allowed).
            structured_block = _extract_structured_block(raw_content)
            if structured_block is None:
                logging.info(f"No structured block found. Raw:\n{raw_content}")
                memory_step.observations = "No structured content returned by model."
                return None

            content_dict = self._parse_structured_block(structured_block, raw_content)
            if content_dict is None:
                logging.info(f"Failed to parse structured content. Raw:\n{raw_content}")
                memory_step.observations = "Failed to parse structured content."
                return None

            # 3. Extract `tools` and `think`.
            if isinstance(content_dict, dict):
                answer_data = content_dict.get("tools")
                memory_step.action_think = content_dict.get("think", NO_THINK)
            elif isinstance(content_dict, list):
                answer_data = content_dict
                memory_step.action_think = NO_THINK
            else:
                memory_step.observations = "Invalid model output structure."
                return None

            # 4. Normalize to a list of tool-call mappings.
            tool_calls_list = []
            if isinstance(answer_data, list):
                tool_calls_list = [tc for tc in answer_data if isinstance(tc, Mapping)]
            elif isinstance(answer_data, Mapping):
                tool_calls_list = [answer_data]
            logging.info(f"raw content was:\n{raw_content}")
            if not tool_calls_list:
                logging.info(f"No valid tool calls parsed, but raw content was:\n{raw_content}")
                memory_step.observations = "Model did not return executable tool calls."
                return None

            memory_step.tool_calls = []
            observations = []

            self.logger.log(
                Panel(
                    Text(
                        f"Function calling number: {len(tool_calls_list)} calls: {tool_calls_list}"
                    )
                ),
                level=LogLevel.INFO,
            )

            # 5. A `final_answer` call short-circuits the step.
            for tool_call in tool_calls_list:
                tool_name = tool_call.get("name")
                tool_arguments = tool_call.get("arguments", {})
                tool_call_id = tool_call.get("id", "")

                if tool_name == "final_answer":
                    memory_step.tool_calls.append(
                        ToolCall(name=tool_name, arguments=tool_arguments, id=tool_call_id)
                    )
                    final_answer_value = (
                        tool_arguments.get("answer", tool_arguments)
                        if isinstance(tool_arguments, Mapping)
                        else tool_arguments
                    )
                    memory_step.observations = str(final_answer_value)
                    self.logger.log(
                        Text(
                            f"Final answer (short-circuit): {final_answer_value}",
                            style=f"bold {YELLOW_HEX}",
                        ),
                        level=LogLevel.INFO,
                    )
                    return final_answer_value

            # 6. Execute the remaining tool calls in parallel.
            with ThreadPoolExecutor() as executor:
                futures = []
                for idx, tool_call in enumerate(tool_calls_list):
                    tool_name = tool_call.get("name")
                    tool_arguments = tool_call.get("arguments", {})
                    tool_call_id = tool_call.get("id", "")

                    memory_step.tool_calls.append(
                        ToolCall(name=tool_name, arguments=tool_arguments, id=tool_call_id)
                    )
                    self.logger.log(
                        Panel(
                            Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")
                        ),
                        level=LogLevel.INFO,
                    )

                    # Worker threads must inherit the current contextvars (e.g. CURRENT_TASK_IDX).
                    ctx = contextvars.copy_context()
                    future = executor.submit(
                        ctx.run, self.execute_tool_call, tool_name, tool_arguments
                    )
                    futures.append((idx, future, tool_name, tool_arguments))

                futures.sort(key=lambda x: x[0])

                for _, future, tool_name, tool_arguments in futures:
                    try:
                        observation = future.result()
                        observations.append(
                            f"Results for tool call '{tool_name}' with arguments '{tool_arguments}':\n{observation}"
                        )
                    except Exception as e:
                        self.logger.error(str(e))
                        observations.append(f"Error for tool call '{tool_name}': {e}")

            memory_step.observations = (
                "\n\n".join(observations) if observations else "No observations"
            )
            return None

        except Exception as e:
            raise AgentGenerationError(
                f"Error in generating tool call with model:\n{e}", self.logger
            ) from e


__all__ = ["MultiStepAgent", "ToolCallingAgent", "PromptTemplates"]
