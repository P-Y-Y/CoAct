#!/usr/bin/env python
# coding=utf-8

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

import importlib
import json
import re
from copy import deepcopy
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from collections import deque
import logging
from logging import getLogger, DEBUG, INFO
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, TypedDict, Union
import yaml
from jinja2 import StrictUndefined, Template
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from .agent_types import AgentType, handle_agent_output_types
from .tools import FinalAnswerTool
from .memory import ActionStep, AgentMemory, PlanningStep, SummaryStep, SystemPromptStep, TaskStep, ToolCall
from .models import (
    ChatMessage,
    MessageRole,
)
from .monitoring import (
    YELLOW_HEX,
    AgentLogger,
    LogLevel,
)
from .tools import Tool
import json_repair
from .utils import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxStepsError,
    parse_json_tool_call,
)
from .search_tools import update_comm_usage, set_current_query_id, reset_current_query_id


logger = getLogger(__name__)


def get_variable_names(self, template: str) -> Set[str]:
    pattern = re.compile(r"\{\{([^{}]+)\}\}")
    return {match.group(1).strip() for match in pattern.finditer(template)}


def populate_template(template: str, variables: Dict[str, Any]) -> str:
    logger.debug(f"[populate_template] ENTRY: template length={len(template)}, variables keys={list(variables.keys())}")
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        result = compiled_template.render(**variables)
        logger.debug(f"[populate_template] SUCCESS: result length={len(result)}")
        return result
    except Exception as e:
        logger.exception(f"[populate_template] EXCEPTION: {type(e).__name__}: {e}")
        logger.debug(f"[populate_template] Template preview: {template[:500]}")
        logger.debug(f"[populate_template] Variables: {variables}")
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")

def parse_model_content(content: Union[str, dict]) -> dict:
    logger.debug(f"[parse_model_content] ENTRY: content type={type(content)}, length={len(str(content)) if hasattr(content, '__len__') else 'N/A'}")
    if isinstance(content, dict):
        logger.debug(f"[parse_model_content] Content is already dict, returning as-is")
        return content
    elif isinstance(content, str):
        try:
            result = json.loads(content)
            logger.debug(f"[parse_model_content] Successfully parsed JSON string")
            return result
        except json.JSONDecodeError as e:
            logger.debug(f"[parse_model_content] JSON decode failed, returning as text: {e}")
            return {"text": content}
    else:
        logger.warning(f"[parse_model_content] Unknown content type: {type(content)}, converting to string")
        return {"unknown_type": str(content)}

class PlanningPromptTemplate(TypedDict):
    """
    Prompt templates for the planning step.

    Args:
        initial_plan (`str`): Initial plan prompt.
    """

    initial_plan: str

class SummaryPromptTemplate(TypedDict):
    """
    Prompt templates for the planning step.

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


class PromptTemplates(TypedDict):
    """
    Prompt templates for the agent.

    Args:
        system_prompt (`str`): System prompt.
        planning ([`~agents.PlanningPromptTemplate`]): Planning prompt templates.
        summary ([`~agents.SummaryPromptTemplate`]): Summary prompt templates.
        final_answer ([`~agents.FinalAnswerPromptTemplate`]): Final answer prompt templates.
    """

    system_prompt: str
    planning: PlanningPromptTemplate
    summary: SummaryPromptTemplate
    final_answer: FinalAnswerPromptTemplate


EMPTY_PROMPT_TEMPLATES = PromptTemplates(
    system_prompt="",
    planning=PlanningPromptTemplate(initial_plan=""),
    summary=SummaryPromptTemplate(),
    final_answer=FinalAnswerPromptTemplate(pre_messages="", post_messages=""),
)


class MultiStepAgent:
    """
    Agent class that solves the given task step by step, using the ReAct framework:
    While the objective is not reached, the agent will perform a cycle of action (given by the LLM) and observation (obtained from the environment).

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        max_steps (`int`, default `6`): Maximum number of steps the agent can take to solve the task.
        verbosity_level (`LogLevel`, default `LogLevel.INFO`): Level of verbosity of the agent's logs.
        grammar (`dict[str, str]`, *optional*): Grammar used to parse the LLM output.
        managed_agents (`list`, *optional*): Managed agents that the agent can call.
        name (`str`, *optional*): Necessary for a managed agent only - the name by which this agent can be called.
        description (`str`, *optional*): Necessary for a managed agent only - the description of this agent.
        provide_run_summary (`bool`, *optional*): Whether to provide a run summary when called as a managed agent.
    """

    def __init__(
            self,
            tools: List[Tool],
            model: Callable[[List[Dict[str, str]]], ChatMessage],
            prompt_templates: Optional[PromptTemplates] = None,
            max_steps: int = 6,
            verbosity_level: LogLevel = LogLevel.INFO,
            grammar: Optional[Dict[str, str]] = None,
            managed_agents: Optional[List] = None,
            summary_interval: Optional[int] = None,
            name: Optional[str] = None,
            description: Optional[str] = None,
            provide_run_summary: bool = False,
            debug: bool = False,
            prompts_type: Optional[str] = "default",
            max_workers: Optional[int] = None,
            query_id: Optional[str] = None,
    ):
        self.agent_name = self.__class__.__name__
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
        self.max_workers = max_workers  # max threads for parallel tool execution
        self.query_id = query_id  # for per-query stats

        for tool in tools:
            assert isinstance(tool, Tool), f"This element is not of class Tool: {str(tool)}"
        self.tools = {tool.name: tool for tool in tools}
        self.tools["final_answer"] = FinalAnswerTool()
        self.system_prompt = self.initialize_system_prompt()
        self.input_messages = None
        self.task = None
        self.memory = AgentMemory(self.system_prompt)
        self.logger = AgentLogger(level=verbosity_level)
        self.prompts_type = prompts_type
        
        # Set logger level to DEBUG if debug mode is enabled
        if self.debug:
            # Set both the module logger and root logger to DEBUG
            root_logger = logging.getLogger()
            root_logger.setLevel(DEBUG)
            # Also set all handlers to DEBUG level
            for handler in root_logger.handlers:
                handler.setLevel(DEBUG)
            logger.setLevel(DEBUG)
            logger.debug(f"[__init__] Debug mode enabled for {self.agent_name}, logger level set to DEBUG")
        else:
            logger.setLevel(INFO)

    @property
    def logs(self):
        logger.warning(
            "The 'logs' attribute is deprecated and will soon be removed. Please use 'self.memory.steps' instead."
        )
        return [self.memory.system_prompt] + self.memory.steps

    def initialize_system_prompt(self):
        """To be implemented in child classes"""
        pass

    def write_memory_to_messages(
            self,
            memory_steps: Optional[List[ActionStep]] = None,
            summary_mode: Optional[bool] = False,
    ) -> List[Dict[str, str]]:
        """
        Reads past llm_outputs, actions, and observations or errors from the memory into a series of messages
        that can be used as input to the LLM. Adds a number of keywords (such as PLAN, error, etc) to help
        the LLM.
        """
        logger.debug(f"[write_memory_to_messages] ENTRY: memory_steps provided={memory_steps is not None}, summary_mode={summary_mode}")
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        logger.debug(f"[write_memory_to_messages] System prompt messages count: {len(messages)}")
        steps_to_process = memory_steps if memory_steps else self.memory.steps
        logger.debug(f"[write_memory_to_messages] Processing {len(steps_to_process)} memory steps")
        for idx, memory_step in enumerate(steps_to_process):
            step_messages = memory_step.to_messages(summary_mode=summary_mode)
            messages.extend(step_messages)
            logger.debug(f"[write_memory_to_messages] Step {idx} ({type(memory_step).__name__}) added {len(step_messages)} messages")
        logger.debug(f"[write_memory_to_messages] SUCCESS: Total messages count: {len(messages)}")
        return messages

    def visualize(self):
        """Creates a rich tree visualization of the agent's structure."""
        self.logger.visualize_agent_tree(self)

    def provide_final_answer(self, task: str) -> Tuple[str, str]:
        """
        Provide the final answer to the task, based on the logs of the agent's interactions.

        Args:
            task (`str`): Task to perform.
            images (`list[str]`, *optional*): Paths to image(s).

        Returns:
            `str`: Final answer to the task.
        """
        logger.debug(f"[provide_final_answer] ENTRY: task='{task}'")
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
        memory_messages = self.write_memory_to_messages()[1:]
        logger.debug(f"[provide_final_answer] Memory messages count: {len(memory_messages)}")
        messages += memory_messages
        post_message_text = populate_template(
            self.prompt_templates["final_answer"]["post_messages"], variables={"task": task}
        )
        logger.debug(f"[provide_final_answer] Post message text length: {len(post_message_text)}")
        messages += [
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": post_message_text,
                    }
                ],
            }
        ]
        logger.debug(f"[provide_final_answer] Total messages count: {len(messages)}")
        try:
            logger.debug(f"[provide_final_answer] Calling model to generate final answer")
            chat_message: ChatMessage = self.model(messages)
            final_answer = chat_message.content
            final_cot_think = chat_message.reasoning_content
            logger.debug(f"[provide_final_answer] Model returned: final_answer length={len(str(final_answer))}, final_cot_think length={len(str(final_cot_think)) if final_cot_think else 0}")
            logger.debug(f"[provide_final_answer] Parsing final_answer JSON")
            final_answer_json = json_repair.loads(final_answer)
            final_answer_think, final_answer_res = final_answer_json.get("think", ""), final_answer_json.get("answer", "")
            logger.debug(f"[provide_final_answer] SUCCESS: extracted think length={len(str(final_answer_think))}, answer length={len(str(final_answer_res))}")
            return final_cot_think, final_answer_think, final_answer_res
        
        except Exception as e:
            logger.exception(f"[provide_final_answer] EXCEPTION: {type(e).__name__}: {e}")
            logger.debug(f"[provide_final_answer] Returning error response")
            return None, None, f"Error in generating final LLM output:\n{e}"

    def execute_tool_call(self, tool_name: str, arguments: Union[Dict[str, str], str]) -> Any:
        """
        Execute tool with the provided input and returns the result.
        This method replaces arguments with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the Tool to execute (should be one from self.tools).
            arguments (Dict[str, str]): Arguments passed to the Tool.
        """
        logger.debug(f"[execute_tool_call] ENTRY: tool_name={tool_name}, arguments_type={type(arguments)}, arguments={arguments}")
        logger.debug(f"[execute_tool_call] Current state: {self.state}")
        
        # Set query_id before running tools so jina/serper stats are attributed correctly
        token = None
        if self.query_id:
            token = set_current_query_id(self.query_id)
            logger.debug(f"[execute_tool_call] Set query_id={self.query_id}")
        
        available_tools = {**self.tools, **self.managed_agents}
        logger.debug(f"[execute_tool_call] Available tools: {list(available_tools.keys())}")
        
        if tool_name not in available_tools:
            error_msg = f"Unknown tool {tool_name}, should be instead one of {list(available_tools.keys())}."
            logger.error(f"[execute_tool_call] ERROR: {error_msg}")
            if token:
                reset_current_query_id(token)
            raise AgentExecutionError(error_msg, self.logger)

        try:
            if isinstance(arguments, str):
                logger.debug(f"[execute_tool_call] Executing tool '{tool_name}' with string arguments")
                if tool_name in self.managed_agents:
                    logger.debug(f"[execute_tool_call] Tool '{tool_name}' is a managed agent")
                    observation = available_tools[tool_name].__call__(arguments)
                else:
                    logger.debug(f"[execute_tool_call] Tool '{tool_name}' is a regular tool")
                    observation = available_tools[tool_name].__call__(arguments, sanitize_inputs_outputs=True)
            elif isinstance(arguments, dict):
                logger.debug(f"[execute_tool_call] Executing tool '{tool_name}' with dict arguments (before state substitution): {arguments}")
                for key, value in arguments.items():
                    if isinstance(value, str) and value in self.state:
                        logger.debug(f"[execute_tool_call] Replacing argument '{key}' value '{value}' with state value: {self.state[value]}")
                        arguments[key] = self.state[value]
                logger.debug(f"[execute_tool_call] Arguments after state substitution: {arguments}")
                if tool_name in self.managed_agents:
                    logger.debug(f"[execute_tool_call] Tool '{tool_name}' is a managed agent")
                    observation = available_tools[tool_name].__call__(**arguments)
                else:
                    logger.debug(f"[execute_tool_call] Tool '{tool_name}' is a regular tool")
                    observation = available_tools[tool_name].__call__(**arguments, sanitize_inputs_outputs=True)
            else:
                error_msg = f"Arguments passed to tool should be a dict or string: got a {type(arguments)}."
                logger.error(f"[execute_tool_call] ERROR: {error_msg}")
                raise AgentExecutionError(error_msg, self.logger)
            logger.debug(f"[execute_tool_call] SUCCESS: tool '{tool_name}' returned observation type={type(observation)}, length={len(str(observation)) if hasattr(observation, '__len__') else 'N/A'}")
            logger.debug(f"[execute_tool_call] Observation preview: {str(observation)[:500]}")
            return observation
        except Exception as e:
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                error_msg = (
                    f"Error when executing tool {tool_name} with arguments {arguments}: {type(e).__name__}: {e}\nYou should only use this tool with a correct input.\n"
                    f"As a reminder, this tool's description is the following: '{tool.description}'.\nIt takes inputs: {tool.inputs} and returns output type {tool.output_type}"
                )
                raise AgentExecutionError(error_msg, self.logger)
            elif tool_name in self.managed_agents:
                error_msg = (
                    f"Error in calling team member: {e}\nYou should only ask this team member with a correct request.\n"
                    f"As a reminder, this team member's description is the following:\n{available_tools[tool_name]}"
                )
                raise AgentExecutionError(error_msg, self.logger)
        finally:
            # Restore previous query_id
            if token:
                reset_current_query_id(token)

    def step(self, memory_step: ActionStep) -> Union[None, Any]:
        """To be implemented in children classes. Should return either None if the step is not final."""
        pass

    def run(
            self,
            task: str,
            stream: bool = False,
            reset: bool = True,
            answer: Optional[str] = None,
            images: Optional[List[str]] = None,
            additional_args: Optional[Dict] = None,
    ):
        logger.debug(f"[run] ENTRY: task='{task}', stream={stream}, reset={reset}, answer={answer}, images={images}")
        self.task = task
        self.answer = answer

        logger.debug(f"[run] Initializing system prompt")
        self.system_prompt = self.initialize_system_prompt()
        logger.debug(f"[run] System prompt length: {len(str(self.system_prompt))}")
        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)

        self.logger.log_task(
            content=self.task.strip(),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else None,
        )

        self.memory.steps.append(TaskStep(task=self.task, task_images=images))
        logger.debug(f"[run] TaskStep added to memory, total steps: {len(self.memory.steps)}")

        if stream:
            logger.debug(f"[run] Running in stream mode")
            # The steps are returned as they are executed through a generator to iterate on.
            return self._run(task=self.task, images=images)
        # Outputs are returned only at the end as a string. We only look at the last step
        logger.debug(f"[run] Running in non-stream mode, collecting all steps")
        result = deque(self._run(task=self.task, images=images), maxlen=1)[0]
        logger.debug(f"[run] Completed, returning result type: {type(result)}")
        return result

    def _run(self, task: str, images: List[str] | None = None) -> Generator[ActionStep | AgentType, None, None]:
        """
        Run the agent in streaming mode and returns a generator of all the steps.

        Args:
            task (`str`): Task to perform.
            images (`list[str]`): Paths to image(s).
        """
        pass
    def planning_step(self, task) -> None:
        """
        Used periodically by the agent to plan the next steps to reach the objective.

        Args:
            task (`str`): Task to perform.
            is_first_step (`bool`): If this step is not the first one, the plan should be an update over a previous plan.
            step (`int`): The number of the current step, used as an indication for the LLM.
        """
        input_messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["initial_plan"],
                            variables={
                                "tools": self.tools,
                            },
                        ),
                    }
                ],
            },
        ]
        task_messages = [{
            "role": MessageRole.USER,
            "content": [{"type": "text", "text": populate_template(self.prompt_templates["planning"]["task_input"], variables={"task": task})}],
        }]
        chat_message_plan: ChatMessage = self.model(input_messages + task_messages)
        think_content = chat_message_plan.reasoning_content
        plans = chat_message_plan.content
        plans_think, plans_answer = "", plans
        logger.debug(f"[planning_step] Extracted plans: length={len(str(plans_answer))}, think_content length={len(str(think_content)) if think_content else 0}")

        final_plan_redaction = textwrap.dedent(
            f"""Here is the plan of action that I will follow to solve the task:\n```\n{plans_answer}\n```\n"""
        )

        self.logger.log(
            Rule("[bold]Initial plan", style="orange"),
            Text(final_plan_redaction),
            level=LogLevel.INFO,
        )

        planning_step = PlanningStep(
            model_input_messages=input_messages,
            plan=plans_answer,
            plan_think=plans_think,
            plan_reasoning=think_content,
        )
        self.memory.steps.append(planning_step)
        logger.debug(f"[planning_step] PlanningStep added to memory, total steps: {len(self.memory.steps)}")

        return planning_step


    def summary_step(self, task, step: int) -> None:
        """
        Used periodically by the agent to summary the steps to reach the objective.

        Args:
            task (`str`): Task to perform.
            step (`int`): The number of the current step, used as an indication for the LLM.
        """
        memory_messages = self.write_memory_to_messages()[1:]

        update_pre_messages = {
            "role": MessageRole.SYSTEM,
            "content": [{"type": "text", "text": self.prompt_templates["summary"]["update_pre_messages"]}],
        }
        update_post_messages = {
            "role": MessageRole.USER,
            "content": [{"type": "text", "text": self.prompt_templates["summary"]["update_post_messages"]}],
        }
        input_messages = [update_pre_messages] + memory_messages + [update_post_messages]
        chat_message_summary: ChatMessage = self.model(input_messages)

        summary_answer = chat_message_summary.content
        summary_cot_content = chat_message_summary.reasoning_content
        logger.debug(f"[summary_step] Extracted summary: length={len(str(summary_answer))}, reasoning length={len(str(summary_cot_content)) if summary_cot_content else 0}")

        final_summary_redaction = textwrap.dedent(
            f"""
            Here is my summary of action to solve the task:
            ```
            {summary_answer}
            ```"""
        )
        summary_step = SummaryStep(
            model_input_messages=input_messages,
            summary=summary_answer,
            summary_reasoning=summary_cot_content,
        )
        self.memory.steps.append(summary_step)
        logger.debug(f"[summary_step] SummaryStep added to memory, total steps: {len(self.memory.steps)}")
        self.logger.log(
            Rule("[bold]Summary", style="orange"),
            Text(final_summary_redaction),
            level=LogLevel.INFO,
        )
        return summary_step


    def to_dict(self) -> Dict[str, Any]:
        """Converts agent into a dictionary."""

        tool_dicts = [tool.to_dict() for tool in self.tools.values()]
        tool_requirements = {req for tool in self.tools.values() for req in tool.to_dict()["requirements"]}
        managed_agents_requirements = {
            req for managed_agent in self.managed_agents.values() for req in managed_agent.to_dict()["requirements"]
        }
        requirements = tool_requirements | managed_agents_requirements
        if hasattr(self, "authorized_imports"):
            BASE_BUILTIN_MODULES = [
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
                {package.split(".")[0] for package in self.authorized_imports if package not in BASE_BUILTIN_MODULES}
            )

        agent_dict = {
            "tools": tool_dicts,
            "model": {
                "class": self.model.__class__.__name__,
                "data": self.model.to_dict(),
            },
            "managed_agents": {
                managed_agent.name: managed_agent.__class__.__name__ for managed_agent in self.managed_agents.values()
            },
            "prompt_templates": self.prompt_templates,
            "max_steps": self.max_steps,
            "verbosity_level": int(self.logger.level),
            "grammar": self.grammar,
            "name": self.name,
            "description": self.description,
            "requirements": list(requirements),
        }
        return agent_dict



class ToolCallingAgent(MultiStepAgent):

    def __init__(
            self,
            tools: List[Tool],
            model: Callable[[List[Dict[str, str]]], ChatMessage],
            prompt_templates: Optional[PromptTemplates] = None,
            summary_interval: Optional[int] = None,
            prompts_type: Optional[str] = "default",
            **kwargs,
    ):
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            summary_interval=summary_interval,
            prompts_type=prompts_type,
            **kwargs,
        )
        try:
            prompt_path = importlib.resources.files(f"FlashOAgents.prompts.{prompts_type}").joinpath("toolcalling_agent.yaml")
            self.prompt_templates = prompt_templates or yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise AgentError(f"No prompt file：{prompts_type}/toolcalling_agent.yaml")
        except yaml.YAMLError as e:
            raise AgentError(f"Yaml parse error：{e}")
        self.summary_interval = summary_interval

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={"tools": self.tools},
        )
        return system_prompt

    def _run(self, task: str, images: List[str] | None = None) -> Generator[ActionStep | AgentType, None, None]:
        """
        Run the agent in streaming mode and returns a generator of all the steps.

        Args:
            task (`str`): Task to perform.
            images (`list[str]`): Paths to image(s).
        """
        logger.debug(f"[_run] ENTRY: task='{task}', images={images}, max_steps={self.max_steps}")
        final_answer = None
        self.step_number = 0
        logger.debug(f"[_run] Starting main loop: step_number={self.step_number}, final_answer={final_answer}")
        while final_answer is None and self.step_number <= self.max_steps:
            logger.debug(f"[_run] Loop iteration: step_number={self.step_number}, final_answer={final_answer}, max_steps={self.max_steps}")
            step_start_time = time.time()
            logger.debug(f"[_run] Creating ActionStep for step {self.step_number} at time {step_start_time}")
            memory_step = ActionStep(
                step_number=self.step_number,
                start_time=step_start_time,
                observations_images=images,
            )
            try:
                if self.step_number == 0:
                    logger.debug(f"[_run] Step 0: Calling planning_step")
                    self.planning_step(task)
                    self.step_number += 1
                    logger.debug(f"[_run] Planning step completed, step_number incremented to {self.step_number}")
                elif (
                    self.summary_interval is not None
                    and self.summary_interval > 0
                    and self.step_number % self.summary_interval == 0
                ):
                    logger.debug(f"[_run] Step {self.step_number}: Calling summary_step (interval={self.summary_interval})")
                    self.summary_step(
                        task,
                        step=self.step_number,
                    )
                    self.step_number += 1
                    logger.debug(f"[_run] Summary step completed, step_number incremented to {self.step_number}")
                self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)
                logger.debug(f"[_run] Calling step() method for step {self.step_number}")
                final_answer = self.step(memory_step)
                logger.debug(f"[_run] step() returned: final_answer={final_answer}, type={type(final_answer)}")
            except AgentError as e:
                logger.exception(f"[_run] AgentError caught at step {self.step_number}: {type(e).__name__}: {e}")
                memory_step.error = e
                raise
            except Exception as e:
                logger.exception(f"[_run] Unexpected exception at step {self.step_number}: {type(e).__name__}: {e}")
                raise
            finally:
                memory_step.end_time = time.time()
                memory_step.duration = memory_step.end_time - step_start_time
                logger.debug(f"[_run] Step {self.step_number} completed in {memory_step.duration:.2f}s")
                self.memory.steps.append(memory_step)
                self.step_number += 1
                logger.debug(f"[_run] Yielding memory_step, step_number incremented to {self.step_number}")
                yield memory_step

        if final_answer is None and self.step_number > self.max_steps:
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

    def reformulate_tool_fuctions(self, tool_list: List[Tool]) -> str:
        json_schema_list = []
        for tool in tool_list:
            required = []
            properties = deepcopy(tool.inputs)
            for key, value in properties.items():
                if value["type"] == "any":
                    value["type"] = "string"
                if not ("nullable" in value and value["nullable"]):
                    required.append(key)
            json_schema_list.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "properties": properties,
                    "required": required,
                }
            })
        return json.dumps(json_schema_list, indent=2, ensure_ascii=False)
    

    def step(self, memory_step: ActionStep, memory_messages=None) -> Union[None, Any]:
        """Single agent step.

        For tool-calling agents, if the model output in this step contains no valid tool call,
        the step is treated as a failed generation and retried a few times inside this function.
        """
        logger.debug(f"[step] ENTRY: step_number={memory_step.step_number}, memory_messages provided={memory_messages is not None}")
        
        # Set query_id at step level so communication stats are attributed correctly
        query_token = None
        if self.query_id:
            query_token = set_current_query_id(self.query_id)
            logger.debug(f"[step] Set query_id={self.query_id}")
        
        memory_messages = self.write_memory_to_messages() if memory_messages is None else memory_messages
        logger.debug(f"[step] Memory messages count: {len(memory_messages)}")
        self.input_messages = memory_messages

        # Add new step in logs
        memory_step.model_input_messages = memory_messages.copy()

        logger.debug(f"[step] Preparing instruction message with task: {self.task}")
        tool_functions_json = self.reformulate_tool_fuctions(list(self.tools.values()))
        logger.debug(f"[step] Tool functions JSON length: {len(tool_functions_json)}")
        logger.debug(f"[step] Available tools: {list(self.tools.keys())}")
        
        instruction_message = [{
            "role": MessageRole.USER,
            "content": [{
                "type": "text",
                "text": populate_template(
                    self.prompt_templates["step"]["pre_messages"],
                    variables={
                        "tool_functions_json": tool_functions_json,
                        "task": self.task
                    }
                )
            }]
        }]
        logger.debug(f"[step] Instruction message prepared, content length: {len(str(instruction_message))}")

        max_retries = 3
        retry_delay = 2  # seconds
        tool_execution_timeout = 300  # 5 minutes timeout for tool execution
        logger.debug(f"[step] Retry configuration: max_retries={max_retries}, retry_delay={retry_delay}s, tool_timeout={tool_execution_timeout}s")

        try:
            model_message: ChatMessage | None = None
            tool_calls_list: List[Dict[str, Any]] = []

            for attempt in range(max_retries):
                try:
                    logger.debug(
                        "[ToolCallingAgent.step] Attempt %s/%s: calling model with %s messages",
                        attempt + 1,
                        max_retries,
                        len(memory_messages + instruction_message),
                    )
                    model_message = self.model(memory_messages + instruction_message)
                    memory_step.model_output_messages = model_message

                    # Parse model output JSON (consistent with downstream logic)
                    logger.debug(f"[step] Model returned content type: {type(model_message.content)}, length: {len(str(model_message.content))}")
                    logger.debug(f"[step] Model content preview (first 500 chars): {str(model_message.content)[:500]}")
                    if hasattr(model_message, 'reasoning_content'):
                        logger.debug(f"[step] Model reasoning_content type: {type(model_message.reasoning_content)}, length: {len(str(model_message.reasoning_content)) if model_message.reasoning_content else 0}")
                    
                    try:
                        content_dict = json_repair.loads(model_message.content)
                        logger.debug(f"[step] Successfully parsed JSON, type: {type(content_dict)}")
                    except Exception as e:
                        logger.warning(
                            "[ToolCallingAgent.step] json_repair.loads failed on attempt %s/%s: %s",
                            attempt + 1,
                            max_retries,
                            e,
                        )
                        logger.debug(f"[step] Failed content was: {str(model_message.content)[:1000]}")
                        if attempt < max_retries - 1:
                            logger.debug(f"[step] Retrying after {retry_delay}s delay...")
                            time.sleep(retry_delay)
                            continue
                        raise Exception(f"Unsupported step output: {type(model_message.content)}: {e}")

                    # --- Extract answer_data and record think ---
                    # Three list formats are accepted:
                    # 1) single group (original logic): [{think, tools: [{name, arguments}, ...]}] -> list[0]["tools"]
                    # 2) multiple groups: [{think, tools: [...]}, {think, tools: [...]}, ...] -> flatten all tools
                    # 3) plain tool list: [{name, arguments}, ...] -> the whole list is answer_data
                    logger.debug(f"[step] Extracting answer_data from content_dict type: {type(content_dict)}")
                    if isinstance(content_dict, list):
                        logger.debug(f"[step] content_dict is list with length: {len(content_dict)}")
                        if not content_dict:
                            answer_data = []
                            memory_step.action_think = "No 'think' field in response"
                        elif isinstance(content_dict[0], dict) and "tools" in content_dict[0]:
                            # Multi-group if every item is a dict with a "tools" key
                            all_have_tools = all(
                                isinstance(item, dict) and "tools" in item for item in content_dict
                            )
                            if all_have_tools and len(content_dict) > 1:
                                # Multiple {think, tools: [...]} groups: flatten into one tool-call list
                                answer_data = []
                                for item in content_dict:
                                    sub = item.get("tools", [])
                                    if isinstance(sub, list):
                                        answer_data.extend([t for t in sub if isinstance(t, dict)])
                                    elif isinstance(sub, dict):
                                        answer_data.append(sub)
                                memory_step.action_think = content_dict[0].get(
                                    "think", "No 'think' field in response"
                                )
                                logger.debug(
                                    "[step] Multi-group format: flattened %s groups into %s tool calls",
                                    len(content_dict),
                                    len(answer_data),
                                )
                            else:
                                # Single group (original logic): take tools of the first item
                                answer_data = content_dict[0]["tools"]
                                memory_step.action_think = content_dict[0].get(
                                    "think", "No 'think' field in response"
                                )
                                logger.debug(
                                    "[step] Single-group format: extracted %s tool calls from list[0]['tools']",
                                    len(answer_data) if isinstance(answer_data, list) else 1,
                                )
                        else:
                            # Treat the list itself as the tool list (items are {name, arguments})
                            answer_data = content_dict
                            memory_step.action_think = "No 'think' field in response"
                            logger.debug(f"[step] Using entire list as answer_data (direct tool calls)")
                    elif isinstance(content_dict, dict):
                        answer_data = content_dict.get("tools", content_dict)
                        memory_step.action_think = content_dict.get("think", "No 'think' field in response")
                        logger.debug(f"[step] content_dict is dict, extracted tools field: {'tools' in content_dict}, think field: {'think' in content_dict}")
                    else:
                        answer_data = None
                        memory_step.action_think = "No 'think' field in response"
                        logger.warning(f"[step] content_dict is unexpected type: {type(content_dict)}")
                    logger.debug(f"[step] answer_data type: {type(answer_data)}, value preview: {str(answer_data)[:200] if answer_data else None}")

                    # --- Normalize to tool_calls_list: list[dict] of {name, arguments} ---
                    logger.debug(f"[step] Normalizing tool_calls_list from answer_data type: {type(answer_data)}")
                    tool_calls_list = []
                    if isinstance(answer_data, list):
                        # Collect dict items
                        raw_list = [tc for tc in answer_data if isinstance(tc, dict)]
                        # Flatten: {think, tools: [...]} contributes its tools; anything else is a single tool call
                        for tc in raw_list:
                            if "tools" in tc and isinstance(tc.get("tools"), list):
                                tool_calls_list.extend([t for t in tc["tools"] if isinstance(t, dict)])
                            elif "tools" in tc and isinstance(tc.get("tools"), dict):
                                tool_calls_list.append(tc["tools"])
                            elif tc.get("name") or "arguments" in tc:
                                # Already {name, arguments} or usable as a single call
                                tool_calls_list.append(tc)
                            # Otherwise skip (e.g. empty dict {})
                        logger.debug(
                            f"[step] answer_data is list: {len(raw_list)} raw items -> {len(tool_calls_list)} tool calls after flatten/filter"
                        )
                    elif isinstance(answer_data, dict):
                        tool_calls_list = [answer_data]
                        logger.debug(f"[step] answer_data is dict, wrapped in list")
                    elif answer_data is None:
                        tool_calls_list = []
                        logger.debug(f"[step] answer_data is None, tool_calls_list is empty")
                    else:
                        pass  # falls through to the invalid-type branch below

                    # Keep only calls whose name is non-empty and in self.tools (avoids running tool '')
                    available_tool_names = set(self.tools.keys())
                    _before = len(tool_calls_list)
                    tool_calls_list = [
                        tc for tc in tool_calls_list
                        if isinstance(tc, dict) and tc.get("name") and tc.get("name") in available_tool_names
                    ]
                    if _before != len(tool_calls_list):
                        logger.debug(
                            "[step] Filtered out %s invalid tool call(s) (empty name or unknown tool), %s remaining",
                            _before - len(tool_calls_list),
                            len(tool_calls_list),
                        )

                    if not isinstance(answer_data, list) and not isinstance(answer_data, dict) and answer_data is not None:
                        # answer_data is a str / other type: the model did not return the expected tool-call structure
                        logger.warning(
                            "[ToolCallingAgent.step] Invalid `tools` field type: %s, raw content preview: %s",
                            type(answer_data),
                            str(model_message.content)[:300],
                        )
                        logger.debug(f"[step] Invalid answer_data value: {str(answer_data)[:500]}")
                        if attempt < max_retries - 1:
                            logger.debug(f"[step] Retrying due to invalid answer_data type")
                            time.sleep(retry_delay)
                            continue
                        raise Exception(
                            f"Model step output `tools` field has invalid type: {type(answer_data)}. "
                            f"Raw content: {model_message.content}"
                        )
                    logger.debug(f"[step] Final tool_calls_list length: {len(tool_calls_list)}")
                    for idx, tc in enumerate(tool_calls_list):
                        logger.debug(f"[step] Tool call {idx}: name={tc.get('name', 'N/A')}, arguments keys={list(tc.get('arguments', {}).keys()) if isinstance(tc.get('arguments'), dict) else 'N/A'}")

                    # No valid tool call: treat as failed generation and retry
                    if not tool_calls_list:
                        logger.warning(
                            "[ToolCallingAgent.step] No valid tool calls found (attempt %s/%s). "
                            "Content preview: %s",
                            attempt + 1,
                            max_retries,
                            str(model_message.content)[:300],
                        )
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        raise Exception(
                            "No valid tool calls were found in model output after "
                            f"{max_retries} attempts. Raw content: {model_message.content}"
                        )

                    # Generation succeeded; leave the retry loop
                    logger.info(
                        "[ToolCallingAgent.step] Successfully extracted %s tool calls on attempt %s/%s",
                        len(tool_calls_list),
                        attempt + 1,
                        max_retries,
                    )
                    break

                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            "[ToolCallingAgent.step] Error in generation/parsing on attempt %s/%s: %s. "
                            "Retrying in %s seconds...",
                            attempt + 1,
                            max_retries,
                            repr(e),
                            retry_delay,
                        )
                        time.sleep(retry_delay)
                        continue
                    raise

            # ===== Tool execution (only after a valid tool_calls_list) =====
            logger.debug(f"[step] Starting tool execution phase with {len(tool_calls_list)} tool calls")
            memory_step.tool_calls = []
            final_answer_value = None
            observations = []

            self.logger.log(
                Panel(Text(f"Function calling number: {len(tool_calls_list)} calls: {str(tool_calls_list)}")),
                level=LogLevel.INFO,
            )

            # Parallel tool execution. (Please ensure that the tool implement has sufficient concurrency!)
            logger.debug(f"[step] Creating ThreadPoolExecutor for parallel tool execution, max_workers={self.max_workers}")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                tool_info_list = []
                
                for idx, tool_call in enumerate(tool_calls_list):
                    logger.debug(f"[step] Processing tool call {idx}/{len(tool_calls_list)}: {tool_call}")
                    # Defensive check
                    if not isinstance(tool_call, dict):
                        raise Exception(
                            f"Invalid tool_call item type: {type(tool_call)}. "
                            f"Raw item: {tool_call}. Full content: {model_message.content}"
                        )

                    tool_name = tool_call.get("name", "")
                    tool_arguments = tool_call.get("arguments", {})
                    tool_call_id = tool_call.get("id", "")
                    logger.debug(f"[step] Tool call {idx}: name='{tool_name}', id='{tool_call_id}', arguments type={type(tool_arguments)}")
                    
                    tool_call_obj = ToolCall(name=tool_name, arguments=tool_arguments, id=tool_call_id)
                    memory_step.tool_calls.append(tool_call_obj)
                    
                    self.logger.log(
                        Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")),
                        level=LogLevel.INFO,
                    )
                    
                    if tool_name == "final_answer":
                        logger.debug(f"[step] Final answer tool detected, extracting answer")
                        if isinstance(tool_arguments, dict):
                            answer = tool_arguments.get("answer", tool_arguments)
                        else:
                            answer = tool_arguments
                        logger.debug(f"[step] Final answer extracted: type={type(answer)}, value preview: {str(answer)[:200]}")
                        
                        final_answer_value = answer
                        self.logger.log(
                            Text(f"Final answer: {final_answer_value}", style=f"bold {YELLOW_HEX}"),
                            level=LogLevel.INFO,
                        )
                        observations.append(str(final_answer_value))
                        logger.debug(f"[step] Breaking tool execution loop due to final_answer")
                        break

                    logger.debug(f"[step] Submitting tool '{tool_name}' to executor")
                    future = executor.submit(self.execute_tool_call, tool_name, tool_arguments)
                    futures.append((idx, future, tool_name, tool_arguments))
                    tool_info_list.append((idx, tool_name, tool_arguments))
                    logger.debug(f"[step] Tool '{tool_name}' submitted, future added to list")
                
                if final_answer_value is not None:
                    logger.debug(f"[step] Final answer already set, returning early: {final_answer_value}")
                    memory_step.observations = "\n\n".join(observations) if observations else "No observations"
                    return final_answer_value
                
                if futures:
                    logger.debug(f"[step] Waiting for {len(futures)} tool execution futures to complete")
                    futures.sort(key=lambda x: x[0])
                    
                    for idx, future, tool_name, tool_arguments in futures:
                        logger.debug(f"[step] Waiting for tool '{tool_name}' (idx={idx}) to complete (timeout={tool_execution_timeout}s)")
                        
                        # Input communication bytes (central agent -> tool)
                        try:
                            input_str = json.dumps(tool_arguments, ensure_ascii=False) if isinstance(tool_arguments, dict) else str(tool_arguments)
                            input_bytes = len(input_str.encode('utf-8'))
                        except Exception:
                            input_bytes = 0
                        
                        try:
                            # Add timeout to prevent infinite waiting
                            observation = future.result(timeout=tool_execution_timeout)
                            logger.debug(f"[step] Tool '{tool_name}' completed successfully, observation type: {type(observation)}")
                            updated_information = str(observation).strip()
                            logger.debug(f"[step] Tool '{tool_name}' observation length: {len(updated_information)}")
                            
                            # Output communication bytes (tool -> central agent)
                            output_bytes = len(updated_information.encode('utf-8'))
                            
                            # Update communication stats
                            update_comm_usage(input_bytes, output_bytes)
                            
                            observations.append(
                                f"Results for tool call '{tool_name}' with arguments '{tool_arguments}':\n{updated_information}"
                            )
                            self.logger.log(
                                f"Observations: {updated_information.replace('[', '|')}",
                                level=LogLevel.INFO,
                            )
                        except FutureTimeoutError as e:
                            error_msg = f"Tool '{tool_name}' execution timed out after {tool_execution_timeout} seconds"
                            logger.error(f"[step] {error_msg}")
                            self.logger.error(f"Tool execution timeout: {error_msg}")
                            
                            # Count communication on timeout too (output = 0)
                            update_comm_usage(input_bytes, 0)
                            
                            observations.append(
                                f"Timeout error for tool call '{tool_name}' with arguments '{tool_arguments}':\n{error_msg}"
                            )
                            # Cancel the future if possible
                            try:
                                future.cancel()
                            except Exception:
                                pass  # Ignore errors when canceling
                        except Exception as e:
                            logger.exception(f"[step] Exception in tool '{tool_name}' execution: {type(e).__name__}: {e}")
                            observation = str(e)
                            self.logger.error(f"Tool execution error: {observation}")
                            
                            # Count communication on error too
                            output_bytes = len(str(observation).encode('utf-8'))
                            update_comm_usage(input_bytes, output_bytes)
                            
                            observations.append(
                                f"Error for tool call '{tool_name}' with arguments '{tool_arguments}':\n{observation}"
                            )
                else:
                    logger.debug(f"[step] No futures to wait for")
            memory_step.observations = "\n\n".join(observations) if observations else "No observations"
            logger.debug(f"[step] Total observations count: {len(observations)}, combined length: {len(memory_step.observations)}")
            
            # Handle final answer if present
            if final_answer_value is not None:
                logger.debug(f"[step] Returning final_answer_value: {final_answer_value}")
                return final_answer_value
            
            logger.debug(f"[step] No final answer, returning None")
            return None

        except Exception as e:
            logger.exception(f"[step] EXCEPTION in step method: {type(e).__name__}: {e}")
            logger.debug(f"[step] Exception occurred at step_number={memory_step.step_number}")
            raise AgentGenerationError(f"Error in generating tool call with model:\n{e}", self.logger) from e
        finally:
            # Restore previous query_id
            if query_token:
                reset_current_query_id(query_token)
                logger.debug(f"[step] Reset query_id")

