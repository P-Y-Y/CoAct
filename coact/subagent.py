"""Sub-agent: a tool-calling executor that carries out one planned action at a time."""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from coact.framework.agents import ToolCallingAgent
from coact.framework.models import Model
from coact.framework.monitoring import LogLevel
from coact.framework.tools import Tool
from coact.prompts import OWN_RESULTS_SECTION, SUBTASK_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    action_id: str
    action: str
    result: str
    is_success: bool


class SubAgent:
    def __init__(
        self,
        name: str,
        query_idx: int,
        model: Model,
        tools: list[Tool],
        max_steps: int,
        step_budget: int,
    ) -> None:
        self.name = name
        self.query_idx = query_idx
        self.model = model
        self.tools = tools
        self.max_steps = max_steps
        self.step_budget = step_budget
        # One worker thread per sub-agent: the framework agent is synchronous.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self.reset_round()

    def reset_round(self) -> None:
        """Forget the bookkeeping of the previous planning round."""
        self.tried: set[str] = set()
        self.completed: list[str] = []
        self.round_results: list[ActionResult] = []

    async def run(
        self, run_id: str, task: str, history: list[dict], action_id: str, action: str
    ) -> ActionResult:
        """Execute one action with a fresh agent and record the outcome.

        ``run_id`` identifies this assignment (round and agent) in communication accounting.
        """
        logger.info(f"SubAgent {self.name} start executing actions {[action_id]}")
        prompt = SUBTASK_PROMPT.format(
            task=task,
            history=json.dumps(history, indent=2),
            own_results=self._own_results_section(),
            action=action,
        )

        agent = ToolCallingAgent(
            tools=self.tools,
            query_idx=self.query_idx,
            model=self.model,
            max_steps=self.max_steps,
            step_budget=self.step_budget,
            summary_interval=None,
            verbosity_level=LogLevel.ERROR,
        )
        # Run in a copied context so the worker thread sees CURRENT_TASK_IDX for log tagging.
        ctx = contextvars.copy_context()
        output = await asyncio.get_running_loop().run_in_executor(
            self._executor, ctx.run, agent.run, prompt
        )

        result = ActionResult(action_id, action, str(output), "success" in str(output).lower())
        logger.info(
            f"SubAgent {self.name} finished executing actions {[action_id]} "
            f"with success={result.is_success}"
        )
        logger.info(f"sub to center communication size:{self._message_size(run_id, result)} bytes")

        if result.is_success:
            self.completed.append(action_id)
            self.round_results.append(result)
        return result

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def _own_results_section(self) -> str:
        if not self.round_results:
            return ""
        results = json.dumps([asdict(r) for r in self.round_results], indent=2)
        return OWN_RESULTS_SECTION.format(results=results)

    def _message_size(self, run_id: str, result: ActionResult) -> int:
        message = {
            "task_id": run_id,
            "agent": self.name,
            "subtask": result.action,
            "result": result.result,
            "is_success": result.is_success,
            "action_ids": [result.action_id],
        }
        return len(str(message))
