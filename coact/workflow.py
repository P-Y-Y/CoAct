"""The central planner: plan -> dispatch to sub-agents in parallel -> re-plan."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import asdict

import numpy as np

from coact.framework.context import get_steps
from coact.framework.models import Model
from coact.models import embed
from coact.prompts import ACTION_TEMPLATE, FINALIZE_PROMPT, PLANNING_PROMPT
from coact.scheduler import RandomScheduler, Scheduler, SemanticScheduler
from coact.subagent import ActionResult, SubAgent
from coact.utils import extract_json

logger = logging.getLogger(__name__)


class CentralWorkflow:
    """Solves one query with a planner LLM and a pool of parallel sub-agents.

    Every round the planner either answers or decomposes the remaining information gaps
    into atomic actions. Sub-agents execute those actions concurrently; successful
    results are appended to the shared history that the next round plans from.
    """

    def __init__(
        self,
        model: Model,
        subagents: list[SubAgent],
        query_idx: int,
        scheduler: str = "semantic",
        max_rounds: int = 10,
        step_budget: int = 120,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.subagents = subagents
        self.query_idx = query_idx
        self.scheduler = scheduler
        self.max_rounds = max_rounds
        self.step_budget = step_budget
        self.rng = random.Random(seed)

    async def run(self, task: str) -> str:
        history: list[dict] = []
        for round_id in range(1, self.max_rounds + 1):
            logger.info(str(history))
            plan = self._plan(task, history)
            if plan.get("completed") is True:
                return plan.get("final_answer", "")

            actions = plan.get("actions") or []
            if not actions:
                logger.info("No actions returned")
            results = await self._execute_round(round_id, task, history, actions)
            if results:
                history.append({"round": round_id, "results": [asdict(r) for r in results]})
        raise RuntimeError("Max rounds reached")

    def close(self) -> None:
        for agent in self.subagents:
            agent.close()

    # ------------------------------------------------------------------ planning

    def _plan(self, task: str, history: list[dict]) -> dict:
        exhausted = get_steps(self.query_idx) >= self.step_budget
        template = FINALIZE_PROMPT if exhausted else PLANNING_PROMPT
        prompt = template.format(task=task, history=json.dumps(history, indent=2))

        logger.info("llm call start")
        response = self.model(messages=[{"role": "user", "content": prompt}]).content
        logger.info("llm call end,result:" + response)

        plan = normalize_plan(extract_json(response))
        logger.info(f"Planning results are: {plan}")
        return plan

    # ----------------------------------------------------------------- execution

    async def _execute_round(
        self, round_id: int, task: str, history: list[dict], actions: list[dict]
    ) -> list[ActionResult]:
        if not actions:
            return []
        scheduler = await self._make_scheduler(actions)
        ids = [a["action_id"] for a in actions]
        done: set[str] = set()  # actions completed by any agent this round
        results_by_agent: list[list[ActionResult]] = [[] for _ in self.subagents]

        for agent in self.subagents:
            agent.reset_round()

        async def drive(agent_idx: int, action_idx: int | None) -> None:
            agent = self.subagents[agent_idx]
            while action_idx is not None:
                steps = get_steps(self.query_idx)
                logger.info(f"counter:{steps}")
                if steps > self.step_budget:
                    logger.info(
                        f"Agent {agent.name}: counter > {self.step_budget}, skipping task execution"
                    )
                    return

                action = actions[action_idx]
                action_text = ACTION_TEMPLATE.format(**action)
                run_id = f"round{round_id}_agent{agent_idx}"
                size = (
                    len(run_id)
                    + len(task)
                    + len(json.dumps(history))
                    + len(action_text)
                    + len(str([action["action_id"]]))
                )
                logger.info(f"center to sub communication size:{size} bytes")

                result = await agent.run(run_id, task, history, action["action_id"], action_text)
                if result.is_success:
                    results_by_agent[agent_idx].append(result)
                    done.add(result.action_id)

                action_idx = self._next_action(agent_idx, agent, scheduler, ids, done)

        starts = scheduler.initial(len(self.subagents))
        for agent_idx, action_idx in enumerate(starts):
            self.subagents[agent_idx].tried.add(ids[action_idx])
        await asyncio.gather(*(drive(i, a) for i, a in enumerate(starts)))

        return [r for agent_results in results_by_agent for r in agent_results]

    def _next_action(
        self, agent_idx: int, agent: SubAgent, scheduler: Scheduler, ids: list[str], done: set[str]
    ) -> int | None:
        """Pick an action no agent has finished yet and this agent has not tried yet."""
        unfinished = [i for i, action_id in enumerate(ids) if action_id not in done]
        if not unfinished:
            logger.info(f"Agent {agent.name}: all tasks have been completed by some agent")
            return None
        candidates = [i for i in unfinished if ids[i] not in agent.tried]
        if not candidates:
            logger.info(f"Agent {agent.name}: no more uncompleted tasks to try")
            return None

        own_done = [i for i, action_id in enumerate(ids) if action_id in agent.completed]
        all_done = [i for i, action_id in enumerate(ids) if action_id in done]
        choice = scheduler.pick_next(agent_idx, candidates, own_done, all_done)
        agent.tried.add(ids[choice])
        return choice

    async def _make_scheduler(self, actions: list[dict]) -> Scheduler:
        if self.scheduler == "random":
            return RandomScheduler(len(actions), self.rng)

        objectives = [action["objective"] for action in actions]
        for objective in objectives:
            logger.info(f"Embedding Text: {objective}")
        embeddings = await asyncio.to_thread(embed, objectives)
        return SemanticScheduler(np.array(embeddings).reshape(len(objectives), -1))


def normalize_plan(plan: dict) -> dict:
    """Map a plan that came back in tool-calling format onto the planner schema."""
    if "think" not in plan or "tools" not in plan:
        return plan
    logger.info("Detected tool calling format in planning response")
    tools = plan.get("tools") or []
    if tools and tools[0].get("name") == "final_answer":
        answer = tools[0].get("arguments", {}).get("answer", "")
        return {"completed": True, "final_answer": answer, "actions": []}
    logger.warning(f"Unexpected tool calling format in planning: {plan}")
    return {"completed": False, "final_answer": "", "actions": []}
