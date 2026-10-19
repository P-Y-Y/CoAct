"""Strategies for assigning a round's planned actions to sub-agents.

Each round, every sub-agent starts on one action from ``initial(k)`` and, after finishing,
pulls another via ``pick_next`` until no unfinished action is left that it has not tried.
"""

from __future__ import annotations

import itertools
import random
from typing import Protocol

import numpy as np


class Scheduler(Protocol):
    def initial(self, k: int) -> list[int]:
        """Return up to ``k`` action indices to start the round with, one per sub-agent."""

    def pick_next(
        self, agent_idx: int, candidates: list[int], own_done: list[int], all_done: list[int]
    ) -> int:
        """Choose the next action for a sub-agent among ``candidates``.

        ``own_done`` / ``all_done`` are the actions completed this round by this agent and by
        any agent, respectively.
        """


class RandomScheduler:
    """Ablation baseline: random start actions, random re-assignment."""

    def __init__(self, num_actions: int, rng: random.Random) -> None:
        self.num_actions = num_actions
        self.rng = rng

    def initial(self, k: int) -> list[int]:
        order = list(range(self.num_actions))
        self.rng.shuffle(order)
        return order[:k]

    def pick_next(
        self, agent_idx: int, candidates: list[int], own_done: list[int], all_done: list[int]
    ) -> int:
        return self.rng.choice(candidates)


class SemanticScheduler:
    """Assign actions by the semantic (dis)similarity of their objectives.

    Start actions are the ``k`` most mutually unrelated ones, so sub-agents begin in
    different directions. Afterwards each agent prefers the action most related to what
    *it* has completed relative to what *all* agents have completed, so it can reuse its
    own context.
    """

    def __init__(self, embeddings: np.ndarray) -> None:
        unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        cos = unit @ unit.T
        np.fill_diagonal(cos, 0.0)
        self.exp_cos = np.exp(cos)
        # Softmax over each row: probability that action i "points to" action j.
        self.log_prob = np.log(self.exp_cos / self.exp_cos.sum(axis=1, keepdims=True))

    def dissimilarity(self, i: int, j: int) -> float:
        return -(self.log_prob[i, j] + self.log_prob[j, i])

    def initial(self, k: int) -> list[int]:
        n = len(self.exp_cos)
        if n <= k:
            return list(range(n))
        return list(
            max(
                itertools.combinations(range(n), k),
                key=lambda combo: sum(
                    self.dissimilarity(i, j) for i, j in itertools.combinations(combo, 2)
                ),
            )
        )

    def pick_next(
        self, agent_idx: int, candidates: list[int], own_done: list[int], all_done: list[int]
    ) -> int:
        if not own_done:
            return candidates[agent_idx % len(candidates)]

        def affinity(candidate: int) -> float:
            own = self.exp_cos[candidate, own_done].sum()
            total = self.exp_cos[candidate, all_done].sum()
            return own / total if total > 0 else 0.0

        return max(candidates, key=affinity)
