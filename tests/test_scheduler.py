import itertools
import random

import numpy as np

from coact.scheduler import RandomScheduler, SemanticScheduler


def test_initial_picks_most_mutually_dissimilar_actions():
    # Actions 0/1 and 2/3 are near-duplicates; 4 is orthogonal to both groups.
    embeddings = np.array([[1, 0, 0], [1, 0.05, 0], [0, 1, 0], [0.05, 1, 0], [0, 0, 1]], float)
    picked = SemanticScheduler(embeddings).initial(3)
    assert 4 in picked
    assert len({0, 1} & set(picked)) == 1 and len({2, 3} & set(picked)) == 1


def test_initial_returns_everything_when_few_actions():
    assert SemanticScheduler(np.eye(2)).initial(3) == [0, 1]


def test_pick_next_prefers_actions_related_to_own_work():
    embeddings = np.array([[1, 0], [0, 1], [1, 0.1], [0.1, 1]], float)
    scheduler = SemanticScheduler(embeddings)
    # Agent finished 0, another agent finished 1: candidate 2 (close to 0) wins.
    assert scheduler.pick_next(0, [2, 3], own_done=[0], all_done=[0, 1]) == 2


def test_pick_next_round_robin_without_own_results():
    scheduler = SemanticScheduler(np.eye(4))
    assert scheduler.pick_next(1, [2, 3], own_done=[], all_done=[0]) == 3


def test_random_scheduler_is_seeded():
    picks = [RandomScheduler(6, random.Random(0)).initial(3) for _ in range(2)]
    assert picks[0] == picks[1] and len(set(picks[0])) == 3
    assert all(i in range(6) for i in itertools.chain(*picks))
