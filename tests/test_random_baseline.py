import numpy as np

from analysis.random_baseline import random_baseline


def test_random_baseline_returns_one_score_per_episode_in_the_game_range():
    scores = random_baseline("ALE/Pong-v5", episodes=2, seed=0)
    assert len(scores) == 2 and all(-21 <= s <= 21 for s in scores)


def test_random_baseline_is_reproducible_for_the_same_seed():
    a = random_baseline("ALE/Pong-v5", episodes=1, seed=3)
    b = random_baseline("ALE/Pong-v5", episodes=1, seed=3)
    assert np.allclose(a, b)
