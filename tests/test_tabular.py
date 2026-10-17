from collections import defaultdict

import gymnasium as gym
import numpy as np

from rl_lab.tabular.agent import epsilon_at, evaluate_greedy, td_target, train_tabular


def _q(n=2):
    return defaultdict(lambda: np.zeros(n))


def test_q_learning_target_uses_max_of_next_state():
    q = _q()
    q["s2"] = np.array([1.0, 5.0])
    assert td_target("q_learning", q, 1.0, "s2", 0, False, 0.5) == 1.0 + 0.5 * 5.0


def test_sarsa_target_uses_action_actually_taken():
    q = _q()
    q["s2"] = np.array([1.0, 5.0])
    assert td_target("sarsa", q, 1.0, "s2", 0, False, 0.5) == 1.0 + 0.5 * 1.0


def test_terminal_target_ignores_next_state():
    assert td_target("sarsa", _q(), 2.0, "x", 0, True, 0.9) == 2.0


def test_epsilon_schedule_is_linear_then_flat():
    assert epsilon_at(0, 100, 1.0, 0.1) == 1.0
    assert epsilon_at(50, 100, 1.0, 0.1) == 0.1
    assert epsilon_at(99, 100, 1.0, 0.1) == 0.1


def test_tuple_observations_work_as_q_keys():  # Review Focus 1: Blackjack
    env = gym.make("Blackjack-v1")
    q, returns = train_tabular(env, "q_learning", 200, 0.1, 1.0, 1.0, 0.05, seed=0)
    assert len(returns) == 200
    assert all(isinstance(k, tuple) for k in q)


def test_q_learning_solves_deterministic_frozenlake():
    env = gym.make("FrozenLake-v1", is_slippery=False)
    q, _ = train_tabular(env, "q_learning", 2000, 0.5, 0.99, 1.0, 0.05, seed=0)
    assert np.mean(evaluate_greedy(env, q, 20, seed=1)) == 1.0


class _Endless(gym.Env):
    """Ambiente que nunca termina (como o CliffWalking-v1 sem TimeLimit)."""

    observation_space = gym.spaces.Discrete(1)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        return 0, {}

    def step(self, action):
        return 0, -1.0, False, False, {}


def test_greedy_evaluation_stops_on_environments_without_time_limit():
    q = defaultdict(lambda: np.zeros(2))
    scores = evaluate_greedy(_Endless(), q, episodes=2, seed=0, max_steps=50)
    assert scores == [-50.0, -50.0]
