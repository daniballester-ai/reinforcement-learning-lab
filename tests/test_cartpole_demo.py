import gymnasium as gym
import numpy as np
import pandas as pd

from rl_lab.cartpole_demo import RandomActionWrapper, heuristic_action, run_demo


def test_heuristic_pushes_toward_the_lean():
    assert heuristic_action(np.array([0, 0, 0.1, 0.0])) == 1
    assert heuristic_action(np.array([0, 0, -0.1, 0.0])) == 0


def test_wrapper_with_epsilon_zero_keeps_the_action():
    env = RandomActionWrapper(gym.make("CartPole-v1"), epsilon=0.0)
    assert all(env.action(1) == 1 for _ in range(50))


def test_wrapper_with_epsilon_one_samples_both_actions():
    env = RandomActionWrapper(gym.make("CartPole-v1"), epsilon=1.0, seed=0)
    assert {env.action(1) for _ in range(200)} == {0, 1}


def test_run_demo_writes_all_conditions(tmp_path):
    run_demo(str(tmp_path), episodes=3, seed=0)
    df = pd.read_csv(tmp_path / "cartpole_demo" / "results.csv")
    assert set(df["condition"]) == {"random", "heuristic_eps0.0", "heuristic_eps0.1",
                                    "heuristic_eps0.3", "heuristic_eps0.5"}
    assert len(df) == 5 * 3
