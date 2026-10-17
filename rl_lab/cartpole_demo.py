import os

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.wrappers import RecordVideo

EPSILONS = (0.0, 0.1, 0.3, 0.5)


def heuristic_action(obs):
    return 1 if obs[2] + 0.5 * obs[3] > 0 else 0


class RandomActionWrapper(gym.ActionWrapper):
    """Com probabilidade epsilon troca a ação do agente por uma ação aleatória (slide 68)."""

    def __init__(self, env, epsilon, seed=0):
        super().__init__(env)
        self.epsilon = epsilon
        self._rng = np.random.default_rng(seed)

    def action(self, action):
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.env.action_space.n))
        return action


def _run(env, policy, episodes, seed):
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(policy(obs))
            total += reward
            done = terminated or truncated
        returns.append(total)
    return returns


def run_demo(root, episodes=100, seed=0):
    out = os.path.join(root, "cartpole_demo")
    os.makedirs(out, exist_ok=True)
    rows = []
    rng = np.random.default_rng(seed)

    def random_policy(_obs):
        return int(rng.integers(2))

    for ep, r in enumerate(_run(gym.make("CartPole-v1"), random_policy, episodes, seed)):
        rows.append(("random", ep, r))
    for eps in EPSILONS:
        env = RandomActionWrapper(gym.make("CartPole-v1"), eps, seed)
        for ep, r in enumerate(_run(env, heuristic_action, episodes, seed)):
            rows.append((f"heuristic_eps{eps}", ep, r))
    pd.DataFrame(rows, columns=["condition", "episode", "episode_return"]).to_csv(
        os.path.join(out, "results.csv"), index=False)

    for name, policy in (("random", random_policy), ("heuristic", heuristic_action)):
        env = RecordVideo(gym.make("CartPole-v1", render_mode="rgb_array"),
                          os.path.join(out, "videos"), name_prefix=name, episode_trigger=lambda i: True)
        _run(env, policy, 1, seed)
        env.close()


if __name__ == "__main__":
    run_demo("runs")
