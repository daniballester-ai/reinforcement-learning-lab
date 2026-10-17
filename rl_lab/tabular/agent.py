from collections import defaultdict

import numpy as np


def td_target(algo, q, reward, next_obs, next_action, terminated, gamma):
    if terminated:
        return reward
    if algo == "q_learning":
        return reward + gamma * np.max(q[next_obs])
    if algo == "sarsa":
        return reward + gamma * q[next_obs][next_action]
    raise ValueError(f"algoritmo tabular desconhecido: {algo}")


def epsilon_at(episode, episodes, eps_start, eps_end, decay_frac=0.5):
    span = max(1, int(episodes * decay_frac))
    return max(eps_end, eps_start - (eps_start - eps_end) * episode / span)


def _epsilon_greedy(q_row, eps, n_actions, rng):
    if rng.random() < eps:
        return int(rng.integers(n_actions))
    best = np.flatnonzero(q_row == q_row.max())
    return int(rng.choice(best))  # desempate aleatório


def train_tabular(env, algo, episodes, alpha, gamma, eps_start, eps_end, seed, decay_frac=0.5):
    rng = np.random.default_rng(seed)
    n = env.action_space.n
    q = defaultdict(lambda: np.zeros(n))
    returns = []
    for ep in range(episodes):
        eps = epsilon_at(ep, episodes, eps_start, eps_end, decay_frac)
        obs, _ = env.reset(seed=seed if ep == 0 else None)
        action = _epsilon_greedy(q[obs], eps, n, rng)
        total, done = 0.0, False
        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_action = _epsilon_greedy(q[next_obs], eps, n, rng)
            target = td_target(algo, q, reward, next_obs, next_action, terminated, gamma)
            q[obs][action] += alpha * (target - q[obs][action])
            obs, action = next_obs, next_action
            total += reward
            done = terminated or truncated
        returns.append(total)
    return q, returns


def evaluate_greedy(env, q, episodes, seed, max_steps=1000):
    """Política gulosa; max_steps protege contra ambientes sem limite de tempo (CliffWalking-v1)."""
    rng = np.random.default_rng(seed)
    n = env.action_space.n
    out = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total, done, steps = 0.0, False, 0
        while not done and steps < max_steps:
            action = _epsilon_greedy(q[obs], 0.0, n, rng)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            steps += 1
            done = terminated or truncated
        out.append(total)
    return out
