import numpy as np

from rl_lab.envs.factory import make_env


def random_baseline(env_id, episodes, seed):
    """Retorno de uma política uniformemente aleatória, com o mesmo pré-processamento do treino."""
    env = make_env(env_id, seed + 10_000)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(episodes):
        env.reset()
        total, done = 0.0, False
        while not done:
            _, reward, terminated, truncated, _ = env.step(int(rng.integers(env.action_space.n)))
            total += reward
            done = terminated or truncated
        scores.append(total)
    return scores


if __name__ == "__main__":
    for game in ("ALE/Pong-v5", "ALE/SpaceInvaders-v5"):
        s = random_baseline(game, episodes=30, seed=0)
        print(f"{game}: média {np.mean(s):.2f}, desvio {np.std(s):.2f} (30 episódios)")
