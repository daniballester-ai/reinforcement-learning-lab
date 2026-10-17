from dataclasses import dataclass


@dataclass
class DeepConfig:
    env_id: str
    total_steps: int
    hidden: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    train_freq: int = 1
    target_update: int = 500
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 10_000
    eval_every: int = 5_000
    eval_episodes: int = 10
    final_eval_episodes: int = 100
    eval_eps: float = 0.0
    checkpoint_every: int = 50_000
    loss: str = "mse"
    clip_reward: bool = False
    shaping: bool = False


TABULAR = {
    "frozenlake": dict(env_id="FrozenLake-v1", kwargs={"is_slippery": True}, episodes=20_000, gamma=0.99),
    "frozenlake_noslip": dict(env_id="FrozenLake-v1", kwargs={"is_slippery": False}, episodes=20_000, gamma=0.99),
    "cliffwalking": dict(env_id="CliffWalking-v1", kwargs={}, episodes=1_000, gamma=0.99),
    "taxi": dict(env_id="Taxi-v4", kwargs={}, episodes=10_000, gamma=0.99),
    "blackjack": dict(env_id="Blackjack-v1", kwargs={}, episodes=500_000, gamma=1.0),
}


from dataclasses import replace  # noqa: E402

_ATARI = dict(total_steps=1_000_000, lr=1e-4, batch_size=32, buffer_size=100_000, learning_starts=20_000,
              train_freq=4, target_update=8_000, eps_end=0.01, eps_decay_steps=250_000, eval_every=100_000,
              eval_episodes=5, final_eval_episodes=30, eval_eps=0.05, checkpoint_every=50_000,
              loss="huber", clip_reward=True)

_DEEP = {
    "cartpole": DeepConfig(env_id="CartPole-v1", total_steps=100_000),
    "cartpole_prof": DeepConfig(env_id="CartPole-v1", total_steps=100_000, lr=1e-4),
    "mountaincar": DeepConfig(env_id="MountainCar-v0", total_steps=200_000),
    "mountaincar_shaped": DeepConfig(env_id="MountainCar-v0", total_steps=200_000, shaping=True),
    "pong": DeepConfig(env_id="ALE/Pong-v5", **_ATARI),
    "spaceinvaders": DeepConfig(env_id="ALE/SpaceInvaders-v5", **_ATARI),
}


def deep_preset(name):
    return replace(_DEEP[name])
