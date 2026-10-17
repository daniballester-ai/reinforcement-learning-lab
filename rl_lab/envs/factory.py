import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation


class EnergyShaping(gym.Wrapper):
    """Recompensa modelada por energia mecânica: r + γ·Φ(s') − Φ(s)."""

    def __init__(self, env, gamma=0.99, scale=1000.0):
        super().__init__(env)
        self.gamma = gamma
        self.scale = scale
        self._phi_prev = 0.0

    def _phi(self, obs):
        x, v = float(obs[0]), float(obs[1])
        return self.scale * (0.5 * v * v + 0.0025 * np.sin(3 * x))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._phi_prev = self._phi(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        phi = self._phi(obs)
        shaped = reward + self.gamma * phi - self._phi_prev
        self._phi_prev = phi
        info["raw_reward"] = reward
        return obs, float(shaped), terminated, truncated, info


def make_env(env_id, seed, shaping=False):
    if env_id.startswith("ALE/"):
        import ale_py

        gym.register_envs(ale_py)
        env = gym.make(env_id, frameskip=1)
        env = AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, grayscale_obs=True)
        env = FrameStackObservation(env, 4)
    else:
        env = gym.make(env_id)
        if shaping and env_id == "MountainCar-v0":
            env = EnergyShaping(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env
