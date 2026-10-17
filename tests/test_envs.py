import numpy as np
import pytest

from rl_lab.envs.factory import make_env


def test_atari_observation_shape_and_dtype():  # Review Focus 5
    pytest.importorskip("ale_py")
    for env_id in ("ALE/Pong-v5", "ALE/SpaceInvaders-v5"):
        env = make_env(env_id, seed=0)
        obs, _ = env.reset()
        assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8
        obs, *_ = env.step(0)
        assert obs.shape == (4, 84, 84)
        assert env.action_space.n >= 3


def test_energy_shaping_keeps_raw_reward_and_is_finite():
    env = make_env("MountainCar-v0", seed=0, shaping=True)
    env.reset()
    for _ in range(20):
        _, reward, _, _, info = env.step(env.action_space.sample())
        assert np.isfinite(reward)
        assert info["raw_reward"] == -1.0


def test_plain_cartpole_has_no_shaping_info():
    env = make_env("CartPole-v1", seed=0)
    env.reset()
    _, _, _, _, info = env.step(0)
    assert "raw_reward" not in info
