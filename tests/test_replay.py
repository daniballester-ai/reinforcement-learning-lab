import numpy as np
import torch

from rl_lab.deep.networks import AtariCNN, MLP
from rl_lab.deep.replay import ReplayBuffer


def _fill(buf, n):
    for t in range(n):
        buf.add(np.array([t], np.float32), 0, float(t), 0.0)


def test_next_obs_is_next_slot_and_newest_is_never_sampled():
    buf = ReplayBuffer(10, (1,), np.float32)
    _fill(buf, 6)
    obs, _, _, nxt, _ = buf.sample(200)
    assert np.all(nxt[:, 0] == obs[:, 0] + 1)
    assert obs.max() <= 4


def test_wraparound_never_pairs_newest_with_oldest():
    buf = ReplayBuffer(5, (1,), np.float32)
    _fill(buf, 12)  # guarda 7..11
    obs, _, _, nxt, _ = buf.sample(500)
    assert np.all(nxt[:, 0] == obs[:, 0] + 1)
    assert obs.min() >= 7 and obs.max() <= 10


def test_len_is_capped_at_capacity():
    buf = ReplayBuffer(5, (1,), np.float32)
    _fill(buf, 12)
    assert len(buf) == 5


def test_network_output_shapes():
    assert MLP(4, 2)(torch.zeros(3, 4)).shape == (3, 2)
    assert AtariCNN(6)(torch.zeros(2, 4, 84, 84, dtype=torch.uint8)).shape == (2, 6)
