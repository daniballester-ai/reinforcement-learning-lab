import numpy as np
import torch
import torch.nn as nn

from rl_lab.deep.dqn import DQNAgent, td_targets
from rl_lab.deep.networks import MLP


class Fixed(nn.Module):
    """Rede de brinquedo que devolve sempre a mesma linha de Q-valores."""

    def __init__(self, row):
        super().__init__()
        self.row = torch.tensor([row])

    def forward(self, x):
        return self.row.expand(x.shape[0], -1)


def _args():
    return torch.tensor([0.0]), torch.zeros(1, 1), torch.tensor([0.0])


def test_double_dqn_selects_with_online_and_evaluates_with_target():
    online, target = Fixed([9.0, 0.0]), Fixed([1.0, 10.0])
    r, nobs, d = _args()
    assert td_targets(online, target, r, nobs, d, 1.0, double=False).item() == 10.0
    assert td_targets(online, target, r, nobs, d, 1.0, double=True).item() == 1.0


def test_terminal_transition_does_not_bootstrap():
    online, target = Fixed([9.0, 0.0]), Fixed([1.0, 10.0])
    r, nobs, _ = _args()
    out = td_targets(online, target, torch.tensor([2.0]), nobs, torch.tensor([1.0]), 0.99, double=True)
    assert out.item() == 2.0


def test_learning_reduces_loss_on_a_fixed_batch():
    torch.manual_seed(0)
    agent = DQNAgent(MLP(4, 2), 2, 1e-2, 0.99, double=False, loss="mse", device=torch.device("cpu"))
    rng = np.random.default_rng(0)
    batch = (rng.normal(size=(32, 4)).astype(np.float32), rng.integers(0, 2, 32),
             rng.normal(size=32).astype(np.float32), rng.normal(size=(32, 4)).astype(np.float32),
             np.zeros(32, np.float32))
    first = agent.learn(batch)
    for _ in range(100):
        last = agent.learn(batch)
    assert last < first


def test_greedy_action_follows_q_values():
    agent = DQNAgent(MLP(4, 3), 3, 1e-3, 0.99, double=True, loss="mse", device=torch.device("cpu"))
    agent.q_net = Fixed([0.0, 5.0, 1.0])
    assert agent.act(np.zeros(4, np.float32), 0.0, np.random.default_rng(0)) == 1
