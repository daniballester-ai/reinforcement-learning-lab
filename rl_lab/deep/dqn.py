import copy

import numpy as np
import torch
import torch.nn.functional as F


def td_targets(q_net, target_net, rewards, next_obs, dones, gamma, double):
    with torch.no_grad():
        if double:  # Double DQN: a rede online escolhe a ação, a rede-alvo a avalia
            best = q_net(next_obs).argmax(dim=1, keepdim=True)
            next_q = target_net(next_obs).gather(1, best).squeeze(1)
        else:  # DQN com rede-alvo: max direto da rede-alvo
            next_q = target_net(next_obs).max(dim=1).values
        return rewards + gamma * (1.0 - dones) * next_q


class DQNAgent:
    def __init__(self, q_net, n_actions, lr, gamma, double, loss, device):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.double = double
        self.q_net = q_net.to(device)
        self.target_net = copy.deepcopy(q_net).to(device)
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = F.smooth_l1_loss if loss == "huber" else F.mse_loss

    def act(self, obs, eps, rng):
        if rng.random() < eps:
            return int(rng.integers(self.n_actions))
        with torch.no_grad():
            q = self.q_net(torch.as_tensor(np.asarray(obs), device=self.device).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def learn(self, batch):
        obs, act, rew, nobs, done = (torch.as_tensor(x, device=self.device) for x in batch)
        q = self.q_net(obs).gather(1, act.long().unsqueeze(1)).squeeze(1)
        target = td_targets(self.q_net, self.target_net, rew, nobs, done, self.gamma, self.double)
        loss = self.loss_fn(q, target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.opt.step()
        return loss.item()

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def state_dict(self):
        return {"q": self.q_net.state_dict(), "target": self.target_net.state_dict(),
                "opt": self.opt.state_dict()}

    def load_state_dict(self, d):
        self.q_net.load_state_dict(d["q"])
        self.target_net.load_state_dict(d["target"])
        self.opt.load_state_dict(d["opt"])
