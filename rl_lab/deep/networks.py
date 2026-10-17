import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x.float())


class AtariCNN(nn.Module):
    """CNN do artigo da DeepMind (Nature, 2015) para entrada 4×84×84 uint8."""

    def __init__(self, n_actions, in_frames=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_frames, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.net(x.float() / 255.0)
