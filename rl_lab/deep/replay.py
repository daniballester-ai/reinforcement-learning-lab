import numpy as np


class ReplayBuffer:
    """Guarda (obs_t, a_t, r_t, done_t); obs_{t+1} é lido da posição t+1.

    Isso evita guardar cada quadro empilhado duas vezes (Atari: ~2,8 GB em vez de ~5,6 GB
    com 100 mil transições). A posição mais nova nunca é sorteada, pois seu sucessor ainda
    não foi escrito. Quando done=1 o sucessor é mascarado no alvo TD, então o valor lido
    (primeiro estado do episódio seguinte) é irrelevante.
    """

    def __init__(self, capacity, obs_shape, obs_dtype, seed=0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def add(self, obs, action, reward, done):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        start = self.pos if self.size == self.capacity else 0  # posição mais antiga
        idx = (start + self.rng.integers(0, self.size - 1, size=batch_size)) % self.capacity
        nxt = (idx + 1) % self.capacity
        return self.obs[idx], self.actions[idx], self.rewards[idx], self.obs[nxt], self.dones[idx]
