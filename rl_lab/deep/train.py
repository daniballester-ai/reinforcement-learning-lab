import json
import os
from dataclasses import asdict

import numpy as np
import torch

from rl_lab.common import CsvLog, pick_device, set_seed
from rl_lab.deep.dqn import DQNAgent
from rl_lab.deep.networks import AtariCNN, MLP
from rl_lab.deep.replay import ReplayBuffer
from rl_lab.envs.factory import make_env


def build_network(env_id, obs_shape, n_actions, hidden):
    if env_id.startswith("ALE/"):
        return AtariCNN(n_actions, in_frames=obs_shape[0])
    return MLP(obs_shape[0], n_actions, hidden)


def _epsilon(step, cfg):
    frac = min(1.0, step / cfg.eps_decay_steps)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def evaluate(agent, cfg, seed, episodes, rng_seed):
    env = make_env(cfg.env_id, seed + 10_000)
    rng = np.random.default_rng(rng_seed)
    scores = []
    for _ in range(episodes):
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(agent.act(obs, cfg.eval_eps, rng))
            total += reward
            done = terminated or truncated
        scores.append(total)
    return float(np.mean(scores)), float(np.std(scores))


def _truncate_log(path, max_step):
    """Remove do log as linhas com passo > max_step (resquícios de uma execução interrompida)."""
    import csv

    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    keep = [rows[0]] + [r for r in rows[1:] if int(float(r[0])) <= max_step]
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(keep)


def _save_checkpoint(path, agent, step, episode):
    tmp = path + ".tmp"
    torch.save({"agent": agent.state_dict(), "step": step, "episode": episode}, tmp)
    os.replace(tmp, path)  # atômico: uma queda de sessão nunca deixa checkpoint pela metade


def train(cfg, algo, seed, out_dir, resume=False, device=None):
    assert algo in ("dqn", "ddqn"), algo
    device = device or pick_device()
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)
    env = make_env(cfg.env_id, seed, cfg.shaping)
    obs_shape, n_actions = env.observation_space.shape, env.action_space.n
    agent = DQNAgent(build_network(cfg.env_id, obs_shape, n_actions, cfg.hidden), n_actions, cfg.lr,
                     cfg.gamma, algo == "ddqn", cfg.loss, device)
    buf = ReplayBuffer(cfg.buffer_size, obs_shape, env.observation_space.dtype, seed)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({**asdict(cfg), "algo": algo, "seed": seed}, f, indent=2)
    ep_log = CsvLog(os.path.join(out_dir, "episodes.csv"), ["step", "episode", "episode_return", "epsilon"])
    ev_log = CsvLog(os.path.join(out_dir, "eval.csv"), ["step", "mean_return", "std_return"])

    ckpt = os.path.join(out_dir, "checkpoint.pt")
    step = episode = 0
    if resume and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device, weights_only=False)
        agent.load_state_dict(state["agent"])
        step, episode = state["step"], state["episode"]
    _truncate_log(ep_log.path, step)  # sem checkpoint, step=0 e os logs recomeçam do zero
    _truncate_log(ev_log.path, step)
    # O buffer não é salvo (pesado demais); após retomar, reaquece com a política atual.
    rng = np.random.default_rng(seed + step)
    warmup_until = step + cfg.learning_starts

    obs, _ = env.reset()
    ep_return = 0.0
    while step < cfg.total_steps:
        eps = _epsilon(step, cfg)
        action = agent.act(obs, eps, rng)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        buf.add(obs, action, float(np.sign(reward)) if cfg.clip_reward else float(reward), float(done))
        ep_return += info.get("raw_reward", reward)
        obs = next_obs
        step += 1
        if done:
            episode += 1
            ep_log.write(step=step, episode=episode, episode_return=ep_return, epsilon=eps)
            obs, _ = env.reset()
            ep_return = 0.0
        if step >= warmup_until and len(buf) > cfg.batch_size and step % cfg.train_freq == 0:
            agent.learn(buf.sample(cfg.batch_size))
        if step % cfg.target_update == 0:
            agent.sync_target()
        if step % cfg.eval_every == 0:
            mean, std = evaluate(agent, cfg, seed, cfg.eval_episodes, seed + step)
            ev_log.write(step=step, mean_return=mean, std_return=std)
        if step % cfg.checkpoint_every == 0 or step == cfg.total_steps:
            _save_checkpoint(ckpt, agent, step, episode)

    mean, std = evaluate(agent, cfg, seed, cfg.final_eval_episodes, seed + 1)
    with open(os.path.join(out_dir, "final.json"), "w") as f:
        json.dump({"mean_return": mean, "std_return": std, "episodes": cfg.final_eval_episodes}, f)
