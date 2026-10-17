import argparse
import json
import os

import gymnasium as gym
import numpy as np
import pandas as pd

from rl_lab import configs
from dataclasses import replace

from rl_lab.common import set_seed
from rl_lab.deep.train import train
from rl_lab.tabular.agent import evaluate_greedy, train_tabular


def run_dir(root, preset, tag, seed):
    path = os.path.join(root, preset, tag, f"seed{seed}")
    os.makedirs(path, exist_ok=True)
    return path


def run_tabular(preset, algo, seed, alpha, root, episodes=None, eval_episodes=1000):
    cfg = configs.TABULAR[preset]
    out = run_dir(root, preset, f"{algo}_a{alpha}", seed)
    if os.path.exists(os.path.join(out, "eval.json")):
        return out
    episodes = episodes or cfg["episodes"]
    env = gym.make(cfg["env_id"], **cfg["kwargs"])
    set_seed(seed)
    q, returns = train_tabular(env, algo, episodes, alpha, cfg["gamma"], 1.0, 0.05, seed)
    pd.DataFrame({"episode": range(1, len(returns) + 1), "episode_return": returns}).to_csv(
        os.path.join(out, "returns.csv"), index=False
    )
    if isinstance(env.observation_space, gym.spaces.Discrete):
        np.save(os.path.join(out, "q.npy"), np.array([q[s] for s in range(env.observation_space.n)]))
    scores = evaluate_greedy(env, q, eval_episodes, seed)
    with open(os.path.join(out, "eval.json"), "w") as f:
        json.dump({"mean_return": float(np.mean(scores)), "std_return": float(np.std(scores)),
                   "episodes": eval_episodes}, f)
    return out


def _apply_overrides(cfg, overrides):
    values = {}
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not hasattr(cfg, key):
            raise SystemExit(f"campo desconhecido em --set: {key}")
        old = getattr(cfg, key)
        values[key] = (raw.lower() in ("1", "true", "yes")) if isinstance(old, bool) else type(old)(raw)
    return replace(cfg, **values)


def run_deep(preset, algo, seed, root, resume=False, overrides=None):
    cfg = _apply_overrides(configs.deep_preset(preset), overrides)
    out = run_dir(root, preset, algo, seed)
    if os.path.exists(os.path.join(out, "final.json")):
        return out
    train(cfg, algo, seed, out, resume=resume)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--algo", required=True, choices=["q_learning", "sarsa", "dqn", "ddqn"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--out", default="runs")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--set", action="append", default=[], metavar="CAMPO=VALOR")
    a = p.parse_args(argv)
    if a.algo in ("q_learning", "sarsa"):
        print(run_tabular(a.env, a.algo, a.seed, a.alpha, a.out))
    else:
        print(run_deep(a.env, a.algo, a.seed, a.out, a.resume, a.set))


if __name__ == "__main__":
    main()
