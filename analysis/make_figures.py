import os

import numpy as np
import pandas as pd

from analysis.summarize import final_table, load_runs, mean_std_curve, plot_curves, plot_policy_heatmap
from rl_lab.configs import TABULAR

ROOT, FIG, TAB = "runs", "relatorio/figuras", "relatorio/tabelas"
GRIDS = {"frozenlake": (4, 4), "frozenlake_noslip": (4, 4), "cliffwalking": (4, 12)}


def r1():
    for preset in TABULAR:
        tags = [f"{a}_a{al}" for a in ("q_learning", "sarsa") for al in (0.05, 0.1, 0.5)]
        final_table(ROOT, preset, tags, "eval.json").to_csv(f"{TAB}/r1_{preset}.csv", index=False)
        curves = {}
        for algo in ("q_learning", "sarsa"):
            dfs = load_runs(ROOT, preset, f"{algo}_a0.1", "returns.csv")
            if dfs:
                curves[algo] = mean_std_curve(dfs, "episode", "episode_return", window=max(10, len(dfs[0]) // 50))
        if curves:
            plot_curves(curves, f"{preset}: Q-learning × SARSA (α=0,1)", "episódio", f"{FIG}/r1_{preset}.png")
        if preset in GRIDS:
            for algo in ("q_learning", "sarsa"):
                path = f"{ROOT}/{preset}/{algo}_a0.1/seed0/q.npy"
                if os.path.exists(path):
                    plot_policy_heatmap(np.load(path), GRIDS[preset], f"{preset} — {algo}",
                                        f"{FIG}/r1_{preset}_{algo}_policy.png", preset=preset)


def r2():
    path = f"{ROOT}/cartpole_demo/results.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.groupby("condition")["episode_return"].agg(["mean", "std"]).reset_index().to_csv(
            f"{TAB}/r2_cartpole.csv", index=False)


def r3():
    for preset in ("cartpole", "cartpole_prof", "mountaincar", "mountaincar_shaped", "pong", "spaceinvaders"):
        tags = ["dqn", "ddqn"]
        final_table(ROOT, preset, tags, "final.json").to_csv(f"{TAB}/r3_{preset}.csv", index=False)
        curves = {}
        for algo in tags:
            dfs = load_runs(ROOT, preset, algo, "episodes.csv")
            if dfs:
                curves[algo] = mean_std_curve(dfs, "step", "episode_return", window=20)
        if curves:
            plot_curves(curves, f"{preset}: DQN × Double DQN", "passos", f"{FIG}/r3_{preset}.png")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    r1()
    r2()
    r3()
