import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_runs(root, preset, tag, filename):
    base = Path(root) / preset / tag
    out = []
    for d in sorted(base.glob("seed*")):
        f = d / filename
        if f.exists():
            out.append(pd.read_csv(f))
    return out


def mean_std_curve(dfs, xcol, ycol, window=20, n_points=200):
    """Suaviza cada execução, interpola numa grade comum e agrega entre sementes."""
    smoothed = [(df[xcol].to_numpy(float), df[ycol].rolling(window, min_periods=1).mean().to_numpy(float))
                for df in dfs]
    lo = max(x.min() for x, _ in smoothed)
    hi = min(x.max() for x, _ in smoothed)
    grid = np.linspace(lo, hi, n_points)
    stack = np.vstack([np.interp(grid, x, y) for x, y in smoothed])
    return grid, stack.mean(axis=0), stack.std(axis=0)


def final_table(root, preset, tags, filename):
    """Média e desvio ENTRE sementes do retorno final; std_episodes é o desvio médio entre os
    episódios de avaliação de cada execução (a única dispersão informativa quando há 1 semente)."""
    rows = []
    for tag in tags:
        base = Path(root) / preset / tag
        res = [json.load(open(f)) for f in sorted(base.glob(f"seed*/{filename}"))]
        if res:
            vals = [r["mean_return"] for r in res]
            rows.append({"tag": tag, "n_seeds": len(vals), "mean": float(np.mean(vals)),
                         "std": float(np.std(vals)),
                         "std_episodes": float(np.mean([r.get("std_return", 0.0) for r in res]))})
    return pd.DataFrame(rows, columns=["tag", "n_seeds", "mean", "std", "std_episodes"])


def plot_curves(curves, title, xlabel, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, (x, mean, std) in curves.items():
        ax.plot(x, mean, label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("recompensa por episódio (média móvel)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


ARROWS = {"frozenlake": "←↓→↑", "frozenlake_noslip": "←↓→↑", "cliffwalking": "↑→↓←"}


def arrow_for(preset, action):
    return ARROWS[preset][action]


def plot_policy_heatmap(q, shape, title, path, preset="frozenlake"):
    values = q.max(axis=1).reshape(shape)
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(values, cmap="viridis")
    for (i, j), best in np.ndenumerate(q.argmax(axis=1).reshape(shape)):
        ax.text(j, i, arrow_for(preset, best), ha="center", va="center", color="white")
    ax.set_title(title)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
