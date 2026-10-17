import json

import numpy as np
import pandas as pd

from analysis.summarize import final_table, load_runs, mean_std_curve, plot_curves


def _make_run(root, preset, tag, seed, rows, final=None):
    d = root / preset / tag / f"seed{seed}"
    d.mkdir(parents=True)
    pd.DataFrame(rows, columns=["step", "episode_return"]).to_csv(d / "episodes.csv", index=False)
    if final is not None:
        json.dump({"mean_return": final, "std_return": 0.0, "episodes": 5}, open(d / "final.json", "w"))


def test_curve_handles_runs_of_different_length(tmp_path):  # Review Focus 4
    _make_run(tmp_path, "cartpole", "dqn", 0, [(i * 10, float(i)) for i in range(1, 50)])
    _make_run(tmp_path, "cartpole", "dqn", 1, [(i * 10, float(i)) for i in range(1, 30)])
    dfs = load_runs(str(tmp_path), "cartpole", "dqn", "episodes.csv")
    x, mean, std = mean_std_curve(dfs, "step", "episode_return", window=5)
    assert len(dfs) == 2 and not np.isnan(mean).any() and not np.isnan(std).any()
    assert x.max() <= 290


def test_constant_returns_give_zero_std_and_plot_does_not_crash(tmp_path):  # MountainCar sem sucesso
    for seed in range(3):
        _make_run(tmp_path, "mountaincar", "dqn", seed, [(200 * i, -200.0) for i in range(1, 40)], final=-200.0)
    x, mean, std = mean_std_curve(load_runs(str(tmp_path), "mountaincar", "dqn", "episodes.csv"),
                                  "step", "episode_return", window=5)
    assert np.allclose(mean, -200.0) and np.allclose(std, 0.0)
    plot_curves({"dqn": (x, mean, std)}, "t", "passos", str(tmp_path / "c.png"))
    assert (tmp_path / "c.png").exists()


def test_single_seed_has_zero_std_in_table(tmp_path):
    _make_run(tmp_path, "pong", "dqn", 0, [(10, 1.0)], final=7.0)
    t = final_table(str(tmp_path), "pong", ["dqn"], "final.json")
    assert t.loc[0, "n_seeds"] == 1 and t.loc[0, "mean"] == 7.0 and t.loc[0, "std"] == 0.0


def test_missing_runs_yield_empty_table_not_an_error(tmp_path):
    assert final_table(str(tmp_path), "nope", ["dqn"], "final.json").empty
    assert load_runs(str(tmp_path), "nope", "dqn", "episodes.csv") == []


def test_arrow_glyphs_follow_each_env_action_order():
    from analysis.summarize import arrow_for

    assert [arrow_for("frozenlake", a) for a in range(4)] == ["←", "↓", "→", "↑"]
    assert [arrow_for("frozenlake_noslip", a) for a in range(4)] == ["←", "↓", "→", "↑"]
    assert [arrow_for("cliffwalking", a) for a in range(4)] == ["↑", "→", "↓", "←"]


def test_table_reports_the_std_between_evaluation_episodes_for_single_seed_runs(tmp_path):
    d = tmp_path / "pong" / "dqn" / "seed0"
    d.mkdir(parents=True)
    json.dump({"mean_return": -17.7, "std_return": 1.8, "episodes": 30}, open(d / "final.json", "w"))
    t = final_table(str(tmp_path), "pong", ["dqn"], "final.json")
    assert t.loc[0, "std"] == 0.0 and t.loc[0, "n_seeds"] == 1  # entre sementes: indefinido com 1 semente
    assert abs(t.loc[0, "std_episodes"] - 1.8) < 1e-9  # dispersão dos episódios de avaliação
