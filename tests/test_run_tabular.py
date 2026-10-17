import json
import os

from rl_lab.run import run_tabular


def test_run_tabular_writes_artifacts_and_is_idempotent(tmp_path):
    out = run_tabular("taxi", "sarsa", seed=0, alpha=0.1, root=str(tmp_path), episodes=30, eval_episodes=5)
    assert out.endswith(os.path.join("taxi", "sarsa_a0.1", "seed0"))
    res = json.load(open(os.path.join(out, "eval.json")))
    assert set(res) == {"mean_return", "std_return", "episodes"} and res["episodes"] == 5
    assert os.path.exists(os.path.join(out, "returns.csv"))
    assert os.path.exists(os.path.join(out, "q.npy"))  # Taxi é Discrete
    mtime = os.path.getmtime(os.path.join(out, "eval.json"))
    run_tabular("taxi", "sarsa", seed=0, alpha=0.1, root=str(tmp_path), episodes=30, eval_episodes=5)
    assert os.path.getmtime(os.path.join(out, "eval.json")) == mtime


def test_blackjack_skips_q_array(tmp_path):
    out = run_tabular("blackjack", "q_learning", seed=0, alpha=0.1, root=str(tmp_path), episodes=50, eval_episodes=5)
    assert not os.path.exists(os.path.join(out, "q.npy"))
