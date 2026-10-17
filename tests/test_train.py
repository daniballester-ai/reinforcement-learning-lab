import json
import os

import pandas as pd
import torch

from rl_lab.common import set_seed
from rl_lab.configs import DeepConfig
from rl_lab.deep.train import build_network, train

CPU = torch.device("cpu")


def _cfg(total_steps=600):
    return DeepConfig(env_id="CartPole-v1", total_steps=total_steps, learning_starts=100, buffer_size=1000,
                      batch_size=32, target_update=100, eps_decay_steps=300, eval_every=300,
                      eval_episodes=2, final_eval_episodes=2, checkpoint_every=300)


def test_train_writes_logs_and_final_json(tmp_path):
    train(_cfg(), "ddqn", seed=0, out_dir=str(tmp_path), device=CPU)
    for name in ("config.json", "episodes.csv", "eval.csv", "checkpoint.pt", "final.json"):
        assert os.path.exists(tmp_path / name)
    assert list(pd.read_csv(tmp_path / "eval.csv")["step"]) == [300, 600]
    assert json.load(open(tmp_path / "final.json"))["episodes"] == 2


def test_resume_continues_without_duplicating_eval_rows(tmp_path):  # Review Focus 2
    train(_cfg(300), "dqn", seed=0, out_dir=str(tmp_path), device=CPU)
    train(_cfg(600), "dqn", seed=0, out_dir=str(tmp_path), resume=True, device=CPU)
    assert list(pd.read_csv(tmp_path / "eval.csv")["step"]) == [300, 600]
    assert torch.load(tmp_path / "checkpoint.pt", weights_only=False)["step"] == 600


def test_dqn_and_ddqn_start_from_identical_weights():  # Review Focus 3
    nets = []
    for _ in ("dqn", "ddqn"):
        set_seed(7)
        nets.append(build_network("CartPole-v1", (4,), 2, 128))
    for a, b in zip(nets[0].parameters(), nets[1].parameters()):
        assert torch.equal(a, b)


def test_restart_after_interruption_does_not_duplicate_log_rows(tmp_path):
    # simula queda: o log tem linhas além do último checkpoint (passo 300)
    train(_cfg(300), "dqn", seed=0, out_dir=str(tmp_path), device=CPU)
    with open(tmp_path / "eval.csv", "a") as f:
        f.write("450,10.0,1.0\n")
    with open(tmp_path / "episodes.csv", "a") as f:
        f.write("450,99,10.0,0.05\n")
    train(_cfg(600), "dqn", seed=0, out_dir=str(tmp_path), resume=True, device=CPU)
    assert list(pd.read_csv(tmp_path / "eval.csv")["step"]) == [300, 600]
    steps = pd.read_csv(tmp_path / "episodes.csv")["step"]
    assert steps.is_monotonic_increasing and steps.is_unique


def test_restart_without_checkpoint_starts_logs_from_scratch(tmp_path):
    cfg = _cfg(200)
    cfg.checkpoint_every = 10_000  # só o checkpoint final (passo 200) existiria
    (tmp_path / "eval.csv").write_text("step,mean_return,std_return\n100,1.0,0.0\n")
    (tmp_path / "episodes.csv").write_text("step,episode,episode_return,epsilon\n100,1,5.0,0.5\n")
    train(cfg, "dqn", seed=0, out_dir=str(tmp_path), resume=True, device=CPU)
    assert (pd.read_csv(tmp_path / "eval.csv")["step"] == 100).sum() == 0  # linha velha removida
    assert (pd.read_csv(tmp_path / "episodes.csv")["episode_return"] == 5.0).sum() == 0
