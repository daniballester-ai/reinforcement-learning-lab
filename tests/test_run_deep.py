import json

import pytest

from rl_lab import configs
from rl_lab.run import main

SMOKE = ["--set", "total_steps=400", "--set", "learning_starts=100", "--set", "buffer_size=1000",
         "--set", "eps_decay_steps=200", "--set", "eval_every=200", "--set", "eval_episodes=1",
         "--set", "final_eval_episodes=1", "--set", "checkpoint_every=200", "--set", "target_update=100"]


def test_cli_runs_a_deep_preset_end_to_end(tmp_path):
    main(["--env", "cartpole", "--algo", "dqn", "--seed", "0", "--out", str(tmp_path), *SMOKE])
    out = tmp_path / "cartpole" / "dqn" / "seed0"
    assert json.load(open(out / "final.json"))["episodes"] == 1
    assert json.load(open(out / "config.json"))["total_steps"] == 400


def test_preset_values_match_the_spec():
    assert configs.deep_preset("cartpole").total_steps == 100_000
    assert configs.deep_preset("mountaincar").total_steps == 200_000
    assert configs.deep_preset("cartpole_prof").lr == 1e-4
    assert configs.deep_preset("mountaincar_shaped").shaping is True
    for game, env_id in (("pong", "ALE/Pong-v5"), ("spaceinvaders", "ALE/SpaceInvaders-v5")):
        c = configs.deep_preset(game)
        assert (c.env_id, c.total_steps, c.lr, c.batch_size, c.buffer_size) == (env_id, 1_000_000, 1e-4, 32, 100_000)
        assert (c.learning_starts, c.train_freq, c.target_update, c.eps_end, c.eps_decay_steps) == (20_000, 4, 8_000, 0.01, 250_000)
        assert (c.loss, c.clip_reward, c.final_eval_episodes, c.eval_eps) == ("huber", True, 30, 0.05)


def test_unknown_override_field_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        main(["--env", "cartpole", "--algo", "dqn", "--out", str(tmp_path), "--set", "nao_existe=1"])
