from rl_lab.sweeps import classic_jobs


def test_main_experiments_run_before_the_optional_extras():
    jobs = classic_jobs("runs", shaped=True)
    assert len(jobs) == 40
    first_20 = {j[0] for j in jobs[:20]}
    assert first_20 == {"cartpole", "mountaincar"}  # 2 presets x 2 algoritmos x 5 sementes
    assert {j[0] for j in jobs[20:]} == {"cartpole_prof", "mountaincar_shaped"}


def test_extras_are_ordered_seed_first_so_partial_runs_stay_balanced():
    extras = classic_jobs("runs", shaped=True)[20:]
    assert [j[2] for j in extras[:4]] == [0, 0, 0, 0]  # a semente 0 de todas as combinações vem antes da 1
