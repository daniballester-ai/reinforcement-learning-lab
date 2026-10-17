import argparse
from concurrent.futures import ProcessPoolExecutor

import torch

from rl_lab import configs, run as R

ALPHAS = (0.05, 0.1, 0.5)


def _tabular_job(job):
    preset, algo, seed, alpha, root = job
    R.run_tabular(preset, algo, seed, alpha, root)
    return job


def _classic_job(job):
    torch.set_num_threads(1)  # uma thread por processo; os workers dividem os núcleos
    preset, algo, seed, root = job
    R.run_deep(preset, algo, seed, root, resume=True)
    return job


def tabular_jobs(root, seeds=range(10)):
    return [(p, a, s, alpha, root) for p in configs.TABULAR for a in ("q_learning", "sarsa")
            for alpha in ALPHAS for s in seeds]


def classic_jobs(root, seeds=range(5), shaped=False):
    """Essenciais primeiro (CartPole e MountainCar); extras depois, semente a semente."""
    def tier(presets):
        return [(p, a, s, root) for s in seeds for p in presets for a in ("dqn", "ddqn")]

    return tier(["cartpole", "mountaincar"]) + tier(["cartpole_prof"] + (["mountaincar_shaped"] if shaped else []))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("which", choices=["tabular", "classic"])
    p.add_argument("--root", default="runs")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--shaped", action="store_true")
    a = p.parse_args(argv)
    if a.which == "tabular":
        jobs, fn = tabular_jobs(a.root), _tabular_job
    else:
        jobs, fn = classic_jobs(a.root, shaped=a.shaped), _classic_job
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, job in enumerate(ex.map(fn, jobs), 1):
            print(f"[{i}/{len(jobs)}] {job[:4]}", flush=True)


if __name__ == "__main__":
    main()
