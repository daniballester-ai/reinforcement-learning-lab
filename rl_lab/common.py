import csv
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CsvLog:
    """CSV append-only com cabeçalho fixo; reabrir o mesmo arquivo continua de onde parou."""

    def __init__(self, path: str, fields):
        self.path = path
        self.fields = list(fields)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(self.fields)

    def write(self, **row) -> None:
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([row[k] for k in self.fields])
