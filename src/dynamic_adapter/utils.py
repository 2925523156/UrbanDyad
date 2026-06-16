from __future__ import annotations

import csv
import random
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_run_dir(output_root: Path, prefix: str = "run") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(output_root / f"{prefix}_{timestamp}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


class CSVMetricLogger:
    def __init__(self, path: Path, fieldnames: Iterable[str], append: bool = False) -> None:
        self.path = path
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        should_write_header = True
        if append and self.path.exists() and self.path.stat().st_size > 0:
            should_write_header = False
        mode = "a" if append else "w"
        with self.path.open(mode, encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if should_write_header:
                writer.writeheader()

    def log(self, row: Mapping[str, object]) -> None:
        with self.path.open("a", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(row)
