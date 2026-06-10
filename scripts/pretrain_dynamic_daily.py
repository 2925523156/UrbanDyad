from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "src" / "dynamic_adapter"

sys.path.insert(0, str(MODULE_DIR))
runpy.run_path(str(MODULE_DIR / "train_daily.py"), run_name="__main__")
