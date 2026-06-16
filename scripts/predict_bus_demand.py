from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "downstream" / "hourly_demand_prediction.py"

if "--target-col" not in sys.argv:
    sys.argv.extend(["--target-col", "bus_demand"])

runpy.run_path(str(SCRIPT), run_name="__main__")
