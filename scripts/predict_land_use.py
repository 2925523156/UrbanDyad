from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "downstream" / "static_population_landuse_prediction.py"

if "--task" not in sys.argv:
    sys.argv.extend(["--task", "landuse"])

runpy.run_path(str(SCRIPT), run_name="__main__")
