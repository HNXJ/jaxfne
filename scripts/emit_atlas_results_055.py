"""Emit generated Atlas result files for the 0.5.5 coverage rows.

Runs the canonical runners (no new physics) and writes each summary under
``artifacts/atlas/results/``. Usage: ``python scripts/emit_atlas_results_055.py
[--only at01 at05 at07 reduction]``. Each file carries ``_meta`` (runner,
jaxfne version, wall time); nothing is edited after the run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jaxfne  # noqa: E402
from artifacts.atlas import at01_at06_052 as A  # noqa: E402
from artifacts.atlas import at07_at04_053 as B  # noqa: E402

OUT = ROOT / "artifacts" / "atlas" / "results"

JOBS = {
    "at01": ("at01_055.json", "at01_at06_052:run_at01", lambda: A.run_at01()),
    "at05": ("at05_055.json", "at01_at06_052:run_at05", lambda: A.run_at05()),
    "reduction": ("at_reduction_055.json", "at01_at06_052:run_reduction", lambda: A.run_reduction()),
    "at07": ("at07_055.json", "at07_at04_053:run_at07", lambda: B.run_at07()),
}


def _plain(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (bool, int, float, str)) or x is None:
        return x
    return repr(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(JOBS), default=sorted(JOBS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for key in args.only:
        name, runner, fn = JOBS[key]
        t0 = time.time()
        res = fn()
        res = _plain(res)
        res["_meta"] = {"runner": runner, "jaxfne_version": jaxfne.__version__,
                        "wall_s": round(time.time() - t0, 1)}
        (OUT / name).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        print(key, "->", name, f"{res['_meta']['wall_s']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
