#!/usr/bin/env python3
"""W11 -- bounded-degree scaling atlas over (N, K_max, B).

Answers, per cell, what the two representations actually cost, so that any
v0.4.23 representation change is judged against measurement rather than
against the intuition that a sparse request produces a sparse model.

Three things this tool refuses to take on trust:

* **Realized, not requested, degree.** ``max_i k_i^in`` is computed from the
  realized edge table. A configuration that asked for ``max_in_degree=K`` and
  got something else is a finding, not a rounding error.
* **Realized, not requested, backend.** A materialized connection rule forces
  ``recurrent_backend="edge_list"`` regardless of what the runtime asked for
  (``_construct_core.py:577,713``). Each cell records both, and flags the
  cells where they differ -- those are the cells where a "dense" row in a
  benchmark table would be a lie.
* **Resident, not logical, memory.** ``M_logical_expanded`` is the dense
  ``N x N`` cost the topology would imply; ``M_resident`` is what is actually
  held. Their ratio is the headroom a representation change can address, and
  it is meaningless without the per-component split (M_W, M_edge, M_H, M_B).

Each cell runs in a fresh subprocess: peak RSS is not measurable in a process
that has already allocated the previous cell's arrays, and JAX's compilation
cache would otherwise make t_compile a function of cell order.

Usage:
    python scripts/perf/w11_scaling_atlas.py --out atlas.jsonl
    python scripts/perf/w11_scaling_atlas.py --cell 1000 100 edge_list   # one cell
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if sys.path[:1] != [str(ROOT)]:
    sys.path.insert(0, str(ROOT))

from scripts.perf.w10_allocation_map import (  # noqa: E402
    RSSSampler,
    build_config,
    walk_arrays,
)

# K_max = 0 is the sentinel for "unbounded" (no max_in_degree rule at all);
# argparse cannot carry `inf` through an int grid and `None` through a CLI.
K_UNBOUNDED = 0

DEFAULT_NS = (100, 1000, 5000, 10000)
DEFAULT_KS = (25, 50, 100, 200, K_UNBOUNDED)
DEFAULT_BACKENDS = ("dense", "edge_list")

# Component classification of the persistent arrays. walk_arrays yields paths
# of the form ``model.params['emitter'].W``, so match on the top-level params
# key plus (for the emitter) the attribute -- W is the whole point of the split
# and must never be pooled with the emitter's per-neuron state vectors.
_COMPONENT_BY_KEY = {
    "edge_list": "M_edge",
    "H": "M_H",
    "hdp": "M_H",
    "field": "M_B",
    "B": "M_B",
    "positions": "M_geometry",
}
_COMPONENTS = ("M_W", "M_edge", "M_H", "M_B", "M_geometry", "M_emitter_state", "M_other")

_KEY_RE = re.compile(r"^model\.params\['([^']+)'\](.*)$")


def _component_of(path: str) -> str:
    m = _KEY_RE.match(path)
    if not m:
        return "M_other"
    key, rest = m.group(1), m.group(2)
    if key == "emitter":
        return "M_W" if rest == ".W" else "M_emitter_state"
    return _COMPONENT_BY_KEY.get(key, "M_other")


def _split_bytes(model) -> tuple[dict[str, int], dict[str, int]]:
    """(component totals, the unclassified paths behind M_other).

    M_other is reported with its constituent paths rather than as a lump: an
    unattributed byte count is exactly the thing a representation change would
    later be judged against without anyone knowing what it contained.
    """
    out = dict.fromkeys(_COMPONENTS, 0)
    unclassified: dict[str, int] = {}
    for path, array in walk_arrays(model.params, path="model.params"):
        nbytes = int(np.asarray(array).nbytes)
        component = _component_of(path)
        out[component] += nbytes
        if component == "M_other":
            unclassified[path] = nbytes
    return out, unclassified


def _realized_degree(edge_list) -> tuple[int, int]:
    """(E, max_i k_i^in) from the realized edge table, never from config."""
    post = np.asarray(edge_list.post)
    if post.size == 0:
        return 0, 0
    counts = np.bincount(post)
    return int(post.size), int(counts.max())


def run_cell(n: int, k_max: int, backend: str, duration_ms: float, dt_ms: float,
             seed: int) -> dict:
    import jaxfne as jtfne

    k = None if k_max == K_UNBOUNDED else k_max
    # p_connect=0.0 suppresses the baseline recurrence so a max_in_degree rule
    # is the sole source of edges -- the only way E is attributable to K_max.
    cfg = build_config(n, 0.0 if k else None, k, duration_ms, dt_ms, seed)
    cfg = cfg.runtime(recurrent_backend=backend)

    rec: dict = {
        "N": n,
        "K_max_requested": "inf" if k is None else k,
        "backend_requested": backend,
    }

    with RSSSampler() as rss_construct:
        t0 = time.perf_counter()
        model = jtfne.construct(cfg)
        rec["t_construct_s"] = time.perf_counter() - t0
    rec["M_peak_construct_bytes"] = int(rss_construct.peak - rss_construct.baseline)

    edge_list = model.params.get("edge_list")
    E, k_in = _realized_degree(edge_list) if edge_list is not None else (0, 0)
    rec["E"] = E
    rec["E_per_N"] = E / n if n else 0.0
    rec["K_max_realized"] = k_in
    rec["K_max_honoured"] = (k is None) or (k_in <= k)

    split, unclassified = _split_bytes(model)
    rec.update(split)
    rec["M_other_paths"] = unclassified
    rec["M_persistent_bytes"] = sum(split.values())
    # What the same topology would cost if held densely, for the ratio that
    # bounds any representation change.
    rec["M_logical_expanded_bytes"] = n * n * 4
    rec["M_resident_over_logical"] = (
        rec["M_persistent_bytes"] / rec["M_logical_expanded_bytes"]
        if rec["M_logical_expanded_bytes"] else float("nan")
    )

    with RSSSampler() as rss_run:
        t0 = time.perf_counter()
        result = jtfne.simulate(model, duration_ms=duration_ms, dt_ms=dt_ms, seed=0)
        rec["t_first_run_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        jtfne.simulate(model, duration_ms=duration_ms, dt_ms=dt_ms, seed=0)
        rec["t_second_run_s"] = time.perf_counter() - t0
    rec["M_peak_run_bytes"] = int(rss_run.peak - rss_run.baseline)

    # t_compile is the part of the first run that the second run does not pay.
    n_steps = max(1, int(round(duration_ms / dt_ms)))
    rec["t_compile_s"] = max(0.0, rec["t_first_run_s"] - rec["t_second_run_s"])
    rec["t_step_s"] = rec["t_second_run_s"] / n_steps

    meta = getattr(result, "metadata", {}) or {}
    rec["backend_realized"] = meta.get("recurrent_backend", "unreported")
    rec["backend_substituted"] = rec["backend_realized"] != backend
    rec["M_recording_bytes"] = sum(
        int(np.asarray(a).nbytes) for _, a in walk_arrays(result, path="result")
    )
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ns", type=int, nargs="+", default=list(DEFAULT_NS))
    ap.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS),
                    help=f"max_in_degree values; {K_UNBOUNDED} means unbounded")
    ap.add_argument("--backends", nargs="+", default=list(DEFAULT_BACKENDS))
    ap.add_argument("--duration-ms", type=float, default=20.0)
    ap.add_argument("--dt-ms", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cell", nargs=3, metavar=("N", "K", "BACKEND"),
                    help="run exactly one cell in this process and print JSON")
    args = ap.parse_args(argv)

    if args.cell:
        rec = run_cell(int(args.cell[0]), int(args.cell[1]), args.cell[2],
                       args.duration_ms, args.dt_ms, args.seed)
        print(json.dumps(rec))
        return 0

    out = args.out.open("w", encoding="utf-8") if args.out else None
    try:
        for n in args.ns:
            for k in args.ks:
                for backend in args.backends:
                    cmd = [sys.executable, str(Path(__file__).resolve()),
                           "--cell", str(n), str(k), backend,
                           "--duration-ms", str(args.duration_ms),
                           "--dt-ms", str(args.dt_ms), "--seed", str(args.seed)]
                    t0 = time.perf_counter()
                    proc = subprocess.run(cmd, capture_output=True, text=True,
                                          cwd=str(ROOT), env={**os.environ})
                    if proc.returncode != 0:
                        # A refused cell is evidence, not an interruption: the
                        # refusal message is the measurement for that cell.
                        rec = {"N": n, "K_max_requested": "inf" if k == K_UNBOUNDED else k,
                               "backend_requested": backend, "status": "refused",
                               "returncode": proc.returncode,
                               "error": proc.stderr.strip().splitlines()[-1:] or [""]}
                    else:
                        rec = json.loads(proc.stdout.strip().splitlines()[-1])
                        rec["status"] = "ok"
                    rec["wall_s"] = time.perf_counter() - t0
                    line = json.dumps(rec)
                    print(line, flush=True)
                    if out:
                        out.write(line + "\n")
                        out.flush()
    finally:
        if out:
            out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
