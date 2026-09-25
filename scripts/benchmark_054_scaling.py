#!/usr/bin/env python3
"""0.5.4 ENGINE item 5: toy k-area composition scaling probe.

Chains k laminar columns (k = 1..4) with feedforward cross edges, runs one
short simulation per k, and records wall time + peak RSS. This is a
measurement, not a gate: it proves 0.5.5 is not the first multi-area scale
test. The frozen 0.5.1 matrix (artifacts/perf/matrix_051*.json) is untouched;
output goes to artifacts/perf/scaling_054.json (tracked).

Usage:
    python scripts/benchmark_054_scaling.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "artifacts" / "perf" / "scaling_054.json"

DURATION_MS, DT_MS, SEED = 40.0, 0.5, 7
N_PER_AREA = 12


def _rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 1e6


class _PeakSampler:
    def __init__(self) -> None:
        self.samples: list = []
        self._stop = threading.Event()

    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()

    def _loop(self):
        import time as _t

        while not self._stop.is_set():
            self.samples.append(_rss_mb())
            _t.sleep(0.01)

    def peak(self) -> float:
        return max(self.samples) if self.samples else _rss_mb()


def _cell(k: int) -> dict:
    import jax
    import numpy as np

    import jaxfne as jtfne

    def _col(name, seed):
        cfg = (
            jtfne.Configuration()
            .runtime(duration_ms=DURATION_MS, dt_ms=DT_MS, seed=seed, recurrent_backend="edge_list")
            .column(name, layers=["L2/3", "L4"], n=N_PER_AREA)
            .cell_types({"E": 0.8, "PV": 0.2})
            .connectivity()
            .set_emitter(family="izhikevich")
            .probes(["spikes", "V_m"], n_contacts=8)
        )
        return jtfne.construct(cfg)

    members = [_col(f"V{i}", i) for i in range(k)]
    edges = [
        dict(
            source={"model": i, "area": f"V{i}", "cell_type": "E"},
            target={"model": i + 1, "area": f"V{i + 1}"},
            probability=0.3,
            weight=0.5,
            sign="excitatory",
        )
        for i in range(k - 1)
    ]
    t0 = time.perf_counter()
    with _PeakSampler() as peak:
        if k == 1:
            model = members[0]
        else:
            model = jtfne.connect(*members, namespace=tuple(f"A{i}" for i in range(k)), edges=edges)
        t_construct = time.perf_counter()
        sig = jtfne.simulate(model, duration_ms=DURATION_MS, dt_ms=DT_MS, seed=SEED)
        jax.block_until_ready(sig.V_m)
        t_sim = time.perf_counter()
    el = model.params["edge_list"]
    vm = np.asarray(sig.V_m)
    return {
        "k_areas": k,
        "n_neurons": int(model.params["emitter"].n_neurons),
        "n_edges": int(el.n_edges),
        "construct_ms": (t_construct - t0) * 1000.0,
        "simulate_ms": (t_sim - t_construct) * 1000.0,
        "peak_rss_mb": peak.peak(),
        "spikes_total": int(np.asarray(sig.spikes).sum()),
        "finite": bool(np.isfinite(vm).all()),
    }


def main() -> None:
    cells = [_cell(k) for k in (1, 2, 3, 4)]
    OUT.write_text(json.dumps({"cells": cells}, indent=2), encoding="utf-8")
    for c in cells:
        print(c)


if __name__ == "__main__":
    main()
