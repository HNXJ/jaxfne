#!/usr/bin/env python3
"""Measure the realized H of the two frozen HDP presets (docs/tutorials/07_add_dynamics.md table).

Protocol: the jaxfne.hdp_network laminar column, N=500, 2000 ms, dt 0.5 ms, seeds 0-4.
DEFAULT_HDP runs at the base drive; DEFAULT_HDP_DESYNC at DRIVE_SCALE_DESYNC times it.
Statistics over the last 1000 ms: per-neuron time-mean H pooled over seeds (mean, std,
min, max), population rate, per-neuron rate std, and kappa (mean pairwise spike-train
correlation, jaxfne._signals._compute_kappa_synchrony_metric).

Usage:
    python scripts/measure_hdp_presets.py --out artifacts/protocols/hdp/hdp_presets/realized_h.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import jaxfne  # noqa: E402
from jaxfne._signals import _compute_kappa_synchrony_metric  # noqa: E402
from jaxfne.hdp_network import (  # noqa: E402
    BASE_DRIVE_BY_CELL_TYPE_DEFAULT,
    DEFAULT_HDP,
    DEFAULT_HDP_DESYNC,
    DRIVE_SCALE_DESYNC,
    HDPColumnConfig,
    apply_drive_correction,
    build_model,
    run,
)

assert "site-packages" not in jaxfne.__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

N_NEURONS = 500
DURATION_MS = 2000.0
DT_MS = 0.5
LATE_MS = 1000.0
SEEDS = (0, 1, 2, 3, 4)
PRESETS = {
    "DEFAULT_HDP": (DEFAULT_HDP, 1.0),
    "DEFAULT_HDP_DESYNC": (DEFAULT_HDP_DESYNC, DRIVE_SCALE_DESYNC),
}


def measure(preset: dict, drive_scale: float, seed: int) -> dict:
    base = {ct: v * drive_scale for ct, v in BASE_DRIVE_BY_CELL_TYPE_DEFAULT.items()}
    cfg = HDPColumnConfig(n_neurons=N_NEURONS, duration_ms=DURATION_MS, dt_ms=DT_MS,
                          seed=seed, base_drive_by_cell_type=base)
    model = apply_drive_correction(build_model(cfg), cfg)
    out = run(model, cfg, duration_ms=DURATION_MS, seed=seed, hdp_kwargs=preset)
    late_n = int(round(LATE_MS / DT_MS))
    spikes = np.asarray(out["spikes"])[-late_n:]
    H = np.asarray(out["diagnostics"]["H_trace"])[-late_n:]
    assert spikes.sum() > 0, "fixture must spike"
    rates = spikes.sum(axis=0) / (LATE_MS / 1000.0)
    return {
        "H_neuron_mean": H.mean(axis=0).tolist(),
        "rate_hz": float(rates.mean()),
        "rate_std_hz": float(rates.std()),
        "kappa": _compute_kappa_synchrony_metric(spikes),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    result = {"jaxfne_version": jaxfne.__version__, "n_neurons": N_NEURONS,
              "duration_ms": DURATION_MS, "dt_ms": DT_MS, "late_window_ms": LATE_MS,
              "seeds": list(SEEDS), "presets": {}}
    for name, (preset, drive_scale) in PRESETS.items():
        runs = [measure(preset, drive_scale, s) for s in SEEDS]
        h = np.concatenate([r["H_neuron_mean"] for r in runs])
        summary = {
            "drive_scale": drive_scale,
            "H_mean": float(h.mean()), "H_std": float(h.std()),
            "H_min": float(h.min()), "H_max": float(h.max()),
            "rate_hz": [r["rate_hz"] for r in runs],
            "rate_std_hz": [r["rate_std_hz"] for r in runs],
            "kappa": [r["kappa"] for r in runs],
        }
        result["presets"][name] = summary
        print(f"{name}: H={summary['H_mean']:.4f}+-{summary['H_std']:.4f} "
              f"[{summary['H_min']:.3f},{summary['H_max']:.3f}] "
              f"rate={np.mean(summary['rate_hz']):.2f}Hz "
              f"rate_std={np.mean(summary['rate_std_hz']):.2f}Hz "
              f"kappa={np.mean(summary['kappa']):.4f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
