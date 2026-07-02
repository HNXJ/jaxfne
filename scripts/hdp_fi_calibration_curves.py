"""F-I (drive-current -> steady-state rate) calibration curves, per cell type.

Per user spec: the deterministic relationship between total drive current
to a neuron and its firing rate, so the desired equilibrium H / drive
correction can be set analytically instead of via per-cell-type bisection
(as scripts/hdp_100_stability_sweep.py's DRIVE_CORRECTION_BY_CELL_TYPE was
originally found).

K_HDP=0 throughout (weights frozen) to isolate the pure I->rate
relationship from plasticity. Drive is scaled by a uniform per-cell-type
multiplicative factor (same mechanism as apply_drive_correction), and rate
is read from a 2s tail window after a 1s settle.

Usage: PYTHONPATH=. python3 scripts/hdp_fi_calibration_curves.py
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jaxfne.hdp_network import (
    HDPColumnConfig, build_model, run, BASE_HDP_KWARGS_DEFAULT, DEFAULT_HDP,
    BASE_DRIVE_BY_CELL_TYPE_DEFAULT,
)

OUTPUT_DIR = Path("outputs/hdp_fi_calibration_curves")

N_NEURONS = 250
DURATION_MS = 3000.0
SETTLE_MS = 1000.0
DT_MS = 0.5
SEED = 0
CELL_TYPES = ["E", "PV", "SST", "VIP"]
DRIVE_FACTORS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # F-023: explicit base_drive_by_cell_type preserves this script's prior
    # always-4.0-for-all behavior now that HDPColumnConfig's dataclass
    # default is None (emitter-preset passthrough) instead of
    # BASE_DRIVE_BY_CELL_TYPE_DEFAULT -- see skills/FRICTIONS_STACK.md F-023.
    cfg = HDPColumnConfig(
        n_neurons=N_NEURONS, duration_ms=DURATION_MS, dt_ms=DT_MS, seed=SEED,
        base_drive_by_cell_type=dict(BASE_DRIVE_BY_CELL_TYPE_DEFAULT),
    )
    model = build_model(cfg)
    labels = np.asarray(model.params["emitter"].labels)
    base_drive = np.asarray(model.params["emitter"].drive)

    hdp_kwargs = dict(BASE_HDP_KWARGS_DEFAULT)
    hdp_kwargs.update(DEFAULT_HDP)
    hdp_kwargs["K_HDP"] = 0.0  # frozen weights -- pure F-I, no plasticity confound

    curves: dict[str, list[list[float]]] = {ct: [] for ct in CELL_TYPES}
    tail_steps = int(round((DURATION_MS - SETTLE_MS) / DT_MS))
    for factor in DRIVE_FACTORS:
        corrected = base_drive.copy()
        for ct in CELL_TYPES:
            corrected[labels == ct] = base_drive[labels == ct] * factor
        model_f = model.with_emitter_parameters(drive_per_neuron=jnp.asarray(corrected))
        out = run(model_f, cfg, duration_ms=DURATION_MS, seed=SEED, hdp_kwargs=hdp_kwargs)
        spikes = np.asarray(out["spikes"])
        tail = spikes[-tail_steps:]
        duration_s = tail.shape[0] * DT_MS / 1000.0
        for ct in CELL_TYPES:
            I = float(base_drive[labels == ct][0] * factor)
            rate = float(tail[:, labels == ct].sum() / ((labels == ct).sum() * duration_s))
            curves[ct].append([round(I, 4), round(rate, 4)])
        print(f"drive_factor={factor} done")

    with open(OUTPUT_DIR / "fi_calibration_curves.json", "w") as f:
        json.dump(curves, f, indent=2, allow_nan=False)
    print(json.dumps(curves, indent=2))
    print(f"\nDone. Results written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
