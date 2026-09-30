"""Regenerate frozen MVC-2 recovery metrics (P-016 re-freeze).

The etude runner named in the bundle manifest (scripts/hdp_mvc_etude.py) no
longer exists, so this script reproduces the frozen ``mvc2_recovery`` values
with the test's own computation: it imports the helpers from
tests/test_hdp_population_restoring.py and runs the off / scalar / vector
arms exactly as ``test_population_restoring_etude_regression_metrics`` does
(seed 17, dt 0.1 ms, 15 s, perturbation x1.2 at 3000 ms).

Read-only: prints the six asserted values as JSON. The caller copies the
moved leaf (vector ``terminal_error_weighted``) into
artifacts/etudes/hdp_controllability_reachability/metrics.json.

Cause of the move (P-016): commit 865e74b ("0.5.3 item 3 (W1): P-010
chain-noise fix + continuation all-state") changed the plain-path noise
schedule, moving vector terminal error 0.0296 -> ~0.0028 (smaller, out of
the old pin). See artifacts/programme/bisect_p016_receipt.md and
artifacts/programme/refreeze_p016_p020_2026-09-30.md.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Force the repo package: `python scripts/...` puts scripts/ first on sys.path,
# which would import any site-packages jaxfne instead of this tree (P-016/P-020
# refreeze driver hole, 2026-09-30). Prove the import below via jaxfne.__file__.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np

import jaxfne as jtfne

assert "site-packages" not in jtfne.__file__, jtfne.__file__
from test_hdp_population_restoring import (
    _mcc3_model,
    _population_hdp_params,
    _recovery_metrics,
    _windowed_population_rates,
)


def main() -> int:
    model, mei_mask, e_mask = _mcc3_model()
    dt_ms = 0.1
    duration_ms = 15000.0
    perturb_ms = 3000.0
    burn_ms = 20.0
    drive = np.asarray(model.params["emitter"].drive, dtype=np.float32)

    def paradigm(scale: float):
        extra = (scale - 1.0) * drive
        events = []
        for i in range(drive.shape[0]):
            if abs(extra[i]) < 1e-12:
                continue
            events.append(
                {
                    "onset_ms": perturb_ms,
                    "duration_ms": duration_ms - perturb_ms,
                    "amplitude": float(extra[i]),
                    "target_indices": [i],
                    "is_drive_event": True,
                }
            )
        return jtfne.StimulusSchedule(events=tuple(events), n_neurons=drive.shape[0])

    sig_off_measure = model.simulate(
        jtfne.simulation(
            duration_ms=duration_ms,
            dt_ms=dt_ms,
            seed=17,
            runtime=jtfne.RuntimeConfig(enable_hdp=False, recurrent_backend="edge_list", jit=False),
        ),
        paradigm=paradigm(1.2),
    )
    r_e_m, r_i_m = _windowed_population_rates(np.asarray(sig_off_measure.spikes), dt_ms)
    i0 = int(round(burn_ms / dt_ms))
    i1 = int(round(perturb_ms / dt_ms))
    r0_e = float(np.mean(r_e_m[i0:i1]))
    r0_i = float(np.mean(r_i_m[i0:i1]))

    def run(enable_hdp: bool, hp=None):
        runtime = jtfne.RuntimeConfig(
            enable_hdp=enable_hdp,
            recurrent_backend="edge_list",
            jit=False,
            hdp_params=hp or {},
        )
        sig = model.simulate(
            jtfne.simulation(duration_ms=duration_ms, dt_ms=dt_ms, seed=17, runtime=runtime),
            paradigm=paradigm(1.2),
        )
        sp = np.asarray(sig.spikes)
        r_e, r_i = _windowed_population_rates(sp, dt_ms)
        return _recovery_metrics(
            r_e,
            r_i,
            dt_ms=dt_ms,
            r0_e=r0_e,
            r0_i=r0_i,
            perturb_start_ms=perturb_ms,
            final_window_ms=(13000.0, 15000.0),
        )

    r_ei_off, term_off = run(False)
    r_ei_scalar, term_scalar = run(True, {"K_HDP": 0.01, "h_state_dim": 1})
    r_ei_vec, term_vec = run(
        True, _population_hdp_params(mei_mask, e_mask, r0_e=r0_e, r0_i=r0_i)
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    print(
        json.dumps(
            {
                "head": head,
                "jaxfne_file": jtfne.__file__,
                "jaxfne_version": jtfne.__version__,
                "seed": 17,
                "dt_ms": dt_ms,
                "duration_ms": duration_ms,
                "r0_e_hz": r0_e,
                "r0_i_hz": r0_i,
                "mvc2_recovery": {
                    "off": {"R_EI": r_ei_off, "terminal_error_weighted": term_off},
                    "scalar": {
                        "R_EI": r_ei_scalar,
                        "terminal_error_weighted": term_scalar,
                    },
                    "vector": {"R_EI": r_ei_vec, "terminal_error_weighted": term_vec},
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
