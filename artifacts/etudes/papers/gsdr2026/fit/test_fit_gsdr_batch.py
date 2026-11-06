"""Batched GSDR candidate evaluation through ``simulate_hh_batch`` (F5).

For 3 candidates at ``t_max_ms=600`` (baseline window [0, 200] ms and stim
window [500, 600] ms both hold >= 1 kappa bin), the batched path returns
per-candidate losses that agree with the serial ``candidate_loss`` path under the
declared tolerance (Hamm 2026-11-06: opt-in with tolerance): loss relative
difference < 1e-2, the same candidate ranking, spike counts within 2 per trial.
Bit-equality does not hold: the vmapped float32 HH trajectory leaves the serial one
after about 200 steps (rounding amplified by the dynamics), so ``--batch`` serves
pilots and sweeps; signed and scored fits stay on the serial path.
"""

import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_RUN_DIR = _HERE.parent / "run"
_repo_root = Path(__file__).resolve().parents[5]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_RUN_DIR) not in sys.path:
    sys.path.insert(0, str(_RUN_DIR))

from fit_gsdr import batch_trial_vs, candidate_loss, candidate_losses_batch  # noqa: E402

import run_gsdr as rg  # noqa: E402

T_MAX_MS = 600.0
FIT_SEEDS = (1,)
N_CANDS = 3


def _stub_y():
    rng = np.random.default_rng(0)
    grid = np.linspace(15.0, 60.0, 40)
    return {"grid": grid,
            "baseline_psd": rng.uniform(0.0, 1.0, 40),
            "stim_psd": rng.uniform(0.0, 1.0, 40),
            "kappa_baseline": 0.1,
            "kappa_stim": 0.2}


def _candidates():
    import jax

    g0 = np.asarray(rg.init_conductances(jax.random.PRNGKey(1)),
                    dtype=np.float64) / rg.US_TO_MSCM2
    center = np.clip(g0, 0.0, 10.0)
    rng = np.random.default_rng(0)
    cands = [center]
    for _ in range(N_CANDS - 1):
        cands.append(np.clip(center + 0.5 * rng.standard_normal(center.shape),
                             0.0, 10.0))
    return cands


def test_batch_matches_serial_losses_and_spikes():
    y = _stub_y()
    cands = _candidates()
    trials = [(0.0, 1)] + [(0.6, s) for s in FIT_SEEDS]

    t0 = time.time()
    serial_losses, serial_comps = [], []
    for g in cands:
        loss, comp = candidate_loss(
            g, rg, y, fit_seeds=FIT_SEEDS, morphology="single",
            t_max_ms=T_MAX_MS)
        serial_losses.append(loss)
        serial_comps.append(comp)
    serial_wall = time.time() - t0

    t0 = time.time()
    batch_losses, batch_comps = candidate_losses_batch(
        cands, rg, y, fit_seeds=FIT_SEEDS, morphology="single",
        t_max_ms=T_MAX_MS)
    batch_wall = time.time() - t0
    print("SERIALvsBATCH candidates=%d trials=%d serial=%.1fs batch=%.1fs" % (
        N_CANDS, len(trials), serial_wall, batch_wall), flush=True)

    assert len(batch_losses) == len(serial_losses) == N_CANDS
    bl = np.asarray(batch_losses, dtype=np.float64)
    sl = np.asarray(serial_losses, dtype=np.float64)
    assert np.all(np.isfinite(sl)) and np.ptp(sl) > 0, "fixture must give distinct finite losses"
    assert np.allclose(bl, sl, rtol=1e-2), \
        "batched losses must agree with serial within rtol 1e-2, got %r vs %r" % (
            batch_losses, serial_losses)
    assert list(np.argsort(bl)) == list(np.argsort(sl)), \
        "batched losses must rank the candidates as the serial path does"

    for amp, ou in trials:
        serial_vs = [rg.run_trial(amp, g_seed=ou, ou_seed=ou,
                                  g_override=np.asarray(g, dtype=np.float32),
                                  morphology="single",
                                  t_max_ms=T_MAX_MS)[1]["v"]
                     for g in cands]
        batch_v = np.asarray(batch_trial_vs(
            cands, rg, amp, ou, morphology="single", t_max_ms=T_MAX_MS))
        assert batch_v.shape == (N_CANDS,) + np.shape(serial_vs[0]), \
            "batch V must stack the serial shapes, got %s" % (batch_v.shape,)
        for b in range(N_CANDS):
            nb = int(np.sum(rg.detect_spikes(batch_v[b])))
            ns = int(np.sum(rg.detect_spikes(serial_vs[b])))
            assert abs(nb - ns) <= 2, \
                "candidate %d trial (amp %s, ou %d): spike counts %d vs %d differ by > 2" % (
                    b, amp, ou, nb, ns)
