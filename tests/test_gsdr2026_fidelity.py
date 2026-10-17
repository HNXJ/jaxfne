"""GSDR lane fidelity tests: reference-unit conversions (PORT_SPEC.md D5/D9/units).

Reference values observed in the Fig. 6 run sources (read as data):
``HNXJ/GSDR`` ``misc/main.py`` + ``gsdr/models.py`` on jaxley 0.14.0 —
stimulus in nA, synapse conductances in uS, OU channel current in mA/cm2,
all combined in uA/cm2 via ``convert_point_process_to_distributed``
(``I/A*1e5``) or the channel ``*1000``. The lane kernel works in uA/cm2
throughout, so the runner converts at the boundary. CPU-only, float32.

Fail-first: on the pre-fidelity tree the runner passes raw reference numbers
as densities, so every conversion test below fails there.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


def _load_run_gsdr():
    path = Path(__file__).resolve().parents[1] / "artifacts" / "etudes" / "papers" / "gsdr2026" / "run" / "run_gsdr.py"
    spec = importlib.util.spec_from_file_location("gsdr2026_run_gsdr", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run_gsdr = _load_run_gsdr()

AREA_CELL_UM2 = 80.0 * math.pi  # 4 comps x 20pi (length assignment is a no-op)
PP_TO_DENSITY = 1e5 / AREA_CELL_UM2  # nA/um2 -> uA/cm2 (and uS -> mS/cm2)


def test_stimulus_matrix_is_density():
    """0.6 nA over the whole-cell area, not raw nA."""
    mat = run_gsdr.stimulus_matrix(0.6)
    assert mat.shape == (run_gsdr.N_STEPS, run_gsdr.N_CELLS)
    peak = float(np.abs(mat).max())
    # rel=1e-3: the 120 Hz sine sampled at dt=0.1 never lands exactly on its
    # peak (sampling slop ~8e-5); the conversion itself is exact.
    assert peak == pytest.approx(0.6 * PP_TO_DENSITY, rel=1e-3), \
        "stimulus peak must be 0.6 nA as density, got %r" % peak
    assert float(np.abs(mat[:4000]).max()) == 0.0, "pre-window must be silent"


def test_build_spec_weights_are_density():
    """Rule weights are reference uS converted to mS/cm2."""
    from jaxfne._construct_hh import construct_hh_model

    cfg, _ = run_gsdr.build_spec()
    model = construct_hh_model(cfg)
    w = np.asarray(model.static["hh"]["weight"])
    rec = np.asarray(model.static["hh"]["receptor"])
    assert float(w[rec == 0].mean()) == pytest.approx(2.5 * PP_TO_DENSITY, rel=1e-6), \
        "AMPA weights must be 2.5 uS as density"
    assert float(w[rec == 1].mean()) == pytest.approx(5.0 * PP_TO_DENSITY, rel=1e-6), \
        "GABAa weights must be 5.0 uS as density"


def test_init_conductances_density_scale():
    """Init draws stay in reference uS bounds, converted to mS/cm2."""
    import jax

    g = np.asarray(run_gsdr.init_conductances(jax.random.PRNGKey(0)))
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    raw_ampa = np.clip(np.asarray(jax.random.uniform(
        k1, (run_gsdr.N_AMPA_EDGES,), minval=-4.0, maxval=10.0)), 0.0, 8.0)
    raw_gaba = np.clip(np.asarray(jax.random.uniform(
        k2, (run_gsdr.N_GABAA_GLOBAL_EDGES + run_gsdr.N_GABAA_LOCAL_EDGES,),
        minval=2.0, maxval=10.0)), 0.0, 10.0)
    expected = np.concatenate([raw_ampa, raw_gaba]) * PP_TO_DENSITY
    np.testing.assert_allclose(g, expected.astype(np.float32), rtol=1e-5)


def test_ou_sigma_milliamp_to_microamp():
    """OU sigma is a channel current (mA/cm2); the lane needs uA/cm2."""
    import jax

    _, sigma = run_gsdr.ou_hyperparams(jax.random.fold_in(jax.random.PRNGKey(0), 0))
    s = np.asarray(sigma)
    assert s.max() > 1.0, "sigma must carry the x1000 scale, got max %r" % s.max()
    assert s.max() <= 50.0, "sigma must stay within 0.05 mA/cm2, got %r" % s.max()


def test_single_cell_fi_smoke_finite_with_real_rate():
    """Driven single cell at dt=0.1: finite trace and a physiological rate."""
    import jax

    from jaxfne.emitters_hh import make_hh_params, simulate_hh

    params = make_hh_params(["E"])
    n_steps = 6000
    i_ext = np.full((n_steps, 1), 10.0, dtype=np.float32)
    out = simulate_hh(params, n_steps=n_steps, dt_ms=0.1,
                      receptor_e_rev=(0.0,), receptor_tau_r=(0.2,),
                      receptor_tau_d=(2.0,), receptor_slope=(5.0,),
                      receptor_vth=(-20.0,), I_ext=i_ext,
                      ou_key=jax.random.PRNGKey(0))
    v = np.asarray(out["V_mV"])[:, 0]
    assert np.all(np.isfinite(v)), "V trace must stay finite under drive"
    rate = float(np.asarray(out["spikes"])[:, 0].sum()) / 0.6
    assert 10.0 < rate < 150.0, "I=10 must give a physiological rate, got %r" % rate
