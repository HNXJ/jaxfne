"""CAMINOS drive gain through the public API route (RESIDUAL_PROTOCOL.md step 3).

The gain a sets the mean external drive per E and PV cell to
``560 + a * (round(p*NE) - 560)`` sources at 4 Hz. The core draws one Poisson count
per cell with mean ``n_sources * rate_hz * dt / 1000`` (``adex_external_drive_kwargs``),
so the gain is realized with ``n_sources = round(p*NE)`` fixed and ``rate_hz`` scaled.
The model-level tests read the realized ``ext_lambda`` through the core accessor.
CPU only; no simulation runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pytest

DT_MS = 0.1  # PORT_SPEC Methods, dt = 0.1 ms
V6_SOURCES = 560  # the fixed count of the v6 drive
RATE_HZ = 4.0


def _lam(n_sources, rate_hz):
    """Per-cell Poisson mean per step, as the core defines it."""
    return n_sources * rate_hz * DT_MS / 1000.0


def test_no_gain_keeps_the_v7_drive():
    import caminos_api as ca

    d = ca.drive_spec(0.02, 1)
    assert d["n_sources"] == 160
    assert d["rate_hz"] == RATE_HZ


def test_gain_one_is_the_v7_drive():
    import caminos_api as ca

    d = ca.drive_spec(0.02, 1, gain=1.0)
    assert d["n_sources"] == 160
    assert d["rate_hz"] == pytest.approx(RATE_HZ, rel=1e-12)


def test_gain_zero_has_the_v6_mean_drive():
    import caminos_api as ca

    d = ca.drive_spec(0.02, 1, gain=0.0)
    assert d["n_sources"] * d["rate_hz"] == pytest.approx(V6_SOURCES * RATE_HZ, rel=1e-12)


def test_fractional_gain_scales_the_mean_drive_linearly():
    import caminos_api as ca

    # 560 + 0.5 * (160 - 560) = 360 sources at 4 Hz
    d = ca.drive_spec(0.02, 1, gain=0.5)
    assert d["n_sources"] * d["rate_hz"] == pytest.approx(360 * RATE_HZ, rel=1e-12)


def test_gain_at_p_007_is_the_v6_drive():
    import caminos_api as ca

    # round(0.07 * 8000) = 560, so the gain has no effect at the reference p
    d = ca.drive_spec(0.07, 1, gain=0.3)
    assert d["n_sources"] == 560
    assert d["rate_hz"] == pytest.approx(RATE_HZ, rel=1e-12)


def test_nonpositive_drive_is_refused():
    import caminos_api as ca

    # 560 + a * (160 - 560) <= 0 at a = 1.5
    with pytest.raises(ValueError):
        ca.drive_spec(0.02, 1, gain=1.5)


@pytest.fixture(scope="module")
def model_p002():
    import jaxfne as jtfne

    import caminos_api as ca

    return jtfne.construct(ca.build_spec(0.02, 1))


@pytest.mark.parametrize(
    ("gain", "expected_mean"),
    [(None, 160 * RATE_HZ), (1.0, 160 * RATE_HZ), (0.0, V6_SOURCES * RATE_HZ),
     (0.5, 360 * RATE_HZ)],
)
def test_core_realizes_the_gain_as_its_poisson_mean(model_p002, gain, expected_mean):
    import caminos_api as ca
    from jaxfne._construct_adex import adex_external_drive_kwargs

    sim = ca.build_sim(0.02, 1, 12.0, gain=gain)
    kw = adex_external_drive_kwargs(model_p002, sim)
    assert kw["ext_lambda"] == pytest.approx(_lam(1, expected_mean), rel=1e-6)
    assert len(kw["ext_targets"]) > 0


# --- ref. 42 phase method for the PV->SOM delay (dispatcher ruling, 2026-10-13) ---

F_TEST_HZ = 50.0  # period 20 ms: a whole number of 0.1 ms steps
DT_TEST_S = 1e-4
N_TEST_STEPS = 120000  # 12 s: 2 s dropped, 10 s analysed


def _synthetic_counts(delay_ms, f_hz=F_TEST_HZ, n_steps=N_TEST_STEPS, dt_s=DT_TEST_S):
    """Known sinusoid with spikes at known phases.

    E counts are round(50 + 40 cos(2 pi f t)), so the LFP phase is 2 pi f t. PV fires at
    phase 0 (t = k / f); SOM fires ``delay_ms`` after each PV spike. The true delay is
    ``delay_ms``.
    """
    import numpy as np

    fs = 1.0 / dt_s
    period = int(round(fs / f_hz))
    assert period * f_hz == pytest.approx(fs), "fixture: the period must be whole steps"
    t = np.arange(n_steps) * dt_s
    e = np.rint(50.0 + 40.0 * np.cos(2.0 * np.pi * f_hz * t)).astype(np.int64)
    pv = np.zeros(n_steps, dtype=np.int64)
    pv[::period] = 1
    som = np.zeros(n_steps, dtype=np.int64)
    idx = np.arange(0, n_steps, period) + int(round(delay_ms * 1e-3 * fs))
    idx = idx[(idx >= 0) & (idx < n_steps)]
    som[idx] = 1
    return np.column_stack([e, pv, som])


def test_ref42_recovers_a_known_pv_to_som_delay():
    import caminos_api as ca

    out = ca.ref42_phase_delay(_synthetic_counts(8.5), DT_TEST_S, F_TEST_HZ)
    assert out["delay_ms"] == pytest.approx(8.5, abs=0.1)


def test_ref42_sign_follows_the_order_of_firing():
    import caminos_api as ca

    out = ca.ref42_phase_delay(_synthetic_counts(-8.5), DT_TEST_S, F_TEST_HZ)
    assert out["delay_ms"] == pytest.approx(-8.5, abs=0.1)


def test_ref42_reads_the_core_raster_through_spikes_to_counts():
    import numpy as np

    import caminos_api as ca

    base = _synthetic_counts(8.5)
    n_e, n_pv, n_som = 100, 4, 4
    e_raster = np.arange(n_e)[None, :] < base[:, [0]]  # E cells firing per step = E count
    pv_raster = np.zeros((base.shape[0], n_pv), dtype=bool)
    pv_raster[:, 0] = base[:, 1] > 0
    som_raster = np.zeros((base.shape[0], n_som), dtype=bool)
    som_raster[:, 0] = base[:, 2] > 0
    spikes = np.concatenate([e_raster, pv_raster, som_raster], axis=1)
    counts = ca.spikes_to_counts(spikes, {"E": n_e, "PV": n_pv, "SOM": n_som})
    assert np.array_equal(counts, base)
    out = ca.ref42_phase_delay(counts, DT_TEST_S, F_TEST_HZ)
    assert out["delay_ms"] == pytest.approx(8.5, abs=0.1)


def test_ref42_refuses_a_band_that_reaches_zero_frequency():
    import caminos_api as ca

    with pytest.raises(ValueError):
        ca.ref42_phase_delay(_synthetic_counts(8.5), DT_TEST_S, F_TEST_HZ, half_band_hz=60.0)


def test_lag_summary_bands_the_mean_against_y():
    import score_lag_ref42 as s

    close = s.summarize([8.5, 8.6, 8.4], 8.5)
    assert close["band"] == "close"
    assert close["e"] == pytest.approx(0.0, abs=1e-12)
    off = s.summarize([7.2, 7.2, 7.2], 8.5)
    assert off["band"] == "off"
    assert off["e"] == pytest.approx(abs(7.2 - 8.5) / 8.5)
    assert s.summarize([8.5, None, 8.5], 8.5)["band"] == "undefined"


# --- declared synaptic and drive delay (lane 3 sensitivity: option (a), 1 step) ---


def test_default_delay_is_the_two_ms_port_value():
    import caminos_api as ca

    assert ca.drive_spec(0.02, 1)["delay_ms"] == 2.0
    assert ca.drive_spec(0.02, 1, delay_ms=None)["delay_ms"] == 2.0


def test_delay_override_reaches_the_drive_spec():
    import caminos_api as ca

    assert ca.drive_spec(0.02, 1, delay_ms=0.1)["delay_ms"] == 0.1


def test_default_delay_is_twenty_steps_on_every_connection_and_the_drive(model_p002):
    import numpy as np

    import caminos_api as ca
    from jaxfne._construct_adex import adex_external_drive_kwargs, adex_kernel_args

    steps = np.asarray(adex_kernel_args(model_p002, ca.DT_MS)["edge_delay_steps"])
    assert steps.size > 0, "fixture must build edges"
    assert np.all(steps == 20)
    kw = adex_external_drive_kwargs(model_p002, ca.build_sim(0.02, 1, 12.0))
    assert kw["ext_delay_steps"] == 20


@pytest.fixture(scope="module")
def model_p002_delay01():
    import jaxfne as jtfne

    import caminos_api as ca

    return jtfne.construct(ca.build_spec(0.02, 1, delay_ms=0.1))


def test_core_realizes_the_declared_delay_as_one_step_on_every_connection(model_p002_delay01):
    import numpy as np

    import caminos_api as ca
    from jaxfne._construct_adex import adex_kernel_args

    steps = np.asarray(adex_kernel_args(model_p002_delay01, ca.DT_MS)["edge_delay_steps"])
    assert steps.size > 0, "fixture must build edges"
    assert np.all(steps == 1)


def test_core_realizes_the_declared_delay_as_one_step_on_the_drive(model_p002_delay01):
    import caminos_api as ca
    from jaxfne._construct_adex import adex_external_drive_kwargs

    kw = adex_external_drive_kwargs(model_p002_delay01,
                                    ca.build_sim(0.02, 1, 12.0, delay_ms=0.1))
    assert kw["ext_delay_steps"] == 1


def test_delayed_runs_declare_no_fixed_point(model_p002, model_p002_delay01):
    import caminos_api as ca
    from jaxfne._construct_adex import adex_kernel_args

    for model in (model_p002, model_p002_delay01):
        kw = adex_kernel_args(model, ca.DT_MS)
        assert kw.get("intra_step_rounds") is None
        assert "edge0_pre" not in kw or len(kw["edge0_pre"]) == 0
    assert "same_step" not in ca.drive_spec(0.02, 1)
    assert "same_step" not in ca.drive_spec(0.02, 1, delay_ms=0.1)


# --- true zero delay (core gap G12 closed in T-22; Hamm's ruling 2026-10-14, point 4) ---


@pytest.fixture(scope="module")
def model_p002_delay0():
    import jaxfne as jtfne

    import caminos_api as ca

    return jtfne.construct(ca.build_spec(0.02, 1, delay_ms=0.0))


def test_zero_delay_runs_every_connection_through_the_fixed_point(model_p002_delay0):
    import numpy as np

    import caminos_api as ca
    from jaxfne._construct_adex import adex_kernel_args

    kw = adex_kernel_args(model_p002_delay0, ca.DT_MS)
    assert len(kw["edge0_pre"]) > 0, "fixture must build zero-delay edges"
    assert kw["intra_step_rounds"] == ca.ZERO_DELAY_ROUNDS
    assert np.asarray(kw["edge_delay_steps"]).size == 0


def test_zero_delay_drive_acts_in_its_own_step(model_p002_delay0):
    import caminos_api as ca
    from jaxfne._construct_adex import adex_external_drive_kwargs

    sim = ca.build_sim(0.02, 1, 12.0, delay_ms=0.0)
    assert sim.poisson_drive["same_step"] is True
    assert adex_external_drive_kwargs(model_p002_delay0, sim)["ext_delay_steps"] == 0


def test_rounds_override_reaches_the_kernel():
    import jaxfne as jtfne

    import caminos_api as ca
    from jaxfne._construct_adex import adex_kernel_args

    model = jtfne.construct(ca.build_spec(0.02, 1, delay_ms=0.0, intra_step_rounds=3))
    assert adex_kernel_args(model, ca.DT_MS)["intra_step_rounds"] == 3


@pytest.mark.parametrize("delay_ms", [None, 0.1, 2.0])
def test_rounds_override_with_a_delayed_run_is_refused(delay_ms):
    import caminos_api as ca

    with pytest.raises(ValueError, match="intra_step_rounds"):
        ca.build_spec(0.02, 1, delay_ms=delay_ms, intra_step_rounds=4)
    with pytest.raises(ValueError, match="intra_step_rounds"):
        ca.drive_spec(0.02, 1, delay_ms=delay_ms, intra_step_rounds=4)


@pytest.mark.parametrize("bad", [0, -1, True, 2.0])
def test_rounds_override_must_be_a_positive_int(bad):
    import caminos_api as ca

    with pytest.raises(ValueError, match="positive integer"):
        ca.build_spec(0.02, 1, delay_ms=0.0, intra_step_rounds=bad)


@pytest.mark.parametrize("delay_ms, rounds, recorded", [(0.0, None, "default"), (0.0, 3, 3),
                                                        (0.1, None, None), (None, None, None)])
def test_manifest_records_rounds_only_on_the_zero_delay_path(tmp_path, delay_ms, rounds, recorded):
    import json

    import numpy as np

    import caminos_api as ca

    ca.write_run(tmp_path, p=0.02, seed=1, T_s=0.001, counts=np.zeros((10, 3), np.float32),
                 sizes={"E": 1, "PV": 1, "SOM": 1}, wall_s=0.0, start_time="t", sha="s",
                 delay_ms=delay_ms, intra_step_rounds=rounds)
    entry = json.loads((tmp_path / "manifest.json").read_text())["runs"][0]
    want = ca.ZERO_DELAY_ROUNDS if recorded == "default" else recorded
    keys = list(entry)
    if want is None:
        assert "intra_step_rounds" not in entry
    else:
        assert entry["intra_step_rounds"] == want
        assert keys[keys.index("delay_ms") + 1] == "intra_step_rounds"


@pytest.mark.parametrize("delay_ms, used, want", [(0.0, 5, 5), (0.0, None, None),
                                                  (0.1, 5, None)])
def test_manifest_records_rounds_used_on_the_zero_delay_path(tmp_path, delay_ms, used, want):
    import json

    import numpy as np

    import caminos_api as ca

    ca.write_run(tmp_path, p=0.02, seed=1, T_s=0.001, counts=np.zeros((10, 3), np.float32),
                 sizes={"E": 1, "PV": 1, "SOM": 1}, wall_s=0.0, start_time="t", sha="s",
                 delay_ms=delay_ms, max_rounds_used=used)
    entry = json.loads((tmp_path / "manifest.json").read_text())["runs"][0]
    if want is None:
        assert "max_rounds_used" not in entry
    else:
        assert entry["max_rounds_used"] == want
        keys = list(entry)
        assert keys[keys.index("intra_step_rounds") + 1] == "max_rounds_used"


def test_delay_scorer_checks_the_declared_delay_and_reports_rounds(tmp_path):
    """score_delay_sensitivity.check_manifest takes the declared delay (0 for T-22 runs),
    refuses runs at another delay, and reports the cap and rounds used per seed."""
    import numpy as np

    import caminos_api as ca
    import score_delay_sensitivity as sds

    for seed in sds.SENS_SEEDS:
        ca.write_run(tmp_path, p=0.07, seed=seed, T_s=0.001, counts=np.zeros((10, 3), np.float32),
                     sizes={"E": 1, "PV": 1, "SOM": 1}, wall_s=0.0, start_time="t", sha="s",
                     delay_ms=0.0, max_rounds_used=seed % 3 + 1)
    got = sds.check_manifest(tmp_path, 0.0)
    assert got["delay_ms"] == 0.0
    assert got["intra_step_rounds"] == [ca.ZERO_DELAY_ROUNDS]
    assert got["max_rounds_used_per_seed"] == {str(k): k % 3 + 1 for k in sds.SENS_SEEDS}
    with pytest.raises(SystemExit, match="another delay"):
        sds.check_manifest(tmp_path, 0.1)
