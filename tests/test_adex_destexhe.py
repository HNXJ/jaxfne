"""Unit tests for the Destexhe 2009 aeIF cell table (jaxfne/adex_destexhe.py).

The table carries the paper's section 2.2 values in jaxfne units (SOURCE_P1.md
section 1); the apply path swaps them onto an adex-family Model without
touching edges, receptors or the kernel.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.adex_destexhe import (
    DESTEXHE_CELL_DEFAULTS,
    apply_destexhe_params,
    destexhe_overrides,
)
from jaxfne.emitters_adex import ADEX_CELL_TYPE_DEFAULTS, AdExParams


def _small_model():
    cfg = (
        jtfne.Configuration()
        .network(name="destexhe-small", n=20, cell_types={"E": 0.7, "PV": 0.2, "SOM": 0.1})
        .emitter(family="adex", adex_decay="exp", adex_step_order="default")
        .mechanisms(name="ampa", kind="receptor", params={"tau_ms": 5.0, "reversal_mV": 0.0})
        .mechanisms(name="gaba", kind="receptor", params={"tau_ms": 10.0, "reversal_mV": -80.0})
        .connections(
            name="e_e",
            source={"cell_type": "E"},
            target={"cell_type": "E"},
            probability=0.2,
            weight=6.0,
            mechanism="ampa",
            delay_ms=0.2,
            sampling="bernoulli",
        )
        .connections(
            name="pv_e",
            source={"cell_type": "PV"},
            target={"cell_type": "E"},
            probability=0.2,
            weight=67.0,
            mechanism="gaba",
            delay_ms=0.2,
            sampling="bernoulli",
        )
        .connectivity(edge_seed=7)
    )
    return jtfne.construct(cfg)


def test_table_shared_geometry():
    for name, row in DESTEXHE_CELL_DEFAULTS.items():
        assert row["C_pF"] == 200.0, name
        assert row["gL_nS"] == 10.0, name
        assert row["EL_mV"] == -60.0, name
        assert row["VT_mV"] == -50.0, name
        assert row["Delta_mV"] == 2.5, name
        assert row["Vreset_mV"] == -60.0, name
        assert row["tauw_ms"] == 600.0, name
        assert row["Tref_ms"] == 2.5, name


def test_table_adaptation_values():
    assert DESTEXHE_CELL_DEFAULTS["RS"]["a_nS"] == 1.0
    assert DESTEXHE_CELL_DEFAULTS["RS"]["b_pA"] == 40.0
    assert DESTEXHE_CELL_DEFAULTS["RSweak"]["b_pA"] == 5.0
    assert DESTEXHE_CELL_DEFAULTS["FS"]["b_pA"] == 0.0
    assert DESTEXHE_CELL_DEFAULTS["LTS"]["a_nS"] == 20.0
    assert DESTEXHE_CELL_DEFAULTS["TC"]["a_nS"] == 40.0
    assert DESTEXHE_CELL_DEFAULTS["RE"]["a_nS"] == 80.0
    assert DESTEXHE_CELL_DEFAULTS["RE"]["b_pA"] == 30.0


def test_overrides_refuse_unknown_type():
    with pytest.raises(ValueError, match="unknown Destexhe cell type"):
        destexhe_overrides({"E": "PY"})


def test_overrides_refuse_empty():
    with pytest.raises(ValueError, match="at least one"):
        destexhe_overrides({})


def test_destexhe_overrides_carry_authors_detection_threshold():
    """F6 plumbing: every carrier override carries Vpeak_mV at the authors' value.

    Authors' code detects at VTR = -50 mV
    (``demo_cx05_N500b_LTS.py:86,152-153``: ``VTR = -50``,
    ``'v_thresh': VTR, 'v_spike': VTR+1e-6``); the core default is
    VT + 5*Delta = -37.5 mV.
    """
    ov = destexhe_overrides({"E": "RSweak", "PV": "FS"})
    assert ov["E"]["Vpeak_mV"] == -50.0, ov["E"]
    assert ov["PV"]["Vpeak_mV"] == -50.0, ov["PV"]


def test_apply_realizes_vpeak_at_vt():
    """The applied model carries Vpeak_mV = -50 mV on every mapped cell."""
    model = _small_model()
    out = apply_destexhe_params(model, {"E": "RSweak", "PV": "FS"})
    got = out.params["emitter"]
    labels = list(got.labels)
    for carrier in ("E", "PV"):
        assert float(got.Vpeak_mV[labels.index(carrier)]) == -50.0, carrier
    # Unmapped carriers keep the core default: NaN = absent = VT + 5*Delta.
    som_vals = [float(v) for v, lab in zip(np.asarray(got.Vpeak_mV), labels) if lab == "SOM"]
    assert som_vals and all(np.isnan(som_vals)), som_vals


def test_gated_fork_removed_core_window_is_the_drive():
    """The time-gated fork is gone: the core drive window carries the kick."""
    import jaxfne.adex_destexhe as lane

    assert not hasattr(lane, "simulate_adex_gated_drive"), (
        "simulate_adex_gated_drive must be deleted; use the core ext_t_* window"
    )


def test_apply_refuses_non_adex_model():
    from jaxfne._model import Model

    cfg = jtfne.Configuration().network(name="izhi", n=4)
    model = Model(cfg=cfg, params={"emitter": object()}, static={})
    with pytest.raises(ValueError, match="adex-family"):
        apply_destexhe_params(model, {"E": "RSweak"})


def test_apply_refuses_absent_carrier():
    model = _small_model()
    with pytest.raises(ValueError, match="absent from the model"):
        apply_destexhe_params(model, {"E": "RSweak", "TC": "TC"})


def test_apply_carries_values_and_keeps_rest():
    model = _small_model()
    n_before = int(model.static["adex"]["pre"].shape[0])
    out = apply_destexhe_params(model, {"E": "RSweak", "PV": "FS"})
    assert out.cfg is model.cfg
    assert out.static is model.static
    assert tuple(out.params["emitter"].labels) == tuple(model.params["emitter"].labels)
    assert int(out.static["adex"]["pre"].shape[0]) == n_before
    labels = list(out.params["emitter"].labels)
    e_idx = labels.index("E")
    pv_idx = labels.index("PV")
    som_idx = labels.index("SOM")
    got = out.params["emitter"]
    assert float(got.b_pA[e_idx]) == 5.0
    assert float(got.a_nS[e_idx]) == 1.0
    assert float(got.EL_mV[e_idx]) == -60.0
    assert float(got.Delta_mV[e_idx]) == 2.5
    assert float(got.b_pA[pv_idx]) == 0.0
    # Unmapped carriers keep the route's own defaults.
    assert float(got.a_nS[som_idx]) == ADEX_CELL_TYPE_DEFAULTS["SOM"]["a_nS"]
    assert str(jnp.asarray(got.C_pF).dtype) == "float32"


def test_applied_model_runs_kernel():
    from jaxfne._construct_adex import adex_kernel_args
    from jaxfne.emitters_adex import simulate_adex

    model = apply_destexhe_params(_small_model(), {"E": "RSweak", "PV": "FS"})
    kwargs = adex_kernel_args(model, 0.1)
    out = simulate_adex(**kwargs, n_steps=50, dt_ms=0.1, record_v=False)
    spikes = np.asarray(out["spikes"])
    assert spikes.shape == (50, 20)
    assert spikes.dtype == bool


def _tiny_window_world():
    """Four Destexhe cells with two receptors, edges and an external drive."""
    import jax

    from jaxfne.adex_destexhe import DESTEXHE_CELL_DEFAULTS
    from jaxfne.emitters_adex import AdExParams

    rows = [dict(DESTEXHE_CELL_DEFAULTS["RSweak"])] * 2 + [dict(DESTEXHE_CELL_DEFAULTS["FS"])] * 2
    fields = (
        "C_pF",
        "gL_nS",
        "EL_mV",
        "VT_mV",
        "Delta_mV",
        "Vreset_mV",
        "tauw_ms",
        "a_nS",
        "b_pA",
        "Tref_ms",
        "Vpeak_mV",
    )
    cols = {k: jnp.asarray([r[k] for r in rows], dtype=jnp.float32) for k in fields}
    params = AdExParams(labels=("E", "E", "PV", "PV"), **cols)
    base = {
        "params": params,
        "edge_pre": np.array([0, 1, 2, 3]),
        "edge_post": np.array([1, 0, 0, 2]),
        "edge_weight_nS": np.array([6.0, 6.0, 67.0, 67.0]),
        "edge_receptor": np.array([0, 0, 1, 1]),
        "edge_delay_steps": np.array([1, 2, 1, 3]),
        "receptor_reversal_mV": (0.0, -80.0),
        "receptor_tau_ms": (5.0, 10.0),
        "decay": "exp",
        "step_order": "default",
    }
    drive = {
        "ext_targets": np.array([0, 1]),
        "ext_lambda": 0.08,
        "ext_weight_nS": 6.0,
        "ext_receptor": 0,
        "ext_delay_steps": 2,
        "key": jax.random.PRNGKey(0),
    }
    stim = np.zeros((60, 4), dtype=np.float32)
    stim[5:15, 0] = 300.0
    return base, drive, stim


def test_core_window_full_open_matches_all_drive():
    """ext_t [0, 60) reproduces the all-steps drive run bit-exactly."""
    from jaxfne.emitters_adex import simulate_adex

    base, drive, stim = _tiny_window_world()
    ref = simulate_adex(**base, **drive, n_steps=60, dt_ms=0.1, I_ext_pA=stim, record_v=False)
    win = simulate_adex(
        **base,
        **drive,
        n_steps=60,
        dt_ms=0.1,
        I_ext_pA=stim,
        ext_t_start_steps=0,
        ext_t_stop_steps=60,
        record_v=False,
    )
    assert np.array_equal(np.asarray(win["spikes"]), np.asarray(ref["spikes"]))
    for key in ("V", "w", "g"):
        assert np.array_equal(np.asarray(win[key]), np.asarray(ref[key])), key


def test_core_window_prefix_matches_all_drive():
    """Window [0, 20): steps inside reproduce the all-steps run bit-for-bit."""
    from jaxfne.emitters_adex import simulate_adex

    base, drive, stim = _tiny_window_world()
    ref = simulate_adex(**base, **drive, n_steps=60, dt_ms=0.1, I_ext_pA=stim, record_v=False)
    win = simulate_adex(
        **base,
        **drive,
        n_steps=60,
        dt_ms=0.1,
        I_ext_pA=stim,
        ext_t_start_steps=0,
        ext_t_stop_steps=20,
        record_v=False,
    )
    assert np.array_equal(np.asarray(win["spikes"][:20]), np.asarray(ref["spikes"][:20])), (
        "windowed prefix must equal the all-steps drive run exactly"
    )


def test_core_window_pre_window_matches_no_drive():
    """Window [20, 60): pre-window steps equal the no-drive run (drive zeroed)."""
    from jaxfne.emitters_adex import simulate_adex

    base, drive, stim = _tiny_window_world()
    ref = simulate_adex(**base, n_steps=60, dt_ms=0.1, I_ext_pA=stim, record_v=False)
    win = simulate_adex(
        **base,
        **drive,
        n_steps=60,
        dt_ms=0.1,
        I_ext_pA=stim,
        ext_t_start_steps=20,
        ext_t_stop_steps=60,
        record_v=False,
    )
    assert np.array_equal(np.asarray(win["spikes"][:20]), np.asarray(ref["spikes"][:20]))


def test_core_window_refuses_bad_window():
    from jaxfne.emitters_adex import simulate_adex

    base, drive, stim = _tiny_window_world()
    with pytest.raises(ValueError, match="ext_t_start_steps"):
        simulate_adex(
            **base,
            **drive,
            n_steps=60,
            dt_ms=0.1,
            I_ext_pA=stim,
            ext_t_start_steps=20,
            ext_t_stop_steps=20,
            record_v=False,
        )
    with pytest.raises(ValueError, match="ext_t_start_steps"):
        simulate_adex(
            **base,
            **drive,
            n_steps=60,
            dt_ms=0.1,
            I_ext_pA=stim,
            ext_t_start_steps=-1,
            ext_t_stop_steps=20,
            record_v=False,
        )
    with pytest.raises(ValueError, match="ext_t_stop_steps"):
        simulate_adex(
            **base,
            **drive,
            n_steps=60,
            dt_ms=0.1,
            I_ext_pA=stim,
            ext_t_start_steps=0,
            record_v=False,
        )
    with pytest.raises(ValueError, match="need the external drive"):
        simulate_adex(
            **base,
            n_steps=60,
            dt_ms=0.1,
            I_ext_pA=stim,
            ext_t_start_steps=0,
            ext_t_stop_steps=20,
            record_v=False,
        )


def test_subset_drive_fraction_size_and_labels():
    """A seeded fraction of the E targets, deterministic in the seed."""
    from artifacts.etudes.papers.destexhe2009 import destexhe_api as D
    from jaxfne._construct_adex import adex_external_drive_kwargs
    from jaxfne.adex_destexhe import subset_drive_kwargs

    import jaxfne as jtfne

    model = D.build_spec(seed=1)
    sim = jtfne.Simulation(duration_ms=D.KICK_MS, dt_ms=D.DT_MS, poisson_drive=D.kick_drive(1))
    full = adex_external_drive_kwargs(model, sim)
    sub = subset_drive_kwargs(model, sim, 0.05)
    assert sub["ext_targets"].size == 80, sub["ext_targets"].size
    assert set(np.asarray(sub["ext_targets"]).tolist()) <= set(
        np.asarray(full["ext_targets"]).tolist()
    )
    for key in ("ext_lambda", "ext_weight_nS", "ext_receptor", "ext_delay_steps"):
        assert sub[key] == full[key], key
    again = subset_drive_kwargs(model, sim, 0.05)
    assert np.array_equal(again["ext_targets"], sub["ext_targets"])
    other_sim = jtfne.Simulation(
        duration_ms=D.KICK_MS, dt_ms=D.DT_MS, poisson_drive=D.kick_drive(2)
    )
    other = subset_drive_kwargs(model, other_sim, 0.05)
    assert not np.array_equal(other["ext_targets"], sub["ext_targets"])


def test_subset_drive_refuses_bad_fraction():
    from artifacts.etudes.papers.destexhe2009 import destexhe_api as D
    from jaxfne.adex_destexhe import subset_drive_kwargs

    import jaxfne as jtfne

    model = D.build_spec(seed=1)
    sim = jtfne.Simulation(duration_ms=D.KICK_MS, dt_ms=D.DT_MS, poisson_drive=D.kick_drive(1))
    for bad in (0.0, -0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="fraction"):
            subset_drive_kwargs(model, sim, bad)
