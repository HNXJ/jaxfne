"""Tests for the multi-compartment HH route in ``jaxfne/emitters_hh.py``.

Fail-first: written for the morphology route; on the baseline tree the new
keyword arguments do not exist, so every test here fails there (``TypeError``
on the unexpected keyword). Reference values: jaxley 0.14.0, read as data
(``compute_axial_conductances`` in
``site-packages/jaxley/utils/morph_attributes.py``, the implicit dhs voltage
solve in ``site-packages/jaxley/solver_voltage.py``, the point-process
conversion in ``site-packages/jaxley/utils/cell_utils.py``, HH default gate
state 0.2 in ``site-packages/jaxley/channels/hh.py``). CPU-only, small grids.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from jaxfne.emitters_hh import (
    make_hh_params,
    reference_4comp_morphology,
    simulate_hh,
)

DT_MS = 0.025


def _one_cell_oracle():
    """4-comp jaxley cell, step into branch 1 loc 0.0, all comps recorded."""
    import jaxley as jx
    from jaxley.channels import HH

    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=2)
    cell = jx.Cell(branch, parents=[-1, 0])
    cell.insert(HH())
    cell.branch(0).loc(0.0).record()
    cell.branch(0).loc(1.0).record()
    cell.branch(1).loc(0.0).record()
    cell.branch(1).loc(1.0).record()
    return cell


def test_multicomp_oracle_current_step_matches_jaxley():
    jaxley = pytest.importorskip("jaxley")
    orig_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        n_steps = 200
        stim_on = int(1.0 // DT_MS)
        cell = _one_cell_oracle()
        assert cell.recordings.rec_index.to_numpy().tolist() == [0, 1, 2, 3], \
            "recordings must map row i to compartment i, got %r" % (
                cell.recordings.rec_index.to_numpy().tolist())
        stim = np.zeros(n_steps, dtype=np.float64)
        stim[stim_on:] = 0.02
        data_stim = cell.branch(1).loc(0.0).data_stimulate(stim)
        v_oracle = np.asarray(
            jaxley.integrate(
                cell,
                data_stimuli=data_stim,
                t_max=n_steps * DT_MS,
                delta_t=DT_MS,
                solver="bwd_euler",
                voltage_solver="jaxley.dhs",
            )
        )
        assert v_oracle.shape == (4, n_steps + 1)

        morph = reference_4comp_morphology()
        params = make_hh_params(["E"])
        assert np.asarray(params.gNa_mS_cm2).dtype == np.float64
        stim_nA = np.zeros((n_steps, 1), dtype=np.float64)
        stim_nA[stim_on:, 0] = 0.02
        out = simulate_hh(
            params,
            n_steps=n_steps,
            dt_ms=DT_MS,
            I_ext_nA=stim_nA,
            stim_comp=2,
            record_comp=0,
            morphology=morph,
            gate_init=0.2,
            record_v=True,
        )
        v_ours = np.asarray(out["V_mV"])[:, 0, :]
        # Tolerance: float64 eps (2.2e-16) on ~100 mV scales accumulated
        # over 200 implicit solves; measured worst |diff| on this grid is
        # 8.29e-12, so 1e-9 keeps ~120x headroom while resolving mV-scale
        # dynamics to 9 decimals.
        for i in range(4):
            np.testing.assert_allclose(
                v_ours[:, i], v_oracle[i, 1:], rtol=1e-12, atol=1e-9,
                err_msg="compartment %d drifts from the jaxley oracle" % i,
            )
    finally:
        jax.config.update("jax_enable_x64", orig_x64)


def _two_cell_common(n_steps=800):
    morph = reference_4comp_morphology()
    stim = np.zeros((n_steps, 2), dtype=np.float64)
    stim[int(1.0 // DT_MS):, 0] = 0.02
    return morph, dict(
        params=make_hh_params(["E", "E"]),
        n_steps=n_steps,
        dt_ms=DT_MS,
        receptor_e_rev=(0.0,),
        receptor_tau_r=(0.2,),
        receptor_tau_d=(2.0,),
        receptor_slope=(5.0,),
        receptor_vth=(-20.0,),
        I_ext_nA=stim,
        stim_comp=2,
        record_comp=0,
        morphology=morph,
        record_v=True,
    )


def test_multicomp_two_cell_ampa_propagates():
    # No oracle: jaxley 0.14.0 `synapses/__init__.py` exports only
    # IonotropicSynapse/TanhConductance/TanhRate, no graded AMPA, so the
    # synapse leg has no reference implementation. Assert qualitative
    # propagation instead: the follower is silent without the edge and
    # spikes with the AMPA edge onto compartment 3.
    morph, common = _two_cell_common()
    assert morph.area_um2[3] == pytest.approx(20.0 * np.pi)
    silent = simulate_hh(**common)
    assert np.asarray(silent["V_mV"]).shape == (800, 2, 4)
    assert int(np.asarray(silent["spikes"])[:, 0].sum()) > 0, \
        "driven cell must spike (fixture must spike)"
    assert int(np.asarray(silent["spikes"])[:, 1].sum()) == 0, \
        "follower must stay silent without the edge"
    driven = simulate_hh(edge_pre=[0], edge_post=[1], edge_weight_uS=[0.05],
                         edge_post_comp=3, edge_receptor=[0], **common)
    assert int(np.asarray(driven["spikes"])[:, 1].sum()) > 0, \
        "AMPA edge onto compartment 3 must make the follower spike"


def test_multicomp_point_process_matches_density():
    # uS through the post-compartment area must equal the density form.
    morph, common = _two_cell_common()
    w_us = 0.05
    dens = w_us / morph.area_um2[3] * 1e5
    via_us = simulate_hh(edge_pre=[0], edge_post=[1], edge_weight_uS=[w_us],
                         edge_post_comp=3, edge_receptor=[0], **common)
    via_dens = simulate_hh(edge_pre=[0], edge_post=[1], edge_weight=[dens],
                           edge_post_comp=3, edge_receptor=[0], **common)
    np.testing.assert_array_equal(np.asarray(via_us["V_mV"]),
                                  np.asarray(via_dens["V_mV"]))


def _one_cell_kwargs(**over):
    morph = reference_4comp_morphology()
    kw = dict(
        params=make_hh_params(["E"]),
        n_steps=8,
        dt_ms=DT_MS,
        receptor_e_rev=(0.0,),
        receptor_tau_r=(0.2,),
        receptor_tau_d=(2.0,),
        receptor_slope=(5.0,),
        receptor_vth=(-20.0,),
        morphology=morph,
    )
    kw.update(over)
    return kw


def test_multicomp_refuses_comp_without_morphology():
    kw = _one_cell_kwargs()
    del kw["morphology"]
    with pytest.raises(ValueError, match="morphology is required"):
        simulate_hh(stim_comp=2, **kw)


def test_multicomp_refuses_comp_out_of_range():
    with pytest.raises(ValueError, match="must index compartments"):
        simulate_hh(**_one_cell_kwargs(stim_comp=4))


def test_multicomp_refuses_mixed_current_forms():
    with pytest.raises(ValueError, match="alternative input forms"):
        simulate_hh(**_one_cell_kwargs(
            I_ext=np.zeros((8, 1)), I_ext_nA=np.zeros((8, 1))))


def test_multicomp_refuses_mixed_weight_forms():
    with pytest.raises(ValueError, match="alternative input forms"):
        simulate_hh(**_one_cell_kwargs(
            edge_pre=[0], edge_post=[0], edge_receptor=[0],
            edge_weight=[1.0], edge_weight_uS=[0.01]))


def test_multicomp_refuses_non_morphology():
    with pytest.raises(ValueError, match="must be an HHMorphology"):
        simulate_hh(**_one_cell_kwargs(morphology="chain"))


def _two_cell_cfg():
    import jaxfne as jtfne

    return (
        jtfne.Configuration()
        .network(name="hh2", n=2, cell_types={"E": 1.0})
        .mechanisms(name="ampa", kind="receptor",
                    params={"tau_ms": 2.0, "reversal_mV": 0.0,
                            "tau_r_ms": 0.2, "slope_mV": 5.0,
                            "v_th_mV": -20.0, "s0": 0.1})
        .connections(name="e", source={"cell_type": "E"}, target={},
                     probability=None, weight=2.5, mechanism="ampa",
                     sign="excitatory")
    ).connectivity(edge_seed=0)


def test_construct_hh_forwards_morphology_and_comps():
    from jaxfne._construct_hh import construct_hh_model, hh_kernel_args

    morph = reference_4comp_morphology()
    model = construct_hh_model(_two_cell_cfg(), morphology=morph,
                               edge_post_comp=3, stim_comp=2)
    st = model.static["hh"]
    assert st["morphology"]["parents"] == (-1, 0, 1, 2)
    assert [int(v) for v in st["edge_post_comp"]] == [3] * len(st["post"])
    assert [int(v) for v in st["stim_comp"]] == [2, 2]
    args = hh_kernel_args(model, dt_ms=DT_MS)
    assert args["morphology"].parents == (-1, 0, 1, 2)
    assert [int(v) for v in args["edge_post_comp"]] == [3] * len(st["post"])


def test_construct_hh_refuses_comp_out_of_range():
    from jaxfne._construct_hh import construct_hh_model

    with pytest.raises(ValueError, match="edge_post_comp"):
        construct_hh_model(_two_cell_cfg(),
                           morphology=reference_4comp_morphology(),
                           edge_post_comp=99)


def _float32_two_cell():
    import jax.numpy as jnp

    morph = reference_4comp_morphology()
    params = make_hh_params(["E", "E"])
    params = params._replace(**{f: jnp.asarray(getattr(params, f), dtype=jnp.float32)
                                for f in params._fields})
    n_steps = 16
    stim = np.zeros((n_steps, 2), dtype=np.float32)
    stim[int(1.0 // DT_MS):, 0] = 0.5
    base = dict(
        params=params, n_steps=n_steps, dt_ms=DT_MS,
        edge_pre=[0], edge_post=[1], edge_receptor=[0],
        receptor_e_rev=(0.0,), receptor_tau_r=(0.2,), receptor_tau_d=(2.0,),
        receptor_slope=(5.0,), receptor_vth=(-20.0,),
        morphology=morph, edge_pre_comp=0, edge_post_comp=3,
        stim_comp=2, record_comp=0, record_v=True,
    )
    return base, stim


def test_multicomp_jit_float32_matches_eager():
    import jax

    base, stim = _float32_two_cell()
    eager = simulate_hh(edge_weight=[5.0], I_ext=stim, **base)
    jitted = jax.jit(lambda: simulate_hh(edge_weight=[5.0], I_ext=stim, **base))()
    jax.block_until_ready(jitted["V_mV"])
    np.testing.assert_array_equal(np.asarray(jitted["V_mV"]),
                                  np.asarray(eager["V_mV"]))


def test_multicomp_grad_wrt_weight_finite():
    import jax
    import jax.numpy as jnp

    base, stim = _float32_two_cell()

    def loss(w):
        out = simulate_hh(edge_weight=w, I_ext=stim, **base)
        return jnp.sum(out["V_mV"])

    g = jax.grad(loss)(jnp.asarray([5.0], dtype=jnp.float32))
    g = np.asarray(g)
    assert g.shape == (1,)
    assert np.all(np.isfinite(g))


def test_multicomp_empty_uS_matches_empty_density():
    morph, common = _two_cell_common()
    via_none = simulate_hh(**common)
    via_us_empty = simulate_hh(edge_weight_uS=[], **common)
    np.testing.assert_array_equal(np.asarray(via_us_empty["V_mV"]),
                                  np.asarray(via_none["V_mV"]))
    assert np.asarray(via_us_empty["spikes"]).shape == \
        np.asarray(via_none["spikes"]).shape


def test_multicomp_gate_init_default_is_reference():
    morph, common = _two_cell_common()
    default = simulate_hh(**common)
    explicit = simulate_hh(gate_init=0.2, **common)
    np.testing.assert_array_equal(np.asarray(default["V_mV"]),
                                  np.asarray(explicit["V_mV"]))
    other = simulate_hh(gate_init=0.05, **common)
    assert not np.array_equal(np.asarray(other["V_mV"]),
                              np.asarray(default["V_mV"]))


def test_construct_hh_forwards_weight_unit_and_gate_init():
    from jaxfne._construct_hh import construct_hh_model, hh_kernel_args

    morph = reference_4comp_morphology()
    model = construct_hh_model(_two_cell_cfg(), morphology=morph,
                               edge_post_comp=3, stim_comp=2, record_comp=0,
                               weight_unit="uS", gate_init=0.2)
    st = model.static["hh"]
    assert st["weight_unit"] == "uS"
    assert st["gate_init"] == 0.2
    args = hh_kernel_args(model, dt_ms=DT_MS)
    assert "edge_weight_uS" in args and "edge_weight" not in args
    assert args["gate_init"] == 0.2
    assert [int(v) for v in args["edge_post_comp"]] == [3] * len(st["post"])


def test_construct_hh_forwards_density_default_without_gate_init():
    from jaxfne._construct_hh import construct_hh_model, hh_kernel_args

    model = construct_hh_model(_two_cell_cfg())
    st = model.static["hh"]
    assert st["weight_unit"] == "density"
    assert st["gate_init"] is None
    args = hh_kernel_args(model, dt_ms=DT_MS)
    assert "edge_weight" in args and "edge_weight_uS" not in args
    assert "gate_init" not in args
    assert "morphology" not in args


def test_construct_hh_refuses_bad_weight_unit_and_gate_init():
    from jaxfne._construct_hh import construct_hh_model

    with pytest.raises(ValueError, match="weight_unit"):
        construct_hh_model(_two_cell_cfg(), weight_unit="nA")
    with pytest.raises(ValueError, match="needs morphology"):
        construct_hh_model(_two_cell_cfg(), weight_unit="uS")
    with pytest.raises(ValueError, match="needs morphology"):
        construct_hh_model(_two_cell_cfg(), stim_comp=2)
    with pytest.raises(ValueError, match="gate_init"):
        construct_hh_model(_two_cell_cfg(), gate_init=float("nan"))
