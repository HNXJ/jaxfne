"""Vogels 2011 inhibitory STDP rule (``jaxfne/istdp_vogels2011.py``).

Paper rule (authors' ``isp_brian.py``, read as text only): traces
``dA/dt = -A/tau`` (tau = 20 ms); presynaptic spike ``A_pre += 1`` and
``w += (A_post - alpha)*eta``; postsynaptic spike ``A_post += 1`` and
``w += A_pre*eta``; ``eta = 1e-2``, ``alpha = 2*rho0*tau`` (rho0 = 3 Hz);
I-to-E only. The kernel integrates rate-form drives (jump/dt), so the
hand values below divide each event jump by ``dt = 0.1`` ms.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.emitters_adex import make_adex_params, simulate_adex_hdp_registered
from jaxfne.hdp_rule import (
    HDPRuleContext,
    check_hdp_rule_params,
    get_hdp_rule,
    list_registered_hdp_rules,
)
from jaxfne.istdp_vogels2011 import (
    ISTDP_RULE_NAME,
    istdp_alpha,
    istdp_plasticity_mask,
    resolve_istdp_params,
)
from jaxfne.public_surface import symbol_tier

assert "site-packages" not in jtfne.__file__

DT_MS = 0.1
N_STEPS = 100
ETA = 1e-2
TAU_MS = 20.0
ALPHA = 2.0 * 3.0 * (TAU_MS / 1000.0)  # 0.12 at the paper values


@pytest.fixture(autouse=True)
def _float64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _ctx(*, spikes, pre_sp, aux=None, rule_params=None):
    n = 2
    return HDPRuleContext(
        H=jnp.ones((n,)),
        aux=jnp.zeros((n, 2)) if aux is None else jnp.asarray(aux),
        v=jnp.zeros((n,)),
        u=jnp.zeros((n,)),
        spikes=jnp.asarray(spikes, dtype=jnp.float64),
        prev_spikes=jnp.zeros((n,)),
        syn_state=jnp.zeros((1,)),
        w=jnp.asarray([3.0]),
        pre=jnp.asarray([1]),
        post=jnp.asarray([0]),
        exc_mask=jnp.asarray([True]),
        dt=jnp.asarray(DT_MS),
        n_neurons=n,
        rule_params=dict(rule_params or {}),
        key=jax.random.PRNGKey(0),
        pre_sp=None if pre_sp is None else jnp.asarray(pre_sp, dtype=jnp.float64),
    )


def _step(ctx):
    _, rule_fn = get_hdp_rule(ISTDP_RULE_NAME)
    return rule_fn(ctx)


def test_rule_is_registered_and_advanced():
    """The rule is registered, ADVANCED-tier, and outside __all__."""
    assert ISTDP_RULE_NAME in list_registered_hdp_rules()
    assert hasattr(jtfne, "istdp_vogels2011")
    assert "istdp_vogels2011" not in jtfne.__all__
    assert symbol_tier("istdp_vogels2011") == "ADVANCED"


def test_alpha_default_is_paper_value():
    """alpha = 2*rho0*tau is 0.12 at the paper values."""
    assert istdp_alpha() == pytest.approx(0.12)
    assert resolve_istdp_params({})["alpha"] == pytest.approx(0.12)


def test_pre_spike_depresses_without_post_trace():
    """Lone presynaptic spike on neuron 1: A_pre[1] += 1 (and its own
    A_post[1] += 1, every neuron carries both traces), w += (0-alpha)*eta."""
    upd = _step(_ctx(spikes=[0.0, 1.0], pre_sp=[1.0]))
    np.testing.assert_allclose(np.asarray(upd.dH), [0.0, 0.0], rtol=0, atol=0)
    np.testing.assert_allclose(
        np.asarray(upd.d_aux), [[0.0, 0.0], [1.0 / DT_MS, 1.0 / DT_MS]], rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(upd.d_theta["edge_weight"]),
        [ETA * (0.0 - ALPHA) / DT_MS],
        rtol=1e-12,
    )


def test_post_spike_potentiates_with_pre_trace():
    """Postsynaptic spike with A_pre = 0.5: A_post += 1, w += 0.5*eta."""
    aux = [[0.0, 0.0], [0.5, 0.0]]
    upd = _step(_ctx(spikes=[1.0, 0.0], pre_sp=[0.0], aux=aux))
    np.testing.assert_allclose(
        np.asarray(upd.d_aux), [[0.0, 1.0 / DT_MS], [-0.5 / TAU_MS, 0.0]], rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(upd.d_theta["edge_weight"]), [ETA * 0.5 / DT_MS], rtol=1e-12,
    )


def test_rho0_sets_alpha():
    """rho0_hz = 6 doubles the depression term (alpha = 0.24)."""
    upd = _step(_ctx(spikes=[0.0, 1.0], pre_sp=[1.0], rule_params={"rho0_hz": 6.0}))
    np.testing.assert_allclose(
        np.asarray(upd.d_theta["edge_weight"]),
        [ETA * (0.0 - 0.24) / DT_MS],
        rtol=1e-12,
    )


def test_unknown_param_key_is_refused():
    """A mistyped rule param fails closed instead of simulating default."""
    with pytest.raises(ValueError, match="unrecognized keys"):
        check_hdp_rule_params(ISTDP_RULE_NAME, {"eta_typo": 0.1})
    with pytest.raises(ValueError, match="unrecognized keys"):
        simulate_adex_hdp_registered(
            make_adex_params(("E", "PV")),
            n_steps=10,
            dt_ms=DT_MS,
            edge_pre=[1],
            edge_post=[0],
            edge_weight_nS=[3.0],
            edge_receptor=[0],
            edge_delay_steps=[1],
            receptor_reversal_mV=(0.0,),
            receptor_tau_ms=(2.0,),
            decay="exp",
            hdp_rule=ISTDP_RULE_NAME,
            hdp_rule_params={"eta_typo": 0.1},
        )


def test_mask_builder_marks_only_i_to_e():
    """The mask is True exactly on I-to-E edges."""
    got = np.asarray(istdp_plasticity_mask(
        [False, True, True], [True, True, False]))
    assert got.tolist() == [False, True, False]


def test_mask_builder_refuses_empty_and_mismatched():
    """No I-to-E edge (or ragged inputs) is refused, never silently static."""
    with pytest.raises(ValueError, match="no edge"):
        istdp_plasticity_mask([False, False], [True, True])
    with pytest.raises(ValueError, match="equal length"):
        istdp_plasticity_mask([True], [True, False])


def _two_edge_run():
    """Two forced spikes: pre (PV cell 1) at t=30, post (E cell 0) at t=45.

    A -500 pA holding current blocks every other spike, so the hand values
    below are exact: the post jump (+1 at t=45) decays 4 steps, then the
    delayed pre arrival (t=50) reads it.
    """
    I_pA = np.full((N_STEPS, 2), -500.0)
    I_pA[30, 1] = 1e5
    I_pA[45, 0] = 1e5
    return simulate_adex_hdp_registered(
        make_adex_params(("E", "PV")),
        n_steps=N_STEPS,
        dt_ms=DT_MS,
        edge_pre=[0, 1],
        edge_post=[1, 0],
        edge_weight_nS=[3.0, 3.0],
        edge_receptor=[0, 0],
        edge_delay_steps=[20, 20],
        receptor_reversal_mV=(0.0,),
        receptor_tau_ms=(2.0,),
        decay="exp",
        I_ext_pA=I_pA,
        record_v=False,
        hdp_rule=ISTDP_RULE_NAME,
        plasticity_mask=istdp_plasticity_mask([False, True], [False, True]),
        seed=0,
    )


def test_kernel_moves_i_to_e_and_freezes_e_edges():
    """End to end through the registered AdEx kernel: the E-presynaptic edge
    is bit-identical, the I-to-E edge grows by the hand pairing update, H
    stays 1, and only the two forced spikes exist."""
    out = _two_edge_run()
    spikes = np.asarray(out["spikes"], dtype=bool)
    assert [(int(t), int(c)) for t, c in np.argwhere(spikes)] == [(30, 1), (45, 0)]
    w_final = np.asarray(out["w_final_nS"], dtype=float)
    assert w_final[0] == 3.0, "masked E edge must stay bit-identical"
    decayed = (1.0 - DT_MS / TAU_MS) ** 4
    expected = 3.0 + DT_MS * (ETA * (decayed - ALPHA) / DT_MS)
    np.testing.assert_allclose(w_final[1], expected, rtol=1e-9)
    assert w_final[1] > 3.0, "paired I-to-E weight must potentiate"
    np.testing.assert_allclose(
        np.asarray(out["H_final"], dtype=float), [1.0, 1.0], rtol=0, atol=0,
    )
    aux = np.asarray(out["aux_final"], dtype=float)
    assert aux.shape == (2, 2), "aux carries (A_pre, A_post) per neuron"
    assert aux[1, 0] > 0.5, "pre trace must hold the t=50 arrival"


def test_kernel_refuses_bad_mask_shape():
    """A mask outside the edge order fails closed at the kernel."""
    with pytest.raises(ValueError, match="plasticity_mask must have shape"):
        simulate_adex_hdp_registered(
            make_adex_params(("E", "PV")),
            n_steps=10,
            dt_ms=DT_MS,
            edge_pre=[1],
            edge_post=[0],
            edge_weight_nS=[3.0],
            edge_receptor=[0],
            edge_delay_steps=[1],
            receptor_reversal_mV=(0.0,),
            receptor_tau_ms=(2.0,),
            decay="exp",
            hdp_rule=ISTDP_RULE_NAME,
            plasticity_mask=[True, False],
            seed=0,
        )
