"""Minor-batch behavior pins: deprecation warning, refusals, strict load, shim API."""

import json

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne


def test_net_alias_warns_and_returns_model():
    """Net is a deprecated alias: warns on use, still returns Model."""
    with pytest.warns(DeprecationWarning, match="Use Model instead"):
        assert jtfne.Net is jtfne.Model


def test_triangular_drive_grid_refusal_and_shape():
    """Non-integral duration/dt refuses; integral grids keep their shape."""
    drive = jtfne.triangular_drive(duration_ms=10.0, dt_ms=0.5)
    assert drive.shape == (20,)
    with pytest.raises(ValueError, match="whole number"):
        jtfne.triangular_drive(duration_ms=10.0, dt_ms=0.3)


def test_sequential_joint_windows_codes_refuse():
    """event_windows + event_codes together refuse (windows would win silently)."""
    with pytest.raises(ValueError, match="together"):
        jtfne.general_sequential_oddball_paradigm(
            event_windows={"a": (0.0, 100.0)},
            event_codes={"b": (0.0, 100.0)},
        )


def test_sequential_comparison_collision_refuses():
    """A comparison label colliding with a sequence code refuses."""
    with pytest.raises(ValueError, match="collides"):
        jtfne.general_sequential_oddball_paradigm(
            event_windows={"p1": (0.0, 100.0)},
            sequence_event_labels=("p1",),
            conditions={"c": {"sequence": ("p1",)}},
            comparison_label="p1",
            comparison_code=102,
        )


def test_load_strict_refuses_future_schema(tmp_path):
    """strict=True refuses forward-reads with possibly dropped fields."""
    p = tmp_path / "future.json"
    p.write_text(json.dumps({"schema_version": "future-v9"}), encoding="utf-8")
    with pytest.raises(ValueError, match="strict"):
        jtfne.load(p, strict=True)


def test_clip_compat_install_uninstall_roundtrip():
    """Shim install/uninstall report bools; uninstall mirrors install state."""
    from jaxfne.bridges import _install_jax_clip_compat, uninstall_jax_clip_compat

    uninstall_jax_clip_compat()  # clear any residue from other tests
    installed = _install_jax_clip_compat()
    assert isinstance(installed, bool)
    assert uninstall_jax_clip_compat() is installed


def test_stdp_stream_dt_guard():
    """dt above the synaptic time constant refuses (negative decay factor)."""
    n = 8
    rng = np.random.RandomState(0)
    stim = jnp.asarray(rng.uniform(0.0, 8.0, size=(10, n)), dtype=jnp.float32)
    zeros = jnp.zeros(n)
    state = jtfne.STDPState(W=jnp.zeros((n, n)), trace_pre=zeros, trace_post=zeros)
    with pytest.raises(ValueError, match="SYN_TAU_MS"):
        jtfne.run_stdp_stream(
            v_init=jnp.full(n, -65.0),
            u_init=zeros,
            s_init=zeros,
            stdp_state=state,
            stim_drive=stim,
            noise=jnp.zeros((10, n)),
            solver_config=jtfne.SolverConfig(method="euler", dt=6.0),
            plasticity_config=jtfne.STDPPlasticityConfig(A_plus=0.01, A_minus=0.012),
            plasticity_scale=0.1,
            exc_mask=jnp.ones(n, dtype=bool),
            inh_mask=jnp.zeros(n, dtype=bool),
            a=jnp.full(n, 0.02),
            b=jnp.full(n, 0.2),
            c=jnp.full(n, -65.0),
            d=jnp.full(n, 8.0),
        )

def test_shared_stdp_kernel_hand_computed():
    """Shared kernel matches a hand-computed 2-neuron update."""
    from jaxfne.plasticity import stdp_weight_update

    W = jnp.zeros((2, 2))
    out = stdp_weight_update(
        W,
        trace_pre=jnp.array([1.0, 2.0]),
        trace_post=jnp.zeros(2),
        spiked=jnp.array([True, True]),
        exc_mask=jnp.array([True, True]),
        A_plus=1.0, A_minus=1.0, plasticity_scale=1.0,
        w_min=0.0, w_max=10.0,
    )
    # LTP only: dW_ltp = [[1, 2], [1, 2]]; diagonal masked (no autapse).
    assert jnp.array_equal(out, jnp.array([[0.0, 2.0], [1.0, 0.0]]))
