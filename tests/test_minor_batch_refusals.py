"""Minor-batch behavior pins: deprecation warning, refusals, strict load, shim API."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.emitters import _dtype_from_policy
from jaxfne.fields.proxy import project_laminar_sources


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


@pytest.mark.parametrize(
    "dt, taus, match",
    [(6.0, {}, "SYN_TAU_MS"), (2.0, {"tau_plus": 1.0}, "tau_plus"), (2.0, {"tau_minus": 1.5}, "tau_minus")],
)
def test_stdp_stream_dt_guard(dt, taus, match):
    """dt above the synaptic or an STDP trace time constant refuses (negative decay factor)."""
    n = 8
    rng = np.random.RandomState(0)
    stim = jnp.asarray(rng.uniform(0.0, 8.0, size=(10, n)), dtype=jnp.float32)
    zeros = jnp.zeros(n)
    state = jtfne.STDPState(W=jnp.zeros((n, n)), trace_pre=zeros, trace_post=zeros)
    with pytest.raises(ValueError, match=match):
        jtfne.run_stdp_stream(
            v_init=jnp.full(n, -65.0),
            u_init=zeros,
            s_init=zeros,
            stdp_state=state,
            stim_drive=stim,
            noise=jnp.zeros((10, n)),
            solver_config=jtfne.SolverConfig(method="euler", dt=dt),
            plasticity_config=jtfne.STDPPlasticityConfig(A_plus=0.01, A_minus=0.012, **taus),
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

# Silent fallbacks become errors (Hamm 2026-10-13): D3 float64 without x64, D4
# off-grid duration, F3 unconsumed compartment_axis, F7 select_neurons fallbacks.


@pytest.fixture()
def x64_off():
    orig = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", orig)


_NEURONS = [
    {"neuron_id": 0, "area": "V1", "layer": "L4", "cell_type": "E"},
    {"neuron_id": 1, "area": "V1", "layer": "L4", "cell_type": "E"},
    {"neuron_id": 2, "area": "V1", "layer": "L4", "cell_type": "PV"},
    {"neuron_id": 3, "area": "V4", "layer": "L2", "cell_type": "E"},
]
_MECHS = [{"name": "ampa", "kind": "synapse", "params": {"tau_ms": 5.0, "sign": 1.0}}]
_RULE = {
    "name": "rec",
    "source": {"area": "V1"},
    "target": {"area": "V1"},
    "probability": 0.5,
    "weight": 0.1,
    "mechanism": "ampa",
    "allow_self_connections": True,
}


class TestD3Float64Refusal:
    def test_dtype_from_policy_float64_raises_without_x64(self, x64_off):
        with pytest.raises(ValueError, match="jax_enable_x64"):
            _dtype_from_policy("float64")

    def test_dtype_from_policy_float32_passes_without_x64(self, x64_off):
        assert _dtype_from_policy("float32") == jnp.float32

    def test_compile_connection_rules_float64_raises_without_x64(self, x64_off):
        with pytest.raises(ValueError, match="jax_enable_x64"):
            jtfne.compile_connection_rules(_NEURONS, [_RULE], _MECHS, seed=0, dtype="float64")

    def test_to_edge_list_float64_raises_without_x64(self, x64_off):
        compiled = jtfne.compile_connection_rules(_NEURONS, [_RULE], _MECHS, seed=0)
        with pytest.raises(ValueError, match="jax_enable_x64"):
            compiled.to_edge_list(dtype="float64")

    def test_project_laminar_sources_float64_raises_without_x64(self, x64_off):
        sources = jnp.ones((10, 4), dtype=jnp.float32)
        positions = jnp.asarray(
            [[0.0, 0.0, 0.1], [0.0, 0.0, 0.4], [0.0, 0.0, 0.7], [0.0, 0.0, 0.9]]
        )
        with pytest.raises(ValueError, match="jax_enable_x64"):
            project_laminar_sources(sources, positions, n_contacts=4, dtype="float64")

    def test_drive_schedule_to_array_jax_float64_raises_without_x64(self, x64_off):
        from jaxfne._signals import StimulusSchedule

        sched = StimulusSchedule(events=(), n_neurons=2)
        with pytest.raises(ValueError, match="jax_enable_x64"):
            sched.to_array_jax(10, 0.5, dtype="float64")

    def test_runtime_config_actual_dtype_float64_raises_without_x64(self, x64_off):
        rt = jtfne.RuntimeConfig(dtype="float64")
        with pytest.raises(ValueError, match="jax_enable_x64"):
            _ = rt.actual_dtype


class TestD4OffGridDuration:
    def test_simulation_off_grid_raises(self):
        with pytest.raises(ValueError, match="grid-aligned"):
            jtfne.Simulation(duration_ms=100.0, dt_ms=0.3)

    def test_simulation_grid_aligned_ok(self):
        sim = jtfne.Simulation(duration_ms=200.0, dt_ms=0.5)
        assert sim.n_steps == 400

    def test_make_stimulus_off_grid_raises(self):
        with pytest.raises(ValueError, match="grid-aligned"):
            jtfne.make_stimulus(duration_ms=100.0, dt_ms=0.3)

    def test_hdp_run_off_grid_raises(self):
        from jaxfne import hdp_network as hn

        cfg = hn.HDPColumnConfig(n_neurons=8, duration_ms=10.0, dt_ms=0.5, seed=6)
        model = hn.build_model(cfg)
        with pytest.raises(ValueError, match="grid-aligned"):
            hn.run(model, cfg, 10.1, 6, hdp_kwargs=hn.disable_plasticity({"K_HDP": 0.01}))


class TestF3CompartmentAxis:
    def test_nondefault_axis_raises_without_jaxley(self):
        # Constructor stores the value; no jaxley import needed.
        with pytest.raises(ValueError, match="compartment_axis"):
            jtfne.bridges.JaxleyBridge(model="mock_model", compartment_axis="first")

    def test_default_axis_ok_without_jaxley(self):
        bridge = jtfne.bridges.JaxleyBridge(model="mock_model")
        assert bridge.compartment_axis == "last"


class TestF7SelectNeurons:
    def test_missing_table_raises(self):
        with pytest.raises(ValueError, match="neuron_table"):
            jtfne.select_neurons(object(), area="V1")

    def test_raising_table_raises(self):
        class _Bad:
            def neuron_table(self):
                raise RuntimeError("boom")

        with pytest.raises(ValueError, match="neuron_table"):
            jtfne.select_neurons(_Bad())

    def test_zero_match_returns_empty(self):
        rows = [{"neuron_id": 0, "area": "V1", "layer": "L4", "cell_type": "E"}]

        class _Model:
            def neuron_table(self):
                return list(rows)

        out = jtfne.select_neurons(_Model(), area="V4")
        assert isinstance(out, np.ndarray) and out.size == 0

    def test_match_returns_ids(self):
        rows = [
            {"neuron_id": 0, "area": "V1", "layer": "L4", "cell_type": "E"},
            {"neuron_id": 1, "area": "V4", "layer": "L4", "cell_type": "E"},
        ]

        class _Model:
            def neuron_table(self):
                return list(rows)

        out = jtfne.select_neurons(_Model(), area="V1")
        assert list(out) == [0]
