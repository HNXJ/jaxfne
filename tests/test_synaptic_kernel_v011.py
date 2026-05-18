"""Tests for v0.0.11 receptor-indexed exponential synaptic kernel.

The new kernel is opt-in via ``runtime(recurrent_backend="edge_list",
synaptic_kernel="receptor_exponential")``. All assertions are computational
correctness checks. No empirical-validation, biological-mechanism, calibrated-
amplitude, PDE, or field-physics claim is introduced by these tests.
"""

import json
import math

import jax
import jax.numpy as jnp
import pytest

import jaxfne as jtfne


def _cfg(n: int = 6):
    return (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=n, cell_types={"E": 0.7, "PV": 0.2, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )


# Test A — runtime synaptic kernel validation
def test_runtime_synaptic_kernel_validation():
    assert jtfne.runtime().synaptic_kernel == "exponential"
    assert jtfne.runtime(synaptic_kernel="exponential").synaptic_kernel == "exponential"
    rt = jtfne.runtime(synaptic_kernel="receptor_exponential")
    assert rt.synaptic_kernel == "receptor_exponential"
    with pytest.raises(ValueError):
        jtfne.runtime(synaptic_kernel="bad_kernel_name")
    with pytest.raises(ValueError):
        jtfne.RuntimeConfig(synaptic_kernel="not_a_kernel")


# Test B — receptor tau lookup
def test_receptor_tau_lookup():
    specs = jtfne.standard_receptor_specs()
    assert {"AMPA", "GABA_A", "NMDA", "GABA_B"} <= set(specs)
    assert specs["AMPA"].sign == 1
    assert specs["NMDA"].sign == 1
    assert specs["GABA_A"].sign == -1
    assert specs["GABA_B"].sign == -1
    # receptor_index stability
    assert specs["AMPA"].receptor_index == 0
    assert specs["GABA_A"].receptor_index == 1
    assert specs["NMDA"].receptor_index == 2
    assert specs["GABA_B"].receptor_index == 3
    # Standard tau table aligns with declared metadata
    table = jtfne.standard_receptor_tau_table()
    assert table.shape == (4,)
    assert float(table[0]) == pytest.approx(specs["AMPA"].tau_ms)
    assert float(table[1]) == pytest.approx(specs["GABA_A"].tau_ms)
    assert float(table[2]) == pytest.approx(specs["NMDA"].tau_ms)
    assert float(table[3]) == pytest.approx(specs["GABA_B"].tau_ms)


# Test C — receptor exponential impulse decay
def test_receptor_exponential_impulse_decay():
    # Build a tiny isolated edge: one pre, one post, one impulse, NMDA tau.
    edges = jtfne.EdgeList(
        pre=jnp.array([0], dtype=jnp.int32),
        post=jnp.array([1], dtype=jnp.int32),
        weight=jnp.array([1.0], dtype=jnp.float32),
        receptor_index=jnp.array([2], dtype=jnp.int32),  # NMDA → tau=100 ms
        tau_ms=jnp.array([2.0], dtype=jnp.float32),  # intentionally wrong; v0.0.11 must ignore
    )
    # 2-neuron Izhikevich params: pre fires once via huge drive, post is silent.
    n = 2
    big_drive = jnp.array([200.0, -1e6], dtype=jnp.float32)  # neuron 1 cannot fire
    params = jtfne.IzhikevichParams(
        a=jnp.full((n,), 0.02, dtype=jnp.float32),
        b=jnp.full((n,), 0.2, dtype=jnp.float32),
        c=jnp.full((n,), -65.0, dtype=jnp.float32),
        d=jnp.full((n,), 8.0, dtype=jnp.float32),
        drive=big_drive,
        sign=jnp.array([1.0, -1.0], dtype=jnp.float32),
        W=jnp.zeros((n, n), dtype=jnp.float32),
        v0=jnp.array([-65.0, -65.0], dtype=jnp.float32),
        u0=jnp.array([-13.0, -13.0], dtype=jnp.float32),
        source_scale=jnp.asarray(1.0, dtype=jnp.float32),
        labels=("E", "PV"),
    )
    dt_ms = 1.0
    n_steps = 50
    key = jax.random.PRNGKey(0)
    voltages, spikes, sources, final_state = jtfne.simulate_receptor_exponential_izhikevich(
        params, edges, n_steps, dt_ms, key, dtype="float32"
    )
    assert final_state["syn_state"].shape == (1,)
    # tau used must be NMDA (100ms), not the bogus 2ms stored in tau_ms
    assert float(final_state["tau_per_edge"][0]) == pytest.approx(100.0, rel=1e-4)
    # syn_state must be finite and bounded; decay is exp(-dt/tau)
    decay_step = math.exp(-dt_ms / 100.0)
    # Upper bound when a spike arrives every step at the edge
    geom_sum_cap = 1.0 / (1.0 - decay_step)
    assert float(final_state["syn_state"][0]) <= geom_sum_cap + 1e-3
    assert jnp.all(jnp.isfinite(final_state["syn_state"]))


# Test D — segment_sum no-double-count manual equivalence
def test_segment_sum_no_double_count_manual_equivalence():
    n_neurons = 4
    # Multiple edges deliberately converge on the same post-neuron (1, 2).
    pre = jnp.array([0, 1, 2, 3, 2], dtype=jnp.int32)
    post = jnp.array([1, 1, 2, 2, 2], dtype=jnp.int32)
    weight = jnp.array([0.7, -0.3, 0.5, 1.1, -0.2], dtype=jnp.float32)
    syn = jnp.array([1.0, 0.5, 0.8, 0.2, 0.4], dtype=jnp.float32)
    seg = jax.ops.segment_sum(weight * syn, post, n_neurons)
    manual = jnp.zeros((n_neurons,), dtype=jnp.float32)
    for e in range(int(pre.shape[0])):
        manual = manual.at[int(post[e])].add(weight[e] * syn[e])
    assert jnp.allclose(seg, manual, atol=1e-6, rtol=1e-6)
    # Sanity: post has repeats but segment_sum is still correct.
    assert int(jnp.sum(post == 2)) >= 2
    assert int(jnp.sum(post == 1)) >= 2


# Test E — deterministic fixed seed
def test_receptor_exponential_deterministic_fixed_seed():
    model = jtfne.construct(_cfg(n=5))
    rt = jtfne.runtime(recurrent_backend="edge_list", synaptic_kernel="receptor_exponential", seed=7)
    sim = jtfne.simulation(duration_ms=3.0, dt_ms=0.1, seed=7, runtime=rt)
    s1 = model.simulate(sim)
    s2 = model.simulate(sim)
    assert jnp.array_equal(s1.V_m, s2.V_m)
    assert jnp.array_equal(s1.spikes, s2.spikes)
    assert jnp.allclose(s1.sources, s2.sources)
    assert s1.metadata["synaptic_kernel"] == s2.metadata["synaptic_kernel"] == "receptor_exponential"


# Test F — finite state
def test_receptor_exponential_finite_state():
    model = jtfne.construct(_cfg(n=6))
    rt = jtfne.runtime(recurrent_backend="edge_list", synaptic_kernel="receptor_exponential")
    sim = jtfne.simulation(duration_ms=4.0, dt_ms=0.1, seed=11, runtime=rt)
    signals = model.simulate(sim)
    assert jnp.all(jnp.isfinite(signals.V_m))
    assert jnp.all(jnp.isfinite(signals.spikes))
    assert jnp.all(jnp.isfinite(signals.sources))
    manifest = model.manifest(signals=signals)
    json.dumps(manifest, allow_nan=False)


# Test G — default backward compatibility
def test_default_exponential_backward_compatibility():
    # Default runtime() must still be "exponential" and produce v0.0.9 metadata.
    assert jtfne.runtime().synaptic_kernel == "exponential"
    model = jtfne.construct(_cfg(n=5))
    rt = jtfne.runtime(recurrent_backend="edge_list")
    sim = jtfne.simulation(duration_ms=2.0, dt_ms=0.1, seed=3, runtime=rt)
    signals = model.simulate(sim)
    assert signals.metadata["recurrent_backend"] == "edge_list"
    assert signals.metadata["synaptic_kernel"] == "exponential"
    manifest = model.manifest(signals=signals)
    assert manifest["backend_metadata"]["used_synaptic_kernel"] == "exponential"
    assert manifest["backend_metadata"]["used_recurrent_backend"] == "edge_list"
    # Dense path must also be unaffected.
    sim_dense = jtfne.simulation(duration_ms=2.0, dt_ms=0.1, seed=3, runtime=jtfne.runtime())
    signals_dense = model.simulate(sim_dense)
    assert signals_dense.metadata["recurrent_backend"] == "dense"
    assert signals_dense.metadata["synaptic_kernel"] == "exponential"


# Test H — manifest JSON safety
def test_receptor_exponential_manifest_json_safe():
    model = jtfne.construct(_cfg(n=5))
    rt = jtfne.runtime(recurrent_backend="edge_list", synaptic_kernel="receptor_exponential")
    sim = jtfne.simulation(duration_ms=3.0, dt_ms=0.1, seed=5, runtime=rt)
    signals = model.simulate(sim)
    readout = model.probe(signals, modes=["spikes", "V_m", "CSD", "LFP"])
    manifest = model.manifest(signals=signals, readout=readout)
    assert "backend_metadata" in manifest
    bm = manifest["backend_metadata"]
    assert bm["used_recurrent_backend"] == "edge_list"
    assert bm["used_synaptic_kernel"] == "receptor_exponential"
    assert bm["edge_list_physical_amplitude_claim_allowed"] is False
    assert bm["edge_count"] > 0
    json.dumps(manifest, allow_nan=False)


# Test I — truth gates unchanged
def test_receptor_exponential_truth_gates_unchanged():
    model = jtfne.construct(_cfg(n=5))
    rt = jtfne.runtime(recurrent_backend="edge_list", synaptic_kernel="receptor_exponential")
    sim = jtfne.simulation(duration_ms=2.0, dt_ms=0.1, seed=9, runtime=rt)
    signals = model.simulate(sim)
    manifest = model.manifest(signals=signals)
    assert manifest["truth_mode"] == "truth_safe_unverified"
    assert manifest["claim_level"] == "computational_scaffold"
    assert manifest["source_projection_mode"] == "proxy_no_field_solve"
    assert manifest["field_solver_status"] == "laminar_proxy_no_pde"
    labels = manifest.get("v005_claim_labels", {})
    assert labels.get("field_claim_level", "proxy_readout_only") == "proxy_readout_only"
    assert labels.get("physical_amplitude_claim_allowed", False) is False
    assert labels.get("empirical_validation_status", "not_empirically_validated") == "not_empirically_validated"
    assert labels.get("mechanism_claim_status", "not_claimed") == "not_claimed"
    # Signals-level gate
    assert signals.metadata["field_claim_level"] == "proxy_readout_only"


# Test J — no new source mode
def test_receptor_exponential_no_new_source_mode():
    model = jtfne.construct(_cfg(n=5))
    rt_exp = jtfne.runtime(recurrent_backend="edge_list", synaptic_kernel="exponential")
    rt_rec = jtfne.runtime(recurrent_backend="edge_list", synaptic_kernel="receptor_exponential")
    sim_exp = jtfne.simulation(duration_ms=2.0, dt_ms=0.1, seed=4, runtime=rt_exp)
    sim_rec = jtfne.simulation(duration_ms=2.0, dt_ms=0.1, seed=4, runtime=rt_rec)
    m_exp = model.manifest(signals=model.simulate(sim_exp))
    m_rec = model.manifest(signals=model.simulate(sim_rec))
    # Source mode/projection metadata identical between kernels.
    for key in ("source_projection_mode", "source_decomposition", "field_solver_status", "source_calibration_status"):
        assert m_exp[key] == m_rec[key]
    # No per-receptor physical source export added.
    assert "per_receptor_source_terms" not in m_rec
    assert "physical_source_modes" not in m_rec
