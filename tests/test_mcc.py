"""Permanent Minimal Complete Circuit integration tests.

MCCs sit above focused unit tests and below scientific/publication benchmarks.
They intentionally use the public package chain while keeping topology and
assertions small enough for CPU CI.
"""

from __future__ import annotations

import json
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne import _pipeline
from jaxfne._model import _model_with_scalar_parameter

from mcc_fixtures import (
    MCC_COVERAGE_MAP,
    edge_list_runtime,
    hdp_runtime,
    mcc_model,
    mcc_tensor_model,
    mcc_stimulus,
    objective_dict,
)


EXPECTED_NEURONS = 10
EXPECTED_EDGES = 90
EXPECTED_EDGE_CLASSES = {
    ("E", "E"),
    ("E", "PV"),
    ("PV", "E"),
    ("PV", "PV"),
}


@pytest.fixture(scope="module")
def model() -> Any:
    return mcc_model()


@pytest.fixture(scope="module")
def tensor_model() -> Any:
    return mcc_tensor_model()


def _topology(model: Any) -> tuple[np.ndarray, np.ndarray, Any]:
    rows = model.neuron_table()
    labels = np.asarray([row["cell_type"] for row in rows])
    layers = np.asarray([row["layer"] for row in rows])
    edges = model.params["edge_list"]
    pre = np.asarray(edges.pre)
    post = np.asarray(edges.post)
    return labels, layers, edges


def _assert_topology(model: Any) -> None:
    labels, layers, edges = _topology(model)
    assert labels.shape == (EXPECTED_NEURONS,)
    assert set(layers.tolist()) == {"L2/3", "L4"}
    assert np.asarray(model.params["positions"]).shape == (EXPECTED_NEURONS, 3)
    pre = np.asarray(edges.pre)
    post = np.asarray(edges.post)
    actual_pairs = {(int(source), int(target)) for source, target in zip(pre, post)}
    expected_pairs = {
        (source, target)
        for source in range(EXPECTED_NEURONS)
        for target in range(EXPECTED_NEURONS)
        if source != target
    }
    assert int(edges.n_edges) == EXPECTED_EDGES
    assert len(actual_pairs) == EXPECTED_EDGES
    assert np.all(pre != post)
    assert actual_pairs == expected_pairs

    pairs = {
        (str(labels[source]), str(labels[target]))
        for source, target in zip(pre, post)
    }
    assert pairs == EXPECTED_EDGE_CLASSES

    for (source_type, target_type), expected_sign in (
        (("E", "E"), 1),
        (("E", "PV"), 1),
        (("PV", "E"), -1),
        (("PV", "PV"), -1),
    ):
        mask = (labels[np.asarray(edges.pre)] == source_type) & (
            labels[np.asarray(edges.post)] == target_type
        )
        weights = np.asarray(edges.weight)[mask]
        assert weights.size > 0
        assert np.all(np.sign(weights) == expected_sign)


def _assert_finite_array(name: str, value: Any) -> None:
    array = jnp.asarray(value)
    assert bool(jnp.all(jnp.isfinite(array))), f"{name} contains NaN/Inf"


def _assert_signal_outputs_finite(signals: Any) -> None:
    for name in ("time_ms", "V_m", "spikes", "sources"):
        value = getattr(signals, name)
        if value is not None:
            _assert_finite_array(f"signals.{name}", value)
    assert signals.field is not None
    for name in ("source_proxy", "phi_e_proxy", "csd_proxy", "lfp_proxy"):
        value = getattr(signals.field, name)
        if value is not None:
            _assert_finite_array(f"signals.field.{name}", value)


def _assert_readout_finite(readout: dict[str, Any]) -> None:
    for name, value in readout.items():
        if hasattr(value, "shape"):
            _assert_finite_array(f"readout.{name}", value)


def _run_hdp_segment(
    model: Any,
    *,
    duration_ms: float,
    seed: int,
    noise_scale: float,
    continuation: Any = None,
) -> tuple[Any, Any]:
    return jtfne.simulate(
        model,
        duration_ms=duration_ms,
        dt_ms=0.5,
        seed=seed,
        runtime=hdp_runtime(noise_scale=noise_scale),
        record_sources=True,
        record_fields=False,
        continuation=continuation,
        return_state=True,
    )


@pytest.fixture(scope="module")
def hdp_cases(model: Any) -> dict[float, tuple[Any, Any, Any, Any, Any, Any]]:
    cases: dict[float, tuple[Any, Any, Any, Any, Any, Any]] = {}
    for noise_scale in (0.0, 0.2):
        full, full_state = _run_hdp_segment(
            model, duration_ms=12.0, seed=17, noise_scale=noise_scale
        )
        first, first_state = _run_hdp_segment(
            model, duration_ms=6.0, seed=17, noise_scale=noise_scale
        )
        second, second_state = _run_hdp_segment(
            model,
            duration_ms=6.0,
            seed=999,
            noise_scale=noise_scale,
            continuation=first_state,
        )
        cases[noise_scale] = (
            full,
            full_state,
            first,
            first_state,
            second,
            second_state,
        )
    return cases


@pytest.fixture(scope="module")
def mcc1_result(model: Any) -> dict[str, Any]:
    """Run MCC-1's two repeatability executions once for all assertions."""
    stimulus = mcc_stimulus()
    runtime = edge_list_runtime()
    first = jtfne.simulate(
        model,
        duration_ms=10.0,
        dt_ms=0.5,
        seed=23,
        runtime=runtime,
        paradigm=stimulus,
        record_sources=True,
        record_fields=True,
    )
    repeat = jtfne.simulate(
        model,
        duration_ms=10.0,
        dt_ms=0.5,
        seed=23,
        runtime=runtime,
        paradigm=stimulus,
        record_sources=True,
        record_fields=True,
    )
    readout = model.probe(first, modes=["spikes", "V_m", "CSD", "LFP"])
    objective = jtfne.Objective(name="mcc1_rate_objective").loss(
        "target_rate",
        metric="spike_rate_hz_mean",
        target=0.0,
        weight=1.0,
    )
    evaluation = model.evaluate(first, objective)
    manifest = model.manifest(
        signals=first,
        readout=readout,
        objective=objective_dict(objective),
        evaluation=evaluation,
    )
    receipt = model.run_receipt(first, tags={"mcc": "MCC-1"})
    return {
        "stimulus": stimulus,
        "drive": stimulus.to_array(n_steps=20, dt_ms=0.5),
        "first": first,
        "repeat": repeat,
        "readout": readout,
        "objective": objective,
        "evaluation": evaluation,
        "manifest": manifest,
        "receipt": receipt,
    }


def _assert_segmented_matches(
    full: Any,
    full_state: Any,
    first: Any,
    second: Any,
    second_state: Any,
) -> None:
    for name, full_value, segmented_value in (
        (
            "V_m",
            full.V_m,
            jnp.concatenate((first.V_m, second.V_m), axis=0),
        ),
        (
            "spikes",
            full.spikes,
            jnp.concatenate((first.spikes, second.spikes), axis=0),
        ),
        (
            "sources",
            full.sources,
            jnp.concatenate((first.sources, second.sources), axis=0),
        ),
    ):
        assert jnp.array_equal(full_value, segmented_value), name

    for name in ("v", "u", "prev_spikes", "syn_state", "H", "w"):
        assert jnp.array_equal(
            getattr(full_state.dynamic, name),
            getattr(second_state.dynamic, name),
        ), name
    assert second_state.step_index == full_state.step_index == 24


def test_mcc_coverage_map_is_machine_readable() -> None:
    assert set(MCC_COVERAGE_MAP) >= {
        "CircuitSpec",
        "Model",
        "Signals",
        "HDP",
        "continuation",
        "objective",
        "optimizer",
        "manifest",
        "validation",
    }
    json.dumps(MCC_COVERAGE_MAP, allow_nan=False)


def test_mcc1_configuration_topology(model: Any) -> None:
    _assert_topology(model)


def test_mcc1_stimulus_and_signal_shapes(mcc1_result: dict[str, Any]) -> None:
    drive = mcc1_result["drive"]
    first = mcc1_result["first"]
    assert drive.shape == (20, EXPECTED_NEURONS)
    assert jnp.all(drive[:, [1, 2, 3, 4, 6, 7, 8, 9]] == 0.0)
    assert jnp.any(drive[:, [0, 5]] != 0.0)
    assert first.V_m.shape == (20, EXPECTED_NEURONS)
    assert first.spikes.shape == first.V_m.shape
    assert first.sources.shape == first.V_m.shape
    assert first.field is not None
    assert first.field.lfp_proxy.shape[0] == 20
    assert first.field.csd_proxy.shape[0] == 20
    assert first.metadata["field_claim_level"] == "proxy_readout"
    assert first.metadata["stimulus_injection_status"] == "native_drive_schedule_v0.0.12"
    _assert_signal_outputs_finite(first)


def test_mcc1_repeatability(mcc1_result: dict[str, Any]) -> None:
    first = mcc1_result["first"]
    repeat = mcc1_result["repeat"]
    assert jnp.array_equal(first.V_m, repeat.V_m)
    assert jnp.array_equal(first.spikes, repeat.spikes)
    assert jnp.array_equal(first.sources, repeat.sources)


def test_mcc1_probe_and_field_readouts(mcc1_result: dict[str, Any]) -> None:
    readout = mcc1_result["readout"]
    assert set(("spikes", "V_m", "CSD", "LFP")).issubset(readout)
    _assert_readout_finite(readout)


def test_mcc1_objective_manifest_and_receipt(mcc1_result: dict[str, Any]) -> None:
    evaluation = mcc1_result["evaluation"]
    assert evaluation["objective_name"] == "mcc1_rate_objective"
    assert jnp.isfinite(jnp.asarray(evaluation["total_loss"]))

    manifest = mcc1_result["manifest"]
    json.dumps(manifest, allow_nan=False)
    assert manifest["backend_metadata"]["edge_count"] == EXPECTED_EDGES
    assert manifest["field_claim_level"] == "proxy_readout"

    receipt = mcc1_result["receipt"]
    json.dumps(receipt.to_dict(), allow_nan=False)
    assert receipt.truth["physical_amplitude_calibrated"] is False


def test_mcc1_tensor_graph_is_structurally_equivalent(
    model: Any,
    tensor_model: Any,
) -> None:
    """The tensor path preserves MCC-1's exact executable graph."""

    _assert_topology(tensor_model)
    config_labels, config_layers, _ = _topology(model)
    tensor_labels, tensor_layers, tensor_edges = _topology(tensor_model)
    actual_pairs = {
        (int(pre), int(post))
        for pre, post in zip(
            np.asarray(tensor_edges.pre),
            np.asarray(tensor_edges.post),
        )
    }
    expected_pairs = {
        (source, target)
        for source in range(EXPECTED_NEURONS)
        for target in range(EXPECTED_NEURONS)
        if source != target
    }

    assert tensor_model.cfg.metadata["connectivity_mode"] == "explicit"
    assert tensor_edges.n_edges == EXPECTED_EDGES
    assert len(actual_pairs) == EXPECTED_EDGES
    assert actual_pairs == expected_pairs
    assert np.all(np.asarray(tensor_edges.pre) != np.asarray(tensor_edges.post))
    np.testing.assert_array_equal(tensor_labels, config_labels)
    np.testing.assert_array_equal(tensor_layers, config_layers)
    assert tensor_model.cfg.metadata["connectivity_compilation"] == {
        "connectivity_mode": "explicit",
        "default_edge_count": 0,
        "declared_rule_edge_count": EXPECTED_EDGES,
        "total_compiled_edge_count": EXPECTED_EDGES,
    }


@pytest.mark.parametrize("noise_scale", (0.0, 0.2))
def test_mcc2_full_state_continuation_matrix(
    hdp_cases: dict[float, tuple[Any, Any, Any, Any, Any, Any]],
    noise_scale: float,
) -> None:
    """MCC-2 proves deterministic and stochastic segmented continuation."""

    full, full_state, first, _, second, second_state = hdp_cases[noise_scale]
    _assert_segmented_matches(full, full_state, first, second, second_state)
    _assert_finite_array("continuation.dynamic.H", second_state.dynamic.H)
    _assert_finite_array("continuation.dynamic.w", second_state.dynamic.w)
    assert second_state.dynamic.syn_state.shape == (EXPECTED_EDGES,)
    assert second_state.dynamic.H.shape == (EXPECTED_NEURONS,)


def test_mcc2_scalar_hdp_and_weight_adaptation_are_nontrivial(
    model: Any,
    hdp_cases: dict[float, tuple[Any, Any, Any, Any, Any, Any]],
) -> None:
    """The executable kernel remains scalar-H while H and weights evolve."""

    _, state, *_ = hdp_cases[0.0]
    initial_weights = model.params["edge_list"].weight
    assert not jnp.array_equal(state.dynamic.H, jnp.ones_like(state.dynamic.H))
    assert not jnp.array_equal(state.dynamic.w, initial_weights)
    assert state.dynamic.H.ndim == 1
    assert state.dynamic.w.shape == (EXPECTED_EDGES,)


def test_mcc2_manifest_and_receipt_close_the_stateful_run(
    model: Any,
    hdp_cases: dict[float, tuple[Any, Any, Any, Any, Any, Any]],
) -> None:
    full, state, *_ = hdp_cases[0.0]
    manifest = model.manifest(
        signals=full,
        dataset={
            "mcc2": {
                "continuation_step_index": state.step_index,
                "dynamic_shapes": {
                    name: list(getattr(state.dynamic, name).shape)
                    for name in ("v", "u", "prev_spikes", "syn_state", "H", "w")
                },
            }
        },
    )
    json.dumps(manifest, allow_nan=False)
    receipt = model.run_receipt(full, tags={"mcc": "MCC-2"})
    json.dumps(receipt.to_dict(), allow_nan=False)
    assert manifest["dataset"]["mcc2"]["continuation_step_index"] == 24
    assert receipt.truth["physical_amplitude_calibrated"] is False


def test_mcc2_partial_h_w_initialization_remains_compatibility_api(
    model: Any,
    hdp_cases: dict[float, tuple[Any, Any, Any, Any, Any, Any]],
) -> None:
    """Legacy H/W initialization remains available and explicitly partial."""

    _, _, _, first_state, _, _ = hdp_cases[0.0]
    partial_model = model.with_hdp_initial_state(
        H0=first_state.dynamic.H,
        w0=first_state.dynamic.w,
    )
    assert jnp.array_equal(
        partial_model.params["hdp_initial_H"], first_state.dynamic.H
    )
    assert jnp.array_equal(
        partial_model.params["hdp_initial_w"], first_state.dynamic.w
    )
    signals = jtfne.simulate(
        partial_model,
        duration_ms=6.0,
        dt_ms=0.5,
        seed=999,
        runtime=hdp_runtime(noise_scale=0.0),
        record_sources=True,
        record_fields=False,
    )
    assert signals.V_m.shape == (12, EXPECTED_NEURONS)


def test_mcc2_partial_state_negative_control_is_not_exact(
    model: Any,
    hdp_cases: dict[float, tuple[Any, Any, Any, Any, Any, Any]],
) -> None:
    """Reset recurrent state while retaining H/W and prove divergence."""

    _, _, _, first_state, second, _ = hdp_cases[0.0]
    cold_dynamic = _pipeline.dynamic_state_from_model(model)
    partial_state = first_state._replace(
        dynamic=cold_dynamic._replace(
            H=first_state.dynamic.H,
            w=first_state.dynamic.w,
        )
    )
    partial, _ = _run_hdp_segment(
        model,
        duration_ms=6.0,
        seed=999,
        noise_scale=0.0,
        continuation=partial_state,
    )
    assert not jnp.array_equal(partial.V_m, second.V_m)


def test_mcc2_mismatched_prng_is_a_negative_control(
    model: Any,
    hdp_cases: dict[float, tuple[Any, Any, Any, Any, Any, Any]],
) -> None:
    """Changing only the carried stochastic state breaks stochastic equality."""

    _, _, _, first_state, second, _ = hdp_cases[0.2]
    bad_state = first_state._replace(prng_key=jax.random.PRNGKey(9999))
    bad, _ = _run_hdp_segment(
        model,
        duration_ms=6.0,
        seed=17,
        noise_scale=0.2,
        continuation=bad_state,
    )
    assert not jnp.array_equal(bad.V_m, second.V_m)


def test_mcc2_component_nulls_are_not_system_null(model: Any) -> None:
    """N_W and N_H are checked separately; N_system is not promoted."""

    initial_weights = model.params["edge_list"].weight
    weight_null = hdp_runtime(
        noise_scale=0.0,
        K_HDP=0.0,
        K_w_ctrl=0.0,
    )
    _, weight_null_state = jtfne.simulate(
        model,
        duration_ms=6.0,
        dt_ms=0.5,
        seed=17,
        runtime=weight_null,
        record_sources=True,
        record_fields=False,
        return_state=True,
    )
    assert jnp.array_equal(weight_null_state.dynamic.w, initial_weights)
    assert not jnp.array_equal(
        weight_null_state.dynamic.H, jnp.ones_like(weight_null_state.dynamic.H)
    )

    h_null = hdp_runtime(
        noise_scale=0.0,
        K_ctrl=0.0,
        alpha=0.0,
        barrier_c=0.0,
        barrier_d=0.0,
        C_spike=0.0,
    )
    _, h_null_state = jtfne.simulate(
        model,
        duration_ms=6.0,
        dt_ms=0.5,
        seed=17,
        runtime=h_null,
        record_sources=True,
        record_fields=False,
        return_state=True,
    )
    assert jnp.array_equal(h_null_state.dynamic.H, jnp.ones_like(h_null_state.dynamic.H))

    baseline, _ = jtfne.simulate(
        model,
        duration_ms=6.0,
        dt_ms=0.5,
        seed=17,
        runtime=edge_list_runtime(),
        record_sources=True,
        record_fields=False,
        return_state=True,
    )
    component_null, _ = jtfne.simulate(
        model,
        duration_ms=6.0,
        dt_ms=0.5,
        seed=17,
        runtime=h_null,
        record_sources=True,
        record_fields=False,
        return_state=True,
    )
    assert not jnp.array_equal(component_null.V_m, baseline.V_m)


def test_mcc2_h_carrier_preserves_future_trailing_shape() -> None:
    """Current scalar dynamics and future shape-general carrier stay distinct."""

    dynamic = _pipeline.DynamicState(
        v=jnp.zeros((EXPECTED_NEURONS,)),
        u=jnp.zeros((EXPECTED_NEURONS,)),
        prev_spikes=jnp.zeros((EXPECTED_NEURONS,)),
        syn_state=jnp.zeros((EXPECTED_EDGES,)),
        H=jnp.zeros((EXPECTED_NEURONS, 2)),
        w=jnp.zeros((EXPECTED_EDGES,)),
    )
    state = jtfne.ContinuationState(
        dynamic=dynamic,
        prng_key=jax.random.PRNGKey(0),
    )
    leaves, treedef = jax.tree_util.tree_flatten(state)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert restored.dynamic.H.shape == (EXPECTED_NEURONS, 2)


@pytest.fixture(scope="module")
def vector_h_cases(model: Any) -> dict[str, Any]:
    """Execute one independent/coupled vector-H continuation matrix."""
    H0 = jnp.tile(jnp.asarray([[1.0, 0.5]], dtype=jnp.float32), (EXPECTED_NEURONS, 1))
    vector_model = model.with_hdp_initial_state(H0=H0)
    independent_runtime = hdp_runtime(
        noise_scale=0.2,
        h_state_dim=2,
        h_state_readout=(0.5, 0.5),
        h_state_coupling=((0.0, 0.0), (0.0, 0.0)),
        alpha=0.2,
        K_ctrl=0.1,
    )
    coupled_runtime = hdp_runtime(
        noise_scale=0.2,
        h_state_dim=2,
        h_state_readout=(0.5, 0.5),
        h_state_coupling=((-0.02, 0.01), (0.01, -0.02)),
        alpha=0.2,
        K_ctrl=0.1,
    )
    common = dict(
        dt_ms=0.5,
        seed=17,
        record_sources=True,
        record_fields=False,
        return_state=True,
    )
    independent_full, independent_full_state = jtfne.simulate(
        vector_model,
        duration_ms=6.0,
        runtime=independent_runtime,
        **common,
    )
    independent_first, independent_first_state = jtfne.simulate(
        vector_model,
        duration_ms=3.0,
        runtime=independent_runtime,
        **common,
    )
    independent_second, independent_second_state = jtfne.simulate(
        vector_model,
        duration_ms=3.0,
        runtime=independent_runtime,
        seed=999,
        continuation=independent_first_state,
        **{key: value for key, value in common.items() if key != "seed"},
    )
    coupled, coupled_state = jtfne.simulate(
        vector_model,
        duration_ms=6.0,
        runtime=coupled_runtime,
        **common,
    )
    return {
        "full": independent_full,
        "full_state": independent_full_state,
        "first": independent_first,
        "second": independent_second,
        "second_state": independent_second_state,
        "coupled": coupled,
        "coupled_state": coupled_state,
        "model": vector_model,
        "runtime": independent_runtime,
    }


def test_mcc2_vector_h_independent_components_are_finite(
    vector_h_cases: dict[str, Any],
) -> None:
    state = vector_h_cases["full_state"]
    assert state.dynamic.H.shape == (EXPECTED_NEURONS, 2)
    assert state.dynamic.w.shape == (EXPECTED_EDGES,)
    assert jnp.all(jnp.isfinite(state.dynamic.H))
    assert jnp.all(jnp.isfinite(state.dynamic.v))
    assert jnp.all(jnp.isfinite(state.dynamic.w))
    assert vector_h_cases["full"].metadata["hdp"]["h_state"]["h_state_dim"] == 2
    json.dumps(vector_h_cases["full"].metadata, allow_nan=False)


def test_mcc2_vector_h_continuation_matches_uninterrupted_run(
    vector_h_cases: dict[str, Any],
) -> None:
    full = vector_h_cases["full"]
    first = vector_h_cases["first"]
    second = vector_h_cases["second"]
    full_state = vector_h_cases["full_state"]
    second_state = vector_h_cases["second_state"]
    for full_arr, segmented_arr in (
        (full.V_m, jnp.concatenate((first.V_m, second.V_m), axis=0)),
        (full.spikes, jnp.concatenate((first.spikes, second.spikes), axis=0)),
        (full.sources, jnp.concatenate((first.sources, second.sources), axis=0)),
    ):
        assert jnp.array_equal(full_arr, segmented_arr)
    assert jnp.array_equal(full_state.dynamic.H, second_state.dynamic.H)
    assert jnp.array_equal(full_state.dynamic.w, second_state.dynamic.w)


def test_mcc2_vector_h_coupling_is_explicit_and_changes_the_state(
    vector_h_cases: dict[str, Any],
) -> None:
    independent = vector_h_cases["full_state"].dynamic.H
    coupled = vector_h_cases["coupled_state"].dynamic.H
    assert coupled.shape == independent.shape
    assert jnp.all(jnp.isfinite(coupled))
    assert not jnp.array_equal(independent, coupled)
    assert vector_h_cases["coupled"].metadata["hdp"]["h_state"]["coupling"]["enabled"]


def test_mcc2_vector_h_nulls_remain_componentwise_distinct(
    vector_h_cases: dict[str, Any],
) -> None:
    model = vector_h_cases["model"]
    h_null_runtime = hdp_runtime(
        noise_scale=0.2,
        h_state_dim=2,
        h_state_readout=(0.5, 0.5),
        alpha=0.0,
        beta=0.0,
        gamma=0.0,
        delta=0.0,
        rho_passive=0.0,
        K_ctrl=0.0,
        C_spike=0.0,
        barrier_c=0.0,
        barrier_d=0.0,
        h_state_coupling=((0.0, 0.0), (0.0, 0.0)),
    )
    _, h_null_state = jtfne.simulate(
        model,
        duration_ms=2.0,
        dt_ms=0.5,
        seed=17,
        runtime=h_null_runtime,
        record_fields=False,
        return_state=True,
    )
    assert jnp.array_equal(
        h_null_state.dynamic.H,
        model.params["hdp_initial_H"],
    )

    weight_null_runtime = hdp_runtime(
        noise_scale=0.0,
        h_state_dim=2,
        K_HDP=0.0,
        K_w_ctrl=0.0,
        alpha=0.2,
    )
    _, weight_null_state = jtfne.simulate(
        model,
        duration_ms=2.0,
        dt_ms=0.5,
        seed=17,
        runtime=weight_null_runtime,
        record_fields=False,
        return_state=True,
    )
    assert jnp.array_equal(
        weight_null_state.dynamic.w,
        model.params["edge_list"].weight,
    )
    assert not jnp.array_equal(
        weight_null_state.dynamic.H,
        model.params["hdp_initial_H"],
    )


def test_mcc2_vector_h_shape_contract_rejects_incompatible_inputs(
    vector_h_cases: dict[str, Any],
) -> None:
    model = vector_h_cases["model"]
    runtime = vector_h_cases["runtime"]
    with pytest.raises(ValueError, match="H_final must have shape"):
        jtfne.simulate(
            model.with_hdp_initial_state(H0=jnp.ones((EXPECTED_NEURONS - 1, 2))),
            duration_ms=1.0,
            dt_ms=0.5,
            seed=17,
            runtime=runtime,
            record_fields=False,
        )
    with pytest.raises(ValueError, match="h_state_readout must have shape"):
        jtfne.simulate(
            model,
            duration_ms=1.0,
            dt_ms=0.5,
            seed=17,
            runtime=hdp_runtime(
                noise_scale=0.0,
                h_state_dim=2,
                h_state_readout=(1.0,),
            ),
            record_fields=False,
        )
    with pytest.raises(ValueError, match="h_state_coupling must have shape"):
        jtfne.simulate(
            model,
            duration_ms=1.0,
            dt_ms=0.5,
            seed=17,
            runtime=hdp_runtime(
                noise_scale=0.0,
                h_state_dim=2,
                h_state_coupling=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ),
            record_fields=False,
        )


def test_mcc2_scalar_ordinary_and_continuation_dispatch_are_identical(
    model: Any,
) -> None:
    runtime = hdp_runtime(
        noise_scale=0.2,
        rho_passive=0.1,
        hdp_rule="hebbian_product",
    )
    ordinary = jtfne.simulate(
        model,
        duration_ms=2.0,
        dt_ms=0.5,
        seed=17,
        runtime=runtime,
        record_sources=True,
        record_fields=False,
    )
    continuation, _ = jtfne.simulate(
        model,
        duration_ms=2.0,
        dt_ms=0.5,
        seed=17,
        runtime=runtime,
        record_sources=True,
        record_fields=False,
        return_state=True,
    )
    for ordinary_arr, continuation_arr in (
        (ordinary.V_m, continuation.V_m),
        (ordinary.spikes, continuation.spikes),
        (ordinary.sources, continuation.sources),
    ):
        assert jnp.array_equal(ordinary_arr, continuation_arr)


def test_mcc2_vector_h_batch_dispatch_forwards_configuration(
    vector_h_cases: dict[str, Any],
) -> None:
    simulation = jtfne.simulation(
        duration_ms=1.0,
        dt_ms=0.5,
        seed=17,
        runtime=vector_h_cases["runtime"],
        record_fields=False,
    )
    batch = vector_h_cases["model"].simulate_batch(simulation, n_seeds=2)
    assert batch["V_m"].shape == (2, 2, EXPECTED_NEURONS)
    assert batch["spikes"].shape == (2, 2, EXPECTED_NEURONS)
    assert batch["metadata"]["hdp_params"]["h_state_dim"] == 2


@pytest.mark.parametrize("mode", ("E_silence", "I_silence", "disconnected_null"))
def test_mcc2_unsupported_ablations_are_rejected(mode: str, model: Any) -> None:
    with pytest.raises(ValueError, match="does not support ablation"):
        jtfne.simulate(
            model,
            duration_ms=1.0,
            dt_ms=0.5,
            seed=17,
            runtime=edge_list_runtime(),
            ablation=mode,
            record_fields=False,
            return_state=True,
        )


def test_mcc2_unsupported_receptor_continuation_is_rejected(model: Any) -> None:
    with pytest.raises(ValueError, match="supports only synaptic_kernel"):
        jtfne.simulate(
            model,
            duration_ms=1.0,
            dt_ms=0.5,
            seed=17,
            runtime=edge_list_runtime(synaptic_kernel="receptor_exponential"),
            record_fields=False,
            return_state=True,
        )


@pytest.fixture(scope="module")
def mcc3_case(model: Any) -> dict[str, Any]:
    theta_star = 1.6
    bounds = (0.5, 2.0)
    simulation = jtfne.simulation(
        duration_ms=5.0,
        dt_ms=0.5,
        seed=0,
        runtime=edge_list_runtime(),
        record_sources=True,
        record_fields=True,
    )
    target_model = _model_with_scalar_parameter(model, "source_scale", theta_star)
    target_signals = target_model.simulate(simulation)
    target_value = float(jnp.abs(target_signals.field.source_proxy).mean())
    objective = jtfne.Objective(name="mcc3_source_recovery").loss(
        "source_proxy_target",
        metric="source_proxy_abs_mean",
        target=target_value,
        weight=1.0,
    )
    initial_signals = model.simulate(simulation)
    initial_evaluation = model.evaluate(initial_signals, objective)

    result = model.tune(
        objective,
        optimizer="AGSDR",
        simulation=simulation,
        steps=10,
        seed=42,
        parameter="source_scale",
        bounds=bounds,
    )
    probe_scales = (0.5, 0.875, 1.25, 1.625, 2.0)
    probe_values = []
    for scale in probe_scales:
        probe_model = _model_with_scalar_parameter(model, "source_scale", scale)
        probe_signals = probe_model.simulate(simulation)
        probe_values.append(float(jnp.abs(probe_signals.field.source_proxy).mean()))
    repeat_result = model.tune(
        objective,
        optimizer="AGSDR",
        simulation=simulation,
        steps=10,
        seed=42,
        parameter="source_scale",
        bounds=bounds,
    )
    inferred = float(result.summary["best_parameter_value"])
    evidence = {
        "generating_parameters": {"source_scale": theta_star},
        "initial_parameters": {"source_scale": 1.0},
        "inferred_parameters": {"source_scale": inferred},
        "bounds": [float(bounds[0]), float(bounds[1])],
        "target_source_proxy_abs_mean": target_value,
        "initial_score": float(initial_evaluation["total_loss"]),
        "best_score": float(result.best_score),
        "identifiability_probe": {
            "scales": list(probe_scales),
            "observable_values": probe_values,
        },
        "objective_components": objective_dict(objective),
        "optimizer": result.summary["optimizer"],
    }
    manifest = model.manifest(
        signals=initial_signals,
        objective=objective_dict(objective),
        evaluation=initial_evaluation,
        tuning={**result.summary, "mcc3_evidence": evidence},
        dataset={"mcc3_evidence": evidence, "coverage": MCC_COVERAGE_MAP},
    )
    receipt = model.run_receipt(initial_signals, tags={"mcc": "MCC-3"})
    return {
        "theta_star": theta_star,
        "bounds": bounds,
        "objective": objective,
        "initial_evaluation": initial_evaluation,
        "result": result,
        "repeat_result": repeat_result,
        "evidence": evidence,
        "manifest": manifest,
        "receipt": receipt,
    }


def test_mcc3_identifiable_recovery_and_improved_objective(
    mcc3_case: dict[str, Any],
) -> None:
    evidence = mcc3_case["evidence"]
    result = mcc3_case["result"]
    probe = mcc3_case["evidence"]["identifiability_probe"]
    probe_values = probe["observable_values"]
    assert all(np.isfinite(value) for value in probe_values)
    assert all(left < right for left, right in zip(probe_values, probe_values[1:]))
    assert evidence["initial_parameters"]["source_scale"] != evidence["generating_parameters"]["source_scale"]
    assert evidence["best_score"] < evidence["initial_score"]
    assert abs(evidence["inferred_parameters"]["source_scale"] - evidence["generating_parameters"]["source_scale"]) < 0.15
    assert result.summary["tuning_status"] == "blackbox_loop_v0.0.6"
    assert result.summary["tuning_path"] == "scalar_black_box"
    assert result.summary["parameter"] == "source_scale"
    assert result.summary["losses_declared"] == 1

    candidates = result.summary["candidate_values"]
    assert all(mcc3_case["bounds"][0] <= value <= mcc3_case["bounds"][1] for value in candidates)
    assert all(np.isfinite(value) for value in candidates)
    assert all(np.isfinite(value) for value in result.summary["candidate_scores"])
    assert np.isfinite(result.summary["best_score"])


def test_mcc3_optimizer_and_evidence_are_reproducible_and_json_safe(
    mcc3_case: dict[str, Any],
) -> None:
    result = mcc3_case["result"]
    repeat = mcc3_case["repeat_result"]
    assert result.summary["candidate_values"] == repeat.summary["candidate_values"]
    assert result.summary["candidate_scores"] == repeat.summary["candidate_scores"]
    assert result.summary["best_parameter_value"] == repeat.summary["best_parameter_value"]
    assert result.summary["best_score"] == repeat.summary["best_score"]
    json.dumps(result.summary, allow_nan=False)
    json.dumps(mcc3_case["evidence"], allow_nan=False)
    json.dumps(mcc3_case["manifest"], allow_nan=False)
    json.dumps(mcc3_case["receipt"].to_dict(), allow_nan=False)
    assert "mcc3_evidence" in mcc3_case["manifest"]["tuning"]
    assert "best_parameter_value" in mcc3_case["manifest"]["tuning"]
