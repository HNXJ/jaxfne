"""Cheap direct property tests for the 0.4.11 observational TFNE chain."""

from __future__ import annotations

import jax.numpy as jnp

import jaxfne as jtfne
from jaxfne import (
    BasisSpec,
    DEFAULT_SPIKE_IMPULSE_GAIN,
    eeg_proxy_transform,
    meg_proxy_transform,
    project_laminar_sources,
)
from jaxfne._model import _SOURCE_PROXY_METADATA
from jaxfne.emitters import _source_proxy_from_components
from jaxfne.fields.proxy import csd_tensor
from jaxfne.fields.probes import eeg_proxy_probe
from jaxfne.validation import CalibrationSpec, make_calibration_report
from jaxfne.sanity_runtime import _make_probe_metrics


def test_canonical_source_contract_is_single_composition_boundary() -> None:
    current_native = jnp.asarray([[1.0, -2.0], [0.5, 3.0]], dtype=jnp.float32)
    spikes = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    source_scale = jnp.asarray([2.0, 0.5], dtype=jnp.float32)

    actual = _source_proxy_from_components(
        current_native,
        spikes,
        source_scale,
        dtype=jnp.float32,
    )
    expected = source_scale * (
        current_native + DEFAULT_SPIKE_IMPULSE_GAIN * spikes
    )

    assert jnp.array_equal(actual, expected)
    assert _SOURCE_PROXY_METADATA["source_contract"]["gain_owner"] == (
        "jaxfne.presets.DEFAULT_SPIKE_IMPULSE_GAIN"
    )
    assert _SOURCE_PROXY_METADATA["double_count_evidence"]["status"] == "helper_backed"


def test_laminar_projection_exposes_operator_and_normalization_semantics() -> None:
    sources = jnp.asarray([[1.0, 2.0, 3.0]], dtype=jnp.float32)
    positions = jnp.asarray(
        [[0.0, 0.0, 0.1], [0.0, 0.0, 0.5], [0.0, 0.0, 0.9]],
        dtype=jnp.float32,
    )

    density = project_laminar_sources(
        sources, positions, n_contacts=4, mode="density_preserving"
    )
    normalized = project_laminar_sources(
        sources, positions, n_contacts=4, mode="row_normalize"
    )

    assert density.diagnostics["operator_type"] == "linear_projection"
    assert density.diagnostics["representation"] == "relative"
    assert density.diagnostics["normalization_mode"] == "density_preserving"
    assert density.diagnostics["field_admissibility"][
        "kernel_row_normalization_applied"
    ] is False
    assert density.diagnostics["field_admissibility"][
        "kernel_row_normalization_valid"
    ] is None
    assert normalized.diagnostics["normalization_mode"] == "row_normalize"
    assert normalized.diagnostics["field_admissibility"][
        "kernel_row_normalization_applied"
    ] is True
    assert jnp.allclose(
        jnp.sum(normalized.kernel, axis=1),
        jnp.ones((4,), dtype=jnp.float32),
        atol=1e-5,
    )
    assert not jnp.allclose(
        jnp.sum(density.kernel, axis=1),
        jnp.ones((4,), dtype=jnp.float32),
        atol=1e-3,
    )


def test_homeostatic_ei_model_carries_source_status_into_field() -> None:
    cfg = (
        jtfne.Configuration()
        .runtime(seed=0)
        .network(name="ei", n=8)
        .set_emitter("homeostatic_ei")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="mean_zero_neumann",
            gauge="mean_zero",
        )
        .probe(modes=["vm"])
    )
    model = jtfne.construct(cfg)
    signals = model.simulate(
        jtfne.simulation(duration_ms=2.0, dt_ms=0.5, seed=0)
    )

    assert signals.field is not None
    assert signals.field.diagnostics["source_calibration_status"] == (
        "uncalibrated_homeostatic_ei_native_current"
    )
    assert signals.metadata["source_mode_class"] == "specialized"
    receipt = model.run_receipt(signals)
    assert receipt.truth["source_calibration_status"] == (
        "uncalibrated_homeostatic_ei_native_current"
    )
    manifest = model.manifest(signals=signals)
    assert manifest["source_model"]["source_mode_class"] == "specialized"
    assert manifest["source_model"]["source_decomposition"] == (
        "homeostatic_ei_activity_trace"
    )
    assert manifest["source_model"]["source_calibration_status"] == (
        "uncalibrated_homeostatic_ei_native_current"
    )
    assert "includes_spike_impulse" not in manifest["source_model"]


def test_csd_tensor_matches_negative_second_difference_with_edge_padding() -> None:
    phi = jnp.asarray([[0.0, 1.0, 4.0, 9.0, 16.0]], dtype=jnp.float32)
    expected = jnp.asarray([[-1.0, -2.0, -2.0, -2.0, 7.0]], dtype=jnp.float32)

    assert jnp.array_equal(csd_tensor(phi, 1.0), expected)


def test_eeg_meg_use_explicit_relative_source_leadfield_map() -> None:
    source = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    leadfield = jnp.asarray([[1.0, 0.0], [0.5, -1.0]], dtype=jnp.float32)
    expected = source @ leadfield.T

    assert jnp.array_equal(eeg_proxy_transform(source, leadfield), expected)
    assert jnp.array_equal(meg_proxy_transform(source, leadfield), expected)

    report = eeg_proxy_probe(expected).report
    assert report["operator_type"] == "linear_projection"
    assert report["input_representation"] == "canonical_relative_source"
    assert report["representation"] == "relative"
    assert report["calibration_transform"] == "explicit_boundary_transform"


def test_basis_spec_marks_unimplemented_source_modes_reserved() -> None:
    active = BasisSpec(source_mode="proxy_no_field_solve").to_dict()
    reserved = BasisSpec(source_mode="total_membrane_current").to_dict()

    assert active["source_mode_status"] == "active"
    assert active["source_mode_executable"] is True
    assert reserved["source_mode_status"] == "reserved"
    assert reserved["source_mode_executable"] is False


def test_basis_spec_invalid_source_mode_remains_reportable() -> None:
    report = BasisSpec(source_mode="unknown_mode").validate()

    assert report["valid"] is False
    assert report["source_mode_status"] == "invalid"
    assert report["source_mode_executable"] is False


def test_sanity_runtime_metrics_preserve_vm_provenance() -> None:
    metrics = _make_probe_metrics(
        {"lfp_proxy": jnp.ones((2, 3), dtype=jnp.float32)}
    )

    assert metrics["operator_type"] == "linear_projection"
    assert metrics["input_representation"] == "relative_vm_state"
    assert metrics["readout_provenance"] == (
        "sanity_runtime_membrane_voltage_proxy"
    )
    assert metrics["representation"] == "relative"


def test_calibration_declaration_preserves_relative_output_until_applied() -> None:
    report = make_calibration_report(
        CalibrationSpec(
            name="mcc_calibration",
            target="source",
            mode="calibrated_empirical",
            scale=2.0,
            units="arb",
            reference="fixture",
        )
    )

    assert report["input_representation"] == "relative"
    assert report["output_representation"] == "relative"
    assert report["declared_target_representation"] == "calibrated"
    assert report["physical_amplitude_calibrated"] is False
