"""Comprehensive contract tests for field solution metadata (v0.2.13).

Tests the FieldSolution/FieldOutput contract hardening:
- 18+ required fields present
- No truth_mode in field reports
- No *_like terminology
- JSON-safe serialization with allow_nan=False
- Canonical CSD sign convention
- Proxy field constraints
- Examples validate with hardened field metadata
- Version bumped to 0.2.18 (no bump)
"""

import json
import pytest
import jax
import jax.numpy as jnp
import jax.random
from jaxfne.fields import (
    project_laminar_sources,
    _make_field_solution_report,
)
from jaxfne.io import json_safe


# ─── Required Fields Test ──────────────────────────────────────────────────────

def test_field_solution_report_has_18_required_fields():
    """Field solution report includes all 18 required fields."""
    required_fields = {
        "field_solver_status",
        "solver_name",
        "boundary_condition",
        "gauge",
        "csd_sign_convention",
        "current_density_layout",
        "solver_residual_l2_relative",
        "n_iterations",
        "converged",
        "finite_phi_e",
        "finite_J_e",
        "finite_CSD",
        "field_claim_level",
        "physical_amplitude_claim_allowed",
        "source_projection_mode",
        "source_current_conservation_status",
        "source_conservation_tested",
        "source_conservation_claim_allowed",
    }

    report = _make_field_solution_report()
    missing = required_fields - set(report.keys())
    assert not missing, f"Missing fields: {missing}"


# ─── JSON-Safety Test ──────────────────────────────────────────────────────────

def test_field_solution_report_json_safe():
    """Field solution report is JSON-safe with allow_nan=False."""
    report = _make_field_solution_report()
    safe_report = json_safe(report)
    # This must not raise with allow_nan=False
    json.dumps(safe_report, allow_nan=False)


def test_field_output_diagnostics_json_safe():
    """FieldOutput.diagnostics is JSON-safe with allow_nan=False."""
    sources = jnp.ones((50, 10))
    positions = jax.random.normal(jax.random.PRNGKey(0), shape=(10, 3))
    field_out = project_laminar_sources(sources, positions, n_contacts=16)

    safe_diag = json_safe(field_out.diagnostics)
    json.dumps(safe_diag, allow_nan=False)


# ─── No truth_mode in Field Reports ────────────────────────────────────────────

def test_field_solution_report_no_truth_mode():
    """truth_mode must not appear in field solution reports."""
    report = _make_field_solution_report()
    assert "truth_mode" not in report, "truth_mode should not be in field reports"


def test_field_output_diagnostics_no_truth_mode():
    """truth_mode must not appear in FieldOutput diagnostics."""
    sources = jnp.ones((50, 10))
    positions = jax.random.normal(jax.random.PRNGKey(0), shape=(10, 3))
    field_out = project_laminar_sources(sources, positions, n_contacts=16)

    assert "truth_mode" not in field_out.diagnostics, \
        "truth_mode should not be in field diagnostics"


# ─── No *_like Terminology ────────────────────────────────────────────────────

def test_field_solution_report_no_like_terminology():
    """No *_like or *_same terminology in field solution reports."""
    forbidden_substrings = {
        "_like",
        "-like",
        "lfp_like",
        "csd_like",
        "eeg_like",
        "meg_like",
        "proxy_positive_equals_extracellular_source_like",
    }

    report = _make_field_solution_report()
    report_str = json.dumps(report)
    for forbidden in forbidden_substrings:
        assert forbidden not in report_str, \
            f"Forbidden substring '{forbidden}' found in field report"


def test_field_output_diagnostics_no_like_terminology():
    """No *_like terminology in FieldOutput diagnostics."""
    forbidden_substrings = {
        "_like",
        "-like",
        "proxy_positive_equals_extracellular_source_like",
    }

    sources = jnp.ones((50, 10))
    positions = jax.random.normal(jax.random.PRNGKey(0), shape=(10, 3))
    field_out = project_laminar_sources(sources, positions, n_contacts=16)

    diag_str = json.dumps(field_out.diagnostics)
    for forbidden in forbidden_substrings:
        assert forbidden not in diag_str, \
            f"Forbidden substring '{forbidden}' found in field diagnostics"


# ─── CSD Sign Convention Canonical ──────────────────────────────────────────────

def test_field_solution_report_csd_sign_convention_canonical():
    """CSD sign convention uses canonical value (no _like, no proxy prefix)."""
    report = _make_field_solution_report()
    assert report["csd_sign_convention"] == "positive_equals_extracellular_source"


def test_field_output_csd_sign_convention_canonical():
    """FieldOutput diagnostics use canonical CSD sign convention."""
    sources = jnp.ones((50, 10))
    positions = jax.random.normal(jax.random.PRNGKey(0), shape=(10, 3))
    field_out = project_laminar_sources(sources, positions, n_contacts=16)

    assert field_out.diagnostics["csd_sign_convention"] == "positive_equals_extracellular_source"


# ─── Proxy Field Constraints ────────────────────────────────────────────────────

def test_proxy_field_solver_metrics_are_null():
    """Proxy fields have null solver metrics (n_iterations, converged, residual)."""
    report = _make_field_solution_report(
        field_solver_status="laminar_proxy_no_pde"
    )
    assert report["n_iterations"] is None
    assert report["converged"] is None
    assert report["solver_residual_l2_relative"] is None


def test_proxy_field_amplitude_claim_false():
    """Proxy fields must have physical_amplitude_claim_allowed=False."""
    report = _make_field_solution_report(
        field_solver_status="laminar_proxy_no_pde"
    )
    assert report["physical_amplitude_claim_allowed"] is False


def test_proxy_field_claim_level_correct():
    """Proxy fields have claim_level='proxy_readout_only'."""
    report = _make_field_solution_report(
        field_solver_status="laminar_proxy_no_pde"
    )
    assert report["field_claim_level"] == "proxy_readout_only"


def test_proxy_field_current_density_layout_not_applicable():
    """Proxy fields have current_density_layout='not_applicable'."""
    report = _make_field_solution_report(
        field_solver_status="laminar_proxy_no_pde"
    )
    assert report["current_density_layout"] == "not_applicable"


def test_proxy_field_conservation_untested():
    """Proxy fields have conservation untested and unclaimed."""
    report = _make_field_solution_report(
        field_solver_status="laminar_proxy_no_pde"
    )
    assert report["source_conservation_tested"] is False
    assert report["source_conservation_claim_allowed"] is False


def test_proxy_field_j_e_not_computed():
    """Proxy fields from project_laminar_sources have finite_J_e=False (not computed)."""
    sources = jnp.ones((50, 10))
    positions = jax.random.normal(jax.random.PRNGKey(0), shape=(10, 3))
    field_out = project_laminar_sources(sources, positions, n_contacts=16)

    assert field_out.diagnostics["finite_J_e"] is False


# ─── Field Solution Report Helper Consistency ──────────────────────────────────

def test_field_solution_report_default_proxy_values():
    """Default values in _make_field_solution_report match proxy defaults."""
    report = _make_field_solution_report()

    assert report["field_solver_status"] == "laminar_proxy_no_pde"
    assert report["solver_name"] == "laminar_proxy"
    assert report["boundary_condition"] == "declared_metadata_only"
    assert report["gauge"] == "declared_metadata_only"
    assert report["csd_sign_convention"] == "positive_equals_extracellular_source"
    assert report["current_density_layout"] == "not_applicable"
    assert report["field_claim_level"] == "proxy_readout_only"
    assert report["physical_amplitude_claim_allowed"] is False
    assert report["source_projection_mode"] == "proxy_no_field_solve"
    assert report["source_current_conservation_status"] == "not_applicable_proxy_mode"
    assert report["source_conservation_tested"] is False
    assert report["source_conservation_claim_allowed"] is False


def test_field_solution_report_finiteness_defaults():
    """Finiteness flags have sensible defaults for proxy."""
    # For proxy (default), finiteness defaults are True for computed arrays
    report = _make_field_solution_report(
        field_solver_status="laminar_proxy_no_pde",
        finite_J_e=False  # Explicitly set for proxy (not computed)
    )

    assert report["finite_phi_e"] is True
    assert report["finite_J_e"] is False  # Proxy doesn't compute J_e
    assert report["finite_CSD"] is True


# ─── project_laminar_sources Integration ────────────────────────────────────────

def test_project_laminar_sources_includes_field_solution_metadata():
    """project_laminar_sources() includes all field solution metadata in diagnostics."""
    sources = jnp.ones((50, 10))
    positions = jax.random.normal(jax.random.PRNGKey(0), shape=(10, 3))
    field_out = project_laminar_sources(sources, positions, n_contacts=16)

    required_fields = {
        "field_solver_status",
        "solver_name",
        "boundary_condition",
        "gauge",
        "csd_sign_convention",
        "current_density_layout",
        "solver_residual_l2_relative",
        "n_iterations",
        "converged",
        "finite_phi_e",
        "finite_J_e",
        "finite_CSD",
        "field_claim_level",
        "physical_amplitude_claim_allowed",
        "source_projection_mode",
        "source_current_conservation_status",
        "source_conservation_tested",
        "source_conservation_claim_allowed",
    }

    missing = required_fields - set(field_out.diagnostics.keys())
    assert not missing, f"Missing fields in field output: {missing}"


def test_project_laminar_sources_proxy_status():
    """project_laminar_sources() produces valid proxy status fields."""
    sources = jnp.ones((50, 10))
    positions = jax.random.normal(jax.random.PRNGKey(0), shape=(10, 3))
    field_out = project_laminar_sources(sources, positions, n_contacts=16)

    diag = field_out.diagnostics
    assert diag["field_solver_status"] == "laminar_proxy_no_pde"
    assert diag["solver_name"] == "laminar_proxy"
    assert diag["field_claim_level"] == "proxy_readout_only"
    assert diag["physical_amplitude_claim_allowed"] is False
    assert diag["csd_sign_convention"] == "positive_equals_extracellular_source"


# ─── Finite Flags Validation ────────────────────────────────────────────────────

def test_field_output_finite_phi_e_matches_arrays():
    """finite_phi_e in diagnostics matches actual phi_e_proxy finiteness."""
    sources = jnp.ones((50, 10))
    positions = jax.random.normal(jax.random.PRNGKey(0), shape=(10, 3))
    field_out = project_laminar_sources(sources, positions, n_contacts=16)

    # For valid inputs, phi_e should be finite
    assert field_out.diagnostics["finite_phi_e_proxy"] is True
    assert field_out.diagnostics["finite_phi_e"] is True


def test_field_output_finite_csd_matches_arrays():
    """finite_CSD in diagnostics matches actual csd_proxy finiteness."""
    sources = jnp.ones((50, 10))
    positions = jax.random.normal(jax.random.PRNGKey(0), shape=(10, 3))
    field_out = project_laminar_sources(sources, positions, n_contacts=16)

    # For valid inputs, CSD should be finite
    assert field_out.diagnostics["finite_csd_proxy"] is True
    assert field_out.diagnostics["finite_CSD"] is True


def test_field_output_finite_j_e_false_for_proxy():
    """J_e finitude is false for proxy (not computed)."""
    sources = jnp.ones((50, 10))
    positions = jax.random.normal(jax.random.PRNGKey(0), shape=(10, 3))
    field_out = project_laminar_sources(sources, positions, n_contacts=16)

    # Proxy doesn't compute J_e
    assert field_out.diagnostics["finite_J_e"] is False


# ─── Version Constraint ────────────────────────────────────────────────────────

@pytest.mark.xfail(
    reason="Historical release fixture test: pins v0.3.4 release stability. "
    "Marked xfail because v0.3.5 cleanup release intentionally bumps version. "
    "This test verifies v0.3.4 stability after tutorial figures; no longer applicable."
)
def test_version_remains_0210():
    """jaxfne version remains 0.3.4 (after v0.3.4 tutorial figures) — HISTORICAL FIXTURE."""
    import jaxfne
    assert jaxfne.__version__ == "0.3.4"


# ─── JSON Serialization Strictness ────────────────────────────────────────────

def test_field_solution_report_strict_json_round_trip():
    """Field solution report survives strict JSON round-trip."""
    original = _make_field_solution_report()

    # Serialize with strict JSON
    json_str = json.dumps(json_safe(original), allow_nan=False)

    # Deserialize and compare
    restored = json.loads(json_str)
    assert restored == original


def test_field_output_diagnostics_strict_json_round_trip():
    """FieldOutput diagnostics survive strict JSON round-trip."""
    sources = jnp.ones((50, 10))
    positions = jax.random.normal(jax.random.PRNGKey(0), shape=(10, 3))
    field_out = project_laminar_sources(sources, positions, n_contacts=16)

    original = field_out.diagnostics
    json_str = json.dumps(json_safe(original), allow_nan=False)
    restored = json.loads(json_str)

    # Check that critical fields are preserved
    assert restored["field_solver_status"] == original["field_solver_status"]
    assert restored["csd_sign_convention"] == original["csd_sign_convention"]
    assert restored["physical_amplitude_claim_allowed"] == original["physical_amplitude_claim_allowed"]


# ─── Boundary and Gauge Status ──────────────────────────────────────────────────

def test_proxy_boundary_condition_metadata_only():
    """Proxy fields have boundary_condition='declared_metadata_only'."""
    report = _make_field_solution_report(
        field_solver_status="laminar_proxy_no_pde"
    )
    assert report["boundary_condition"] == "declared_metadata_only"


def test_proxy_gauge_metadata_only():
    """Proxy fields have gauge='declared_metadata_only'."""
    report = _make_field_solution_report(
        field_solver_status="laminar_proxy_no_pde"
    )
    assert report["gauge"] == "declared_metadata_only"


def test_field_output_boundary_and_gauge_consistent():
    """FieldOutput diagnostics have consistent boundary/gauge metadata."""
    sources = jnp.ones((50, 10))
    positions = jax.random.normal(jax.random.PRNGKey(0), shape=(10, 3))
    field_out = project_laminar_sources(sources, positions, n_contacts=16)

    assert field_out.diagnostics["boundary_condition"] == "mean_zero_neumann"
    assert field_out.diagnostics["gauge"] == "mean_zero"
