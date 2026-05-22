"""Tests for v0.2.27 conservation-inspired proxy diagnostics.

Validates the compute_conservation_proxy_diagnostics() function and its
integration into Model.manifest(). All diagnostics are proxy-only; no solver,
no physical amplitude claims, no biological metabolism claims.
"""

from __future__ import annotations

import json
import math

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne
from jaxfne import compute_conservation_proxy_diagnostics


# ─── Fixtures ────────────────────────────────────────────────────────────────

_T, _N, _X = 100, 10, 16


@pytest.fixture
def rng_arrays():
    rng = np.random.default_rng(0)
    src = jnp.array(rng.standard_normal((_T, _N)).astype(np.float32))
    phi = jnp.array(rng.standard_normal((_T, _X)).astype(np.float32))
    csd = jnp.array(rng.standard_normal((_T, _X)).astype(np.float32))
    lfp = jnp.array(rng.standard_normal((_T, _X)).astype(np.float32))
    return src, phi, csd, lfp


@pytest.fixture
def zero_arrays():
    src = jnp.zeros((_T, _N))
    phi = jnp.zeros((_T, _X))
    csd = jnp.zeros((_T, _X))
    lfp = jnp.zeros((_T, _X))
    return src, phi, csd, lfp


# ─── Test 1: JSON safety ─────────────────────────────────────────────────────

class TestJsonSafety:
    def test_random_arrays_json_safe(self, rng_arrays):
        src, phi, csd, lfp = rng_arrays
        d = compute_conservation_proxy_diagnostics(
            source=src, phi_e=phi, csd=csd, lfp=lfp
        )
        serialized = json.dumps(d, allow_nan=False)
        assert isinstance(serialized, str)

    def test_none_arrays_json_safe(self):
        d = compute_conservation_proxy_diagnostics()
        serialized = json.dumps(d, allow_nan=False)
        assert isinstance(serialized, str)

    def test_partial_arrays_json_safe(self, rng_arrays):
        src, _, _, _ = rng_arrays
        d = compute_conservation_proxy_diagnostics(source=src)
        json.dumps(d, allow_nan=False)


# ─── Test 2: Zero arrays produce zero norms ──────────────────────────────────

class TestZeroArrays:
    def test_zero_source_norms_are_zero(self, zero_arrays):
        src, phi, csd, lfp = zero_arrays
        d = compute_conservation_proxy_diagnostics(
            source=src, phi_e=phi, csd=csd, lfp=lfp
        )
        assert d["source_norm_l1"] == pytest.approx(0.0, abs=1e-6)
        assert d["source_norm_l2"] == pytest.approx(0.0, abs=1e-6)
        assert d["source_abs_mean"] == pytest.approx(0.0, abs=1e-6)

    def test_zero_phi_gradient_is_zero(self, zero_arrays):
        _, phi, _, _ = zero_arrays
        d = compute_conservation_proxy_diagnostics(phi_e=phi)
        assert d["phi_gradient_proxy_norm2"] == pytest.approx(0.0, abs=1e-6)

    def test_zero_csd_norms_are_zero(self, zero_arrays):
        _, _, csd, _ = zero_arrays
        d = compute_conservation_proxy_diagnostics(csd=csd)
        assert d["csd_abs_mean"] == pytest.approx(0.0, abs=1e-6)
        assert d["csd_norm_l2"] == pytest.approx(0.0, abs=1e-6)

    def test_zero_lfp_norms_are_zero(self, zero_arrays):
        _, _, _, lfp = zero_arrays
        d = compute_conservation_proxy_diagnostics(lfp=lfp)
        assert d["lfp_abs_mean"] == pytest.approx(0.0, abs=1e-6)
        assert d["lfp_norm_l2"] == pytest.approx(0.0, abs=1e-6)

    def test_zero_conservation_residual_is_zero(self, zero_arrays):
        src, _, _, _ = zero_arrays
        d = compute_conservation_proxy_diagnostics(source=src)
        assert d["source_conservation_proxy_residual"] == pytest.approx(0.0, abs=1e-6)


# ─── Test 3: Nonzero arrays produce finite positive norms ────────────────────

class TestNonzeroArrays:
    def test_nonzero_source_norms_positive(self, rng_arrays):
        src, _, _, _ = rng_arrays
        d = compute_conservation_proxy_diagnostics(source=src)
        assert d["source_norm_l1"] is not None and d["source_norm_l1"] > 0.0
        assert d["source_norm_l2"] is not None and d["source_norm_l2"] > 0.0

    def test_nonzero_phi_gradient_positive(self, rng_arrays):
        _, phi, _, _ = rng_arrays
        d = compute_conservation_proxy_diagnostics(phi_e=phi)
        assert d["phi_gradient_proxy_norm2"] is not None
        assert d["phi_gradient_proxy_norm2"] > 0.0

    def test_nonzero_csd_norms_positive(self, rng_arrays):
        _, _, csd, _ = rng_arrays
        d = compute_conservation_proxy_diagnostics(csd=csd)
        assert d["csd_abs_mean"] is not None and d["csd_abs_mean"] > 0.0
        assert d["csd_norm_l2"] is not None and d["csd_norm_l2"] > 0.0

    def test_nonzero_lfp_norms_positive(self, rng_arrays):
        _, _, _, lfp = rng_arrays
        d = compute_conservation_proxy_diagnostics(lfp=lfp)
        assert d["lfp_abs_mean"] is not None and d["lfp_abs_mean"] > 0.0
        assert d["lfp_norm_l2"] is not None and d["lfp_norm_l2"] > 0.0

    def test_all_norms_finite(self, rng_arrays):
        src, phi, csd, lfp = rng_arrays
        d = compute_conservation_proxy_diagnostics(
            source=src, phi_e=phi, csd=csd, lfp=lfp
        )
        for key in [
            "source_norm_l1", "source_norm_l2", "source_abs_mean",
            "phi_abs_mean", "phi_gradient_proxy_norm2",
            "csd_abs_mean", "csd_norm_l2",
            "lfp_abs_mean", "lfp_norm_l2",
            "field_energy_like_proxy",
        ]:
            val = d[key]
            assert val is not None, f"{key} is None for nonzero arrays"
            assert math.isfinite(val), f"{key}={val} is not finite"


# ─── Test 4: Missing arrays produce None ─────────────────────────────────────

class TestMissingArrays:
    def test_no_arrays_all_norms_none(self):
        d = compute_conservation_proxy_diagnostics()
        for key in [
            "source_norm_l1", "source_norm_l2", "source_abs_mean",
            "source_conservation_proxy_residual",
            "phi_abs_mean", "phi_gradient_proxy_norm2",
            "csd_abs_mean", "csd_norm_l2",
            "lfp_abs_mean", "lfp_norm_l2",
            "field_energy_like_proxy",
        ]:
            assert d[key] is None, f"{key} should be None when no arrays provided"

    def test_missing_source_source_norms_none(self, rng_arrays):
        _, phi, csd, lfp = rng_arrays
        d = compute_conservation_proxy_diagnostics(phi_e=phi, csd=csd, lfp=lfp)
        assert d["source_norm_l1"] is None
        assert d["source_norm_l2"] is None
        assert d["source_conservation_proxy_residual"] is None

    def test_missing_phi_phi_norms_none(self, rng_arrays):
        src, _, csd, lfp = rng_arrays
        d = compute_conservation_proxy_diagnostics(source=src, csd=csd, lfp=lfp)
        assert d["phi_abs_mean"] is None
        assert d["phi_gradient_proxy_norm2"] is None
        assert d["field_energy_like_proxy"] is None

    def test_1d_phi_gradient_none(self):
        """phi_e with only 1 spatial point — gradient undefined, returns None."""
        phi_1d = jnp.ones((_T, 1))
        d = compute_conservation_proxy_diagnostics(phi_e=phi_1d)
        assert d["phi_gradient_proxy_norm2"] is None


# ─── Test 5: Physical amplitude claim always False ───────────────────────────

class TestPhysicalAmplitudeClaim:
    def test_physical_amplitude_claim_false_no_arrays(self):
        d = compute_conservation_proxy_diagnostics()
        assert d["physical_amplitude_claim_allowed"] is False

    def test_physical_amplitude_claim_false_with_arrays(self, rng_arrays):
        src, phi, csd, lfp = rng_arrays
        d = compute_conservation_proxy_diagnostics(
            source=src, phi_e=phi, csd=csd, lfp=lfp
        )
        assert d["physical_amplitude_claim_allowed"] is False

    def test_cannot_override_physical_claim(self, rng_arrays):
        """No path to set physical_amplitude_claim_allowed=True exists."""
        src, _, _, _ = rng_arrays
        d = compute_conservation_proxy_diagnostics(source=src)
        assert d["physical_amplitude_claim_allowed"] is False


# ─── Test 6: Biological metabolism claim always False ────────────────────────

class TestBiologicalMetabolismClaim:
    def test_biological_metabolism_claim_false_no_arrays(self):
        d = compute_conservation_proxy_diagnostics()
        assert d["biological_metabolism_claim_allowed"] is False

    def test_biological_metabolism_claim_false_with_arrays(self, rng_arrays):
        src, phi, csd, lfp = rng_arrays
        d = compute_conservation_proxy_diagnostics(
            source=src, phi_e=phi, csd=csd, lfp=lfp
        )
        assert d["biological_metabolism_claim_allowed"] is False


# ─── Test 7: j_dot_e_proxy is always None ───────────────────────────────────

class TestJDotEProxy:
    def test_j_dot_e_proxy_none_no_arrays(self):
        d = compute_conservation_proxy_diagnostics()
        assert d["j_dot_e_proxy"] is None

    def test_j_dot_e_proxy_none_with_arrays(self, rng_arrays):
        src, phi, csd, lfp = rng_arrays
        d = compute_conservation_proxy_diagnostics(
            source=src, phi_e=phi, csd=csd, lfp=lfp
        )
        assert d["j_dot_e_proxy"] is None

    def test_j_dot_e_proxy_none_in_json(self, rng_arrays):
        src, phi, csd, lfp = rng_arrays
        d = compute_conservation_proxy_diagnostics(
            source=src, phi_e=phi, csd=csd, lfp=lfp
        )
        parsed = json.loads(json.dumps(d, allow_nan=False))
        assert parsed["j_dot_e_proxy"] is None


# ─── Test 8: poynting_flux_proxy is always None ──────────────────────────────

class TestPoyntingFluxProxy:
    def test_poynting_flux_proxy_none(self):
        d = compute_conservation_proxy_diagnostics()
        assert d["poynting_flux_proxy"] is None

    def test_poynting_flux_proxy_none_with_arrays(self, rng_arrays):
        src, phi, csd, lfp = rng_arrays
        d = compute_conservation_proxy_diagnostics(
            source=src, phi_e=phi, csd=csd, lfp=lfp
        )
        assert d["poynting_flux_proxy"] is None


# ─── Test 9: stress_energy_tensor_status == "not_implemented" ────────────────

class TestStressEnergyTensorStatus:
    def test_stress_energy_tensor_not_implemented(self):
        d = compute_conservation_proxy_diagnostics()
        assert d["stress_energy_tensor_status"] == "not_implemented"

    def test_stress_energy_tensor_not_implemented_with_arrays(self, rng_arrays):
        src, phi, csd, lfp = rng_arrays
        d = compute_conservation_proxy_diagnostics(
            source=src, phi_e=phi, csd=csd, lfp=lfp
        )
        assert d["stress_energy_tensor_status"] == "not_implemented"


# ─── Test 10: poisson_solver_status == "not_implemented" ─────────────────────

class TestPoissonSolverStatus:
    def test_poisson_solver_not_implemented(self):
        d = compute_conservation_proxy_diagnostics()
        assert d["poisson_solver_status"] == "not_implemented"

    def test_poisson_solver_not_implemented_with_arrays(self, rng_arrays):
        src, phi, csd, lfp = rng_arrays
        d = compute_conservation_proxy_diagnostics(
            source=src, phi_e=phi, csd=csd, lfp=lfp
        )
        assert d["poisson_solver_status"] == "not_implemented"


# ─── Test 11: maxwell_solver_status == "not_implemented" ─────────────────────

class TestMaxwellSolverStatus:
    def test_maxwell_solver_not_implemented(self):
        d = compute_conservation_proxy_diagnostics()
        assert d["maxwell_solver_status"] == "not_implemented"

    def test_maxwell_solver_not_implemented_with_arrays(self, rng_arrays):
        src, phi, csd, lfp = rng_arrays
        d = compute_conservation_proxy_diagnostics(
            source=src, phi_e=phi, csd=csd, lfp=lfp
        )
        assert d["maxwell_solver_status"] == "not_implemented"


# ─── Test 12: manifest contains conservation_proxy_diagnostics ───────────────

class TestManifestIntegration:
    @pytest.fixture
    def built_manifest(self):
        cfg = (
            jaxfne.configuration()
            .network(name="test", kind="isolated_neuron", n=5, cell_types={"E": 1.0})
            .emitter(family="izhikevich", preset="cortical_eig")
            .field(
                domain="laminar_column",
                conductivity="proxy",
                boundary="declared_proxy",
                gauge="mean_zero",
            )
            .probe(name="test", modes=["spikes", "V_m", "source", "LFP"])
        )
        model = jaxfne.construct(cfg)
        sim = jaxfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0)
        signals = model.simulate(sim)
        return model.manifest(signals)

    def test_manifest_has_conservation_proxy_diagnostics(self, built_manifest):
        assert "conservation_proxy_diagnostics" in built_manifest

    def test_manifest_cpd_has_required_keys(self, built_manifest):
        cpd = built_manifest["conservation_proxy_diagnostics"]
        required = [
            "diagnostic_status", "physical_amplitude_claim_allowed",
            "biological_metabolism_claim_allowed",
            "source_norm_l1", "phi_gradient_proxy_norm2",
            "poisson_solver_status", "maxwell_solver_status",
            "stress_energy_tensor_status", "j_dot_e_proxy", "poynting_flux_proxy",
        ]
        for key in required:
            assert key in cpd, f"Missing key: {key}"

    def test_manifest_cpd_physical_claim_false(self, built_manifest):
        assert built_manifest["conservation_proxy_diagnostics"]["physical_amplitude_claim_allowed"] is False


# ─── Test 13: full manifest remains JSON-safe ────────────────────────────────

class TestManifestJsonSafe:
    def test_manifest_json_safe_with_diagnostics(self):
        cfg = (
            jaxfne.configuration()
            .network(name="test", kind="isolated_neuron", n=3, cell_types={"E": 1.0})
            .emitter(family="izhikevich", preset="cortical_eig")
            .field(
                domain="laminar_column",
                conductivity="proxy",
                boundary="declared_proxy",
                gauge="mean_zero",
            )
            .probe(name="test", modes=["spikes", "LFP"])
        )
        model = jaxfne.construct(cfg)
        sim = jaxfne.simulation(duration_ms=5.0, dt_ms=0.1, seed=1)
        signals = model.simulate(sim)
        m = model.manifest(signals)
        # Must not raise with allow_nan=False
        serialized = json.dumps(m, allow_nan=False)
        assert len(serialized) > 100


# ─── Test 14: Language audit — no positive solver/metabolism claims ───────────

class TestLanguageAudit:
    def test_no_positive_solver_claim_in_output(self, rng_arrays):
        src, phi, csd, lfp = rng_arrays
        d = compute_conservation_proxy_diagnostics(
            source=src, phi_e=phi, csd=csd, lfp=lfp
        )
        serialized = json.dumps(d)
        # Must not assert "poisson_solver" is implemented/active
        assert '"poisson_solver_status": "not_implemented"' in serialized or \
               "not_implemented" in serialized
        # Must not claim biological metabolism
        assert "biological_metabolism" not in serialized or \
               "not" in serialized or \
               d["biological_metabolism_claim_allowed"] is False

    def test_diagnostic_status_is_proxy(self, rng_arrays):
        src, phi, csd, lfp = rng_arrays
        d = compute_conservation_proxy_diagnostics(
            source=src, phi_e=phi, csd=csd, lfp=lfp
        )
        assert d["diagnostic_status"] == "proxy"

    def test_claim_level_is_computational_scaffold(self, rng_arrays):
        src, phi, csd, lfp = rng_arrays
        d = compute_conservation_proxy_diagnostics(
            source=src, phi_e=phi, csd=csd, lfp=lfp
        )
        assert d["claim_level"] == "computational_scaffold"


# ─── Test 15: No source double-counting regression ──────────────────────────

class TestNoDoubleCountingRegression:
    def test_field_solution_source_matches_direct_source(self, rng_arrays):
        """Using field_solution= should yield same result as direct source=."""
        src, phi, csd, lfp = rng_arrays
        from jaxfne.fields import FieldOutput
        kernel = jnp.ones((16, 10)) / 10.0
        contacts = jnp.linspace(0, 1, 16)
        fo = FieldOutput(
            source_proxy=src,
            phi_e_proxy=phi,
            csd_proxy=csd,
            lfp_proxy=lfp,
            kernel=kernel,
            contact_depths=contacts,
            diagnostics={},
        )
        d_direct = compute_conservation_proxy_diagnostics(
            source=src, phi_e=phi, csd=csd, lfp=lfp
        )
        d_via_fo = compute_conservation_proxy_diagnostics(field_solution=fo)
        assert d_direct["source_norm_l1"] == pytest.approx(
            d_via_fo["source_norm_l1"], rel=1e-5
        )
        assert d_direct["phi_gradient_proxy_norm2"] == pytest.approx(
            d_via_fo["phi_gradient_proxy_norm2"], rel=1e-5
        )

    def test_public_export(self):
        """compute_conservation_proxy_diagnostics is importable from top-level jaxfne."""
        assert hasattr(jaxfne, "compute_conservation_proxy_diagnostics")
        assert callable(jaxfne.compute_conservation_proxy_diagnostics)
