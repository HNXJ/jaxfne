"""Tests for v0.2.26 computation-basis contracts.

Covers AxisSpec, BasisSpec, validate_basis_spec, basis_claim_gate,
and manifest basis block integration.
"""
from __future__ import annotations

import json
import pytest


# ─── imports ──────────────────────────────────────────────────────────────────

import jaxfne
from jaxfne.core import (
    AxisSpec,
    BasisSpec,
    default_basis_spec,
    _AXIS_STATUS_VALUES,
    _SPACE_BASIS_VALUES,
    _TIME_BASIS_VALUES,
    _FIELD_REGIME_VALUES,
    _SOURCE_MODE_BASIS_VALUES,
    _PROBE_BASIS_VALUES,
    _FUTURE_FIELD_REGIMES,
)
from jaxfne.validation import validate_basis_spec, basis_claim_gate


# ─── Test 1: Default BasisSpec serializes to JSON-safe dict ───────────────────

class TestDefaultBasisSpecJsonSafe:
    def test_default_basis_spec_to_dict_is_json_safe(self):
        b = default_basis_spec()
        d = b.to_dict()
        dumped = json.dumps(d, allow_nan=False)
        assert isinstance(dumped, str)
        parsed = json.loads(dumped)
        assert "space_basis" in parsed
        assert "time_basis" in parsed
        assert "field_regime" in parsed
        assert "source_mode" in parsed
        assert "probe_basis" in parsed
        assert "dimension_status" in parsed
        assert "implemented" in parsed
        assert "future_regime" in parsed
        assert "claim_allowed" in parsed

    def test_default_basis_spec_json_no_nan_inf(self):
        """Ensure no NaN/Inf values sneak into the basis dict."""
        b = default_basis_spec()
        d = b.to_dict()
        raw = json.dumps(d)
        assert "NaN" not in raw
        assert "Infinity" not in raw


# ─── Test 2: Default basis matches laminar proxy behavior ─────────────────────

class TestDefaultBasisMatchesLaminarProxy:
    def test_default_space_basis_is_laminar_depth(self):
        b = default_basis_spec()
        assert b.space_basis == "laminar_depth"

    def test_default_time_basis_is_continuous_ms(self):
        b = default_basis_spec()
        assert b.time_basis == "continuous_ms"

    def test_default_field_regime_is_laminar_proxy(self):
        b = default_basis_spec()
        assert b.field_regime == "laminar_proxy"

    def test_default_source_mode_is_proxy_no_field_solve(self):
        b = default_basis_spec()
        assert b.source_mode == "proxy_no_field_solve"

    def test_default_probe_basis_is_multimodal_proxy(self):
        b = default_basis_spec()
        assert b.probe_basis == "multimodal_proxy"

    def test_default_x_y_collapsed_z_active(self):
        b = default_basis_spec()
        dim = {a.name: a.status for a in b.axes}
        assert dim.get("x") == "collapsed"
        assert dim.get("y") == "collapsed"
        assert dim.get("z") == "active"

    def test_default_implemented_true(self):
        b = default_basis_spec()
        assert b.implemented is True

    def test_default_future_regime_false(self):
        b = default_basis_spec()
        assert b.field_regime not in _FUTURE_FIELD_REGIMES

    def test_default_claim_allowed_false(self):
        b = default_basis_spec()
        assert b.claim_allowed is False


# ─── Test 3: Invalid axis status is rejected ──────────────────────────────────

class TestAxisSpecValidation:
    def test_invalid_axis_status_rejected(self):
        ax = AxisSpec(name="x", status="flying")
        result = ax.validate()
        assert result["valid"] is False
        assert any("invalid_status" in i for i in result["issues"])

    def test_valid_axis_statuses_accepted(self):
        for status in ("active", "collapsed", "indexed"):
            ax = AxisSpec(name="z", status=status)
            result = ax.validate()
            assert result["valid"] is True, f"status={status} should be valid"

    def test_axis_empty_name_rejected(self):
        ax = AxisSpec(name="", status="active")
        result = ax.validate()
        assert result["valid"] is False
        assert "name_empty" in result["issues"]

    def test_axis_negative_size_rejected(self):
        ax = AxisSpec(name="z", status="active", size=-1)
        result = ax.validate()
        assert result["valid"] is False

    def test_axis_to_dict_json_safe(self):
        ax = AxisSpec(name="z", status="active", size=16, units_or_status="proxy")
        d = ax.to_dict()
        json.dumps(d, allow_nan=False)  # should not raise


# ─── Test 4: Invalid space basis is rejected ──────────────────────────────────

class TestInvalidSpaceBasisRejected:
    def test_unknown_space_basis_rejected(self):
        b = BasisSpec(space_basis="hyperbolic_manifold")
        v = validate_basis_spec(b)
        assert v["valid"] is False
        assert any("invalid_space_basis" in i for i in v["issues"])

    def test_all_valid_space_bases_accepted(self):
        for sb in _SPACE_BASIS_VALUES:
            # For laminar_depth we need z active; for xyz we need all active
            if sb == "laminar_depth":
                continue  # already tested via default
            if sb in ("xy", "xyz", "graph"):
                continue  # tested separately below
            b = BasisSpec(space_basis=sb)
            v = validate_basis_spec(b)
            # collapsed with active axes will fail; skip that case
            if sb == "collapsed":
                b2 = BasisSpec(
                    space_basis="collapsed",
                    axes=(
                        AxisSpec(name="x", status="collapsed"),
                        AxisSpec(name="y", status="collapsed"),
                        AxisSpec(name="z", status="collapsed"),
                    ),
                )
                v2 = validate_basis_spec(b2)
                assert v2["valid"] is True, f"collapsed with all-collapsed axes should be valid"
            else:
                # graph doesn't need axis checks
                assert not any("invalid_space_basis" in i for i in v.get("issues", []))


# ─── Test 5: xy basis rejects active z ────────────────────────────────────────

class TestXyBasisActiveZRejected:
    def test_xy_with_active_z_rejected(self):
        b = BasisSpec(
            space_basis="xy",
            axes=(
                AxisSpec(name="x", status="active"),
                AxisSpec(name="y", status="active"),
                AxisSpec(name="z", status="active"),  # z should not be active for xy
            ),
        )
        v = validate_basis_spec(b)
        assert v["valid"] is False
        assert any("xy_basis" in i for i in v["issues"])

    def test_xy_with_collapsed_z_accepted(self):
        b = BasisSpec(
            space_basis="xy",
            axes=(
                AxisSpec(name="x", status="active"),
                AxisSpec(name="y", status="active"),
                AxisSpec(name="z", status="collapsed"),
            ),
        )
        v = validate_basis_spec(b)
        assert v["valid"] is True


# ─── Test 6: xyz basis requires x/y/z active/indexed ─────────────────────────

class TestXyzBasisRequiresAllAxes:
    def test_xyz_with_all_active_accepted(self):
        b = BasisSpec(
            space_basis="xyz",
            axes=(
                AxisSpec(name="x", status="active"),
                AxisSpec(name="y", status="active"),
                AxisSpec(name="z", status="active"),
            ),
        )
        v = validate_basis_spec(b)
        assert v["valid"] is True

    def test_xyz_missing_y_rejected(self):
        b = BasisSpec(
            space_basis="xyz",
            axes=(
                AxisSpec(name="x", status="active"),
                AxisSpec(name="z", status="active"),
            ),
        )
        v = validate_basis_spec(b)
        assert v["valid"] is False
        assert any("xyz_basis" in i for i in v["issues"])


# ─── Test 7: laminar_depth requires active/indexed z ──────────────────────────

class TestLaminarDepthRequiresZ:
    def test_laminar_depth_with_collapsed_z_rejected(self):
        b = BasisSpec(
            space_basis="laminar_depth",
            axes=(
                AxisSpec(name="x", status="collapsed"),
                AxisSpec(name="y", status="collapsed"),
                AxisSpec(name="z", status="collapsed"),  # z must be active or indexed
            ),
        )
        v = validate_basis_spec(b)
        assert v["valid"] is False
        assert any("laminar_depth_basis" in i for i in v["issues"])

    def test_laminar_depth_with_indexed_z_accepted(self):
        b = BasisSpec(
            space_basis="laminar_depth",
            axes=(
                AxisSpec(name="x", status="collapsed"),
                AxisSpec(name="y", status="collapsed"),
                AxisSpec(name="z", status="indexed"),
            ),
        )
        v = validate_basis_spec(b)
        assert v["valid"] is True


# ─── Test 8: collapsed basis rejects active axes ──────────────────────────────

class TestCollapsedBasisRejectsActiveAxes:
    def test_collapsed_with_active_z_rejected(self):
        b = BasisSpec(
            space_basis="collapsed",
            axes=(
                AxisSpec(name="x", status="collapsed"),
                AxisSpec(name="y", status="collapsed"),
                AxisSpec(name="z", status="active"),
            ),
        )
        v = validate_basis_spec(b)
        assert v["valid"] is False
        assert any("collapsed_basis" in i for i in v["issues"])

    def test_collapsed_with_all_collapsed_accepted(self):
        b = BasisSpec(
            space_basis="collapsed",
            axes=(
                AxisSpec(name="x", status="collapsed"),
                AxisSpec(name="y", status="collapsed"),
                AxisSpec(name="z", status="collapsed"),
            ),
        )
        v = validate_basis_spec(b)
        assert v["valid"] is True


# ─── Test 9: future_maxwell declared future, not implemented ──────────────────

class TestFutureMaxwellDeclarationOnly:
    def test_future_maxwell_implemented_false(self):
        b = BasisSpec(field_regime="future_maxwell")
        assert b.implemented is False

    def test_future_maxwell_claim_allowed_false(self):
        b = BasisSpec(field_regime="future_maxwell")
        assert b.claim_allowed is False

    def test_future_maxwell_future_regime_true(self):
        b = BasisSpec(field_regime="future_maxwell")
        d = b.to_dict()
        assert d["future_regime"] is True
        assert d["implemented"] is False
        assert d["claim_allowed"] is False

    def test_future_maxwell_validation_warn_not_implemented(self):
        b = BasisSpec(field_regime="future_maxwell")
        v = validate_basis_spec(b)
        assert v["valid"] is True  # future regimes are valid declarations
        assert v["physical_amplitude_claim_allowed"] is False
        assert any("future_regime_future_maxwell" in w for w in v.get("warnings", []))

    def test_future_maxwell_json_safe(self):
        b = BasisSpec(field_regime="future_maxwell")
        d = b.to_dict()
        json.dumps(d, allow_nan=False)  # must not raise


# ─── Test 10: future_admittive same doctrine ──────────────────────────────────

class TestFutureAdmittiveDeclarationOnly:
    def test_future_admittive_not_implemented(self):
        b = BasisSpec(field_regime="future_admittive")
        assert b.implemented is False
        assert b.claim_allowed is False
        d = b.to_dict()
        assert d["future_regime"] is True
        assert d["implemented"] is False
        assert d["claim_allowed"] is False

    def test_future_admittive_validation_warns(self):
        b = BasisSpec(field_regime="future_admittive")
        v = validate_basis_spec(b)
        assert v["physical_amplitude_claim_allowed"] is False
        assert any("future_regime_future_admittive" in w for w in v.get("warnings", []))


# ─── Test 11: solved_poisson gated in v0.2.x ──────────────────────────────────

class TestSolvedPoissonGated:
    def test_solved_poisson_not_implemented(self):
        b = BasisSpec(field_regime="solved_poisson")
        assert b.implemented is False

    def test_solved_poisson_warn_not_implemented(self):
        b = BasisSpec(field_regime="solved_poisson")
        v = validate_basis_spec(b)
        assert v["valid"] is True  # it is a valid declaration (future)
        assert any("solved_poisson" in w for w in v.get("warnings", []))

    def test_solved_poisson_claim_gate_blocks_in_proxy_mode(self):
        b = BasisSpec(field_regime="solved_poisson")
        g = basis_claim_gate(
            b,
            source_calibration_status="uncalibrated_izhikevich_native_current",
            field_solver_status="laminar_proxy_no_pde",
        )
        assert g["physical_amplitude_claim_allowed"] is False


# ─── Test 12: Manifest contains nested basis block ────────────────────────────

class TestManifestContainsBasisBlock:
    def _run_smoke_manifest(self):
        cfg = (
            jaxfne.configuration()
            .network(name="test", kind="cortical_column", n=8)
            .emitter(family="izhikevich", preset="cortical_eig")
            .field(domain="laminar_column", conductivity="proxy")
            .probe(name="p", modes=["spikes"])
        )
        model = jaxfne.construct(cfg)
        sim = jaxfne.simulation(duration_ms=10.0, dt_ms=0.5, seed=0)
        signals = model.simulate(sim)
        return model.manifest(signals, [])

    def test_manifest_has_basis_key(self):
        manifest = self._run_smoke_manifest()
        assert "basis" in manifest

    def test_manifest_basis_has_required_keys(self):
        manifest = self._run_smoke_manifest()
        basis = manifest["basis"]
        for key in (
            "space_basis",
            "time_basis",
            "field_regime",
            "source_mode",
            "probe_basis",
            "dimension_status",
            "implemented",
            "future_regime",
            "claim_allowed",
        ):
            assert key in basis, f"Missing key: {key}"

    def test_manifest_basis_matches_laminar_proxy(self):
        manifest = self._run_smoke_manifest()
        basis = manifest["basis"]
        assert basis["space_basis"] == "laminar_depth"
        assert basis["field_regime"] == "laminar_proxy"
        assert basis["source_mode"] == "proxy_no_field_solve"
        assert basis["implemented"] is True
        assert basis["future_regime"] is False
        assert basis["claim_allowed"] is False

    def test_manifest_basis_dimension_status(self):
        manifest = self._run_smoke_manifest()
        dim = manifest["basis"]["dimension_status"]
        assert dim["x"] == "collapsed"
        assert dim["y"] == "collapsed"
        assert dim["z"] == "active"


# ─── Test 13: JSON dumps with allow_nan=False passes ─────────────────────────

class TestManifestJsonDumpsAllowNan:
    def test_full_manifest_json_safe(self):
        cfg = (
            jaxfne.configuration()
            .network(name="test", kind="cortical_column", n=8)
            .emitter(family="izhikevich", preset="cortical_eig")
            .field(domain="laminar_column", conductivity="proxy")
            .probe(name="p", modes=["spikes"])
        )
        model = jaxfne.construct(cfg)
        sim = jaxfne.simulation(duration_ms=10.0, dt_ms=0.5, seed=0)
        signals = model.simulate(sim)
        manifest = model.manifest(signals, [])
        # Must not raise
        dumped = json.dumps(manifest, allow_nan=False)
        assert '"basis"' in dumped


# ─── Test 14: source mode remains one of allowed modes ───────────────────────

class TestSourceModeAllowedValues:
    def test_all_allowed_source_modes_valid(self):
        for sm in _SOURCE_MODE_BASIS_VALUES:
            b = BasisSpec(source_mode=sm)
            v = validate_basis_spec(b)
            assert not any("invalid_source_mode" in i for i in v.get("issues", []))

    def test_invalid_source_mode_rejected(self):
        b = BasisSpec(source_mode="quantum_tunneling_current")
        v = validate_basis_spec(b)
        assert v["valid"] is False
        assert any("invalid_source_mode" in i for i in v["issues"])


# ─── Test 15: physical amplitude claim remains false under proxy basis ─────────

class TestPhysicalAmplitudeClaimFalseInProxyBasis:
    def test_default_basis_claim_allowed_false(self):
        b = default_basis_spec()
        assert b.claim_allowed is False

    def test_validation_physical_amplitude_always_false(self):
        b = default_basis_spec()
        v = validate_basis_spec(b)
        assert v["physical_amplitude_claim_allowed"] is False

    def test_claim_gate_always_false_in_proxy(self):
        b = default_basis_spec()
        g = basis_claim_gate(
            b,
            source_calibration_status="uncalibrated_izhikevich_native_current",
            field_solver_status="laminar_proxy_no_pde",
        )
        assert g["physical_amplitude_claim_allowed"] is False

    def test_claim_gate_always_false_for_future_maxwell(self):
        b = BasisSpec(field_regime="future_maxwell")
        g = basis_claim_gate(
            b,
            source_calibration_status="uncalibrated_izhikevich_native_current",
            field_solver_status="laminar_proxy_no_pde",
        )
        assert g["physical_amplitude_claim_allowed"] is False

    def test_to_dict_claim_allowed_false(self):
        b = default_basis_spec()
        d = b.to_dict()
        assert d["claim_allowed"] is False


# ─── Test 16: basis spec is frozen (immutable) ───────────────────────────────

class TestBasisSpecImmutable:
    def test_basis_spec_frozen(self):
        b = BasisSpec()
        with pytest.raises((AttributeError, TypeError)):
            b.space_basis = "xyz"  # type: ignore[misc]

    def test_axis_spec_frozen(self):
        ax = AxisSpec(name="z")
        with pytest.raises((AttributeError, TypeError)):
            ax.status = "collapsed"  # type: ignore[misc]


# ─── Test 17: validate_basis_spec accepts dict input ──────────────────────────

class TestValidateBasisSpecAcceptsDict:
    def test_valid_dict_accepted(self):
        d = {
            "space_basis": "laminar_depth",
            "time_basis": "continuous_ms",
            "field_regime": "laminar_proxy",
            "source_mode": "proxy_no_field_solve",
            "probe_basis": "multimodal_proxy",
        }
        v = validate_basis_spec(d)
        assert v["valid"] is True

    def test_invalid_dict_rejected(self):
        d = {"space_basis": "invalid_basis_value"}
        v = validate_basis_spec(d)
        assert v["valid"] is False


# ─── Test 18: public exports include AxisSpec, BasisSpec ─────────────────────

class TestPublicExports:
    def test_axis_spec_in_public_api(self):
        assert hasattr(jaxfne, "AxisSpec")
        assert jaxfne.AxisSpec is AxisSpec

    def test_basis_spec_in_public_api(self):
        assert hasattr(jaxfne, "BasisSpec")
        assert jaxfne.BasisSpec is BasisSpec

    def test_default_basis_spec_in_public_api(self):
        assert hasattr(jaxfne, "default_basis_spec")
        b = jaxfne.default_basis_spec()
        assert isinstance(b, BasisSpec)
