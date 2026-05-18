"""Tests for v0.0.15 .jcfg.json config/object standard foundation.

A. load_config valid .jcfg.json returns JaxFNEConfig
B. validate_config valid config returns valid=True
C. validate_config missing required section returns issue
D. validate_config truth escalation physical_amplitude_claim_allowed=True returns issue
E. validate_config unsupported schema_version returns issue
F. config_to_simulation maps duration_ms/dt_ms/seed correctly
G. config_to_geometry maps geometry.populations to LaminarSourceGeometry
H. config_to_geometry returns None when geometry absent
I. config_to_configuration maps minimal network/emitter/field/probes to Configuration
J. config_to_trial_batch uses trials.seed_policy correctly
K. JaxFNEConfig.to_dict is JSON-safe and emits JSON key "field"
L. JaxFNEConfig.config_hash is stable for identical inputs
M. config_truth_boundary returns conservative truth fields
N. unknown top-level keys produce warning
O. no physical units introduced by geometry config
"""

import json
import tempfile
from pathlib import Path

import pytest

import jaxfne as jtfne
from jaxfne.core import (
    _JAXFNE_CONFIG_SCHEMA_VERSION,
    _CONSERVATIVE_TRUTH_DEFAULTS,
    JaxFNEConfig,
    ConfigValidationResult,
    LaminarSourceGeometry,
)


_VALID_TRUTH = {
    "truth_mode": "truth_safe_unverified",
    "claim_level": "computational_scaffold",
    "source_calibration_status": "uncalibrated_izhikevich_native_current",
    "field_solver_status": "laminar_proxy_no_pde",
    "physical_amplitude_claim_allowed": False,
    "empirical_validation_status": "not_empirically_validated",
    "mechanism_claim_status": "not_claimed",
}

_MINIMAL_CONFIG = {
    "schema_version": _JAXFNE_CONFIG_SCHEMA_VERSION,
    "run": {"duration_ms": 50.0, "dt_ms": 0.5, "seed": 42},
    "truth": dict(_VALID_TRUTH),
    "network": {"n": 10},
    "emitter": {"family": "izhikevich", "preset": "cortical_eig"},
    "field": {"domain": "laminar_column", "conductivity": "proxy",
               "boundary": "mean_zero_neumann", "gauge": "mean_zero"},
    "probes": [{"name": "laminar_probe", "n_contacts": 16}],
}


def _write_config(d: dict, path: Path) -> None:
    path.write_text(json.dumps(d, allow_nan=False), encoding="utf-8")


# ──────────────────────────────────────────────────────────────

def test_a_load_config_valid_returns_jaxfne_config():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.jcfg.json"
        _write_config(_MINIMAL_CONFIG, p)
        cfg = jtfne.load_config(p)
        assert isinstance(cfg, JaxFNEConfig)
        assert cfg.schema_version == _JAXFNE_CONFIG_SCHEMA_VERSION
        assert cfg.run["duration_ms"] == 50.0
        assert cfg.field_spec["domain"] == "laminar_column"


def test_b_validate_config_valid_returns_valid_true():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.jcfg.json"
        _write_config(_MINIMAL_CONFIG, p)
        cfg = jtfne.load_config(p)
        result = jtfne.validate_config(cfg)
        assert isinstance(result, ConfigValidationResult)
        assert result.valid is True
        assert len(result.issues) == 0


def test_c_validate_config_missing_required_section():
    cfg_dict = dict(_MINIMAL_CONFIG)
    del cfg_dict["network"]
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.jcfg.json"
        _write_config(cfg_dict, p)
        cfg = jtfne.load_config(p)
        result = jtfne.validate_config(cfg)
        assert result.valid is False
        assert any("required_section_missing:network" in i for i in result.issues)


def test_d_validate_config_truth_escalation_blocking():
    cfg_dict = {**_MINIMAL_CONFIG, "truth": {**_VALID_TRUTH, "physical_amplitude_claim_allowed": True}}
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.jcfg.json"
        _write_config(cfg_dict, p)
        cfg = jtfne.load_config(p)
        result = jtfne.validate_config(cfg)
        assert result.valid is False
        assert any("physical_amplitude_claim_allowed" in i for i in result.issues)


def test_e_validate_config_unsupported_schema_version():
    cfg_dict = {**_MINIMAL_CONFIG, "schema_version": "unsupported.version"}
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.jcfg.json"
        _write_config(cfg_dict, p)
        cfg = jtfne.load_config(p)
        result = jtfne.validate_config(cfg)
        assert result.valid is False
        assert any("schema_version_unsupported" in i for i in result.issues)


def test_f_config_to_simulation():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.jcfg.json"
        _write_config(_MINIMAL_CONFIG, p)
        cfg = jtfne.load_config(p)
        sim = jtfne.config_to_simulation(cfg)
        assert sim.duration_ms == 50.0
        assert sim.dt_ms == 0.5
        assert sim.seed == 42


def test_g_config_to_geometry_maps_populations():
    geo_config = {
        **_MINIMAL_CONFIG,
        "network": {"n": 10},
        "geometry": {
            "populations": [
                {"name": "L4_E", "cell_type": "E", "layer": "L4",
                 "depth_min": 0.3, "depth_max": 0.5, "n_units": 7},
                {"name": "L4_PV", "cell_type": "PV", "layer": "L4",
                 "depth_min": 0.3, "depth_max": 0.5, "n_units": 3},
            ],
            "position_units": "relative_laminar_depth_proxy",
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.jcfg.json"
        _write_config(geo_config, p)
        cfg = jtfne.load_config(p)
        geom = jtfne.config_to_geometry(cfg)
        assert isinstance(geom, LaminarSourceGeometry)
        assert geom.n_units_total == 10
        assert len(geom.populations) == 2
        assert geom.position_units == "relative_laminar_depth_proxy"


def test_h_config_to_geometry_returns_none_when_absent():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.jcfg.json"
        _write_config(_MINIMAL_CONFIG, p)
        cfg = jtfne.load_config(p)
        geom = jtfne.config_to_geometry(cfg)
        assert geom is None


def test_i_config_to_configuration():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.jcfg.json"
        _write_config(_MINIMAL_CONFIG, p)
        cfg = jtfne.load_config(p)
        model_cfg = jtfne.config_to_configuration(cfg)
        assert isinstance(model_cfg, jtfne.Configuration)
        v = model_cfg.validate()
        assert v["valid"] is True


def test_j_config_to_trial_batch_seed_policy():
    trials_config = {
        **_MINIMAL_CONFIG,
        "trials": {"n_reps": 2, "base_seed": 100, "seed_policy": "paired_by_replicate"},
    }
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.jcfg.json"
        _write_config(trials_config, p)
        cfg = jtfne.load_config(p)

    paradigm = jtfne.standard_visual_omission()
    conditions = paradigm.conditions[:2]
    batch = jtfne.config_to_trial_batch(cfg, conditions)

    assert isinstance(batch, jtfne.TrialBatch)
    assert len(batch.trials) == 4  # 2 conditions × 2 reps
    # paired_by_replicate: rep 0 → seed 100, rep 1 → seed 101
    assert batch.trials[0].seed == 100  # rep 0
    assert batch.trials[1].seed == 100  # rep 0, different condition
    assert batch.trials[2].seed == 101  # rep 1


def test_k_to_dict_json_safe_emits_field_key():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.jcfg.json"
        _write_config(_MINIMAL_CONFIG, p)
        cfg = jtfne.load_config(p)

    d = cfg.to_dict()
    assert isinstance(d, dict)
    # JSON key must be "field", not "field_spec"
    assert "field" in d
    assert "field_spec" not in d
    # Must round-trip through json.dumps
    json.dumps(d, allow_nan=False)


def test_l_config_hash_stable():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.jcfg.json"
        _write_config(_MINIMAL_CONFIG, p)
        cfg1 = jtfne.load_config(p)
        cfg2 = jtfne.load_config(p)

    assert cfg1.config_hash == cfg2.config_hash
    assert isinstance(cfg1.config_hash, str)
    assert len(cfg1.config_hash) == 16


def test_m_config_truth_boundary_conservative():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.jcfg.json"
        _write_config(_MINIMAL_CONFIG, p)
        cfg = jtfne.load_config(p)

    tb = jtfne.config_truth_boundary(cfg)
    assert tb["truth_mode"] == "truth_safe_unverified"
    assert tb["claim_level"] == "computational_scaffold"
    assert tb["physical_amplitude_claim_allowed"] is False
    assert tb["empirical_validation_status"] == "not_empirically_validated"
    assert tb["mechanism_claim_status"] == "not_claimed"
    # Must be JSON-safe
    json.dumps(tb, allow_nan=False)


def test_n_unknown_top_level_keys_produce_warning():
    cfg_dict = {**_MINIMAL_CONFIG, "unknown_section_xyz": {"value": 42}}
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.jcfg.json"
        _write_config(cfg_dict, p)
        cfg = jtfne.load_config(p)
        result = jtfne.validate_config(cfg)

    assert any("unknown_top_level_key:unknown_section_xyz" in w for w in result.warnings)
    # Valid is still True — unknown keys are warnings, not errors
    # (unless other issues exist)
    assert "unknown_top_level_key" not in " ".join(result.issues)


def test_o_no_physical_units_in_geometry():
    geo_config = {
        **_MINIMAL_CONFIG,
        "network": {"n": 5},
        "geometry": {
            "populations": [
                {"name": "L4_E", "cell_type": "E", "layer": "L4",
                 "depth_min": 0.3, "depth_max": 0.7, "n_units": 5},
            ],
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.jcfg.json"
        _write_config(geo_config, p)
        cfg = jtfne.load_config(p)

    geom = jtfne.config_to_geometry(cfg)
    assert geom is not None
    # Default position_units must be relative proxy, not physical mm/um
    assert geom.position_units == "relative_laminar_depth_proxy"
    # depth values must be in [0, 1]
    for pop in geom.populations:
        assert 0.0 <= pop.depth_min < pop.depth_max <= 1.0
    # No physical unit strings in the serialized geometry
    geo_dict = geom.to_dict()
    geo_str = json.dumps(geo_dict)
    assert "depth_um" not in geo_str
    assert "_mm" not in geo_str
