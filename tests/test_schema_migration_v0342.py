"""v0.3.42 truth-gate schema migration + guard tests.

Canonical schema:
  claim_level: computational_scaffold
  field_solver_status: linear_solver | pde_solver
  field_claim_level: proxy_readout
  physical_amplitude_calibrated: bool

This is the designated *migration test*: it is the one place permitted to name the
retired wording, and it constructs those strings by concatenation so the repository
stays free of literal occurrences even here.
"""
import json
import subprocess
from pathlib import Path

import pytest

import jaxfne as jtfne

# Retired wording, built so this file contains no literal occurrence.
_OLD_AMP_KEY = "physical_amplitude_" + "claim_allowed"
_OLD_SOLVER = "laminar_proxy" + "_no_pde"
_OLD_CLAIM = "proxy_readout" + "_only"
_OLD_TRUTH_MODE = "truth" + "_mode"
_OLD_TRUTH_VALUE = "truth_safe" + "_unverified"
RETIRED = (_OLD_AMP_KEY, _OLD_SOLVER, _OLD_CLAIM, _OLD_TRUTH_MODE, _OLD_TRUTH_VALUE)

REPO = Path(__file__).resolve().parent.parent


def test_migrate_schema_upgrades_legacy_keys():
    legacy = {
        _OLD_TRUTH_MODE: _OLD_TRUTH_VALUE,
        _OLD_AMP_KEY: False,
        "field_solver_status": _OLD_SOLVER,
        "field_claim_level": _OLD_CLAIM,
        "claim_level": "computational_scaffold",
    }
    new = jtfne.migrate_schema(legacy)
    assert _OLD_TRUTH_MODE not in new
    assert new["physical_amplitude_calibrated"] is False
    assert new["field_solver_status"] == "linear_solver"
    assert new["field_claim_level"] == "proxy_readout"
    assert new["claim_level"] == "computational_scaffold"


def test_migrate_schema_forces_amplitude_false():
    """Legacy True amplitude flags are clamped; scaffold never escalates."""
    assert jtfne.migrate_schema({_OLD_AMP_KEY: True})["physical_amplitude_calibrated"] is False
    assert jtfne.migrate_schema({_OLD_AMP_KEY: False})["physical_amplitude_calibrated"] is False


def test_fresh_manifest_uses_new_schema_and_no_retired_terms():
    cfg = jtfne.suite2_single_neuron_config(seed=0, duration_ms=40.0, dt_ms=0.1)
    model = jtfne.construct(cfg)
    manifest = model.manifest()
    blob = json.dumps(manifest)
    for term in RETIRED:
        assert term not in blob, f"retired term {term!r} leaked into manifest"
    truth = manifest.get("truth_boundary", manifest)
    assert truth.get("claim_level") == "computational_scaffold"
    assert truth.get("field_solver_status") in ("linear_solver", "pde_solver")
    assert truth.get("field_claim_level") == "proxy_readout"
    assert truth.get("physical_amplitude_calibrated") is False


def _tracked(patterns):
    """Tracked (committed) files matching git pathspecs — excludes generated/ignored output."""
    out = subprocess.check_output(
        ["git", "ls-files", *patterns], cwd=REPO, text=True
    ).splitlines()
    return [REPO / rel for rel in out if "docs/assets/" not in rel]


@pytest.mark.parametrize("term", RETIRED)
def test_no_retired_terms_in_public_docs(term):
    offenders = [str(p) for p in _tracked(["docs/**/*.md", "docs/*.md"]) if term in p.read_text()]
    assert not offenders, f"retired term {term!r} present in docs: {offenders}"


@pytest.mark.parametrize("term", RETIRED)
def test_no_retired_terms_in_notebooks(term):
    offenders = [str(p) for p in _tracked(["tutorials/**/*.ipynb"]) if term in p.read_text()]
    assert not offenders, f"retired term {term!r} present in notebooks: {offenders}"
