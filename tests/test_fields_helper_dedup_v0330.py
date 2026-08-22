"""v0.3.30 regression: fields helper deduplication.

`jaxfne/fields/diagnostics.py` and `jaxfne/fields/proxy.py` previously carried
byte-identical copies of five helpers, and `fields/__init__.py` imported two of
them from both modules (last-import-wins shadowing). diagnostics.py is now the
single canonical source; proxy.py re-imports the names so `fields.proxy.<name>`
still resolves and the public `jtfne.validate_projection_invariants` has exactly
one definition.

These tests lock that in and prevent planned copy-paste drift.
"""

import ast
import pathlib

import jax.numpy as jnp
import pytest

import jaxfne as jtfne
import jaxfne.fields.proxy as proxy
import jaxfne.fields.diagnostics as diagnostics

DEDUPED = [
    "_dtype_name",
    "_finite_bool",
    "_test_kernel_row_normalization",
    "validate_projection_invariants",
    "_make_field_solution_report",
]

_FIELDS_DIR = pathlib.Path(diagnostics.__file__).parent


def _top_level_defs(module_path):
    tree = ast.parse(pathlib.Path(module_path).read_text(encoding="utf-8"))
    return {
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def test_canonical_defs_live_in_diagnostics():
    """All five helpers must be defined in diagnostics.py."""
    defs = _top_level_defs(_FIELDS_DIR / "diagnostics.py")
    missing = [name for name in DEDUPED if name not in defs]
    assert not missing, f"diagnostics.py missing canonical defs: {missing}"


def test_proxy_does_not_redefine_helpers():
    """proxy.py must NOT locally define the deduped helpers (import only)."""
    defs = _top_level_defs(_FIELDS_DIR / "proxy.py")
    redefined = [name for name in DEDUPED if name in defs]
    assert not redefined, f"proxy.py still locally defines: {redefined}"


def test_proxy_names_are_identical_objects():
    """proxy.<name> must be the same object as diagnostics.<name> (no shadow).

    Covers all five deduped symbols (not just the two public ones) so a planned
    re-definition of any private helper in proxy.py is caught by identity too.
    """
    for name in DEDUPED:
        assert getattr(proxy, name) is getattr(diagnostics, name), (
            f"{name}: proxy and diagnostics resolve to different objects"
        )


def test_public_validator_resolves_to_diagnostics():
    """jtfne.validate_projection_invariants is the canonical diagnostics function."""
    assert jtfne.validate_projection_invariants is diagnostics.validate_projection_invariants
    assert callable(jtfne.validate_projection_invariants)


def test_validate_projection_invariants_behavior_preserved():
    """Golden behavioral check on a small finite fixture."""
    t_steps, n_emitters, n_contacts = 4, 3, 5
    sources = jnp.ones((t_steps, n_emitters), dtype=jnp.float32)
    positions = jnp.zeros((n_emitters, 3), dtype=jnp.float32)
    # Row-stochastic kernel (rows sum to 1).
    kernel = jnp.full((n_contacts, n_emitters), 1.0 / n_emitters, dtype=jnp.float32)
    source_proxy = jnp.ones((t_steps, n_contacts), dtype=jnp.float32)
    phi_e_proxy = jnp.ones((t_steps, n_contacts), dtype=jnp.float32)
    csd_proxy = jnp.ones((t_steps, n_contacts), dtype=jnp.float32)
    lfp_proxy = jnp.ones((t_steps, n_contacts), dtype=jnp.float32)

    report = jtfne.validate_projection_invariants(
        sources=sources,
        positions=positions,
        kernel=kernel,
        source_proxy=source_proxy,
        phi_e_proxy=phi_e_proxy,
        csd_proxy=csd_proxy,
        lfp_proxy=lfp_proxy,
    )

    assert report["source_shape"] == (t_steps, n_emitters)
    assert report["kernel_shape"] == (n_contacts, n_emitters)
    assert report["finite_sources"] is True
    assert report["warnings"] == []
    fa = report["field_admissibility"]
    assert fa["kernel_normalization_valid"] is True
    assert fa["source_conservation_status"] == "proxy_not_solved"
    # Truth-gate metadata must remain proxy-safe.
    assert fa["boundary_condition_status"] == "declared_metadata_only"


def test_make_field_solution_report_proxy_safe():
    """The report helper keeps proxy/no-PDE truth gates."""
    rep = diagnostics._make_field_solution_report()
    assert rep["field_solver_status"] == "linear_solver"
    assert rep["physical_amplitude_calibrated"] is False
    assert rep["field_claim_level"] == "proxy_readout"
