"""0.5.2 ATLAS item 11: AT-04 geometry/orientation arm.

Declared geometry reaches executed positions (relative fractions in
[0,1]); outside [0,1] refuses (H5). Phi->X fails closed (AT-04-R3
OUT_OF_SCOPE); H-perturbation belongs to 0.5.3 (AT-04-R2).
"""

from __future__ import annotations

import artifacts.atlas.at01_at06_052 as A


def test_at04_arms_declare_relative_geometry():
    out = A.run_at04()
    assert out["scenario"] == "AT-04"
    assert out["status"] == "OK"
    assert out["wall_s"] < A.WALL_BUDGET_S
    assert out["level"] == "RELATIVE_PROXY"
    assert set(out["arms"]) == {"stacked", "swapped", "overlap"}
    for name, arm in out["arms"].items():
        g = arm["declared_G"]
        assert all(0.0 <= g[k] <= 1.0 for k in ("a0", "a1", "b0", "b1")), (name, g)
        # Declared geometry reaches executed positions (PARAM-04 repaired).
        assert set(arm["realized_z_ranges"]) == {"A", "B"}
        assert arm["superposition_identity"] is True


def test_at04_geometry_moves_the_executed_field():
    out = A.run_at04()
    famps = {n: out["arms"][n]["field_mean_abs"] for n in out["arms"]}
    assert len(set(famps.values())) > 1, f"geometry inert: {famps}"
    aligns = {n: out["arms"][n]["source_alignment_corr"] for n in out["arms"]}
    for n, v in aligns.items():
        assert -1.0 - 1e-9 <= v <= 1.0 + 1e-9, (n, v)


def test_at04_phi_to_x_fails_closed_h_omitted():
    out = A.run_at04()
    assert out["phi_to_x"]["state"] == "REFUSED"  # AT-04-R3
    assert out["h_perturbation"]["state"] == "OMITTED"  # AT-04-R2 -> 0.5.3
    assert "correlation" in out["x_to_phi_note"].lower()
    assert out["x_to_phi_correlation_across_arms"] is not None


def test_at04_geometry_outside_unit_interval_refused():
    rec = A._refused_geometry()
    assert rec["state"] == "REFUSED", rec
    assert rec["level"] == "RELATIVE_PROXY"


def test_at04_no_physical_units():
    out = A.run_at04()
    blob = repr(out)
    for unit in (" mm", " um", "_mm", "_um", "millimeter"):
        assert unit not in blob, f"physical unit leaked: {unit!r}"
