"""0.5.2 ATLAS item 12: AT-05 emergence + AT-06 electrode locality.

Needs ENGINE items 1 (relative geometry executes) and 5 (probe
semantics), both landed. rho_sync is the executed kappa, never the
intent label. C(R,f) is a proxy ratio under declared assumptions.
"""

from __future__ import annotations

import artifacts.atlas.at01_at06_052 as A


def test_at05_arms_report_executed_rho():
    out = A.run_at05()
    assert out["scenario"] == "AT-05"
    assert out["status"] == "OK"
    assert out["wall_s"] < A.WALL_BUDGET_S
    assert out["level"] == "RELATIVE_PROXY"
    assert set(out["arms"]) == set(A.AT05_ARMS)
    for name, arm in out["arms"].items():
        assert arm["superposition_identity"] is True
        assert arm["rho_sync_executed"] is not None
        assert arm["a_phi_mid_contact"] >= 0


def test_at05_phi_n_neq_n_phi_1():
    """Coherent-vs-incoherent check runs per arm; negatives recorded."""
    out = A.run_at05()
    assert out["emergence_note"].startswith("rho_sync is the executed kappa")
    for name, arm in out["arms"].items():
        ratio = arm["coherent_over_n_single"]
        assert 0.0 <= ratio <= 1.0 + A.AT05_RATIO_MARGIN, (name, ratio)
        assert arm["phi_n_neq_n_phi_1"] == (abs(ratio - 1.0) > A.AT05_RATIO_MARGIN)
    # Ordering by executed rho is a real ordering (not all equal).
    rhos = [out["arms"][k]["rho_sync_executed"] for k in out["arms_ordered_by_executed_rho"]]
    assert rhos == sorted(rhos)
    assert len(set(rhos)) > 1


def test_at05_a_phi_table_covers_n_and_contacts():
    out = A.run_at05()
    ns = {arm["n"] for arm in out["arms"].values()}
    assert ns == {2, 4, 8}
    assert set(out["scaling"]["a_phi_by_n_same_seed"]) == {"2", "4", "8"}
    rhos = [arm["rho_sync_executed"] for arm in out["arms"].values()]
    assert all(v is not None for v in rhos)


def test_at06_electrode_chain_declared():
    out = A.run_at06()
    assert out["scenario"] == "AT-06"
    assert out["status"] == "OK"
    assert out["wall_s"] < A.WALL_BUDGET_S
    assert out["level"] == "RELATIVE_PROXY"
    el = out["electrode"]
    assert el["contacts_declared"] == A.AT06_N_CONTACTS
    assert el["contacts_realized"] == len(el["contact_depths_frac"]) == 4
    assert el["reference"].startswith("common_average")
    assert "record-only" in el["reference"] and "record-only" in el["filter"]
    assert el["conductivity"] == "proxy"
    assert el["distance"] == "relative fractions in [0,1]"
    assert "DECLARED ASSUMPTION" in el["source_depths"]


def test_at06_locality_cells_bounded():
    out = A.run_at06()
    loc = out["locality_c_r_f"]
    assert len(loc) == len(A.AT06_RADII_FRAC) * len(A.AT06_BANDS_HZ) == 6
    for key, cell in loc.items():
        assert 0.0 - 1e-9 <= cell["c_mean"] <= 1.0 + 1e-9, (key, cell)
        assert len(cell["c_per_contact"]) == out["electrode"]["contacts_realized"]
    # Full radius covers every source: ratio 1 within float32 noise.
    full = [v for k, v in loc.items() if k.startswith("R=1.0")]
    assert full, "R=1.0 cells missing"
    for cell in full:
        assert all(abs(v - 1.0) < 1e-6 for v in cell["c_per_contact"]), cell
    assert out["superposition_identity"] is True
