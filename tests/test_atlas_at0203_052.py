"""0.5.2 ATLAS item 10: AT-02 pair transmission + AT-03 E<->I pair.

Delay declared in ms (0.5.2 decision 0b rule); configured ms and realized
steps both recorded from the executed model. Zero-delay limit is
bit-identical; delay-vs-zero differs (arrival timing executes).
"""

from __future__ import annotations

import numpy as np

import artifacts.atlas.at01_at06_052 as A


def test_delay_declared_ms_realized_steps():
    assert A.DELAY_MS == 2.0
    assert round(A.DELAY_MS / A.DT_MS) == 4  # 0b rule at toy dt


def test_at02_arms_and_zero_limit():
    out = A.run_at02()
    assert out["scenario"] == "AT-02"
    assert out["status"] == "OK"
    assert out["wall_s"] < A.WALL_BUDGET_S
    assert out["level"] == "RELATIVE_PROXY"
    d = out["delay"]
    assert d["configured_ms"] == A.DELAY_MS
    assert all(s == 4 for s in d["realized_steps"]), d["realized_steps"]
    assert d["zero_limit_bit_identical"] is True
    assert d["delayed_differs_from_zero"] is True
    for arm in ("delayed", "zero", "absent"):
        assert all(s == (4 if arm == "delayed" else 0) for s in out["arms"][arm]["delay_steps"])
    assert out["arms"]["delayed"]["delay_storage"] == "per_edge"
    # All-zero arms take the uniform_zero fast path; limit identity still holds.
    assert out["arms"]["zero"]["delay_storage"] == "uniform_zero"
    assert out["arms"]["absent"]["delay_storage"] == "uniform_zero"


def test_at02_individual_vs_superposed_fields():
    out = A.run_at02()
    ivs = out["fields"]["individual_vs_superposed"]
    assert ivs["superposition_identity"] is True
    assert ivs["reconstruction_max_err"] <= A.SUPERPOSITION_TOL
    assert ivs["superposed_mean_abs"] > 0


def test_at02_distance_law_proxy_only():
    out = A.run_at02()
    dl = out["fields"]["distance_law_proxy"]
    assert "no physical distance law" in dl["note"]
    assert set(dl["centroids_relative_frac"]) == {"A", "B"}
    assert dl["contacts_declared"] == A.AT02_N_CONTACTS
    # Realized count is read back from the executed field, never assumed.
    assert dl["contacts_realized"] == len(dl["contact_depths_frac"])
    assert dl["contacts_realized"] == len(dl["contact_mean_abs"])
    blob = repr(dl).replace("no physical distance law", "")
    for unit in (" mm", " um", "_mm", "_um"):
        assert unit not in blob


def test_at03_phase_cancellation_frequency():
    out = A.run_at03()
    assert out["scenario"] == "AT-03"
    assert out["status"] == "OK"
    assert out["wall_s"] < A.WALL_BUDGET_S
    assert out["level"] == "RELATIVE_PROXY"
    assert all(s == 4 for s in out["delay"]["realized_steps"])
    assert "grammar" in out["grammar_limitation"].lower() or "limitation" in out
    osc = out["oscillation"]
    assert 0.0 <= osc["phase_lag_e_i_rad"] <= float(np.pi) + 1e-9
    assert np.isfinite(osc["cancellation_index"])
    assert osc["frequency_dependent_field"] is True
    assert out["superposition_identity"] is True
    assert out["pair"]["n_spikes"] >= 0
