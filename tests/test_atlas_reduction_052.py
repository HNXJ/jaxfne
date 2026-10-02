"""0.5.2 ATLAS item 13: reduction row + schema v0 -> v1.

HH -> reduced -> population source against PREDECLARED tolerances;
failures RECORDED (PASS/FAIL booleans), never hidden, never tuned.
Schema v1 extends v0: names stable, cells gain epistemic level,
M_compute wired, E_reduction populated.
"""

from __future__ import annotations

import artifacts.atlas.at01_at06_052 as A
import artifacts.atlas.at01_at10_toy as T0


def test_reduction_tolerances_predeclared():
    assert A.REDUCTION_RATE_TOL_HZ == 10.0
    assert A.REDUCTION_V_PEAK_TOL_MV == 20.0
    assert A.REDUCTION_FIELD_TOL_FRAC == 0.75


def test_reduction_runs_inside_budget():
    out = A.run_reduction()
    assert out["scenario"] == "REDUCTION"
    assert out["status"] == "OK"
    assert out["wall_s"] < A.WALL_BUDGET_S
    assert out["level"] == "RELATIVE_PROXY"
    assert out["tolerances_predeclared"]["rate_hz"] == 10.0


def test_reduction_verdicts_recorded_not_hidden():
    out = A.run_reduction()
    v = out["verdicts"]
    assert (
        set(v)
        >= {
            "hh_to_reduced_rate",
            "hh_to_reduced_v_peak",
            "reduced_to_population_rate",
            "reduced_to_population_source",
        }
        or v.get("hh_to_reduced", {}).get("state") == "REFUSED"
    )
    for key, verdict in v.items():
        if verdict.get("state") == "REFUSED":
            assert verdict["reason"]
        else:
            assert isinstance(verdict["pass"], bool), (key, verdict)
            assert any(k.startswith("diff") or k == "relative_diff" for k in verdict), (
                key,
                verdict,
            )
    # The record is honest about failures: booleans exist regardless of value.
    assert all("pass" in verdict or verdict.get("state") == "REFUSED" for verdict in v.values())


def test_reduction_rungs_present():
    out = A.run_reduction()
    assert set(out["single"]) >= {"rate_hz", "v_peak_mv", "mean_abs_source_per_neuron"}
    assert set(out["population"]) >= {"rate_hz_per_neuron", "mean_abs_source_per_neuron"}
    assert out["hh"]["status"] in ("OK", "REFUSED")


def test_schema_v1_keeps_v0_names_stable():
    assert tuple(A.Y_KEYS_V1) == tuple(T0.Y_KEYS)
    assert A.SCHEMA_VERSION == "v1"
    assert list(A.Y_KEYS_V1).index("Phi_B") == list(T0.Y_KEYS).index("Phi_B")


def test_schema_v1_cells_carry_levels():
    raws = {
        "AT-01": A.run_at01(),
        "AT-02": A.run_at02(),
        "AT-03": A.run_at03(),
        "AT-04": A.run_at04(),
        "AT-05": A.run_at05(),
        "AT-06": A.run_at06(),
    }
    for sid, raw in raws.items():
        rec = A.measure_v1(sid, raw)
        assert set(rec) == set(T0.Y_KEYS), sid
        for key, cell in rec.items():
            assert set(cell) >= {"state", "value", "note", "level"}, (sid, key)
            assert cell["state"] in ("IMPLEMENTED", "OMITTED", "REFUSED"), (sid, key)
            assert cell["level"] == "RELATIVE_PROXY", (sid, key)
            if cell["state"] != "IMPLEMENTED":
                assert cell["value"] is None, (sid, key)
    assert A.measure_v1("AT-01", raws["AT-01"])["Phi_B"]["state"] == "REFUSED"
    assert A.measure_v1("AT-01", raws["AT-01"])["H"]["state"] == "OMITTED"
    assert A.measure_v1("AT-06", raws["AT-06"])["C"]["state"] == "IMPLEMENTED"
    assert A.measure_v1("AT-03", raws["AT-03"])["PSD"]["state"] == "IMPLEMENTED"
    assert A.measure_v1("AT-02", raws["AT-02"])["Q"]["state"] == "IMPLEMENTED"
    m = A.gap_matrix_v1(raws)
    assert set(m) == set(A.SCENARIOS)
    assert set(m["AT-01"]) == set(T0.Y_KEYS)
