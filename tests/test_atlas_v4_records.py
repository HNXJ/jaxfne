"""Schema v4 records from real scenario runs (0.5.5 ENGINE item 1).

Every scenario's v4 record validates, and its IMPLEMENTED set is pinned: a
cell changes state only when a runner starts or stops writing the value.
The v1 static tables claimed M_compute/Q (and some X/SPK/Phi_E) for
AT-02..AT-06 without the runner retaining a value; v4 records them OMITTED.
"""

from __future__ import annotations

import functools

import artifacts.atlas.at01_at06_052 as A052
import artifacts.atlas.at01_at10_toy as T0
import artifacts.atlas.at07_at04_053 as A053
import artifacts.atlas.at08_at09_054 as A054
from artifacts.atlas import y_schema as Y

_TWO_AREA = {
    "C", "C_12", "H", "H_A1", "H_A2", "M_compute", "Phi_A1", "Phi_A2", "Phi_E", "Q",
    "SPK", "SPK_A1", "SPK_A2", "T_compute", "W", "W_12", "W_21", "X", "dphi_12",
}  # fmt: skip
_SINGLE_AREA_PLASTIC = {"C", "H", "M_compute", "Phi_E", "Q", "SPK", "T_compute", "W", "X"}
EXPECTED_IMPLEMENTED: dict[str, set[str]] = {
    "AT-01": {"E_reduction", "M_compute", "SPK", "T_compute", "X"},
    "AT-02": {"M_compute", "Phi_E", "Q", "SPK", "T_compute", "X"},
    "AT-03": {"C", "M_compute", "PSD", "Phi_E", "Q", "SPK", "T_compute", "X", "phi"},
    "AT-04": {"C", "M_compute", "Phi_E", "Q", "SPK", "T_compute", "X"},
    "AT-05": {"C", "M_compute", "Phi_E", "Q", "SPK", "T_compute", "X"},
    "AT-06": {"C", "M_compute", "Q", "SPK", "T_compute", "X"},
    "REDUCTION": {"E_reduction", "Q", "T_compute", "X"},
    "AT-07": _SINGLE_AREA_PLASTIC,
    "AT-04R2": _SINGLE_AREA_PLASTIC,
    "AT-08": _TWO_AREA,
    "AT-09": _TWO_AREA,
    "AT-10": {"SPK", "T_compute"},
}


@functools.lru_cache(maxsize=1)
def _raws() -> tuple[dict, ...]:
    return (
        A052.run_at01(),
        A052.run_at02(),
        A052.run_at03(),
        A052.run_at04(),
        A052.run_at05(),
        A052.run_at06(),
        A052.run_reduction(),
        *A053.run_all().values(),
        *A054.run_all().values(),
        T0.run_scenario("AT-10"),
    )


def test_every_scenario_record_validates_and_is_pinned():
    seen = {}
    for raw in _raws():
        rec = Y.measure_v4(raw)
        assert Y.validate(rec) == [], raw["scenario"]
        if raw["scenario"] == "AT-01" and raw["status"] == "OK_REFUSED_HH":
            continue  # no Jaxley: HH arm refused, so the pinned set (with HH verdicts) does not apply
        seen[raw["scenario"]] = {k for k, c in rec.items() if c["state"] == "IMPLEMENTED"}
        assert rec["Phi_B"]["state"] == "REFUSED"
    for sid, implemented in seen.items():
        assert implemented == EXPECTED_IMPLEMENTED[sid], sid


def test_cross_area_w_split_is_consistent():
    """W_12 / W_21 come from the two owned cross ranges; their max is the cross value."""
    for raw in _raws():
        if raw["scenario"] not in ("AT-08", "AT-09"):
            continue
        for name, arm in raw["arms"].items():
            cross = arm["w_cross_max_abs_change"]
            if cross is None:
                assert arm["w_12_max_abs_change"] is None and arm["w_21_max_abs_change"] is None
                continue
            assert max(arm["w_12_max_abs_change"], arm["w_21_max_abs_change"]) == cross, name
