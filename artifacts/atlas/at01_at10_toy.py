"""AT-01..AT-10 toy-size first pass (0.5.1 ATLAS item 5).

Authority: artifacts/project_sources/8_atlas.md (S1..S10 == AT-01..AT-10).
Consumer/tests of the engine, not an authority over it: scenarios are
TFNE/JDNA *data* over the current public surface only (item 8 firewall).
No new engine capabilities; no synthesized quantities.

Toy sizes only: smallest N, short T, coarse dt. Wall-time budget 120 s per
scenario; over budget -> status OVER_BUDGET, shrink, never force.

AT-01 uses the Jaxley bridge via the public symbol
``jaxfne.hh_jaxley_reference_trace``. If Jaxley is absent the run records
REFUSED, never a substitute.

Import rule (enforced by tests/test_atlas_firewall.py): this file imports
only the top-level ``jaxfne`` package plus stdlib / numpy. No
``jaxfne.<submodule>`` imports.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

import jaxfne as J

# Toy resolution: coarse dt, short T. (dt, T) chosen per phenomenon per
# AT-00-R2, at the cheapest point that still executes.
TOY_DT_MS = 0.5
TOY_SEED = 7
WALL_BUDGET_S = 120.0

SCENARIOS: tuple[str, ...] = (
    "AT-01",
    "AT-02",
    "AT-03",
    "AT-04",
    "AT-05",
    "AT-06",
    "AT-07",
    "AT-08",
    "AT-09",
    "AT-10",
)


def _spike_summary(signals: Any) -> dict[str, Any]:
    """Best-effort spike summary; raises on absence (caller marks OMITTED)."""
    spikes = np.asarray(signals.spikes)
    return {
        "shape": list(spikes.shape),
        "n_spikes": int((spikes > 0).sum()),
    }


def _run_config(cfg: Any, duration_ms: float, dt_ms: float) -> dict[str, Any]:
    """Construct + simulate a public Configuration; return raw outcome."""
    t0 = time.perf_counter()
    model = J.construct(cfg)
    signals = model.simulate(J.simulation(duration_ms=duration_ms, dt_ms=dt_ms, seed=TOY_SEED))
    wall_s = time.perf_counter() - t0
    out: dict[str, Any] = {"wall_s": wall_s, "status": "OK"}
    try:
        out["spikes"] = _spike_summary(signals)
    except Exception as exc:  # absent quantity, never synthesized
        out["spikes"] = {"omitted": f"spike extraction failed: {exc!r}"}
    return out


def run_at01() -> dict[str, Any]:
    """S1: 1 full HH neuron (Jaxley bridge) + reduced-neuron reference."""
    t0 = time.perf_counter()
    try:
        t, v, i_inj = J.hh_jaxley_reference_trace(
            duration_ms=5.0, dt_ms=TOY_DT_MS, current_amplitude=10.0
        )
        hh = {
            "status": "OK",
            "n_steps": int(np.asarray(t).shape[0]),
            "v_min_mv": float(np.min(v)),
            "v_max_mv": float(np.max(v)),
        }
    except ImportError:
        return {
            "scenario": "AT-01",
            "status": "REFUSED",
            "reason": "Jaxley absent; no substitute per 0.5.1 item 5",
            "wall_s": time.perf_counter() - t0,
        }
    reduced = _run_config(
        J.suite2_single_neuron_config(seed=TOY_SEED, duration_ms=5.0, dt_ms=TOY_DT_MS),
        5.0,
        TOY_DT_MS,
    )
    return {
        "scenario": "AT-01",
        "status": reduced["status"],
        "wall_s": time.perf_counter() - t0,
        "hh_reference": hh,
        "reduced": reduced,
    }


def run_at02() -> dict[str, Any]:
    """S2: driven pair N1 -> N2 (toy: 2-neuron column, declared drive)."""
    out = _run_config(
        J.suite2_net1_config(
            seed=TOY_SEED,
            n=2,
            duration_ms=10.0,
            dt_ms=TOY_DT_MS,
            drives={"E": 6.0, "PV": 2.0, "SST": 2.2, "VIP": 2.0},
        ),
        10.0,
        TOY_DT_MS,
    )
    return {"scenario": "AT-02", **out}


def run_at03() -> dict[str, Any]:
    """S3: E<->I recurrent pair (toy: 4-neuron E/I mix, default drive)."""
    out = _run_config(
        J.suite2_net1_config(seed=TOY_SEED, n=4, duration_ms=10.0, dt_ms=TOY_DT_MS),
        10.0,
        TOY_DT_MS,
    )
    return {"scenario": "AT-03", **out}


def run_at04() -> dict[str, Any]:
    """S4: pair + declared drive difference (correlation only).

    Phi -> X feedback is a candidate (AT-04-R3): fails closed, recorded in
    the gap matrix, never executed here.
    """
    lo = _run_config(
        J.suite2_net1_config(
            seed=TOY_SEED,
            n=2,
            duration_ms=10.0,
            dt_ms=TOY_DT_MS,
            drives={"E": 2.0, "PV": 2.0, "SST": 2.2, "VIP": 2.0},
        ),
        10.0,
        TOY_DT_MS,
    )
    hi = _run_config(
        J.suite2_net1_config(
            seed=TOY_SEED,
            n=2,
            duration_ms=10.0,
            dt_ms=TOY_DT_MS,
            drives={"E": 6.0, "PV": 2.0, "SST": 2.2, "VIP": 2.0},
        ),
        10.0,
        TOY_DT_MS,
    )
    return {
        "scenario": "AT-04",
        "status": "OK",
        "wall_s": lo["wall_s"] + hi["wall_s"],
        "drive_lo": lo,
        "drive_hi": hi,
        "phi_to_x": "REFUSED (candidate AT-04-R3, fails closed)",
    }


def run_at05() -> dict[str, Any]:
    """S5: E/I population field emergence (toy: 8-neuron column)."""
    out = _run_config(
        J.suite2_net1_config(seed=TOY_SEED, n=8, duration_ms=20.0, dt_ms=TOY_DT_MS),
        20.0,
        TOY_DT_MS,
    )
    return {"scenario": "AT-05", **out}


def run_at06() -> dict[str, Any]:
    """S6: structured population + electrode chain (toy: 8 neurons)."""
    cfg = (
        J.Configuration()
        .runtime(seed=TOY_SEED, duration_ms=20.0, dt_ms=TOY_DT_MS)
        .column("V1", ["L2/3", "L4"], 8)
        .cell_types({"E": 0.75, "PV": 0.25})
        .connectivity(kind="laminar_signed_metadata", recurrent=True)
        .set_emitter("izhikevich", "cortical_eig")
        .field(domain="laminar_column", conductivity="proxy")
        .probes(["spikes", "V_m", "source", "LFP-proxy", "CSD-proxy"], n_contacts=4)
    )
    out = _run_config(cfg, 20.0, TOY_DT_MS)
    return {"scenario": "AT-06", **out}


def run_at07() -> dict[str, Any]:
    """S7: plastic population (toy: fixed-W matched pair, 8 neurons).

    Plasticity enable/disable/clamp arrives in 0.5.3; this pass records the
    fixed-W baseline twice (same realized network intent, two seeds) and
    marks every plastic arm OMITTED in the gap matrix.
    """
    base_kwargs: dict[str, Any] = {
        "n": 8,
        "duration_ms": 20.0,
        "dt_ms": TOY_DT_MS,
    }
    a = _run_config(J.suite2_net1_config(seed=TOY_SEED, **base_kwargs), 20.0, TOY_DT_MS)
    b = _run_config(J.suite2_net1_config(seed=11, **base_kwargs), 20.0, TOY_DT_MS)
    return {
        "scenario": "AT-07",
        "status": "OK",
        "wall_s": a["wall_s"] + b["wall_s"],
        "fixed_w_a": a,
        "fixed_w_b": b,
        "plastic_arms": "OMITTED (0.5.3 owns AT-07-R1..R3)",
    }


def run_at08() -> dict[str, Any]:
    """S8: two areas, repeated-input adaptation arm (toy: 2x2 neurons)."""
    out = _run_config(
        J.suite2_v1_v4_config(seed=TOY_SEED, n_per_area=2, duration_ms=10.0, dt_ms=TOY_DT_MS),
        10.0,
        TOY_DT_MS,
    )
    return {"scenario": "AT-08", **out}


def run_at09() -> dict[str, Any]:
    """S9: two areas, plastic coupling (toy: fixed coupling, 2x2 neurons).

    Cross-area plasticity W_12(t), W_21(t) arrives in 0.5.4; marked OMITTED.
    """
    out = _run_config(
        J.suite2_v1_v4_config(seed=TOY_SEED, n_per_area=2, duration_ms=10.0, dt_ms=TOY_DT_MS),
        10.0,
        TOY_DT_MS,
    )
    return {
        "scenario": "AT-09",
        **out,
        "plastic_coupling": "OMITTED (0.5.4 owns AT-09-R1..R3)",
    }


def run_at10() -> dict[str, Any]:
    """S10: multi-area synthesis pattern (toy: 3 areas x 2 neurons).

    The G_20 genome -> JDNA development at 20-area scale arrives in 0.5.5;
    this pass exercises the multi-area composition pattern at toy size and
    marks genome development OMITTED.
    """
    cfg = J.build_multi_area_columns(areas=["A1", "A2", "A3"], n_per_area=2)
    cfg = (
        cfg.runtime(seed=TOY_SEED, duration_ms=10.0, dt_ms=TOY_DT_MS)
        .set_emitter("izhikevich", "cortical_eig")
        .field(domain="laminar_column", conductivity="proxy")
        .probes(["spikes", "V_m"])
    )
    out = _run_config(cfg, 10.0, TOY_DT_MS)
    return {
        "scenario": "AT-10",
        **out,
        "genome_development": "OMITTED (0.5.5 owns AT-10-R1..R6)",
    }


_RUNNERS = {
    "AT-01": run_at01,
    "AT-02": run_at02,
    "AT-03": run_at03,
    "AT-04": run_at04,
    "AT-05": run_at05,
    "AT-06": run_at06,
    "AT-07": run_at07,
    "AT-08": run_at08,
    "AT-09": run_at09,
    "AT-10": run_at10,
}


def run_scenario(scenario_id: str) -> dict[str, Any]:
    """Run one toy scenario; never raises: failures become ERROR records."""
    if scenario_id not in _RUNNERS:
        raise KeyError(f"unknown scenario {scenario_id!r}; want one of {SCENARIOS}")
    t0 = time.perf_counter()
    try:
        out = _RUNNERS[scenario_id]()
    except Exception as exc:
        return {
            "scenario": scenario_id,
            "status": "ERROR",
            "reason": repr(exc),
            "wall_s": time.perf_counter() - t0,
        }
    wall_s = time.perf_counter() - t0
    out.setdefault("wall_s", wall_s)
    if out["wall_s"] > WALL_BUDGET_S:
        out["status"] = "OVER_BUDGET"
    return out


def run_all() -> dict[str, dict[str, Any]]:
    """Run every toy scenario in AT order; never raises per scenario."""
    return {sid: run_scenario(sid) for sid in SCENARIOS}
