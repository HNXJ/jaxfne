"""Frozen Atlas measurement vector Y, schema v4 (0.5.5 ENGINE item 1).

Authority: artifacts/project_sources/8_atlas.md (Y vector; absent quantities
are OMITTED, refused capabilities are REFUSED, never synthesized). One
definition shared by every ``artifacts/atlas`` scenario; v0-v3 lived as
per-script copies whose IMPLEMENTED cells came from static tables and
carried only a status note.

v4 keeps the v3 names and order. A cell is IMPLEMENTED only when the run
produced a value for it, and the cell carries that value: ``measure_v4``
reads each scenario's own returned dict (the extractors below name the
exact keys) and ``record`` derives every state from what was found.
Stdlib only; the scenario scripts stay the runners.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

SCHEMA_VERSION = "v4"

# The 13 stable names (v0 order), then area-indexed cells, then cross-area cells.
Y_STABLE: tuple[str, ...] = (
    "X",
    "H",
    "W",
    "Q",
    "Phi_E",
    "Phi_B",
    "SPK",
    "PSD",
    "C",
    "phi",
    "E_reduction",
    "T_compute",
    "M_compute",
)
Y_AREA: tuple[str, ...] = ("SPK_A1", "SPK_A2", "H_A1", "H_A2", "Phi_A1", "Phi_A2")
Y_CROSS: tuple[str, ...] = ("W_12", "W_21", "C_12", "dphi_12")
Y_KEYS_V4: tuple[str, ...] = Y_STABLE + Y_AREA + Y_CROSS

IMPLEMENTED = "IMPLEMENTED"
OMITTED = "OMITTED"
REFUSED = "REFUSED"
CELL_STATES: tuple[str, ...] = (IMPLEMENTED, OMITTED, REFUSED)

RELATIVE_PROXY = "RELATIVE_PROXY"
LEVELS: tuple[str, ...] = (RELATIVE_PROXY, "CALIBRATED", "EXACT")

CELL_FIELDS: tuple[str, ...] = ("state", "value", "level", "note")

NOTES: dict[str, str] = {
    "X": "rate / Vm features of the executed trajectories",
    "H": "final relative state H (mean), per arm",
    "W": "max |W_final - W0| over all edges, per arm",
    "Q": "mean |source| per neuron (the representation every probe consumes)",
    "Phi_E": "field observation at RELATIVE_PROXY",
    "Phi_B": "no calibrated Phi_B beyond the proxy",
    "SPK": "executed spike counts",
    "PSD": "band power / dominant frequency of the executed field",
    "C": "synchrony (kappa), alignment or locality operators",
    "phi": "phase lag of executed group fields",
    "E_reduction": "reduction verdicts against predeclared tolerances",
    "T_compute": "measured wall seconds",
    "M_compute": "measured host peak bytes (tracemalloc; not device memory)",
    "SPK_A1": "A1 spike count",
    "SPK_A2": "A2 spike count",
    "H_A1": "final H mean over A1",
    "H_A2": "final H mean over A2",
    "Phi_A1": "mean |source| over A1 (source-level field proxy)",
    "Phi_A2": "mean |source| over A2 (source-level field proxy)",
    "W_12": "max |dW| over the A1->A2 cross range",
    "W_21": "max |dW| over the A2->A1 cross range",
    "C_12": "mean magnitude-squared coherence 8-25 Hz, per-area rates",
    "dphi_12": "mean cross-spectrum phase 8-25 Hz, per-area rates",
}
PHI_B_REASON = "no calibrated Phi_B beyond the proxy (candidates AT-01-R4, AT-07-R4)"


def _finite_json(value: Any) -> bool:
    """True when ``value`` is JSON-shaped with finite floats."""
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(isinstance(k, str) and _finite_json(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(_finite_json(v) for v in value)
    return False


def pick(source: Mapping[str, Any] | None, *keys: str) -> dict[str, Any] | None:
    """``{k: source[k]}`` for the keys the run produced (non-None); None if none."""
    if not source:
        return None
    out = {k: source[k] for k in keys if source.get(k) is not None}
    return out or None


def per_arm(arms: Mapping[str, Mapping[str, Any]], *keys: str) -> dict[str, Any] | None:
    """Per-arm values: one key gives ``{arm: value}``, several give ``{arm: {key: value}}``.

    Arms that produced none of the keys are left out; None if no arm did.
    """
    out: dict[str, Any] = {}
    for name, arm in arms.items():
        if len(keys) == 1:
            if arm.get(keys[0]) is not None:
                out[name] = arm[keys[0]]
        else:
            got = pick(arm, *keys)
            if got:
                out[name] = got
    return out or None


def record(
    values: Mapping[str, Any],
    refused: Mapping[str, str],
    notes: Mapping[str, str] | None = None,
    levels: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build a v4 record over every name in ``Y_KEYS_V4``.

    ``values`` holds what the run produced (None or absent = not produced);
    ``refused`` maps refused names to the reason. A name is REFUSED if listed
    there, IMPLEMENTED if it has a value, OMITTED otherwise. Levels default to
    RELATIVE_PROXY. Unknown names, a value for a refused name, or a value that
    is not finite JSON raise ValueError.
    """
    notes = NOTES if notes is None else notes
    levels = levels or {}
    unknown = (set(values) | set(refused) | set(levels)) - set(Y_KEYS_V4)
    if unknown:
        raise ValueError(f"names outside schema {SCHEMA_VERSION}: {sorted(unknown)}")
    out: dict[str, dict[str, Any]] = {}
    for key in Y_KEYS_V4:
        value = values.get(key)
        level = levels.get(key, RELATIVE_PROXY)
        if level not in LEVELS:
            raise ValueError(f"{key}: bad level {level!r}")
        if key in refused:
            if value is not None:
                raise ValueError(f"{key}: refused but a value was supplied")
            out[key] = {"state": REFUSED, "value": None, "level": level, "note": refused[key]}
        elif value is not None:
            if not _finite_json(value):
                raise ValueError(f"{key}: value is not finite JSON: {value!r}")
            out[key] = {
                "state": IMPLEMENTED,
                "value": value,
                "level": level,
                "note": notes.get(key, ""),
            }
        else:
            out[key] = {"state": OMITTED, "value": None, "level": level, "note": notes.get(key, "")}
    return out


def validate(rec: Mapping[str, Any]) -> list[str]:
    """Errors for a record that breaks the v4 contract; empty means valid."""
    errors: list[str] = []
    if tuple(rec) != Y_KEYS_V4:
        errors.append(f"keys must be Y_KEYS_V4 in order, got {list(rec)}")
    for key, cell in rec.items():
        if not isinstance(cell, Mapping) or tuple(cell) != CELL_FIELDS:
            errors.append(f"{key}: cell fields must be {CELL_FIELDS}")
            continue
        state, value = cell["state"], cell["value"]
        if state not in CELL_STATES:
            errors.append(f"{key}: bad state {state!r}")
        if cell["level"] not in LEVELS:
            errors.append(f"{key}: bad level {cell['level']!r}")
        if state == IMPLEMENTED and (value is None or not _finite_json(value)):
            errors.append(f"{key}: IMPLEMENTED needs a finite JSON value")
        if state != IMPLEMENTED and value is not None:
            errors.append(f"{key}: {state} must carry value None")
    return errors


def gap_row(rec: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """Cell states of one record (one column of the Y x AT gap matrix)."""
    return {key: cell["state"] for key, cell in rec.items()}


# ---------------------------------------------------------------------------
# Extractors: scenario raw dict -> produced values. Each reads only keys the
# scenario's runner writes (at01_at06_052, at07_at04_053, at08_at09_054,
# at01_at10_toy). A key the runner did not write stays absent -> OMITTED.
# ---------------------------------------------------------------------------


def _verdicts(verdicts: Mapping[str, Any] | None) -> dict[str, Any] | None:
    got = {
        k: dict(v) for k, v in (verdicts or {}).items() if isinstance(v, Mapping) and "pass" in v
    }
    return got or None


def _merge(*parts: dict[str, Any] | None) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    for part in parts:
        out.update(part or {})
    return out or None


def _at01(raw: Mapping[str, Any]) -> dict[str, Any]:
    arms = {"reduced": raw.get("reduced") or {}}
    hh = raw.get("hh_reference") or {}
    if hh.get("status") == "OK":
        arms["hh"] = hh
    return {
        "X": per_arm(arms, "v_rest_mv", "v_peak_mv", "v_min_mv", "rate_hz", "first_spike_ms"),
        "SPK": per_arm(arms, "n_spikes"),
        "E_reduction": _verdicts(raw.get("verdicts")),
        "M_compute": arms["reduced"].get("mem_peak_b"),
    }


def _retained(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """X, SPK, Q, M_compute from the per-arm keys ``at01_at06_052._retained`` writes."""
    return {
        "X": per_arm(arms, "v_mean_mv", "v_peak_mv"),
        "SPK": per_arm(arms, "n_spikes"),
        "Q": per_arm(arms, "Q_mean_abs_per_neuron"),
        "M_compute": per_arm(arms, "mem_peak_b"),
    }


def _at02(raw: Mapping[str, Any]) -> dict[str, Any]:
    fields = raw.get("fields") or {}
    return {
        **_retained(raw.get("arms") or {}),
        "Phi_E": _merge(
            pick(
                fields.get("individual_vs_superposed"),
                "superposed_mean_abs",
                "reconstruction_max_err",
            ),
            pick(fields.get("distance_law_proxy"), "contact_mean_abs"),
        ),
    }


def _at03(raw: Mapping[str, Any]) -> dict[str, Any]:
    pair = raw.get("pair") or {}
    osc = raw.get("oscillation") or {}
    return {
        "X": pick(pair, "rate_e_hz", "rate_i_hz", "v_mean_mv", "v_peak_mv"),
        "SPK": pair.get("n_spikes"),
        "Q": pair.get("Q_mean_abs_per_neuron"),
        "M_compute": pair.get("mem_peak_b"),
        "Phi_E": _merge(pick(raw, "reconstruction_max_err"), pick(osc, "cancellation_index")),
        "PSD": pick(osc, "dominant_freq_hz", "bandpower_4_12", "bandpower_30_80"),
        "C": pair.get("kappa"),
        "phi": osc.get("phase_lag_e_i_rad"),
    }


def _at04_geometry(raw: Mapping[str, Any]) -> dict[str, Any]:
    arms = raw.get("arms") or {}
    corr = per_arm(arms, "source_alignment_corr")
    across = raw.get("x_to_phi_correlation_across_arms")
    return {
        **_retained(arms),
        "X": per_arm(arms, "rate_hz", "v_mean_mv", "v_peak_mv"),
        "Phi_E": per_arm(arms, "field_mean_abs"),
        "C": _merge(
            {"source_alignment_corr": corr} if corr else None,
            {"x_to_phi_across_arms": across} if across is not None else None,
        ),
    }


def _at05(raw: Mapping[str, Any]) -> dict[str, Any]:
    arms = raw.get("arms") or {}
    return {
        **_retained(arms),
        "Phi_E": per_arm(arms, "a_phi_mid_contact"),
        "C": per_arm(arms, "rho_sync_executed"),
    }


def _at06(raw: Mapping[str, Any]) -> dict[str, Any]:
    loc = raw.get("locality_c_r_f") or {}
    c = {k: v["c_mean"] for k, v in loc.items() if isinstance(v, Mapping) and "c_mean" in v}
    pop = raw.get("population") or {}
    return {
        "X": pick(pop, "v_mean_mv", "v_peak_mv"),
        "SPK": pop.get("n_spikes"),
        "Q": pop.get("Q_mean_abs_per_neuron"),
        "M_compute": pop.get("mem_peak_b"),
        "C": c or None,
    }


def _reduction(raw: Mapping[str, Any]) -> dict[str, Any]:
    rungs = {"single": raw.get("single") or {}, "population": raw.get("population") or {}}
    x = {
        "single": pick(rungs["single"], "rate_hz", "v_peak_mv"),
        "population": pick(rungs["population"], "rate_hz_per_neuron"),
    }
    return {
        "X": {k: v for k, v in x.items() if v} or None,
        "Q": per_arm(rungs, "mean_abs_source_per_neuron"),
        "E_reduction": _verdicts(raw.get("verdicts")),
    }


def _single_area_arms(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "X": per_arm(arms, "rate_hz_per_neuron"),
        "SPK": per_arm(arms, "spike_count"),
        "H": per_arm(arms, "H_final_mean"),
        "W": per_arm(arms, "w_final_max_abs_change"),
        "Q": per_arm(arms, "Q_mean_abs_per_neuron"),
        "Phi_E": per_arm(arms, "Phi_mean_abs"),
        "C": per_arm(arms, "kappa"),
        "M_compute": per_arm(arms, "mem_peak_b"),
    }


def _at07(raw: Mapping[str, Any]) -> dict[str, Any]:
    return _single_area_arms(raw.get("arms") or {})


def _at04r2(raw: Mapping[str, Any]) -> dict[str, Any]:
    arms = {k: raw[k] for k in ("baseline", "perturbed") if isinstance(raw.get(k), Mapping)}
    return _single_area_arms(arms)


def _two_area(raw: Mapping[str, Any]) -> dict[str, Any]:
    arms = raw.get("arms") or {}
    return {
        "X": per_arm(arms, "rate_hz_A1", "rate_hz_A2"),
        "SPK": per_arm(arms, "spike_count_A1", "spike_count_A2"),
        "H": per_arm(arms, "H_final_mean_A1", "H_final_mean_A2"),
        "W": per_arm(arms, "w_max_abs_change"),
        "Q": per_arm(arms, "Q_mean_abs_A1", "Q_mean_abs_A2"),
        "Phi_E": per_arm(arms, "Phi_mean_abs"),
        "C": per_arm(arms, "kappa"),
        "M_compute": per_arm(arms, "mem_peak_b"),
        "SPK_A1": per_arm(arms, "spike_count_A1"),
        "SPK_A2": per_arm(arms, "spike_count_A2"),
        "H_A1": per_arm(arms, "H_final_mean_A1"),
        "H_A2": per_arm(arms, "H_final_mean_A2"),
        "Phi_A1": per_arm(arms, "Phi_A1_proxy"),
        "Phi_A2": per_arm(arms, "Phi_A2_proxy"),
        "W_12": per_arm(arms, "w_12_max_abs_change"),
        "W_21": per_arm(arms, "w_21_max_abs_change"),
        "C_12": per_arm(arms, "C_12_band_mean"),
        "dphi_12": per_arm(arms, "dphi_12_band_mean"),
    }


def _at10_toy(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {"SPK": (raw.get("spikes") or {}).get("n_spikes")}


# Keyed by the ``scenario`` field each runner writes into its raw dict.
EXTRACTORS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "AT-01": _at01,
    "AT-02": _at02,
    "AT-03": _at03,
    "AT-04": _at04_geometry,
    "AT-04R2": _at04r2,
    "AT-05": _at05,
    "AT-06": _at06,
    "REDUCTION": _reduction,
    "AT-07": _at07,
    "AT-08": _two_area,
    "AT-09": _two_area,
    "AT-10": _at10_toy,
}


def measure_v4(raw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """v4 record for one scenario run, keyed by the runner's ``scenario`` field.

    A run whose status is ERROR yields only T_compute. Phi_B is REFUSED for
    every scenario until a candidate path promotes it.
    """
    sid = raw.get("scenario")
    if sid not in EXTRACTORS:
        raise KeyError(f"no v4 extractor for scenario {sid!r}")
    values = {} if raw.get("status") == "ERROR" else EXTRACTORS[sid](raw)
    values["T_compute"] = raw.get("wall_s")
    reason = next(
        (
            raw[k]["reason"]
            for k in ("b_beyond_proxy", "phi_b")
            if isinstance(raw.get(k), Mapping) and raw[k].get("reason")
        ),
        PHI_B_REASON,
    )
    return record(values, refused={"Phi_B": reason})
