"""AT-01..AT-06 second pass (0.5.2 ATLAS items 9-13).

Authority: artifacts/project_sources/8_atlas.md (S1..S6 == AT-01..AT-06;
inheritance; Y vector; OMITTED/REFUSED never synthesized; B = magnetic
field; proxy != calibrated; correlation != causal feedback).
Extends the 0.5.1 toy-pass pattern (artifacts/atlas/at01_at10_toy.py):
same firewall, same sentinels, same Y names (v0 names stable in v1).

Consumer/tests of the engine, not an authority over it: scenarios are
data over the current public surface only (firewall:
tests/test_atlas_firewall.py audits this file). No new engine
capabilities; no synthesized quantities.

HUMAN DECISION (0.5.2, implement exactly): coverage rows AT-01-R4 (B
beyond proxy), AT-01-R5 (explicit calibration), AT-04-R3 (Phi->X
feedback) are OUT_OF_SCOPE -- no independent evidence exists yet.
Scenarios FAIL CLOSED where these would apply: AT-01 records no B
beyond the current MEG proxy and no calibrated quantities; AT-04 keeps
Phi->X out (H-perturbation arm belongs to 0.5.3).

INVARIANTS: relative geometry only -- toy/AT fixes use [0,1] fractions,
never mm/um; no conductivity-dependent amplitude, no physical distance
laws without calibration (AT-06 declares assumptions, computes proxy
C(R,f) only); proxy != calibrated in every output; wall-time budget
300 s per scenario (over -> shrink + record, never force).

Import rule (enforced by tests/test_atlas_firewall.py): this file
imports only the top-level ``jaxfne`` package plus stdlib / numpy. No
``jaxfne.<submodule>`` imports (attribute access ``J.tfne`` is use of
the root-namespace symbol, not a submodule import).
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Any

import numpy as np

import jaxfne as J

# ---------------------------------------------------------------------------
# Predeclared constants (declared BEFORE any run; tolerances need human
# authorization to change -- STOP, do not tune after observing results).
# ---------------------------------------------------------------------------

SEED = 7
DT_MS = 0.5
WALL_BUDGET_S = 300.0

# 0.5.2 decision 0b: delay declared in ms; realized as
# delay_steps = round(delay_ms / dt_ms). Configured ms and realized steps
# both recorded. DT 0.5 ms x DELAY 2.0 ms -> 4 steps (never rounds to 0).
DELAY_MS = 2.0

# Item 9 (AT-01 physical anchor) tolerances, declared before the run.
AT01_DURATION_MS = 50.0
AT01_HH_CURRENT_UA = 10.0
AT01_SPIKE_TIME_TOL_MS = 2.0
AT01_RATE_TOL_HZ = 10.0
AT01_V_PEAK_TOL_MV = 20.0
AT01_V_REST_TOL_MV = 10.0
AT01_SPIKE_THRESHOLD_MV = -20.0

# Item 13 (reduction row) tolerances, declared before the run.
REDUCTION_RATE_TOL_HZ = 10.0
REDUCTION_V_PEAK_TOL_MV = 20.0
REDUCTION_FIELD_TOL_FRAC = 0.75

# Relative geometry: fractions in [0,1] only. Never mm/um.
GEO_LO_FRAC = 0.25
GEO_HI_FRAC = 0.75

SCENARIOS: tuple[str, ...] = ("AT-01", "AT-02", "AT-03", "AT-04", "AT-05", "AT-06")

LEVEL_PROXY = "RELATIVE_PROXY"
LEVEL_REDUCED = "REDUCED_PHYSICAL"
LEVEL_CALIBRATED = "CALIBRATED"


# ---------------------------------------------------------------------------
# Sentinels (same contract as the 0.5.1 toy pass).
# ---------------------------------------------------------------------------


class _Omitted:
    _instance = None

    def __new__(cls) -> "_Omitted":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover
        return "OMITTED"

    def __bool__(self) -> bool:
        return False


class _Refused:
    _instance = None

    def __new__(cls) -> "_Refused":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover
        return "REFUSED"

    def __bool__(self) -> bool:
        return False


OMITTED = _Omitted()
REFUSED = _Refused()

# Schema v1 (item 13) keeps every v0 name stable; see Y_KEYS_V1 below.
Y_KEYS: tuple[str, ...] = (
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


# ---------------------------------------------------------------------------
# Shared execution helpers (toy sizes; never force past budget).
# ---------------------------------------------------------------------------


def _run_configuration(
    cfg: Any, duration_ms: float, dt_ms: float, seed: int = SEED
) -> dict[str, Any]:
    """Construct + simulate a public Configuration; raw outcome + cost."""
    t0 = time.perf_counter()
    tracemalloc.start()
    try:
        model = J.construct(cfg)
        signals = model.simulate(J.simulation(duration_ms=duration_ms, dt_ms=dt_ms, seed=seed))
    finally:
        _, peak_b = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    wall_s = time.perf_counter() - t0
    return {
        "model": model,
        "signals": signals,
        "wall_s": wall_s,
        "mem_peak_b": float(peak_b),
    }


def _spike_times(spikes_1d: np.ndarray, dt_ms: float) -> np.ndarray:
    """Step indices where a binary spike train is positive, in ms."""
    idx = np.flatnonzero(np.asarray(spikes_1d) > 0)
    return idx.astype(float) * float(dt_ms)


def _rate_hz(spikes_1d: np.ndarray, dt_ms: float) -> float:
    n = int((np.asarray(spikes_1d) > 0).sum())
    dur_s = (np.asarray(spikes_1d).shape[0] * float(dt_ms)) / 1000.0
    return float(n) / dur_s if dur_s > 0 else 0.0


def _vm_features(v_1d: np.ndarray) -> dict[str, float]:
    v = np.asarray(v_1d, dtype=float)
    n = v.shape[0]
    rest = v[: max(1, n // 10)]
    return {
        "v_rest_mv": float(np.median(rest)),
        "v_peak_mv": float(np.max(v)),
        "v_min_mv": float(np.min(v)),
    }


def _hh_spike_times(v: np.ndarray, dt_ms: float) -> np.ndarray:
    """Upward threshold crossings on an HH Vm trace (mV)."""
    v = np.asarray(v, dtype=float)
    above = v > AT01_SPIKE_THRESHOLD_MV
    crossings = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    return crossings.astype(float) * float(dt_ms)


def _band_power(x: np.ndarray, dt_ms: float, lo_hz: float, hi_hz: float) -> float:
    """Two-sided-band power of a real series via rfft (proxy spectra)."""
    x = np.asarray(x, dtype=float).ravel() - float(np.mean(x))
    n = x.shape[0]
    if n < 4:
        return 0.0
    freqs = np.fft.rfftfreq(n, d=float(dt_ms) / 1000.0)
    psd = np.abs(np.fft.rfft(x)) ** 2 / n
    band = (freqs >= lo_hz) & (freqs < hi_hz)
    return float(psd[band].sum()) if band.any() else 0.0


def _dominant_freq_hz(x: np.ndarray, dt_ms: float) -> float:
    x = np.asarray(x, dtype=float).ravel() - float(np.mean(x))
    n = x.shape[0]
    if n < 4:
        return 0.0
    freqs = np.fft.rfftfreq(n, d=float(dt_ms) / 1000.0)
    psd = np.abs(np.fft.rfft(x)) ** 2
    psd[0] = 0.0
    return float(freqs[int(np.argmax(psd))])


def _phase_at_hz(x: np.ndarray, dt_ms: float, f_hz: float) -> float:
    """Phase (radians) of a series at the fft bin nearest f_hz."""
    x = np.asarray(x, dtype=float).ravel() - float(np.mean(x))
    n = x.shape[0]
    freqs = np.fft.rfftfreq(n, d=float(dt_ms) / 1000.0)
    k = int(np.argmin(np.abs(freqs - f_hz)))
    return float(np.angle(np.fft.rfft(x)[k]))


# ---------------------------------------------------------------------------
# OUT_OF_SCOPE fail-closed records (human decision, 0.5.2).
# ---------------------------------------------------------------------------


def oos_b_beyond_proxy() -> dict[str, Any]:
    """AT-01-R4: no B beyond the current MEG proxy -- REFUSED, never a value."""
    return {
        "state": "REFUSED",
        "level": LEVEL_PROXY,
        "reason": (
            "AT-01-R4 OUT_OF_SCOPE for 0.5.2: no independent evidence for "
            "Phi_B beyond the current MEG proxy; human authorization required"
        ),
    }


def oos_calibration() -> dict[str, Any]:
    """AT-01-R5: no explicit calibration transform -- REFUSED."""
    return {
        "state": "REFUSED",
        "level": LEVEL_PROXY,
        "reason": (
            "AT-01-R5 OUT_OF_SCOPE for 0.5.2: no explicit calibration "
            "transform (units, conductivity, distance); quantities stay "
            "RELATIVE_PROXY, never relabeled CALIBRATED"
        ),
    }


def oos_phi_to_x() -> dict[str, Any]:
    """AT-04-R3: Phi->X feedback -- REFUSED; correlation arm only."""
    return {
        "state": "REFUSED",
        "level": LEVEL_PROXY,
        "reason": (
            "AT-04-R3 OUT_OF_SCOPE for 0.5.2: Phi->X feedback has no "
            "independent evidence; geometry arm runs X->Phi correlation "
            "only; H-perturbation arm belongs to 0.5.3"
        ),
    }


# ---------------------------------------------------------------------------
# Item 9: AT-01 physical anchor.
# ---------------------------------------------------------------------------


def _hh_reference() -> dict[str, Any]:
    """Jaxley HH reference trace; REFUSED (never a substitute) if absent."""
    t0 = time.perf_counter()
    try:
        t, v, i_inj = J.hh_jaxley_reference_trace(
            duration_ms=AT01_DURATION_MS,
            dt_ms=DT_MS,
            current_amplitude=AT01_HH_CURRENT_UA,
        )
    except ImportError:
        return {
            "status": "REFUSED",
            "level": LEVEL_PROXY,
            "reason": "Jaxley absent; no substitute per 0.5.2 item 9",
            "wall_s": time.perf_counter() - t0,
        }
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    spikes = _hh_spike_times(v, DT_MS)
    feats = _vm_features(v)
    return {
        "status": "OK",
        # Physical units via the Jaxley bridge: reduced-physical, explicitly
        # NOT JaxFNE-calibrated (AT-01-R5 stays REFUSED).
        "level": LEVEL_REDUCED,
        "level_note": "mV/ms via Jaxley bridge; not a JaxFNE calibration",
        "n_steps": int(t.shape[0]),
        "rate_hz": float(spikes.shape[0] / (AT01_DURATION_MS / 1000.0)),
        "first_spike_ms": None if spikes.shape[0] == 0 else float(spikes[0]),
        "n_spikes": int(spikes.shape[0]),
        **feats,
        "wall_s": time.perf_counter() - t0,
    }


def run_at01() -> dict[str, Any]:
    """S1: Jaxley HH reference vs reduced neuron, predeclared tolerances.

    Records no B beyond the MEG proxy and no calibrated quantities
    (fail-closed per the 0.5.2 human decision); states the
    no-full-electrodiffusion boundary (AT-01-R7).
    """
    t0 = time.perf_counter()
    hh = _hh_reference()

    red = _run_configuration(
        J.suite2_single_neuron_config(seed=SEED, duration_ms=AT01_DURATION_MS, dt_ms=DT_MS),
        AT01_DURATION_MS,
        DT_MS,
    )
    sig = red["signals"]
    spikes = np.asarray(sig.spikes).ravel()
    vm = np.asarray(sig.V_m).ravel()
    r_times = _spike_times(spikes, DT_MS)
    r_feats = _vm_features(vm)
    reduced = {
        "status": "OK",
        "level": LEVEL_PROXY,
        "level_note": "reduced emitter in native proxy units; proxy != calibrated",
        "n_steps": int(spikes.shape[0]),
        "rate_hz": _rate_hz(spikes, DT_MS),
        "first_spike_ms": None if r_times.shape[0] == 0 else float(r_times[0]),
        "n_spikes": int((spikes > 0).sum()),
        **r_feats,
        "wall_s": red["wall_s"],
        "mem_peak_b": red["mem_peak_b"],
    }

    verdicts: dict[str, dict[str, Any]] = {}
    if hh["status"] == "OK":
        verdicts["rate"] = {
            "diff_hz": abs(hh["rate_hz"] - reduced["rate_hz"]),
            "tolerance_hz": AT01_RATE_TOL_HZ,
            "pass": bool(abs(hh["rate_hz"] - reduced["rate_hz"]) <= AT01_RATE_TOL_HZ),
        }
        verdicts["v_peak"] = {
            "diff_mv": abs(hh["v_peak_mv"] - reduced["v_peak_mv"]),
            "tolerance_mv": AT01_V_PEAK_TOL_MV,
            "pass": bool(abs(hh["v_peak_mv"] - reduced["v_peak_mv"]) <= AT01_V_PEAK_TOL_MV),
        }
        verdicts["v_rest"] = {
            "diff_mv": abs(hh["v_rest_mv"] - reduced["v_rest_mv"]),
            "tolerance_mv": AT01_V_REST_TOL_MV,
            "pass": bool(abs(hh["v_rest_mv"] - reduced["v_rest_mv"]) <= AT01_V_REST_TOL_MV),
        }
        if hh["first_spike_ms"] is None or reduced["first_spike_ms"] is None:
            verdicts["spike_time"] = {
                "state": "OMITTED",
                "reason": "first spike absent on at least one arm; no time to compare",
                "tolerance_ms": AT01_SPIKE_TIME_TOL_MS,
            }
        else:
            diff = abs(hh["first_spike_ms"] - reduced["first_spike_ms"])
            verdicts["spike_time"] = {
                "diff_ms": diff,
                "tolerance_ms": AT01_SPIKE_TIME_TOL_MS,
                "pass": bool(diff <= AT01_SPIKE_TIME_TOL_MS),
            }
    else:  # HH refused -> no comparison to verdict; absence recorded, never filled.
        verdicts["comparison"] = {
            "state": "REFUSED",
            "reason": hh.get("reason", "HH reference refused"),
        }

    out = {
        "scenario": "AT-01",
        "status": "OK" if hh["status"] == "OK" else "OK_REFUSED_HH",
        "wall_s": time.perf_counter() - t0,
        "hh_reference": hh,
        "reduced": reduced,
        "verdicts": verdicts,
        "verdict_tolerances_predeclared": {
            "spike_time_ms": AT01_SPIKE_TIME_TOL_MS,
            "rate_hz": AT01_RATE_TOL_HZ,
            "v_peak_mv": AT01_V_PEAK_TOL_MV,
            "v_rest_mv": AT01_V_REST_TOL_MV,
        },
        "b_beyond_proxy": oos_b_beyond_proxy(),
        "calibrated": oos_calibration(),
        "boundary": (
            "AT-01-R7: reduced emitters are not a claim to solve full "
            "electrodiffusion; HH arm is the Jaxley reference, reduced arm "
            "is RELATIVE_PROXY"
        ),
    }
    if out["wall_s"] > WALL_BUDGET_S:
        out["status"] = "OVER_BUDGET"
    return out
