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


# ---------------------------------------------------------------------------
# Item 10: AT-02 pair transmission + AT-03 E<->I pair (declared delay_ms).
# ---------------------------------------------------------------------------

# TFNE specs use one mechanism per rule in the 0.5.2 grammar; the
# bidirectional AT-03 rule is single-mechanism by grammar limitation (S27
# P_{l,c} is post-0.5.5), so E/I identity there is carried by cell type +
# sign metadata, and the limitation is declared in the record, not hidden.
AT02_SPEC_DELAY = (
    "O[k] := [direction = >; mechanism = AMPA; probability = 1.0; "
    "weight = 0.5; delay = {delay}]; "
    "A := [C = {{E}}; N = 4]; B := [C = {{E}}; N = 4]; x : A O[k] B : y"
)
AT03_SPEC = (
    "O[k] := [direction = <>; mechanism = AMPA; probability = 1.0; "
    "weight = 0.8; delay = {delay}]; "
    "A := [C = {{E}}; N = 3]; B := [C = {{I}}; N = 3]; x : A O[k] B : y"
)
# Absent-delay arm: no template braces (never formatted).
AT02_SPEC_ABSENT = (
    "O[k] := [direction = >; mechanism = AMPA; probability = 1.0; "
    "weight = 0.5]; "
    "A := [C = {E}; N = 4]; B := [C = {E}; N = 4]; x : A O[k] B : y"
)
AT02_DURATION_MS = 60.0
AT03_DURATION_MS = 200.0
AT02_N_CONTACTS = 4
SUPERPOSITION_TOL = 1e-3


def _tfne_pair_run(spec: str, duration_ms: float, dt_ms: float, seed: int = SEED) -> dict[str, Any]:
    """TFNE -> realize -> configure -> field/probes -> construct -> simulate.

    Delay declared in ms on the connection rule (0.5.2 decision 0b);
    configured ms and realized steps both recorded from the executed model.
    """
    t0 = time.perf_counter()
    tracemalloc.start()
    try:
        program = J.tfne.parse(spec)
        realization = J.tfne.realize(J.tfne.resolve(program), program, seed=seed)
        cfg = (
            J.tfne.to_configuration(realization, duration_ms=duration_ms, dt_ms=dt_ms)
            .field(domain="laminar_column", conductivity="proxy")
            .probe(
                name="e1",
                modes=["spikes", "V_m", "source", "LFP-proxy"],
                n_contacts=AT02_N_CONTACTS,
            )
        )
        model = J.construct(cfg)
        signals = model.simulate(J.simulation(duration_ms=duration_ms, dt_ms=dt_ms, seed=seed))
        edges = model.params["edge_list"]
        delay_steps = [int(v) for v in np.asarray(edges.delay_steps).ravel()]
    finally:
        _, peak_b = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return {
        "signals": signals,
        "neuron_table": model.neuron_table(),
        "delay_steps": delay_steps,
        "delay_storage": str(edges.delay_storage),
        "wall_s": time.perf_counter() - t0,
        "mem_peak_b": float(peak_b),
    }


def _field_decomposition(run: dict[str, Any]) -> dict[str, Any]:
    """Per-source vs superposed Phi from the executed kernel (proxy only)."""
    sig = run["signals"]
    K = np.asarray(sig.field.kernel, dtype=float)
    S = np.asarray(sig.sources, dtype=float)
    P = np.asarray(sig.field.lfp_proxy, dtype=float)
    recon = (K @ S.T).T
    err = float(np.abs(recon - P).max())
    # Per-source fields phi_i(t) = S[:, i] outer K[:, i].
    per = S[:, :, None] * K.T[None, :, :]
    return {
        "kernel_shape": list(K.shape),
        "reconstruction_max_err": err,
        "superposition_identity": bool(err <= SUPERPOSITION_TOL),
        "per_source": per,  # [T, N, C]
        "superposed": P,
        "level": LEVEL_PROXY,
    }


def _relative_centroids(
    neuron_table: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Area centroids in relative fractions (never mm/um)."""
    acc: dict[str, dict[str, Any]] = {}
    for row in neuron_table:
        a = acc.setdefault(str(row["area"]), {"x": [], "z": []})
        a["x"].append(float(row["x"]))
        a["z"].append(float(row["z"]))
    return {a: {"x": float(np.mean(v["x"])), "z": float(np.mean(v["z"]))} for a, v in acc.items()}


def run_at02() -> dict[str, Any]:
    """S2: driven pair N1->N2 with declared delay; fields + distance law.

    Arms: delay 2.0 ms vs 0.0 vs absent (zero-delay limit bit-identical).
    Distance law is a proxy observation at relative fractions, never a
    physical law (no calibration exists; AT-01-R5 OUT_OF_SCOPE).
    """
    t0 = time.perf_counter()
    arms = {
        "delayed": _tfne_pair_run(AT02_SPEC_DELAY.format(delay=DELAY_MS), AT02_DURATION_MS, DT_MS),
        "zero": _tfne_pair_run(AT02_SPEC_DELAY.format(delay=0.0), AT02_DURATION_MS, DT_MS),
        "absent": _tfne_pair_run(AT02_SPEC_ABSENT, AT02_DURATION_MS, DT_MS),
    }
    rec: dict[str, Any] = {}
    for name, run in arms.items():
        sig = run["signals"]
        spikes = np.asarray(sig.spikes)
        rec[name] = {
            "spikes_shape": list(spikes.shape),
            "n_spikes": int((spikes > 0).sum()),
            "delay_steps": run["delay_steps"],
            "delay_storage": run["delay_storage"],
            "wall_s": run["wall_s"],
        }
    v_absent = np.asarray(arms["absent"]["signals"].V_m)
    v_zero = np.asarray(arms["zero"]["signals"].V_m)
    s_absent = np.asarray(arms["absent"]["signals"].spikes)
    s_zero = np.asarray(arms["zero"]["signals"].spikes)
    v_delayed = np.asarray(arms["delayed"]["signals"].V_m)

    decomp = _field_decomposition(arms["delayed"])
    per = decomp["per_source"]
    P = decomp["superposed"]
    # Distance law (proxy): contact amplitude vs relative distance from the
    # B-area centroid in (x, z) fractions; no physical units claimed.
    centroids = _relative_centroids(arms["delayed"]["neuron_table"])
    contacts = np.asarray(arms["delayed"]["signals"].field.contact_depths, dtype=float).ravel()
    n_c = int(contacts.shape[0])
    # Contact lateral position is undeclared in the proxy (depths only), so
    # distance is measured in the declared depth fraction axis.
    dist = np.abs(contacts - centroids["B"]["z"])
    amp = np.abs(P).mean(axis=0)
    order = np.argsort(dist)
    falloff_monotone = bool(np.all(np.diff(amp[order]) <= 0))
    per_amp = np.abs(per).mean(axis=0)  # [N, C]
    indiv_total = float(per_amp.sum())
    super_amp = float(amp.sum())

    out = {
        "scenario": "AT-02",
        "status": "OK",
        "wall_s": time.perf_counter() - t0,
        "level": LEVEL_PROXY,
        "level_note": "all fields RELATIVE_PROXY; proxy != calibrated",
        "delay": {
            "configured_ms": DELAY_MS,
            "realized_steps": arms["delayed"]["delay_steps"],
            "rule": "delay_steps = round(delay_ms / dt_ms) per 0.5.2 decision 0b",
            "zero_limit_bit_identical": bool(
                np.array_equal(v_absent, v_zero) and np.array_equal(s_absent, s_zero)
            ),
            "delayed_differs_from_zero": bool(not np.array_equal(v_delayed, v_zero)),
        },
        "arms": rec,
        "fields": {
            "individual_vs_superposed": {
                "reconstruction_max_err": decomp["reconstruction_max_err"],
                "superposition_identity": decomp["superposition_identity"],
                "sum_individual_mean_abs": indiv_total,
                "superposed_mean_abs": super_amp,
            },
            "distance_law_proxy": {
                "centroids_relative_frac": centroids,
                "contacts_declared": AT02_N_CONTACTS,
                "contacts_realized": int(n_c),
                "contacts_note": (
                    "TFNE path realizes the field default (16) against "
                    "declared 4; realized count is read back from the "
                    "executed field, never assumed"
                ),
                "contact_depths_frac": [float(v) for v in contacts],
                "contact_mean_abs": [float(v) for v in amp],
                "depth_distance_frac": [float(v) for v in dist],
                "falloff_monotone_in_depth_frac": falloff_monotone,
                "note": (
                    "proxy observation at relative fractions only; no "
                    "physical distance law without calibration (AT-01-R5 out "
                    "of scope); contact lateral position undeclared in proxy"
                ),
            },
        },
    }
    if out["wall_s"] > WALL_BUDGET_S:
        out["status"] = "OVER_BUDGET"
    return out


def run_at03() -> dict[str, Any]:
    """S3: recurrent E<->I pair with declared delay; phase + field.

    Single-mechanism bidirectional TFNE rule (grammar limitation declared
    in-record); E/I identity from cell type. Cancellation is measured
    dynamically (anti-phase superposition < sum of amplitudes), never by
    relabeling.
    """
    t0 = time.perf_counter()
    run = _tfne_pair_run(AT03_SPEC.format(delay=DELAY_MS), AT03_DURATION_MS, DT_MS)
    sig = run["signals"]
    spikes = np.asarray(sig.spikes)
    table = run["neuron_table"]
    is_e = np.array([r["cell_type"] == "E" for r in table])
    spk_e = spikes[:, is_e]
    spk_i = spikes[:, ~is_e]
    rate_e = float((spk_e > 0).sum() / (spk_e.shape[0] * DT_MS / 1000.0))
    rate_i = float((spk_i > 0).sum() / (spk_i.shape[0] * DT_MS / 1000.0))

    decomp = _field_decomposition(run)
    per = decomp["per_source"]  # [T, N, C]
    P = decomp["superposed"]
    # Population fields per group from the executed kernel decomposition.
    phi_e = per[:, is_e, :].sum(axis=1)
    phi_i = per[:, ~is_e, :].sum(axis=1)
    f_dom = _dominant_freq_hz(P.mean(axis=1), DT_MS)
    phase_e = _phase_at_hz(phi_e.mean(axis=1), DT_MS, f_dom)
    phase_i = _phase_at_hz(phi_i.mean(axis=1), DT_MS, f_dom)
    lag = abs((phase_e - phase_i + np.pi) % (2 * np.pi) - np.pi)
    amp_sum = float(np.abs(phi_e).mean() + np.abs(phi_i).mean())
    amp_super = float(np.abs(P).mean())
    cancel_idx = 1.0 - amp_super / amp_sum if amp_sum > 0 else 0.0
    lo = _band_power(P.mean(axis=1), DT_MS, 4.0, 12.0)
    hi = _band_power(P.mean(axis=1), DT_MS, 30.0, 80.0)

    out = {
        "scenario": "AT-03",
        "status": "OK",
        "wall_s": time.perf_counter() - t0,
        "level": LEVEL_PROXY,
        "level_note": "all fields RELATIVE_PROXY; proxy != calibrated",
        "delay": {
            "configured_ms": DELAY_MS,
            "realized_steps": run["delay_steps"],
            "rule": "delay_steps = round(delay_ms / dt_ms) per 0.5.2 decision 0b",
        },
        "grammar_limitation": (
            "bidirectional rule carries one declared mechanism (TFNE 0.5.2 "
            "grammar; per-direction kinetics need S27 P, post-0.5.5); E/I "
            "identity from cell type; kinetics limitation declared here"
        ),
        "pair": {
            "n_spikes": int((spikes > 0).sum()),
            "rate_e_hz": rate_e,
            "rate_i_hz": rate_i,
            "kappa": float(J.kappa_synchrony(spikes, DT_MS)),
        },
        "oscillation": {
            "dominant_freq_hz": f_dom,
            "phase_lag_e_i_rad": float(lag),
            "cancellation_index": float(cancel_idx),
            "cancellation_note": (
                "dynamic anti-phase measure: 1 - |sum| / sum|.| from "
                "executed kernel decomposition; >0 cancels, <0 reinforces"
            ),
            "bandpower_4_12": lo,
            "bandpower_30_80": hi,
            "frequency_dependent_field": bool(lo != hi),
        },
        "superposition_identity": decomp["superposition_identity"],
        "reconstruction_max_err": decomp["reconstruction_max_err"],
    }
    if out["wall_s"] > WALL_BUDGET_S:
        out["status"] = "OVER_BUDGET"
    return out


# ---------------------------------------------------------------------------
# Item 11: AT-04 geometry/orientation arm (source orientation from item 3).
# ---------------------------------------------------------------------------

# Declared geometry is relative fractions in [0,1] (0.5.2 decision 0a);
# outside [0,1] is refused. Same seed across arms isolates the declared
# geometry effect from JDNA sampling noise.
AT04_SPEC = (
    "O[k] := [direction = <>; mechanism = AMPA; probability = 1.0; "
    "weight = 0.8; delay = {delay}]; "
    "A := [C = {{E}}; N = 3; G = [z0 = {a0}; z1 = {a1}]]; "
    "B := [C = {{I}}; N = 3; G = [z0 = {b0}; z1 = {b1}]]; "
    "x : A O[k] B : y"
)
AT04_ARMS = {
    "stacked": {"a0": 0.0, "a1": 0.5, "b0": 0.5, "b1": 1.0},
    "swapped": {"a0": 0.5, "a1": 1.0, "b0": 0.0, "b1": 0.5},
    "overlap": {"a0": 0.25, "a1": 0.75, "b0": 0.25, "b1": 0.75},
}
AT04_DURATION_MS = 200.0


def _refused_geometry() -> dict[str, Any]:
    """H5 adversarial: G outside [0,1] must refuse, never execute."""
    t0 = time.perf_counter()
    bad = AT04_SPEC.format(delay=DELAY_MS, a0=10.0, a1=20.0, b0=0.0, b1=0.5)
    try:
        program = J.tfne.parse(bad)
        realization = J.tfne.realize(J.tfne.resolve(program), program, seed=SEED)
        J.tfne.to_configuration(realization, duration_ms=10.0, dt_ms=DT_MS)
    except Exception as exc:
        return {
            "state": "REFUSED",
            "level": LEVEL_PROXY,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_s": time.perf_counter() - t0,
        }
    return {
        "state": "UNEXPECTED_EXECUTION",
        "level": LEVEL_PROXY,
        "reason": "G=[10,20] executed; the [0,1] refusal did not fire",
        "wall_s": time.perf_counter() - t0,
    }


def _realized_z_ranges(
    neuron_table: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    by_area: dict[str, list[float]] = {}
    for row in neuron_table:
        by_area.setdefault(str(row["area"]), []).append(float(row["z"]))
    for area, zs in by_area.items():
        out[area] = {"z_min": float(min(zs)), "z_max": float(max(zs))}
    return out


def run_at04() -> dict[str, Any]:
    """S4: geometry/orientation arm; correlation only, Phi->X refused.

    Three declared-geometry arms on one coupled pair (item-3 source
    orientation carried in the source representation Q). Phi->X feedback
    fails closed (AT-04-R3 OUT_OF_SCOPE); the H-perturbation causal arm
    belongs to 0.5.3 (AT-04-R2).
    """
    t0 = time.perf_counter()
    arms: dict[str, Any] = {}
    for name, g in AT04_ARMS.items():
        run = _tfne_pair_run(AT04_SPEC.format(delay=DELAY_MS, **g), AT04_DURATION_MS, DT_MS)
        sig = run["signals"]
        spikes = np.asarray(sig.spikes)
        decomp = _field_decomposition(run)
        per = decomp["per_source"]
        P = decomp["superposed"]
        table = run["neuron_table"]
        is_e = np.array([r["cell_type"] == "E" for r in table])
        phi_e = per[:, is_e, :].sum(axis=1).mean(axis=1)
        phi_i = per[:, ~is_e, :].sum(axis=1).mean(axis=1)
        denom = float(np.std(phi_e) * np.std(phi_i))
        align = float(np.corrcoef(phi_e, phi_i)[0, 1]) if denom > 0 else 0.0
        arms[name] = {
            "declared_G": dict(g),
            "realized_z_ranges": _realized_z_ranges(table),
            "rate_hz": _rate_hz(spikes.ravel(), DT_MS),
            "field_mean_abs": float(np.abs(P).mean()),
            "source_alignment_corr": float(align),
            "superposition_identity": decomp["superposition_identity"],
            "wall_s": run["wall_s"],
        }
    # X -> Phi correlation across arms (observed association only).
    rates = np.array([arms[n]["rate_hz"] for n in AT04_ARMS])
    famps = np.array([arms[n]["field_mean_abs"] for n in AT04_ARMS])
    x_phi_corr = (
        float(np.corrcoef(rates, famps)[0, 1]) if float(np.std(rates) * np.std(famps)) > 0 else 0.0
    )
    out = {
        "scenario": "AT-04",
        "status": "OK",
        "wall_s": time.perf_counter() - t0,
        "level": LEVEL_PROXY,
        "level_note": (
            "geometry in relative fractions; fields RELATIVE_PROXY; proxy != calibrated"
        ),
        "arms": arms,
        "x_to_phi_correlation_across_arms": float(x_phi_corr),
        "x_to_phi_note": (
            "correlation of executed rate vs executed field across "
            "geometry arms; correlation != causal field feedback"
        ),
        "phi_to_x": oos_phi_to_x(),
        "h_perturbation": {
            "state": "OMITTED",
            "reason": "AT-04-R2 causal H arm belongs to 0.5.3",
        },
        "geometry_refusal": _refused_geometry(),
    }
    if out["wall_s"] > WALL_BUDGET_S:
        out["status"] = "OVER_BUDGET"
    return out
