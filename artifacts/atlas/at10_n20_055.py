"""AT-10 on N_20 (0.5.5 ATLAS 6): three phases and the stabilization assay.

N_20 = develop(G_20 v2, K_D = 20): 20 areas x 50 neurons on a declared 1-D
hierarchy (``g20_genome``). Every run uses one realized network (one W0),
the same drive, the same Poisson background and the same pulse train into H01.

Phases (human decision 2026-09-26: 10 s simulated per phase), each a
separate run from W0 (at the time, full-state continuation did not carry
``poisson_drive``; R6 below uses per-chunk continuation):

- ``baseline``: HDP off.
- ``hebbian_hdp``: HDP on, ``HP_HEBB``.
- ``noisy_hdp``: HDP on, ``HP_HEBB`` with ``noise_scale`` 0.5.

Assay (AT-04-R2 pattern): matched stimulation; perturbations are H0 = 0 and
a W kick w0 = k * W0; each runs in an HDP-engaged arm (``HP_HEBB``) and an
HDP-disabled arm (plasticity gains zeroed, H still evolves). Deviation is
|r(t) - r_ref(t)| per 1 s window against the unperturbed engaged run.
Stabilization is declared, before the assay seed ran, as: the disabled arm
stays off (late deviation >= ``MIN_DISABLED_DEV_HZ``) and the engaged arm
returns (late deviation <= ``RETURN_RATIO`` x the disabled one).

Regime and assay constants were fixed before the assay ran: the regime by
the 0.5.5 sweep (todo stack ATLAS 6), the HDP gains by 10 s stability
probes, the kicks by a pilot on ``PILOT_SEED``. Observed after the run: the
Simulation seed does not enter these runs (the only stochastic input is
the Poisson background, seeded by ``NOISE["seed"]``), so the pilot saw the
same trajectories and was not out of sample. The out-of-sample check is
the replicate assay on ``REPLICATE_NOISE_SEEDS``, declared after the
primary result with the criteria unchanged.
Selection status: the cross gain (G_20 v2) was chosen by the sweep because
it propagates, and the HDP gains and kick sizes were chosen on this same
regime and noise realization. The propagation statistics and the assay
verdicts are therefore descriptive results of a selected regime, not
confirmatory tests of a hypothesis fixed in advance; the noise-seed
replicates show robustness to the Poisson realization only (same gain,
same kicks, same network).
Values are relative (RELATIVE_PROXY); nothing is calibrated.
Import rule: top-level ``jaxfne`` only.

R6 (0.5.5, very long T): ``run_r6_long`` continues ``total_ms`` in
``chunk_ms`` full-state chunks with per-chunk Poisson noise (chunk 0 draws
the declared seed, later chunks derive theirs from (seed, chunk_index) in
the engine). Per-chunk records keep per-area rates and H summaries only;
each chunk's H trace is reduced before the next chunk runs and never
stored.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import math
import time
from typing import Any

import numpy as np

import jaxfne as J

from artifacts.atlas import g20_genome as G

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

DT_MS = 0.5
PHASE_MS = 10000.0
RUN_SEED = 7
PILOT_SEED = 101  # kick-size pilot only; never the assay seed
BUILD_SEED = 5  # tensor -> configuration realization seed
DRIVE = {"E": 3.0, "PV": 2.0}
NOISE = {"rate_hz": 200.0, "amplitude": 4.0, "target": "all", "seed": 11}
REPLICATE_NOISE_SEEDS = (12, 13)
STIM_PERIOD_MS, STIM_DUR_MS, STIM_AMP, STIM_FIRST_MS = 200.0, 20.0, 8.0, 100.0
HP_HEBB = {"K_HDP": 0.005, "K_ctrl": 0.15, "K_w_ctrl": 0.05, "alpha": 0.05,
           "tau_0_ms": 5.0, "noise_scale": 0.0}
HP_NOISY = {**HP_HEBB, "noise_scale": 0.5}
# H-space bounds in force for the assay (AT-10-R5): HP_HEBB carries no
# H_min/H_max/w_ceiling keys, so the shared kernel-contract defaults apply
# (jaxfne/_model_simulate.py::_hdp_kernel_kwargs: H_min=0.1, H_max=10.0,
# w_ceiling=50.0; the same floors/ceilings are declared in
# jaxfne/emitters.py, jaxfne/hdp_rule.py and jaxfne/hdp_network.py). Read live
# from HP_HEBB so an explicit key would win; HDP params are unchanged.
H_BOUNDS = (float(HP_HEBB.get("H_min", 0.1)), float(HP_HEBB.get("H_max", 10.0)))
W_CEILING = float(HP_HEBB.get("w_ceiling", 50.0))
KICKS = (0.8, 1.3)
H0_PERTURBED = 0.0
WINDOW_MS = 1000.0
LATE_WINDOWS = 2
RETURN_RATIO = 0.25
MIN_DISABLED_DEV_HZ = 0.5
EVOKED_MS = 40.0
LEVEL = "RELATIVE_PROXY"
RESULTS_PATH = Path(__file__).parent / "results" / "at10_n20_055.json"


def build_model() -> Any:
    """The one realized N_20 network with the declared drive."""
    cfg = J.neuronal_tensor_to_configuration(
        G.develop_n20(), seed=BUILD_SEED, duration_ms=PHASE_MS, dt_ms=DT_MS)
    return J.construct(cfg.drive(baseline_drive_by_cell_type=dict(DRIVE)))


def _areas(model: Any) -> np.ndarray:
    return np.array([r["area"] for r in model.neuron_table()])


def _onsets_ms() -> np.ndarray:
    return np.arange(STIM_FIRST_MS, PHASE_MS - STIM_FIRST_MS, STIM_PERIOD_MS)


def _stimulus(area: np.ndarray) -> Any:
    h01 = [int(i) for i in np.nonzero(area == "H01")[0]]
    events = [{"onset_ms": float(o), "duration_ms": STIM_DUR_MS, "amplitude": STIM_AMP,
               "target_indices": h01} for o in _onsets_ms()]
    return J.stimulus_schedule(events, len(area), drive_amplitude=STIM_AMP,
                               event_duration_ms=STIM_DUR_MS)


def _run_signals(model: Any, stim: Any, hp: "dict | None",
                 noise_seed: int = NOISE["seed"]) -> tuple[Any, Any]:
    """One run; HDP diagnostics copied right after it (w_final, H_trace; no W trace).

    ``record_weight_trace`` is off: a 10 s W trace is (20000 x 68620) floats
    (~5.5 GB in float32) and nothing here reads it; recording does not enter dynamics.
    """
    hdp = None if hp is None else {**hp, "record_weight_trace": False}
    runtime = J.RuntimeConfig(enable_hdp=hp is not None, hdp_params=dict(hdp or {}))
    sim = J.Simulation(duration_ms=PHASE_MS, dt_ms=DT_MS, seed=RUN_SEED, runtime=runtime,
                       poisson_drive={**NOISE, "seed": int(noise_seed)})
    signals = model.simulate(sim, paradigm=stim)
    if hp is None:
        return signals, None
    d = model.last_hdp_diagnostics()
    return signals, {k: None if d.get(k) is None else np.asarray(d[k])
                     for k in ("H_final", "w_final", "H_trace", "w_trace")}


def _run(model: Any, stim: Any, hp: "dict | None", noise_seed: int = NOISE["seed"]) -> tuple[np.ndarray, Any]:
    signals, diag = _run_signals(model, stim, hp, noise_seed)
    return np.asarray(signals.spikes), diag


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x[ok])), np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def _locked_rate(sp: np.ndarray, cols: np.ndarray, starts: list[int]) -> float:
    """Mean rate in [t, t+EVOKED) minus [t-EVOKED, t), over starts (Hz)."""
    w = int(EVOKED_MS / DT_MS)
    vals = [sp[s:s + w, cols].mean() - sp[s - w:s, cols].mean()
            for s in starts if s >= w and s + w <= sp.shape[0]]
    return float(np.mean(vals) / (DT_MS / 1000.0))


def _latency_ms(sp: np.ndarray, cols: np.ndarray, starts: list[int]) -> float:
    """First post-onset time the smoothed PSTH exceeds pre-onset mean + 3 sd.

    The threshold uses the time-course variability of the onset-averaged
    pre-onset trace (not across-onset variability), so it is lenient; the
    4-step boxcar ("same" mode) shifts crossings by at most 1 ms.
    """
    post, pre = int(60 / DT_MS), int(40 / DT_MS)
    use = [s for s in starts if s >= pre and s + post <= sp.shape[0]]
    psth = np.mean([sp[s:s + post, cols].mean(axis=1) for s in use], axis=0)
    base = np.mean([sp[s - pre:s, cols].mean(axis=1) for s in use], axis=0)
    thr = base.mean() + 3.0 * base.std() + 1e-9
    idx = np.nonzero(np.convolve(psth, np.ones(4) / 4, mode="same") > thr)[0]
    return float(idx[0] * DT_MS) if idx.size else float("nan")


def propagation(sp: np.ndarray, area: np.ndarray) -> dict[str, Any]:
    """Pulse-locked response per area vs hierarchy distance from H01.

    Control: the same statistic at onsets shifted by fixed-seed random
    offsets of 120-160 ms (windows of +-EVOKED_MS span 80-200 ms after a
    pulse: after the responses, before the next pulse); an area is significant when its evoked rate
    exceeds the largest control magnitude + 1 Hz. One global bar over all
    areas: conservative by construction (a noisy area raises it for all).
    """
    names = G.area_names()
    on = [int(o / DT_MS) for o in _onsets_ms()]
    rng = np.random.default_rng(0)
    ctrl = [s + int(rng.integers(120, 160) / DT_MS) for s in on]
    d = np.arange(1, len(names)) / (len(names) - 1)
    evoked = np.array([_locked_rate(sp, area == a, on) for a in names[1:]])
    control = np.array([_locked_rate(sp, area == a, ctrl) for a in names[1:]])
    latency = np.array([_latency_ms(sp, area == a, on) for a in names[1:]])
    sig = evoked > np.abs(control).max() + 1.0
    return {
        "rate_hz": float(sp.mean() / (DT_MS / 1000.0)),
        "evoked_hz": dict(zip(names[1:], np.round(evoked, 3).tolist())),
        "latency_ms": dict(zip(names[1:], [None if not np.isfinite(v) else v for v in latency])),
        "control_max_hz": float(np.abs(control).max()),
        "n_significant": int(sig.sum()),
        "spearman_evoked_distance": _spearman(evoked, d),
        "spearman_latency_distance": _spearman(np.where(sig, latency, np.nan), d),
    }


def _window_rates(sp: np.ndarray) -> np.ndarray:
    w = int(WINDOW_MS / DT_MS)
    return np.array([sp[i:i + w].mean() / (DT_MS / 1000.0)
                     for i in range(0, sp.shape[0] - w + 1, w)])


def _w_drift(diag: Any, w0: np.ndarray) -> float:
    return float(np.abs(np.asarray(diag["w_final"])).mean() / np.abs(w0).mean())


def _realized_weights(model: Any) -> np.ndarray:
    w = np.asarray(model.params["edge_list"].weight, dtype=float)
    if w.shape[0] != model.params["edge_list"].n_edges:
        raise ValueError("N_20 must store per-edge weights for the W kick")
    return w


PSD_NPERSEG = 256  # repo Welch convention (vis.fields, vis.plotly.spectra)
RATE_LO_HZ, RATE_HI_HZ = 0.5, 35.0  # boundedness stability bounds


def _area_mean_rates_hz(sp: np.ndarray, area: np.ndarray) -> dict[str, float]:
    return {a: float(sp[:, area == a].mean() / (DT_MS / 1000.0))
            for a in G.area_names()}


def _welch_psd(x: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray, str]:
    """Welch spectrum via ``scipy.signal.welch`` (nperseg=256).

    AT scripts import only the public surface, so the call
    ``jaxfne.vis.core.welch_psd`` makes is repeated here; checked
    bit-identical to it for float64 and float32 input (2026-10-06).
    """
    from scipy import signal as _signal
    arr = np.asarray(x)
    seg = int(min(PSD_NPERSEG, arr.shape[0]))
    f, p = _signal.welch(arr, fs=float(fs_hz), axis=0, nperseg=seg)
    return f, p, "scipy.signal.welch"


def _area_psd(sp: np.ndarray, area: np.ndarray) -> dict[str, Any]:
    """Welch power per area over the full phase (population-mean trace)."""
    freqs = None
    method = ""
    power: dict[str, Any] = {}
    for a in G.area_names():
        f, p, method = _welch_psd(sp[:, area == a].mean(axis=1), 1000.0 / DT_MS)
        freqs = f
        power[a] = np.asarray(p, dtype=float).tolist()
    return {"method": f"{method} nperseg={PSD_NPERSEG}",
            "freqs_hz": np.asarray(freqs, dtype=float).tolist(), "power": power}


PHASES = ("baseline", "hebbian_hdp", "noisy_hdp")


def run_phases(model: Any = None, keep_bundle: bool = False) -> dict[str, Any]:
    """The three phases on one W0. ``keep_bundle`` adds ``{phase: {model, signals, hdp}}``
    (one shared Model; each phase keeps its own HDP diagnostics, None for baseline)."""
    t0 = time.perf_counter()
    model = model if model is not None else build_model()
    area = _areas(model)
    stim = _stimulus(area)
    w0 = _realized_weights(model)
    out: dict[str, Any] = {}
    bundle: dict[str, Any] = {}
    for name, hp in zip(PHASES, (None, HP_HEBB, HP_NOISY)):
        signals, diag = _run_signals(model, stim, hp)
        sp = np.asarray(signals.spikes)
        area_rates = _area_mean_rates_hz(sp, area)
        bounded = all(RATE_LO_HZ <= v <= RATE_HI_HZ for v in area_rates.values())
        out[name] = {**propagation(sp, area),
                     "window_rates_hz": np.round(_window_rates(sp), 3).tolist(),
                     "w_mean_ratio": None if diag is None else _w_drift(diag, w0),
                     "psd": _area_psd(sp, area),
                     "boundedness": {"bounds_hz": [RATE_LO_HZ, RATE_HI_HZ],
                                     "per_area_mean_hz": {a: round(v, 3)
                                                          for a, v in area_rates.items()},
                                     "max_hz": round(max(area_rates.values()), 3),
                                     "verdict": "PASS" if bounded else "FAIL"},
                     "rate_spread_hz": round(max(area_rates.values())
                                             - min(area_rates.values()), 3)}
        if keep_bundle:
            bundle[name] = {"model": model, "signals": signals, "hdp": diag}
    res = {"phases": out, "wall_s": time.perf_counter() - t0}
    if keep_bundle:
        res["bundle"] = bundle
    return res


def run_at10(keep_bundle: bool = False) -> dict[str, Any]:
    """Canonical AT-10-N20 runner (``at_manifest.REGISTRY``): the three phases."""
    return {"scenario": "AT-10-N20", "status": "OK", "level": LEVEL,
            **run_phases(keep_bundle=keep_bundle)}


def _area_mean_H(H_trace: Any, area: np.ndarray) -> np.ndarray:
    """Reduce an H trace (steps x neurons) to per-area means (steps x areas).

    A full trace is 20000 steps x 20000 neurons; the means are steps x 20.
    Recording itself is untouched (record_stride/record_h_subset stay at
    their defaults, so dynamics are identical); only what is kept is reduced.
    """
    H = np.asarray(H_trace, dtype=float)
    return np.stack([H[:, area == a].mean(axis=1) for a in G.area_names()], axis=1)


def _area_H_extremes(H_trace: Any, area: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-area H extremes over time (steps reduced away).

    Per area: min and max over (steps x neurons in area), two arrays of
    length n_areas. Small by construction (20 floats each); the full trace is
    never stored. Used for H boundedness so a per-neuron excursion cannot
    hide inside an area mean.
    """
    H = np.asarray(H_trace, dtype=float)
    names = G.area_names()
    lo = np.array([H[:, area == a].min() for a in names], dtype=float)
    hi = np.array([H[:, area == a].max() for a in names], dtype=float)
    return lo, hi


def _assay_twin(arm: str, ref_H: Any, ref_off_H: Any) -> tuple[Any, str]:
    """Twin selection for one assay arm: an engaged arm twins with the engaged
    reference; a disabled arm twins with the disabled unperturbed reference
    (same noise seed), so the twin differs only in initial state, never in
    plasticity regime. Returns (twin_H, label)."""
    if arm == "engaged":
        return ref_H, "engaged_ref"
    return ref_off_H, "disabled_ref"


def h_space_report(H_ref: Any, H_arm: Any, *, H_bounds: Any,
                   w_ceiling: "float | None" = None, w_final: Any = None,
                   H_extremes: Any = None, twin: str = "engaged_ref") -> dict[str, Any]:
    """H-space separation for one assay arm (AT-10-R5): stability twin plus H boundedness.

    The twin is the reference run in the arm's own plasticity regime vs the
    arm (``twin`` records which: ``"engaged_ref"`` or ``"disabled_ref"``),
    both as per-area-mean H traces. ``boundedness_report`` takes (H_trace,
    w_trace, H_bounds, w_ceiling); no W trace is recorded
    (``record_weight_trace`` is off: a 10 s full W trace is ~5.5 GB), so W
    enters as the final state only (``w_final``-only, flagged ``W_scope``),
    never a fabricated trace. H boundedness is computed from per-area
    extremes over time (``H_extremes``: per-area min/max, one row per area)
    when supplied, so a per-neuron excursion cannot hide inside an area
    mean; violations are then counted per area-extreme (flagged ``H_scope``).
    A zero early twin distance is a vacuous assay: it is refused
    (``REFUSED_DEGENERATE``), not scored. Returns a JSON-safe dict.
    """
    try:
        stability: dict[str, Any] = dict(J.stability_report(H_ref, H_arm))
    except ValueError as exc:
        stability = {"verdict": "REFUSED_DEGENERATE", "reason": str(exc)}
    if H_extremes is not None:
        H_in = np.column_stack([np.asarray(H_extremes[0], dtype=float),
                                np.asarray(H_extremes[1], dtype=float)])
        h_scope = ("per-area extremes over time (min/max over steps x neurons "
                   "in area, one row per area); violations counted per area-extreme")
    else:
        H_in = np.asarray(H_arm, dtype=float)
        h_scope = "input trace as given (no extremes supplied)"
    if w_final is None or w_ceiling is None:
        H = H_in
        lo, hi = float(H_bounds[0]), float(H_bounds[1])
        h_bad = (H < lo) | (H > hi)
        boundedness: dict[str, Any] = {
            "within_bounds": bool(not h_bad.any()),
            "H_violations": int(h_bad.sum()),
            "H_min": float(H.min()),
            "H_max": float(H.max()),
            "H_bounds": [lo, hi],
            "H_scope": h_scope,
            "W_scope": "OMITTED (no w_final/w_ceiling)",
        }
    else:
        boundedness = dict(J.boundedness_report(
            H_in, np.asarray(w_final, dtype=float),
            H_bounds=(float(H_bounds[0]), float(H_bounds[1])),
            w_ceiling=float(w_ceiling)))
        boundedness["H_scope"] = h_scope
        boundedness["W_scope"] = "w_final-only (no W trace recorded)"
    return {"twin": twin, "stability": stability, "boundedness": boundedness}


def run_assay(model: Any = None, noise_seed: int = NOISE["seed"]) -> dict[str, Any]:
    t0 = time.perf_counter()
    model = model if model is not None else build_model()
    area = _areas(model)
    stim = _stimulus(area)
    w0 = _realized_weights(model)
    hp_off = J.hdp_network.disable_plasticity(dict(HP_HEBB), mask=None)
    ref_sp, ref_diag = _run(model, stim, HP_HEBB, noise_seed)
    ref = _window_rates(ref_sp)
    ref_H = _area_mean_H(ref_diag["H_trace"], area)
    # Disabled twin: one extra unperturbed run with plasticity disabled (same
    # noise seed), so a disabled arm's H twin differs only in initial state.
    # A new run; every existing output below is untouched.
    _, ref_off_diag = _run(model, stim, hp_off, noise_seed)
    ref_off_H = _area_mean_H(ref_off_diag["H_trace"], area)
    perturbations = [(f"H0={H0_PERTURBED}", {"H0": np.full(len(area), H0_PERTURBED)})]
    perturbations += [(f"kick={k}", {"w0": k * w0}) for k in KICKS]
    arms: dict[str, Any] = {}
    tests = []
    for label, init in perturbations:
        rec = {}
        for arm, hp in (("engaged", HP_HEBB), ("disabled", hp_off)):
            sp, diag = _run(model.with_hdp_initial_state(**init), stim, hp, noise_seed)
            r = _window_rates(sp)
            dev = np.abs(r - ref)
            w_init = np.asarray(init.get("w0", w0), dtype=np.float32)
            arm_H = _area_mean_H(diag["H_trace"], area)
            twin_H, twin = _assay_twin(arm, ref_H, ref_off_H)
            rec[arm] = {"window_rates_hz": np.round(r, 3).tolist(),
                        "late_dev_hz": float(dev[-LATE_WINDOWS:].mean()),
                        "first_dev_hz": float(dev[0]),
                        "w_mean_ratio": _w_drift(diag, w0),
                        "w_unchanged": bool(np.array_equal(
                            np.asarray(diag["w_final"], dtype=np.float32), w_init)),
                        "h_space": h_space_report(
                            twin_H, arm_H, H_bounds=H_BOUNDS, w_ceiling=W_CEILING,
                            w_final=diag["w_final"],
                            H_extremes=_area_H_extremes(diag["H_trace"], area),
                            twin=twin)}
        stays_off = rec["disabled"]["late_dev_hz"] >= MIN_DISABLED_DEV_HZ
        returns = rec["engaged"]["late_dev_hz"] <= RETURN_RATIO * rec["disabled"]["late_dev_hz"]
        tests.append({"perturbation": label, "disabled_stays_off": bool(stays_off),
                      "engaged_returns": bool(returns),
                      "verdict": "STABILIZED" if stays_off and returns else
                      ("NO_LASTING_EFFECT" if not stays_off else "NOT_STABILIZED")})
        arms[label] = rec
    return {"noise_seed": int(noise_seed), "reference_window_rates_hz": np.round(ref, 3).tolist(), "arms": arms,
            "declared_tests": tests, "wall_s": time.perf_counter() - t0}


# AT-10-R5 valid-twin arm (adversarial-review follow-up): HP_HEBB without the
# restoring term. HP_NOCTRL removes only the restoring term K_ctrl*(1-H)
# (jaxfne/emitters.py ~4179: dH_ctrl = K_ctrl_arr * (1.0 - H)); every other
# HDP term is unchanged.
HP_NOCTRL = {**HP_HEBB, "K_ctrl": 0.0}

# Interior twin perturbation: 0.5 lies strictly inside the H bounds
# [0.1, 10] (H_BOUNDS above), so the H0 shift is not clipped away on the
# first step (unlike H0 = 0.0, which the H_min clip maps to 0.1 before any
# dynamics can act on it). A W kick is not a valid H twin: it changes the
# efficacy state, not the H initial condition, so twin separation confounds
# two state perturbations.
R5_TWIN_H0_VALUE = 0.5
R5_TWIN_EARLY_ENDS = (200, 1000, 2000)


def score_r5_twin_pair(H_ref: Any, H_arm: Any, *, H_trace: Any = None,
                       H_extremes: Any = None, H_bounds: Any = H_BOUNDS,
                       twin: str = "engaged_ref") -> dict[str, Any]:
    """Pure per-pair scoring for ``run_r5_twin_assay`` (no model needed).

    ``H_ref``/``H_arm`` are per-area-mean H traces (steps x areas) from twin
    runs differing only in initial H. Reports ``J.stability_report`` at each
    ``R5_TWIN_EARLY_ENDS`` value (late_start = -200) with the
    ``h_space_report``-style guard (ValueError -> REFUSED_DEGENERATE), plus
    the per-area-extreme boundedness check and ``clip_reached``: whether any
    recorded H equals H_min or H_max. A clip-pinned trace can read
    ``within_bounds`` True while evidencing nothing about genuine
    boundedness, so ``clip_reached`` True disqualifies the bounded verdict
    as evidence. Returns a JSON-safe dict.
    """
    lo, hi = float(H_bounds[0]), float(H_bounds[1])
    sweep: dict[str, Any] = {}
    for early_end in R5_TWIN_EARLY_ENDS:
        try:
            rep = J.stability_report(H_ref, H_arm, early_end=early_end, late_start=-200)
            sweep[str(early_end)] = {"verdict": str(rep["verdict"]),
                                     "contraction_ratio": float(rep["contraction_ratio"]),
                                     "d_early": float(rep["d_early"]),
                                     "d_late": float(rep["d_late"])}
        except ValueError as exc:
            sweep[str(early_end)] = {"verdict": "REFUSED_DEGENERATE", "reason": str(exc)}
    recorded = (np.asarray(H_trace, dtype=float) if H_trace is not None
                else np.asarray(H_arm, dtype=float))
    # Recorded H is float32 (float32(0.1) = 0.10000000149 != 0.1): compare with
    # a float32-scale tolerance, not equality, or a pinned trace reads unclipped.
    clip_reached = bool((np.isclose(recorded, lo, rtol=1e-6, atol=0.0)
                         | np.isclose(recorded, hi, rtol=1e-6, atol=0.0)).any())
    if H_extremes is not None:
        H_in = np.column_stack([np.asarray(H_extremes[0], dtype=float),
                                np.asarray(H_extremes[1], dtype=float)])
        h_scope = ("per-area extremes over time (min/max over steps x neurons "
                   "in area, one row per area); violations counted per area-extreme")
    else:
        H_in = np.asarray(H_arm, dtype=float)
        h_scope = "input trace as given (no extremes supplied)"
    h_bad = (H_in < lo) | (H_in > hi)
    boundedness: dict[str, Any] = {
        "within_bounds": bool(not h_bad.any()),
        "H_violations": int(h_bad.sum()),
        "H_min": float(H_in.min()),
        "H_max": float(H_in.max()),
        "H_bounds": [lo, hi],
        "H_scope": h_scope,
        "clip_reached": clip_reached,
    }
    return {"twin": twin, "sweep": sweep, "clip_reached": clip_reached,
            "H_min_obs": float(recorded.min()), "H_max_obs": float(recorded.max()),
            "boundedness": boundedness}


def run_r5_twin_assay(model: Any = None, noise_seed: int = NOISE["seed"]) -> dict[str, Any]:
    """Two engaged valid-twin pairs: restoring (HP_HEBB) vs no-restoring (HP_NOCTRL).

    Same stimulus, seed and ``_run`` helper as ``run_assay``. Each pair's
    twin differs only in initial H (``model.with_hdp_initial_state(H0=...)``
    with H0 = 0.5 everywhere); the reference is the unperturbed run in the
    pair's own HDP regime. Per-pair H traces are reduced with
    ``_area_mean_H`` and scored with ``score_r5_twin_pair``. ``run_assay``
    outputs are untouched. Returns a JSON-safe dict.
    """
    t0 = time.perf_counter()
    model = model if model is not None else build_model()
    area = _areas(model)
    stim = _stimulus(area)
    H0 = np.full(len(area), R5_TWIN_H0_VALUE)
    pairs: dict[str, Any] = {}
    for key, hp, twin in (("restoring", HP_HEBB, "engaged_ref"),
                          ("no_restoring", HP_NOCTRL, "noctrl_engaged_ref")):
        _, ref_diag = _run(model, stim, hp, noise_seed)
        _, arm_diag = _run(model.with_hdp_initial_state(H0=H0), stim, hp, noise_seed)
        ref_H = _area_mean_H(ref_diag["H_trace"], area)
        arm_H = _area_mean_H(arm_diag["H_trace"], area)
        pairs[key] = {"hp": dict(hp),
                      **score_r5_twin_pair(
                          ref_H, arm_H, H_trace=arm_diag["H_trace"],
                          H_extremes=_area_H_extremes(arm_diag["H_trace"], area),
                          twin=twin)}
    return {"noise_seed": int(noise_seed), "H0": float(R5_TWIN_H0_VALUE),
            "pairs": pairs, "wall_s": time.perf_counter() - t0}


# AT-10-R6 (0.5.5, very long T): total_ms in chunk_ms full-state chunks.
R6_TOTAL_MS = 100_000.0
R6_CHUNK_MS = 25_000.0


def _rss_mb() -> "float | None":
    """Process RSS in MB, or None when psutil is absent (JSON-safe either way)."""
    try:
        import psutil
    except Exception:
        return None
    return float(psutil.Process().memory_info().rss) / 1e6


def _chunk_stimulus(area: np.ndarray, chunk_start_ms: float, chunk_ms: float,
                    total_ms: float) -> Any:
    """Pulse-train slice for one chunk: global H01 onsets overlapping
    ``[chunk_start_ms, chunk_start_ms + chunk_ms)``, shifted by
    ``-chunk_start_ms``.

    ``StimulusSchedule.to_array`` clips out-of-window edges, so a pulse
    straddling a chunk boundary contributes its remainder to the next chunk
    and no drive is lost or doubled; flat pulses are unaffected.
    """
    h01 = [int(i) for i in np.nonzero(area == "H01")[0]]
    end_ms = float(chunk_start_ms) + float(chunk_ms)
    onsets = np.arange(STIM_FIRST_MS, float(total_ms) - STIM_FIRST_MS,
                       STIM_PERIOD_MS)
    events = [{"onset_ms": float(o) - float(chunk_start_ms),
               "duration_ms": STIM_DUR_MS, "amplitude": STIM_AMP,
               "target_indices": h01}
              for o in onsets
              if float(o) < end_ms and float(o) + STIM_DUR_MS > float(chunk_start_ms)]
    return J.stimulus_schedule(events, len(area), drive_amplitude=STIM_AMP,
                               event_duration_ms=STIM_DUR_MS)


def summarize_r6_chunks(chunks: "list[dict[str, Any]]") -> dict[str, Any]:
    """Pure summary over ``run_r6_long`` per-chunk records (no model needed).

    Per-area rates and H means average across chunks; H extremes take the
    min/max across chunks; ``clip_reached`` is OR-ed; ``w_mean_ratio`` keeps
    the last chunk's value (ratio vs W0). Returns a JSON-safe dict.
    """
    if not chunks:
        raise ValueError("summarize_r6_chunks needs at least one chunk record")
    names = list(chunks[0]["per_area_mean_hz"])
    n = len(chunks)
    return {
        "n_chunks": n,
        "per_area_mean_hz": {
            a: float(sum(c["per_area_mean_hz"][a] for c in chunks) / n) for a in names},
        "h_mean": {
            a: float(sum(c["h_mean"][a] for c in chunks) / n) for a in names},
        "h_min": {a: float(min(c["h_min"][a] for c in chunks)) for a in names},
        "h_max": {a: float(max(c["h_max"][a] for c in chunks)) for a in names},
        "clip_reached": bool(any(c["clip_reached"] for c in chunks)),
        "w_mean_ratio_last": chunks[-1]["w_mean_ratio"],
        "wall_s": float(sum(c["wall_s"] for c in chunks)),
    }


def run_r6_long(model: Any = None, total_ms: float = R6_TOTAL_MS,
                chunk_ms: float = R6_CHUNK_MS,
                noise_seed: int = NOISE["seed"]) -> dict[str, Any]:
    """Very-long-time run: ``total_ms`` in ``chunk_ms`` full-state chunks.

    One realized network (``build_model`` unless ``model`` is given), HDP on
    (``HP_HEBB``, no W trace), the H01 pulse train sliced per chunk
    (``_chunk_stimulus``) and per-chunk Poisson noise (chunk 0 draws
    ``noise_seed``; later chunks derive theirs from (seed, chunk_index) in
    the engine). H, w, delays in flight and the PRNG chain thread through
    the continuation state across chunks.

    Each chunk's H trace is reduced to per-area mean/min/max before the next
    chunk runs, so memory stays at one chunk plus the carried states;
    per-chunk records are JSON-safe. Values are relative (RELATIVE_PROXY);
    nothing is calibrated.
    """
    t0 = time.perf_counter()
    model = model if model is not None else build_model()
    area = _areas(model)
    names = G.area_names()
    w0 = _realized_weights(model)
    n_chunks = int(math.ceil(float(total_ms) / float(chunk_ms)))
    if n_chunks < 1:
        raise ValueError(
            f"total_ms={total_ms!r} with chunk_ms={chunk_ms!r} needs >= 1 chunk")
    hp = {**HP_HEBB, "record_weight_trace": False}
    # Continuation requires the edge_list backend (_model_simulate.py:924).
    runtime = J.RuntimeConfig(enable_hdp=True, hdp_params=dict(hp),
                              recurrent_backend="edge_list")
    chunks: list[dict[str, Any]] = []
    state = None
    for c in range(n_chunks):
        start_ms = float(c) * float(chunk_ms)
        dur_ms = min(float(chunk_ms), float(total_ms) - start_ms)
        ct0 = time.perf_counter()
        stim = _chunk_stimulus(area, start_ms, dur_ms, float(total_ms))
        sim = J.Simulation(duration_ms=dur_ms, dt_ms=DT_MS, seed=RUN_SEED,
                           runtime=runtime,
                           poisson_drive={**NOISE, "seed": int(noise_seed)})
        if state is None:
            signals, state = model.simulate(sim, paradigm=stim, return_state=True)
        else:
            signals, state = model.simulate(sim, paradigm=stim,
                                            continuation=state, return_state=True)
        sp = np.asarray(signals.spikes)
        d = model.last_hdp_diagnostics()
        H = np.asarray(d["H_trace"], dtype=float)
        h_mean = {a: float(H[:, area == a].mean()) for a in names}
        h_min = {a: float(H[:, area == a].min()) for a in names}
        h_max = {a: float(H[:, area == a].max()) for a in names}
        lo, hi = float(H_BOUNDS[0]), float(H_BOUNDS[1])
        # Recorded H is float32: float32(H bound) != the float64 bound
        # exactly, so compare with a float32-scale tolerance (R5 idiom).
        clip = bool((np.isclose(H, lo, rtol=1e-6, atol=0.0)
                     | np.isclose(H, hi, rtol=1e-6, atol=0.0)).any())
        chunks.append({
            "chunk": int(c), "start_ms": float(start_ms),
            "duration_ms": float(dur_ms), "chunk_index": int(state.chunk_index),
            "wall_s": time.perf_counter() - ct0, "rss_mb": _rss_mb(),
            "mean_rate_hz": float(sp.mean() / (DT_MS / 1000.0)),
            "per_area_mean_hz": _area_mean_rates_hz(sp, area),
            "h_mean": h_mean, "h_min": h_min, "h_max": h_max,
            "h_bounds": [lo, hi], "clip_reached": clip,
            "w_mean_ratio": float(np.abs(np.asarray(d["w_final"])).mean()
                                  / np.abs(w0).mean()),
        })
    return {"scenario": "AT-10-N20-R6", "level": LEVEL,
            "total_ms": float(total_ms), "chunk_ms": float(chunk_ms),
            "noise_seed": int(noise_seed), "chunks": chunks,
            "summary": summarize_r6_chunks(chunks),
            "wall_s": time.perf_counter() - t0}


def spec() -> dict[str, Any]:
    return {
        "genome": G.G20_NAME, "development_seed": G.G20_DEV_SEED, "build_seed": BUILD_SEED,
        "dt_ms": DT_MS, "phase_ms": PHASE_MS, "run_seed": RUN_SEED, "pilot_seed": PILOT_SEED,
        "drive": DRIVE, "noise": NOISE,
        "stimulus": {"target": "H01", "period_ms": STIM_PERIOD_MS, "duration_ms": STIM_DUR_MS,
                     "amplitude": STIM_AMP, "first_ms": STIM_FIRST_MS},
        "hp_hebb": HP_HEBB, "hp_noisy": HP_NOISY, "kicks": list(KICKS), "h0_perturbed": H0_PERTURBED,
        "replicate_noise_seeds": list(REPLICATE_NOISE_SEEDS), "window_ms": WINDOW_MS, "late_windows": LATE_WINDOWS, "return_ratio": RETURN_RATIO,
        "min_disabled_dev_hz": MIN_DISABLED_DEV_HZ, "level": LEVEL,
    }


def null_threshold(model: Any = None) -> dict[str, Any]:
    """Null without the pulse train (Poisson background kept): the baseline phase, the evoked statistic
    taken at the pulse onsets. Threshold = max |null evoked| over areas + 1 Hz."""
    model = model if model is not None else build_model()
    area = _areas(model)
    sp, _ = _run(model, None, None)
    on = [int(o / DT_MS) for o in _onsets_ms()]
    null = np.array([_locked_rate(sp, area == a, on) for a in G.area_names()[1:]])
    return {"rate_hz": float(sp.mean() / (DT_MS / 1000.0)),
            "evoked_hz": dict(zip(G.area_names()[1:], np.round(null, 3).tolist())),
            "threshold_hz": float(np.abs(null).max() + 1.0)}


def run_all(path: Path = RESULTS_PATH) -> dict[str, Any]:
    model = build_model()
    out = {"scenario": "AT-10-N20", "spec": spec(), **run_phases(model)}
    out["null"] = null_threshold(model)
    for ph in out["phases"].values():
        ph["n_significant_vs_null"] = int(sum(v > out["null"]["threshold_hz"]
                                              for v in ph["evoked_hz"].values()))
    assay = run_assay(model)
    out["assay"] = assay
    out["wall_s"] += assay.pop("wall_s")
    out["replicates"] = [run_assay(model, s) for s in REPLICATE_NOISE_SEEDS]
    out["wall_s"] += sum(r.pop("wall_s") for r in out["replicates"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    res = run_all()
    for name, ph in res["phases"].items():
        print(name, round(ph["rate_hz"], 2), ph["n_significant"],
              round(ph["spearman_evoked_distance"], 2), ph["w_mean_ratio"])
    for a in [res["assay"], *res["replicates"]]:
        for t in a["declared_tests"]:
            print(a["noise_seed"], t)
    print("wall_s", round(res["wall_s"], 1))
