"""AT-10 on N_20 (0.5.5 ATLAS 6): three phases and the stabilization assay.

N_20 = develop(G_20 v2, K_D = 20): 20 areas x 50 neurons on a declared 1-D
hierarchy (``g20_genome``). Every run uses one realized network (one W0),
the same drive, the same Poisson background and the same pulse train into H01.

Phases (human decision 2026-09-26: 10 s simulated per phase), each a
separate run from W0 (full-state continuation does not carry
``poisson_drive``):

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

Regime and assay constants were chosen before the assay seed ran:
the regime by the 0.5.5 sweep (todo stack ATLAS 6), the HDP gains by 10 s
stability probes, the kicks by a pilot on ``PILOT_SEED`` (not ``RUN_SEED``).
Values are relative (RELATIVE_PROXY); nothing is calibrated.
Import rule: top-level ``jaxfne`` only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import jaxfne as J

from artifacts.atlas import g20_genome as G

DT_MS = 0.5
PHASE_MS = 10000.0
RUN_SEED = 7
PILOT_SEED = 101  # kick-size pilot only; never the assay seed
BUILD_SEED = 5  # tensor -> configuration realization seed
DRIVE = {"E": 3.0, "PV": 2.0}
NOISE = {"rate_hz": 200.0, "amplitude": 4.0, "target": "all", "seed": 11}
STIM_PERIOD_MS, STIM_DUR_MS, STIM_AMP, STIM_FIRST_MS = 200.0, 20.0, 8.0, 100.0
HP_HEBB = {"K_HDP": 0.005, "K_ctrl": 0.15, "K_w_ctrl": 0.05, "alpha": 0.05,
           "tau_0_ms": 5.0, "noise_scale": 0.0}
HP_NOISY = {**HP_HEBB, "noise_scale": 0.5}
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


def _run(model: Any, stim: Any, hp: "dict | None", seed: int = RUN_SEED) -> tuple[np.ndarray, Any]:
    runtime = J.RuntimeConfig(enable_hdp=hp is not None, hdp_params=dict(hp or {}))
    sim = J.Simulation(duration_ms=PHASE_MS, dt_ms=DT_MS, seed=seed, runtime=runtime,
                       poisson_drive=dict(NOISE))
    spikes = np.asarray(model.simulate(sim, paradigm=stim).spikes)
    return spikes, (model.last_hdp_diagnostics() if hp is not None else None)


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
    """First post-onset time the smoothed PSTH exceeds pre-onset mean + 3 sd."""
    post, pre = int(60 / DT_MS), int(40 / DT_MS)
    use = [s for s in starts if s >= pre and s + post <= sp.shape[0]]
    psth = np.mean([sp[s:s + post, cols].mean(axis=1) for s in use], axis=0)
    base = np.mean([sp[s - pre:s, cols].mean(axis=1) for s in use], axis=0)
    thr = base.mean() + 3.0 * base.std() + 1e-9
    idx = np.nonzero(np.convolve(psth, np.ones(4) / 4, mode="same") > thr)[0]
    return float(idx[0] * DT_MS) if idx.size else float("nan")


def propagation(sp: np.ndarray, area: np.ndarray) -> dict[str, Any]:
    """Pulse-locked response per area vs hierarchy distance from H01.

    Control: the same statistic at onsets shifted to fixed-seed random
    offsets between pulses; an area is significant when its evoked rate
    exceeds the largest control magnitude + 1 Hz.
    """
    names = G.area_names()
    on = [int(o / DT_MS) for o in _onsets_ms()]
    rng = np.random.default_rng(0)
    ctrl = [s + int(rng.integers(40, 160) / DT_MS) for s in on]
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


def run_phases(model: Any = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    model = model if model is not None else build_model()
    area = _areas(model)
    stim = _stimulus(area)
    w0 = _realized_weights(model)
    out: dict[str, Any] = {}
    for name, hp in (("baseline", None), ("hebbian_hdp", HP_HEBB), ("noisy_hdp", HP_NOISY)):
        sp, diag = _run(model, stim, hp)
        out[name] = {**propagation(sp, area),
                     "window_rates_hz": np.round(_window_rates(sp), 3).tolist(),
                     "w_mean_ratio": None if diag is None else _w_drift(diag, w0)}
    return {"phases": out, "wall_s": time.perf_counter() - t0}


def run_assay(model: Any = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    model = model if model is not None else build_model()
    area = _areas(model)
    stim = _stimulus(area)
    w0 = _realized_weights(model)
    hp_off = J.hdp_network.disable_plasticity(dict(HP_HEBB), mask=None)
    ref = _window_rates(_run(model, stim, HP_HEBB)[0])
    perturbations = [(f"H0={H0_PERTURBED}", {"H0": np.full(len(area), H0_PERTURBED)})]
    perturbations += [(f"kick={k}", {"w0": k * w0}) for k in KICKS]
    arms: dict[str, Any] = {}
    tests = []
    for label, init in perturbations:
        rec = {}
        for arm, hp in (("engaged", HP_HEBB), ("disabled", hp_off)):
            sp, diag = _run(model.with_hdp_initial_state(**init), stim, hp)
            r = _window_rates(sp)
            dev = np.abs(r - ref)
            rec[arm] = {"window_rates_hz": np.round(r, 3).tolist(),
                        "late_dev_hz": float(dev[-LATE_WINDOWS:].mean()),
                        "first_dev_hz": float(dev[0]),
                        "w_mean_ratio": _w_drift(diag, w0)}
        stays_off = rec["disabled"]["late_dev_hz"] >= MIN_DISABLED_DEV_HZ
        returns = rec["engaged"]["late_dev_hz"] <= RETURN_RATIO * rec["disabled"]["late_dev_hz"]
        tests.append({"perturbation": label, "disabled_stays_off": bool(stays_off),
                      "engaged_returns": bool(returns),
                      "verdict": "STABILIZED" if stays_off and returns else
                      ("NO_LASTING_EFFECT" if not stays_off else "NOT_STABILIZED")})
        arms[label] = rec
    return {"reference_window_rates_hz": np.round(ref, 3).tolist(), "arms": arms,
            "declared_tests": tests, "wall_s": time.perf_counter() - t0}


def spec() -> dict[str, Any]:
    return {
        "genome": G.G20_NAME, "development_seed": G.G20_DEV_SEED, "build_seed": BUILD_SEED,
        "dt_ms": DT_MS, "phase_ms": PHASE_MS, "run_seed": RUN_SEED, "pilot_seed": PILOT_SEED,
        "drive": DRIVE, "noise": NOISE,
        "stimulus": {"target": "H01", "period_ms": STIM_PERIOD_MS, "duration_ms": STIM_DUR_MS,
                     "amplitude": STIM_AMP, "first_ms": STIM_FIRST_MS},
        "hp_hebb": HP_HEBB, "hp_noisy": HP_NOISY, "kicks": list(KICKS), "h0_perturbed": H0_PERTURBED,
        "window_ms": WINDOW_MS, "late_windows": LATE_WINDOWS, "return_ratio": RETURN_RATIO,
        "min_disabled_dev_hz": MIN_DISABLED_DEV_HZ, "level": LEVEL,
    }


def run_all(path: Path = RESULTS_PATH) -> dict[str, Any]:
    model = build_model()
    out = {"scenario": "AT-10-N20", "spec": spec(), **run_phases(model)}
    assay = run_assay(model)
    out["assay"] = assay
    out["wall_s"] += assay.pop("wall_s")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    res = run_all()
    for name, ph in res["phases"].items():
        print(name, round(ph["rate_hz"], 2), ph["n_significant"],
              round(ph["spearman_evoked_distance"], 2), ph["w_mean_ratio"])
    for t in res["assay"]["declared_tests"]:
        print(t)
    print("wall_s", round(res["wall_s"], 1))
