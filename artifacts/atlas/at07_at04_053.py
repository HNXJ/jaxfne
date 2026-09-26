"""AT-07 + AT-04-R2 third pass (0.5.3 ATLAS items 8-10).

Authority: artifacts/project_sources/8_atlas.md (S7 + AT-04-R2; Y vector;
inheritance; OMITTED/REFUSED never synthesized; B = magnetic field
observation only; proxy != calibrated; bounded != stable);
0.5.3 ENGINE items 5+6 (clamp/disable controls + intervention grammar
shape) and items 1+2 (H/W ownership + recording budgets).
Extends the 0.5.1 toy-pass pattern (artifacts/atlas/at01_at10_toy.py)
and the 0.5.2 second pass (artifacts/atlas/at01_at06_052.py): same
firewall, same sentinels, same Y names (v1 names stable in v2).

Consumer/tests of the engine, not an authority over it: scenarios are
data over the current public surface only (firewall:
tests/test_atlas_firewall.py audits this file). No new engine
capabilities; no synthesized quantities.

AT-07 (item 8): fixed W vs Hebbian HDP vs noisy HDP under matched
stimulation on the SAME realized network, using the item-5
enable/disable/clamp controls in the item-6 intervention shape
(identical realized network + one-mechanism intervention + declared
observation difference). Reports the observation difference with its
declared test. NO adaptation phenotype is claimed or required.

AT-04-R2 (item 9): H-perturbation arm. Same realized network, fixed W
on both arms (H != HDP separation: W bit-fixed while H differs), H0
baseline vs perturbed. Causal state effect on excitability vs the
X -> Phi correlation. Phi -> X stays REFUSED (AT-04-R3 OUT_OF_SCOPE);
H-freeze is not a 0.5.3 engine control (item 5: H dynamics are never
frozen) and is recorded as unavailable, never emulated.

AT-07-R4 (B as input to dynamics, candidate:true): NOT IMPLEMENTED.
Recorded REFUSED with reason wherever the path would need it.

Schema v2 (item 10): every v1 name stable and in order; each cell keeps
its level; H/W cells move to IMPLEMENTED on AT-07/AT-04 via the H/W
trajectories + recording budgets recorded here.

INVARIANTS: toy/small sizes only (n=8, 1000 ms model-time at dt=0.5);
wall-time budget 600 s per item (over -> shrink + record, never force);
no adaptation phenotype claimed; bounded (!= stable) reported as
finiteness only; proxy != calibrated in every output; B is observation
only.

Import rule (enforced by tests/test_atlas_firewall.py): this file
imports only the top-level ``jaxfne`` package plus stdlib / numpy. No
``jaxfne.<submodule>`` imports (attribute access ``J.hdp_network`` /
``J.public_surface`` is use of a root-namespace symbol, not a
submodule import).
"""

from __future__ import annotations

import hashlib
import json
import time
import tracemalloc
from typing import Any

import numpy as np

import jaxfne as J

# ---------------------------------------------------------------------------
# Predeclared constants (declared BEFORE any run; need human authorization
# to change -- STOP, do not tune after observing results).
# ---------------------------------------------------------------------------

SEED_BUILD = 1
N_NEURONS = 8
DT_MS = 0.5
DUR_MS = 1000.0  # 1 s model-time long horizon at toy size (S7: long T)
RUN_SEED = 4
WALL_BUDGET_S = 600.0  # per-item wall-time budget (0.5.3 acceptance)

# Hebbian HDP gains: the same declarative baseline the item-6 grammar
# tests use (engine-tested engagement point; not tuned here).
BASE_HP: dict[str, Any] = {
    "K_HDP": 0.05,
    "K_ctrl": 0.15,
    "K_w_ctrl": 0.002,
    "alpha": 0.05,
    "tau_0_ms": 5.0,
    "noise_scale": 0.0,
}
# Noisy HDP: explicit stochastic parameter dynamics via the declared RNG
# domain (0.5.3 item 4: same seed reproduces, new chain per run spec).
NOISY_HP: dict[str, Any] = {**BASE_HP, "noise_scale": 0.5}

# H-perturbation arm (item 9): uniform initial H, baseline vs perturbed.
H0_BASELINE = 1.0  # kernel equilibrium default, stated explicitly
H0_PERTURBED = 0.0

# Clamp arm (item 5 control): first half of realized edges pinned.
CLAMP_VALUE = 0.3

# Recording budget demonstration (item 2): stride + subsets, opt-in.
BUDGET_STRIDE = 5
BUDGET_H_SUBSET = (0, 1, 2, 3)

LEVEL_PROXY = "RELATIVE_PROXY"

SCENARIOS: tuple[str, ...] = ("AT-07", "AT-04")


# ---------------------------------------------------------------------------
# Sentinels (same contract as the 0.5.1 toy pass / 0.5.2 second pass).
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

# Schema v2: every v1 name stable and in order (Y_KEYS_V2 == Y_KEYS_V1).
SCHEMA_VERSION = "v2"
Y_KEYS_V2: tuple[str, ...] = (
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
# Shared execution helpers (root surface only; toy sizes; never force).
# ---------------------------------------------------------------------------


# Recording-budget keys ride hdp_params like record_weight_trace but are
# warnings-only "unrecognized" in the semantic validator by design (0.5.3
# item 2 receipt); the kernel's own _validate_recording_budget fails
# closed on them at run time. The gate below stays strict on the rest.
_BUDGET_KEYS = frozenset({"record_stride", "record_h_subset", "record_w_subset"})


def _validate_hp(hp: dict[str, Any]) -> None:
    """Fail closed on unknown/malformed HDP params (item 7b helper)."""
    scoped = {k: v for k, v in hp.items() if k not in _BUDGET_KEYS}
    issues = J.public_surface.validate_hdp_params_semantics(scoped)
    if issues:
        raise ValueError(f"hdp_params invalid: {issues[0]}")


def _fresh_model() -> Any:
    """One realization of the declared network spec (builder suite2_net1)."""
    return J.construct(J.suite2_net1_config(seed=SEED_BUILD, n=N_NEURONS))


def _realized_weights(model: Any) -> np.ndarray:
    return np.asarray(model.params["edge_list"].weight, dtype=float)


def _spec_digest(hp: dict[str, Any]) -> str:
    """Canonical digest of the realized-network spec (grammar shape)."""
    canonical = json.dumps(
        {
            "builder": "suite2_net1",
            "build": {"seed": SEED_BUILD, "n": N_NEURONS},
            "run": {"duration_ms": DUR_MS, "dt_ms": DT_MS, "seed": RUN_SEED},
            "hdp_params": {str(k): v for k, v in hp.items()},
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _run_arm(model: Any, hp: dict[str, Any]) -> dict[str, Any]:
    """Simulate one arm; return signals + diagnostics + cost."""
    _validate_hp(hp)
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        signals = J.simulate(
            model,
            duration_ms=DUR_MS,
            dt_ms=DT_MS,
            seed=RUN_SEED,
            runtime=J.RuntimeConfig(enable_hdp=True, hdp_params=dict(hp)),
        )
    finally:
        _, peak_b = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    wall_s = time.perf_counter() - t0
    diag = model.last_hdp_diagnostics()
    return {
        "model": model,
        "signals": signals,
        "H_final": np.asarray(diag["H_final"], dtype=float),
        "w_final": np.asarray(diag["w_final"], dtype=float),
        "H_trace": None if diag.get("H_trace") is None else np.asarray(diag["H_trace"]),
        "w_trace": None if diag.get("w_trace") is None else np.asarray(diag["w_trace"]),
        "wall_s": wall_s,
        "mem_peak_b": float(peak_b),
    }


_HDP_KEYS = ("H_final", "w_final", "H_trace", "w_trace")


def _bundle_entry(arm: dict[str, Any]) -> dict[str, Any]:
    """Bundle arm: model + signals + this arm's own HDP diagnostics.

    ``model.last_hdp_diagnostics()`` reflects only the model's latest run, so
    consumers read ``hdp`` (captured right after this arm simulated) instead.
    """
    hdp = {k: arm[k] for k in _HDP_KEYS if k in arm}
    return {"model": arm["model"], "signals": arm["signals"], "hdp": hdp or None}


def _arm_summary(name: str, arm: dict[str, Any], w0: np.ndarray) -> dict[str, Any]:
    """Observation summary for one executed arm (chain X,H->dW->dX->dQ->dPhi)."""
    sig = arm["signals"]
    spikes = np.asarray(sig.spikes)
    sources = np.asarray(sig.sources, dtype=float)
    field = sig.field
    phi = np.asarray(field.lfp_proxy, dtype=float)
    counts = (spikes > 0).sum(axis=0).astype(float)
    src_amp = np.abs(sources).mean(axis=0)
    return {
        "arm": name,
        "level": str(field.epistemic_level),
        "level_note": "fields RELATIVE_PROXY; proxy != calibrated",
        "spike_count": int((spikes > 0).sum()),
        "rate_hz_per_neuron": float((spikes > 0).sum() / (spikes.shape[1] * DUR_MS / 1000.0)),
        "per_neuron_counts": [float(v) for v in counts],
        "H_final_mean": float(arm["H_final"].mean()),
        "H_final": [float(v) for v in arm["H_final"].ravel()],
        "w_final_max_abs_change": float(np.abs(arm["w_final"] - w0).max()),
        "Q_mean_abs_per_neuron": float(src_amp.mean()),
        "Phi_mean_abs": float(np.abs(phi).mean()),
        "kappa": float(J.kappa_synchrony(spikes, DT_MS)),
        "H_trace_shape": list(arm["H_trace"].shape),
        "w_trace_shape": list(arm["w_trace"].shape),
        "wall_s": arm["wall_s"],
        "mem_peak_b": arm["mem_peak_b"],
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r; 0.0 when either side is constant (recorded, not forced)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if float(np.std(x) * np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _x_phi_correlation(arm: dict[str, Any]) -> dict[str, Any]:
    """X -> Phi association on one executed arm (correlation only)."""
    sig = arm["signals"]
    spikes = np.asarray(sig.spikes)
    sources = np.asarray(sig.sources, dtype=float)
    phi = np.asarray(sig.field.lfp_proxy, dtype=float)
    per_neuron_r = _pearson((spikes > 0).sum(axis=0), np.abs(sources).mean(axis=0))
    pop_rate_t = (spikes > 0).sum(axis=1).astype(float)
    phi_t = np.abs(phi).mean(axis=1)
    temporal_r = _pearson(pop_rate_t, phi_t)
    return {
        "per_neuron_rate_vs_source_r": per_neuron_r,
        "temporal_poprate_vs_field_r": temporal_r,
        "note": (
            "observed association of executed X vs executed Phi; "
            "correlation != causal field feedback (AT-04-R3 stays REFUSED)"
        ),
    }


def _declare(
    name: str,
    signal: str,
    expect: str,
    a: np.ndarray,
    b: np.ndarray,
    tolerance: float | None = None,
) -> dict[str, Any]:
    """Declared observation difference (item-6 grammar shape); verdicts."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if expect == "equal":
        verdict = "PASS" if np.array_equal(a, b) else "FAIL"
        gap = None if verdict == "PASS" else float(np.abs(a - b).max())
    elif expect == "different":
        verdict = "PASS" if not np.array_equal(a, b) else "FAIL"
        gap = float(np.abs(a - b).max())
    else:  # pragma: no cover - only equal/different declared here
        raise ValueError(f"unsupported expect {expect!r}")
    return {
        "name": name,
        "signal": signal,
        "expect": expect,
        "tolerance": tolerance,
        "verdict": verdict,
        "gap_max_abs": gap,
    }


def _intervention_record(
    name: str,
    hp: dict[str, Any],
    control: dict[str, Any],
    observation: dict[str, Any],
    verdict: str,
) -> dict[str, Any]:
    """Declarative JSON-safe intervention record (item-6 grammar shape).

    Mirrors ``jaxfne.intervene`` record layout (network/control/
    observation + spec digest) using root-surface symbols only; the
    engine object itself is not imported (firewall).
    """
    return {
        "schema": "jaxfne.atlas.intervention_shape.v1",
        "name": name,
        "network": {
            "builder": "suite2_net1",
            "build": {"seed": SEED_BUILD, "n": N_NEURONS},
            "run": {"duration_ms": DUR_MS, "dt_ms": DT_MS, "seed": RUN_SEED},
            "hdp_params": {str(k): v for k, v in hp.items()},
            "spec_digest": _spec_digest(hp),
        },
        "control": control,
        "observation": observation,
        "verdict": verdict,
    }


def _refused_b_as_input() -> dict[str, Any]:
    """AT-07-R4 (candidate:true): B as input to dynamics -- NOT IMPLEMENTED."""
    return {
        "state": "REFUSED",
        "level": LEVEL_PROXY,
        "reason": (
            "AT-07-R4 OUT_OF_SCOPE (human decision 2026-09-25): B as an input "
            "to neural/plastic dynamics (X,H,B,W)->Q has no engine capability "
            "and no independent evidence; B stays observation-only (Phi_B)"
        ),
    }


# ---------------------------------------------------------------------------
# Item 8: AT-07 fixed-W vs Hebbian HDP vs noisy HDP, same realized network.
# ---------------------------------------------------------------------------


def run_at07(keep_bundle: bool = False) -> dict[str, Any]:
    """S7: matched-stimulation plasticity comparison + budgets + schema v2.

    With ``keep_bundle=True`` the output additionally carries ``"bundle"``
    (one ``{"model", "signals"}`` entry per executed arm: ``hebbian``,
    ``fixed``, ``noisy``, ``clamp``, plus the ``repro`` RNG-domain check
    and the ``budgeted`` decimation run). Default ``False`` leaves the
    output unchanged.
    """
    t0 = time.perf_counter()
    _validate_hp(BASE_HP)
    _validate_hp(NOISY_HP)

    models = [_fresh_model() for _ in range(4)]
    w0s = [_realized_weights(m) for m in models]
    identical = all(np.array_equal(w0s[0], w) for w in w0s[1:])
    if not identical:  # fail closed loudly, never silently proceed
        return {
            "scenario": "AT-07",
            "status": "IDENTICAL_NETWORK_VIOLATION",
            "reason": "realized edge weights differ across same-spec builds",
            "wall_s": time.perf_counter() - t0,
        }
    w0 = w0s[0]
    n_edges = int(w0.shape[0])

    # Item-5 controls: disable-all (fixed-W) + half-projection clamp.
    hp_fixed = J.hdp_network.disable_plasticity(dict(BASE_HP), mask=None)
    clamp_mask = J.hdp_network.projection_mask(
        n_edges, values=[0.0] * (n_edges // 2) + [1.0] * (n_edges - n_edges // 2)
    )
    hp_clamp = J.hdp_network.clamp_plasticity(dict(BASE_HP), mask=clamp_mask, value=CLAMP_VALUE)
    models[3] = models[3].with_hdp_initial_state(
        w0=J.hdp_network.pin_projection_weights(w0, clamp_mask, CLAMP_VALUE)
    )

    arms = {
        "hebbian": _run_arm(models[0], BASE_HP),
        "fixed": _run_arm(models[1], hp_fixed),
        "noisy": _run_arm(models[2], NOISY_HP),
        "clamp": _run_arm(models[3], hp_clamp),
    }
    # RNG-domain check (item 4): same noisy spec + seed reproduces exactly.
    repro = _run_arm(_fresh_model(), NOISY_HP)

    summaries = {name: _arm_summary(name, arm, w0) for name, arm in arms.items()}

    # Declared observation differences (expect declared before the run).
    half = n_edges // 2
    tests = [
        _declare("fixed_w_equals_w0", "w_final", "equal", arms["fixed"]["w_final"], w0),
        _declare(
            "hebbian_differs_from_fixed",
            "w_final",
            "different",
            arms["hebbian"]["w_final"],
            arms["fixed"]["w_final"],
        ),
        _declare(
            "noisy_differs_from_hebbian",
            "w_final",
            "different",
            arms["noisy"]["w_final"],
            arms["hebbian"]["w_final"],
        ),
        _declare(
            "noisy_reproduces_same_seed",
            "w_final",
            "equal",
            repro["w_final"],
            arms["noisy"]["w_final"],
        ),
        _declare(
            "clamp_subset_pinned_exact",
            "w_final",
            "equal",
            arms["clamp"]["w_final"][:half],
            # Executed weights are float32: the expectation is stated in
            # the executed dtype (same as the engine item-5 clamp test),
            # never float64(0.3) against float32(0.3).
            np.full(half, np.float32(CLAMP_VALUE)),
        ),
        _declare(
            "clamp_unclamped_moves",
            "w_final",
            "different",
            arms["clamp"]["w_final"][half:],
            w0[half:],
        ),
    ]
    interventions = [
        _intervention_record(
            f"at07-{t['name']}",
            BASE_HP,
            {"mechanism": "plasticity", "mode": "disable-or-clamp"},
            {"signal": t["signal"], "expect": t["expect"]},
            t["verdict"],
        )
        for t in tests
    ]

    # H/W trajectories + recording budgets (items 1+2, AT-07-R2, schema v2).
    full = {"H": arms["hebbian"]["H_trace"], "w": arms["hebbian"]["w_trace"]}
    budgeted_hp = dict(BASE_HP)
    budgeted_hp["record_weight_trace"] = True
    budgeted_hp["record_stride"] = BUDGET_STRIDE
    budgeted_hp["record_h_subset"] = list(BUDGET_H_SUBSET)
    budgeted_hp["record_w_subset"] = [0, 1, 2]
    budgeted = _run_arm(_fresh_model(), budgeted_hp)
    trajectories = {
        "H_trace_shape": list(full["H"].shape),
        "w_trace_shape": list(full["w"].shape),
        "H_final": [float(v) for v in arms["hebbian"]["H_final"].ravel()],
        "budget": {
            "record_stride": BUDGET_STRIDE,
            "record_h_subset": list(BUDGET_H_SUBSET),
            "record_w_subset": [0, 1, 2],
            "H_budget_shape": list(budgeted["H_trace"].shape),
            "w_budget_shape": list(budgeted["w_trace"].shape),
            "kept_frames_equal_full": bool(
                np.array_equal(
                    budgeted["H_trace"], full["H"][::BUDGET_STRIDE][:, list(BUDGET_H_SUBSET)]
                )
                and np.array_equal(budgeted["w_trace"], full["w"][::BUDGET_STRIDE][:, [0, 1, 2]])
            ),
            "note": (
                "decimation is a post-scan slice; dynamics untouched "
                "(V/spikes/sources never strided); full recording stays default"
            ),
        },
    }

    # Chain X,H -> dW -> dX -> dQ -> dPhi across arms (AT-07-R2, observed).
    chain = {
        arm: {
            "dW_max_abs": summaries[arm]["w_final_max_abs_change"],
            "dX_spike_count": summaries[arm]["spike_count"],
            "dQ_mean_abs": summaries[arm]["Q_mean_abs_per_neuron"],
            "dPhi_mean_abs": summaries[arm]["Phi_mean_abs"],
            "H_final_mean": summaries[arm]["H_final_mean"],
        }
        for arm in summaries
    }

    out = {
        "scenario": "AT-07",
        "status": "OK",
        "wall_s": time.perf_counter() - t0,
        "level": LEVEL_PROXY,
        "level_note": "all fields RELATIVE_PROXY; proxy != calibrated",
        "realized_network": {
            "builder": "suite2_net1",
            "build": {"seed": SEED_BUILD, "n": N_NEURONS},
            "run": {"duration_ms": DUR_MS, "dt_ms": DT_MS, "seed": RUN_SEED},
            "n_edges": n_edges,
            "identical_realization": bool(identical),
            "spec_digest": _spec_digest(BASE_HP),
        },
        "arms": summaries,
        "declared_tests": tests,
        "interventions": interventions,
        "trajectories": trajectories,
        "chain": chain,
        "boundedness": {
            "finite": bool(
                all(np.all(np.isfinite(arms[a][k])) for a in arms for k in ("H_final", "w_final"))
            ),
            "hebbian_max_abs_dw": summaries["hebbian"]["w_final_max_abs_change"],
            "note": (
                "bounded trajectories only (finite values); bounded != "
                "returning != homeostatically stabilized -- no stability "
                "assay here, no adaptation phenotype claimed"
            ),
        },
        "b_as_input": _refused_b_as_input(),
        "phi_b": {
            "state": "REFUSED",
            "level": LEVEL_PROXY,
            "reason": "no calibrated Phi_B beyond proxy (candidate AT-07-R4)",
        },
    }
    if out["wall_s"] > WALL_BUDGET_S:
        out["status"] = "OVER_BUDGET"
    if keep_bundle:
        out["bundle"] = {
            name: _bundle_entry(arm)
            for name, arm in {**arms, "repro": repro, "budgeted": budgeted}.items()
        }
    return out


# ---------------------------------------------------------------------------
# Item 9: AT-04 H-perturbation arm (AT-04-R2).
# ---------------------------------------------------------------------------


def run_at04r2(keep_bundle: bool = False) -> dict[str, Any]:
    """S4 causal arm: H0 baseline vs perturbed, matched stimulation.

    Main pair runs with HDP ENGAGED (same gains both arms, same W0): the
    only differing input is the initial relative state H0, so any
    excitability difference is causally attributable to H (W divergence
    downstream of H is a mediator, not a confound: W0 and stimulation
    are identical). Control pair runs with plasticity DISABLED: W is
    then bit-fixed while H differs (H != HDP separation), which tests
    whether the H effect on X needs the HDP-gated path.

    With ``keep_bundle=True`` the output additionally carries ``"bundle"``
    (one ``{"model", "signals"}`` entry per executed arm). Default
    ``False`` leaves the output unchanged.
    """
    t0 = time.perf_counter()
    hp_off = J.hdp_network.disable_plasticity(dict(BASE_HP), mask=None)

    built = [_fresh_model() for _ in range(4)]
    w0s = [_realized_weights(m) for m in built]
    identical = all(bool(np.array_equal(w0s[0], w)) for w in w0s[1:])
    if not identical:
        return {
            "scenario": "AT-04R2",
            "status": "IDENTICAL_NETWORK_VIOLATION",
            "reason": "realized edge weights differ across same-spec builds",
            "wall_s": time.perf_counter() - t0,
        }
    w0 = w0s[0]
    model_base = built[0].with_hdp_initial_state(H0=np.full(N_NEURONS, H0_BASELINE))
    model_pert = built[1].with_hdp_initial_state(H0=np.full(N_NEURONS, H0_PERTURBED))
    model_dis_base = built[2].with_hdp_initial_state(H0=np.full(N_NEURONS, H0_BASELINE))
    model_dis_pert = built[3].with_hdp_initial_state(H0=np.full(N_NEURONS, H0_PERTURBED))

    arm_base = _run_arm(model_base, BASE_HP)
    arm_pert = _run_arm(model_pert, BASE_HP)
    arm_dis_base = _run_arm(model_dis_base, hp_off)
    arm_dis_pert = _run_arm(model_dis_pert, hp_off)
    sum_base = _arm_summary("baseline_H0", arm_base, w0)
    sum_pert = _arm_summary("perturbed_H0", arm_pert, w0)
    corr_base = _x_phi_correlation(arm_base)
    corr_pert = _x_phi_correlation(arm_pert)

    spk_dis_base = np.asarray(arm_dis_base["signals"].spikes)
    spk_dis_pert = np.asarray(arm_dis_pert["signals"].spikes)
    tests = [
        _declare(
            "h_perturbation_changes_excitability",
            "spike_count",
            "different",
            np.atleast_1d(sum_base["spike_count"]),
            np.atleast_1d(sum_pert["spike_count"]),
        ),
        _declare(
            "disabled_control_x_unaffected_by_h0",
            "spikes",
            "equal",
            spk_dis_base,
            spk_dis_pert,
        ),
        _declare(
            "disabled_control_w_fixed",
            "w_final",
            "equal",
            arm_dis_pert["w_final"],
            w0,
        ),
    ]
    interventions = [
        _intervention_record(
            f"at04r2-{t['name']}",
            dict(BASE_HP),
            {"mechanism": "relative-state-H0", "mode": "perturb-initial-state"},
            {"signal": t["signal"], "expect": t["expect"]},
            t["verdict"],
        )
        for t in tests
    ]

    out = {
        "scenario": "AT-04R2",
        "status": "OK",
        "wall_s": time.perf_counter() - t0,
        "level": LEVEL_PROXY,
        "level_note": (
            "fields RELATIVE_PROXY; proxy != calibrated; "
            "H is finite-dimensional relative state (H != homeostasis)"
        ),
        "realized_network": {
            "builder": "suite2_net1",
            "build": {"seed": SEED_BUILD, "n": N_NEURONS},
            "run": {"duration_ms": DUR_MS, "dt_ms": DT_MS, "seed": RUN_SEED},
            "identical_realization": identical,
            "spec_digest": _spec_digest(dict(BASE_HP)),
        },
        "perturbation": {
            "H0_baseline": H0_BASELINE,
            "H0_perturbed": H0_PERTURBED,
            "plasticity": (
                "engaged with identical gains on the main pair (H0 is the "
                "only differing input); disabled on the control pair (W "
                "bit-fixed while H differs; H != HDP)"
            ),
            "w_fixed_disabled_pair": bool(
                np.array_equal(arm_dis_base["w_final"], w0)
                and np.array_equal(arm_dis_pert["w_final"], w0)
            ),
        },
        "baseline": {**sum_base, "x_to_phi": corr_base},
        "perturbed": {**sum_pert, "x_to_phi": corr_pert},
        "disabled_control": {
            "spike_count_baseline": int((spk_dis_base > 0).sum()),
            "spike_count_perturbed": int((spk_dis_pert > 0).sum()),
            "note": (
                "H0 differs, X identical: the H effect on excitability "
                "needs the HDP-gated path (H != HDP separation cuts both "
                "ways); the main pair above is the positive causal test"
            ),
        },
        "declared_tests": tests,
        "interventions": interventions,
        "h_freeze": {
            "state": "REFUSED",
            "reason": (
                "no H-freeze control exists in the 0.5.3 engine (item 5: H "
                "dynamics are never frozen by enable/disable/clamp); the "
                "H0 perturbation above is the executed causal H arm, never "
                "an emulated clamp"
            ),
        },
        "phi_to_x": {
            "state": "REFUSED",
            "level": LEVEL_PROXY,
            "reason": (
                "AT-04-R3 OUT_OF_SCOPE: Phi->X feedback has no independent "
                "evidence; this arm runs X->Phi correlation plus the causal "
                "H perturbation only"
            ),
        },
        "phi_b": {
            "state": "REFUSED",
            "level": LEVEL_PROXY,
            "reason": "B is observation-only; no calibrated Phi_B exists",
        },
    }
    if out["wall_s"] > WALL_BUDGET_S:
        out["status"] = "OVER_BUDGET"
    if keep_bundle:
        out["bundle"] = {
            "baseline": _bundle_entry(arm_base),
            "perturbed": _bundle_entry(arm_pert),
            "disabled_baseline": _bundle_entry(arm_dis_base),
            "disabled_perturbed": _bundle_entry(arm_dis_pert),
        }
    return out


_RUNNERS = {"AT-07": run_at07, "AT-04": run_at04r2}


def run_scenario(scenario_id: str) -> dict[str, Any]:
    """Run one 0.5.3 scenario; never raises: failures become ERROR records."""
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
    """Run every 0.5.3 scenario in order; never raises per scenario."""
    return {sid: run_scenario(sid) for sid in SCENARIOS}


# ---------------------------------------------------------------------------
# Item 10: measurement schema v1 -> v2 (H/W trajectories + budgets).
# Every v1 name is stable and in order (Y_KEYS_V2 == Y_KEYS_V1, tested).
# ---------------------------------------------------------------------------

# Which Y keys each 0.5.3 scenario implements (mechanical table, derived
# from what run_at07 / run_at04r2 actually compute; Phi_B stays REFUSED
# until a candidate path promotes it; PSD/phi stay with the oscillator
# and cross-area measures in 0.5.2/0.5.4; E_reduction scale matrix is
# 0.5.5).
_V2_IMPLEMENTED: dict[str, tuple[str, ...]] = {
    "AT-07": ("X", "H", "W", "Q", "Phi_E", "SPK", "C", "T_compute", "M_compute"),
    "AT-04": ("X", "H", "W", "Q", "Phi_E", "SPK", "C", "T_compute", "M_compute"),
}
_V2_NOTES: dict[str, str] = {
    "X": "Vm/rate features from executed trajectories",
    "H": "H trajectory + final under recording budgets (0.5.3 items 1-2)",
    "W": "W trajectory + final; fixed/clamp bit-exact, plastic moves (item 5)",
    "Q": "single source representation consumed by every probe",
    "Phi_E": "field contract at RELATIVE_PROXY",
    "Phi_B": "no calibrated Phi_B beyond proxy (candidate AT-07-R4)",
    "SPK": "raster shape + count from executed signals",
    "PSD": "spectral operators stay with the AT-03 oscillator (0.5.2)",
    "C": "synchrony (kappa) / X-Phi association on executed fields",
    "phi": "phase operators stay with AT-03 / cross-area (0.5.2/0.5.4)",
    "E_reduction": "reduction scale matrix is 0.5.5",
    "T_compute": "measured wall_s per scenario",
    "M_compute": "measured host peak bytes (tracemalloc; not device memory)",
}


def measure_v2(scenario_id: str, raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Schema-v2 Y record: v1 names stable, each cell gains a level.

    IMPLEMENTED keys come from the mechanical _V2_IMPLEMENTED table;
    value is the scenario's own status note (never synthesized).
    """
    if scenario_id not in SCENARIOS:
        raise KeyError(f"unknown scenario {scenario_id!r}")
    implemented = _V2_IMPLEMENTED.get(scenario_id, ("T_compute",))
    record: dict[str, dict[str, Any]] = {}
    for key in Y_KEYS_V2:
        if key == "Phi_B":
            record[key] = {
                "state": "REFUSED",
                "value": None,
                "level": LEVEL_PROXY,
                "note": _V2_NOTES[key],
            }
        elif key in implemented:
            record[key] = {
                "state": "IMPLEMENTED",
                "value": {"scenario": scenario_id, "status": raw.get("status")},
                "level": LEVEL_PROXY,
                "note": _V2_NOTES[key],
            }
        else:
            record[key] = {
                "state": "OMITTED",
                "value": None,
                "level": LEVEL_PROXY,
                "note": _V2_NOTES[key],
            }
    return record


def gap_matrix_v2(results: dict[str, dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Y x AT cell states for the 0.5.3 scenarios (drives gap_053.md)."""
    return {
        sid: {key: cell["state"] for key, cell in measure_v2(sid, results[sid]).items()}
        for sid in SCENARIOS
        if sid in results
    }
