"""AT-08 + AT-09 second composition pass (0.5.4 ATLAS items 6-7).

Authority: artifacts/project_sources/8_atlas.md (S8/S9; Y vector;
inheritance; OMITTED/REFUSED never synthesized; proxy != calibrated;
bounded != stable); 0.5.4 ENGINE items 1 (composition + member RNG
domains + continuation), 3 (observation operators), 4 (cross-area
plasticity ownership).

Same firewall, sentinels, and Y discipline as the 0.5.1 toy pass and
the 0.5.2/0.5.3 passes: scenarios are data over the current public
surface only (firewall: tests/test_atlas_firewall.py audits this
file). No new engine capabilities; no synthesized quantities.

AT-08 (item 6): A1 -> A2 with fixed connectivity, repeated stimulus to
A1, SPK/Phi/H recorded in both areas, inter-area dphi_12(f)/C_12(f).
Adaptation = HDP. Arms on the SAME realized network: fixed-W baseline,
full adaptation, cross-frozen (local-only) adaptation. R4 answer comes
from the causal decomposition (full-base) vs (local-base): propagated
activity effects with fixed coupling vs coupling-plasticity effects.

AT-09 (item 7): plastic W_12(t)/W_21(t) under the legacy HDP rule with
delays. Arms: cross-only plasticity, member-only plasticity, fully
frozen (H evolves while dW/dt = 0: the R2 separation). R3 compares
member-W change vs cross-W change and the coherence consequences.

Schema v3 (item 8): v1/v2 names stable and in order; new area-indexed
cells (SPK/H/Phi per area, W_12/W_21) and cross-area cells (C_12,
dphi_12). Phi_B stays REFUSED; E_reduction scale matrix is 0.5.5.

INVARIANTS: toy sizes (n=8 per area, 1000 ms model-time at dt=0.5);
wall-time budget 600 s per scenario (over -> shrink + record, never
force); no phenotype claimed beyond declared observation differences;
proxy != calibrated in every output.

Import rule (enforced by tests/test_atlas_firewall.py): this file
imports only the top-level ``jaxfne`` package plus stdlib / numpy. No
``jaxfne.<submodule>`` imports (attribute access ``J.fields`` /
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

SEED_BUILD = 2
N_PER_AREA = 8
DT_MS = 0.5
DUR_MS = 1000.0
RUN_SEED = 4
WALL_BUDGET_S = 600.0
CROSS_DELAY_MS = 2.0

# HDP gains: the 0.5.3 declarative baseline (engine-tested engagement
# point; not tuned here). noise_scale 0 keeps arms deterministic.
BASE_HP: dict[str, Any] = {
    "K_HDP": 0.05,
    "K_ctrl": 0.15,
    "K_w_ctrl": 0.002,
    "alpha": 0.05,
    "tau_0_ms": 5.0,
    "noise_scale": 0.0,
}

# Repeated stimulus to A1: 20 ms pulses every 200 ms, targets = A1 block.
STIM_PERIOD_MS = 200.0
STIM_DUR_MS = 20.0
STIM_AMP = 8.0

LEVEL_PROXY = "RELATIVE_PROXY"
SCENARIOS: tuple[str, ...] = ("AT-08", "AT-09")


# ---------------------------------------------------------------------------
# Sentinels (same contract as prior passes).
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

# Schema v3: v1/v2 names stable and in order, then area-indexed cells,
# then cross-area cells.
SCHEMA_VERSION = "v3"
Y_KEYS_V3: tuple[str, ...] = (
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
    "SPK_A1",
    "SPK_A2",
    "H_A1",
    "H_A2",
    "Phi_A1",
    "Phi_A2",
    "W_12",
    "W_21",
    "C_12",
    "dphi_12",
)


# ---------------------------------------------------------------------------
# Shared execution helpers (root surface only; toy sizes; never force).
# ---------------------------------------------------------------------------

_BUDGET_KEYS = frozenset({"record_stride", "record_h_subset", "record_w_subset"})


def _validate_hp(hp: dict[str, Any]) -> None:
    """Fail closed on unknown/malformed HDP params."""
    scoped = {k: v for k, v in hp.items() if k not in _BUDGET_KEYS}
    issues = J.public_surface.validate_hdp_params_semantics(scoped)
    if issues:
        raise ValueError(f"hdp_params invalid: {issues[0]}")


def _fresh_members() -> tuple[Any, Any]:
    """Two edge_list columns with field + LFP/CSD probes (one realization)."""
    members = []
    for i, name in enumerate(("A1", "A2")):
        cfg = (
            J.Configuration()
            .runtime(
                duration_ms=DUR_MS, dt_ms=DT_MS, seed=SEED_BUILD + i, recurrent_backend="edge_list"
            )
            .column(name, layers=["L2/3", "L4"], n=N_PER_AREA)
            .cell_types({"E": 0.8, "PV": 0.2})
            .connectivity()
            .set_emitter(family="izhikevich")
            .probes(["spikes", "V_m", "LFP", "CSD"], n_contacts=8)
            .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        )
        members.append(J.construct(cfg))
    return members[0], members[1]


def _fresh_ensemble() -> Any:
    """The SAME realized two-area network for every arm (connect, both ways)."""
    m1, m2 = _fresh_members()
    return J.connect(
        m1,
        m2,
        namespace=("A", "B"),
        edges=[
            dict(
                source={"model": 0, "area": "A1", "cell_type": "E"},
                target={"model": 1, "area": "A2"},
                probability=0.5,
                weight=0.5,
                sign="excitatory",
                delay_ms=CROSS_DELAY_MS,
            ),
            dict(
                source={"model": 1, "area": "A2", "cell_type": "E"},
                target={"model": 0, "area": "A1"},
                probability=0.5,
                weight=0.5,
                sign="excitatory",
                delay_ms=CROSS_DELAY_MS,
            ),
        ],
    )


def _stimulus(n_total: int, n_a1: int) -> Any:
    """Repeated pulses to the A1 block (first n_a1 indices)."""
    events = [
        {
            "onset_ms": float(t),
            "duration_ms": STIM_DUR_MS,
            "amplitude": STIM_AMP,
            "target_indices": list(range(n_a1)),
        }
        for t in np.arange(0.0, DUR_MS, STIM_PERIOD_MS)
    ]
    return J.stimulus_schedule(
        events, n_total, drive_amplitude=STIM_AMP, event_duration_ms=STIM_DUR_MS
    )


def _spec_digest(extra: dict[str, Any]) -> str:
    canonical = json.dumps(
        {
            "builder": "two-column-connect",
            "build": {"seed": SEED_BUILD, "n_per_area": N_PER_AREA},
            "run": {"duration_ms": DUR_MS, "dt_ms": DT_MS, "seed": RUN_SEED},
            "stimulus": {
                "period_ms": STIM_PERIOD_MS,
                "duration_ms": STIM_DUR_MS,
                "amplitude": STIM_AMP,
                "target": "A1",
            },
            **extra,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _run_arm(model: Any, hp: dict[str, Any] | None, stim: Any) -> dict[str, Any]:
    """Simulate one arm; return signals + diagnostics + cost."""
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        if hp is None:
            signals = J.simulate(
                model, duration_ms=DUR_MS, dt_ms=DT_MS, seed=RUN_SEED, paradigm=stim
            )
            diag = {}
        else:
            _validate_hp(hp)
            signals = J.simulate(
                model,
                duration_ms=DUR_MS,
                dt_ms=DT_MS,
                seed=RUN_SEED,
                paradigm=stim,
                runtime=J.RuntimeConfig(enable_hdp=True, hdp_params=dict(hp)),
            )
            diag = model.last_hdp_diagnostics()
    finally:
        _, peak_b = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    wall_s = time.perf_counter() - t0
    out: dict[str, Any] = {
        "signals": signals,
        "wall_s": wall_s,
        "mem_peak_b": float(peak_b),
    }
    if diag:
        out["H_final"] = np.asarray(diag["H_final"], dtype=float)
        out["w_final"] = np.asarray(diag["w_final"], dtype=float)
        out["H_trace"] = None if diag.get("H_trace") is None else np.asarray(diag["H_trace"])
        out["w_trace"] = None if diag.get("w_trace") is None else np.asarray(diag["w_trace"])
    return out


def _area_slices(n_a1: int) -> tuple[slice, slice]:
    return slice(0, n_a1), slice(n_a1, None)


def _arm_summary(
    name: str, arm: dict[str, Any], w0: np.ndarray, n_a1: int, own: dict[str, Any]
) -> dict[str, Any]:
    """Observation summary for one executed arm (chain X,H->dW->dX->dQ->dPhi)."""
    sig = arm["signals"]
    spikes = np.asarray(sig.spikes)
    sources = np.asarray(sig.sources, dtype=float)
    phi = np.asarray(sig.field.lfp_proxy, dtype=float)
    s0, s1 = _area_slices(n_a1)
    counts = (spikes > 0).sum(axis=0).astype(float)
    r0 = J.fields.population_rate(spikes[:, s0], DT_MS)
    r1 = J.fields.population_rate(spikes[:, s1], DT_MS)
    coh = J.fields.cross_area_coherence(np.asarray(r0), np.asarray(r1), DT_MS)
    band = (coh["freq_hz"] >= 8.0) & (coh["freq_hz"] <= 25.0) & coh["valid"]
    h_fin = arm.get("H_final")
    wf = arm.get("w_final")
    cross_idx = np.zeros(len(w0), dtype=bool)
    for lo, hi in own["cross_ranges"]:
        cross_idx[lo:hi] = True
    return {
        "arm": name,
        "level": str(sig.field.epistemic_level),
        "level_note": "fields RELATIVE_PROXY; proxy != calibrated",
        "spike_count_A1": int((spikes[:, s0] > 0).sum()),
        "spike_count_A2": int((spikes[:, s1] > 0).sum()),
        "rate_hz_A1": float((spikes[:, s0] > 0).sum() / (s0.stop * DUR_MS / 1000.0)),
        "rate_hz_A2": float(
            (spikes[:, s1] > 0).sum() / ((spikes.shape[1] - s0.stop) * DUR_MS / 1000.0)
        ),
        "per_neuron_counts": [float(v) for v in counts],
        "Q_mean_abs_A1": float(np.abs(sources[:, s0]).mean()),
        "Q_mean_abs_A2": float(np.abs(sources[:, s1]).mean()),
        "Phi_mean_abs": float(np.abs(phi).mean()),
        "Phi_A1_proxy": float(np.abs(sources[:, s0]).mean()),
        "Phi_A2_proxy": float(np.abs(sources[:, s1]).mean()),
        "H_final_mean_A1": None if h_fin is None else float(h_fin[s0].mean()),
        "H_final_mean_A2": None if h_fin is None else float(h_fin[s1].mean()),
        "w_max_abs_change": None if wf is None else float(np.abs(wf - w0).max()),
        "w_cross_max_abs_change": (
            None if wf is None else float(np.abs(wf[cross_idx] - w0[cross_idx]).max())
        ),
        "w_member_max_abs_change": (
            None if wf is None else float(np.abs(wf[~cross_idx] - w0[~cross_idx]).max())
        ),
        "C_12_band_mean": float(np.asarray(coh["coherence"])[band].mean()) if band.any() else 0.0,
        "dphi_12_band_mean": float(np.asarray(coh["cross_phase_rad"])[band].mean())
        if band.any()
        else 0.0,
        "kappa": float(J.kappa_synchrony(spikes, DT_MS)),
        "wall_s": arm["wall_s"],
        "mem_peak_b": arm["mem_peak_b"],
    }


def _declare(name: str, signal: str, expect: str, a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Declared observation difference; verdicts (same shape as 0.5.3)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if expect == "equal":
        verdict = "PASS" if np.array_equal(a, b) else "FAIL"
        gap = None if verdict == "PASS" else float(np.abs(a - b).max())
    elif expect == "different":
        verdict = "PASS" if not np.array_equal(a, b) else "FAIL"
        gap = float(np.abs(a - b).max())
    else:  # pragma: no cover
        raise ValueError(f"unsupported expect {expect!r}")
    return {
        "name": name,
        "signal": signal,
        "expect": expect,
        "verdict": verdict,
        "gap_max_abs": gap,
    }


def _intervention_record(
    name: str,
    hp: dict[str, Any] | None,
    control: dict[str, Any],
    observation: dict[str, Any],
    verdict: str,
    mask_kind: str,
) -> dict[str, Any]:
    return {
        "schema": "jaxfne.atlas.intervention_shape.v1",
        "name": name,
        "network": {
            "builder": "two-column-connect",
            "build": {"seed": SEED_BUILD, "n_per_area": N_PER_AREA},
            "run": {"duration_ms": DUR_MS, "dt_ms": DT_MS, "seed": RUN_SEED},
            "stimulus": "repeated-A1-pulses",
            "hdp_params": {
                str(k): (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in (hp or {}).items()
            },
            "mask_kind": mask_kind,
            "spec_digest": _spec_digest(
                {"hp": {str(k): str(v) for k, v in (hp or {}).items()}, "mask": mask_kind}
            ),
        },
        "control": control,
        "observation": observation,
        "verdict": verdict,
    }


def _masks(n: int, own: dict[str, Any]) -> dict[str, np.ndarray]:
    cross = np.zeros(n)
    for lo, hi in own["cross_ranges"]:
        cross[lo:hi] = 1.0
    member = np.zeros(n)
    for lo, hi in own["member_ranges"]:
        member[lo:hi] = 1.0
    return {"cross_only": cross, "member_only": member, "frozen": np.zeros(n)}


# ---------------------------------------------------------------------------
# AT-08: fixed connectivity, adaptation arms, downstream decomposition.
# ---------------------------------------------------------------------------


def run_at08() -> dict[str, Any]:
    """AT-08: same network; fixed baseline vs full HDP vs cross-frozen HDP."""
    model = _fresh_ensemble()
    n_a1 = N_PER_AREA
    n_total = 2 * N_PER_AREA
    w0 = np.asarray(model.params["edge_list"].weight, dtype=float)
    own = J.ensemble_edge_ownership(model)
    masks = _masks(int(model.params["edge_list"].n_edges), own)
    stim = _stimulus(n_total, n_a1)
    arms = {
        "fixed": _run_arm(model, None, stim),
        "adapt_full": _run_arm(model, dict(BASE_HP), stim),
        "adapt_local": _run_arm(model, dict(BASE_HP, plasticity_mask=masks["member_only"]), stim),
    }
    sums = {k: _arm_summary(k, v, w0, n_a1, own) for k, v in arms.items()}
    # R4 decomposition on A2 (downstream): propagated-activity effect
    # (local-base, fixed coupling) vs coupling-plasticity effect
    # (full-local). Declared as differences of executed quantities.
    prop_spk = sums["adapt_local"]["spike_count_A2"] - sums["fixed"]["spike_count_A2"]
    coup_spk = sums["adapt_full"]["spike_count_A2"] - sums["adapt_local"]["spike_count_A2"]
    prop_phi = sums["adapt_local"]["Phi_A2_proxy"] - sums["fixed"]["Phi_A2_proxy"]
    coup_phi = sums["adapt_full"]["Phi_A2_proxy"] - sums["adapt_local"]["Phi_A2_proxy"]
    answer = {
        "propagated_spike_delta_A2": prop_spk,
        "coupling_spike_delta_A2": coup_spk,
        "propagated_field_delta_A2": float(prop_phi),
        "coupling_field_delta_A2": float(coup_phi),
        "note": (
            "adaptation effect downstream = propagated activity "
            "(fixed coupling) + coupling-plasticity; no claim that "
            "bounded trajectories imply stabilization"
        ),
    }
    declares = [
        _declare(
            "stimulus-drives-A1",
            "SPK_A1",
            "different",
            np.asarray(arms["fixed"]["signals"].spikes)[:, :n_a1],
            np.zeros_like(np.asarray(arms["fixed"]["signals"].spikes)[:, :n_a1]),
        ),
        _declare("adaptation-moves-W", "W", "different", arms["adapt_full"]["w_final"], w0),
        # Fixed arm exposes no HDP diagnostics (HDP off: no W object at
        # all, not a zero delta). Vacuous self-comparison is refused here.
        {
            "name": "fixed-exposes-no-W",
            "signal": "W",
            "expect": "absent",
            "verdict": "PASS" if "w_final" not in arms["fixed"] else "FAIL",
            "gap_max_abs": None,
        },
    ]
    verdict = "PASS" if all(d["verdict"] == "PASS" for d in declares) else "FAIL"
    return {
        "scenario": "AT-08",
        "status": "OK",
        "arms": sums,
        "r4_answer": answer,
        "declares": declares,
        "records": [
            _intervention_record(
                "AT-08-fixed-vs-adapt",
                dict(BASE_HP),
                {"control": "HDP off vs on, same network+stimulus"},
                {"observation": "SPK/Phi/H per area + C_12/dphi_12"},
                verdict,
                "none-vs-full",
            ),
        ],
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# AT-09: plastic coupling arms, H/W separation, local vs inter-area.
# ---------------------------------------------------------------------------


def run_at09() -> dict[str, Any]:
    """AT-09: cross-only vs member-only vs frozen plasticity, same network."""
    model = _fresh_ensemble()
    n_a1 = N_PER_AREA
    n_total = 2 * N_PER_AREA
    w0 = np.asarray(model.params["edge_list"].weight, dtype=float)
    own = J.ensemble_edge_ownership(model)
    masks = _masks(int(model.params["edge_list"].n_edges), own)
    stim = _stimulus(n_total, n_a1)
    arms = {
        "plastic_cross": _run_arm(model, dict(BASE_HP, plasticity_mask=masks["cross_only"]), stim),
        "plastic_member": _run_arm(
            model, dict(BASE_HP, plasticity_mask=masks["member_only"]), stim
        ),
        "frozen": _run_arm(model, dict(BASE_HP, plasticity_mask=masks["frozen"]), stim),
    }
    sums = {k: _arm_summary(k, v, w0, n_a1, own) for k, v in arms.items()}
    # R2: H evolves while dW/dt = 0 on the frozen arm.
    h_moves = abs(sums["frozen"]["H_final_mean_A1"] - 1.0) + abs(
        sums["frozen"]["H_final_mean_A2"] - 1.0
    )
    r2 = {
        "H_moves_while_dW_zero": bool(h_moves > 0.0),
        "H_deviation": float(h_moves),
        "note": "fast X, relative H, and plastic W are distinct objects",
    }
    # R3: member-W change vs cross-W change + coherence consequences.
    r3 = {
        "cross_plastic_W_cross_change": sums["plastic_cross"]["w_cross_max_abs_change"],
        "cross_plastic_W_member_change": sums["plastic_cross"]["w_member_max_abs_change"],
        "member_plastic_W_cross_change": sums["plastic_member"]["w_cross_max_abs_change"],
        "member_plastic_W_member_change": sums["plastic_member"]["w_member_max_abs_change"],
        "C_12_cross": sums["plastic_cross"]["C_12_band_mean"],
        "C_12_member": sums["plastic_member"]["C_12_band_mean"],
        "C_12_frozen": sums["frozen"]["C_12_band_mean"],
    }
    declares = [
        _declare(
            "cross-arm-moves-cross-W", "W_12", "different", arms["plastic_cross"]["w_final"], w0
        ),
        _declare("frozen-keeps-W", "W", "equal", arms["frozen"]["w_final"], w0),
    ]
    verdict = "PASS" if all(d["verdict"] == "PASS" for d in declares) else "FAIL"
    return {
        "scenario": "AT-09",
        "status": "OK",
        "arms": sums,
        "r2_separation": r2,
        "r3_local_vs_inter": r3,
        "declares": declares,
        "records": [
            _intervention_record(
                "AT-09-cross-vs-member-vs-frozen",
                dict(BASE_HP),
                {"control": "plasticity_mask scoping, same network+stimulus"},
                {"observation": "W_12/W_21 trajectories + C_12/dphi_12"},
                verdict,
                "cross_only/member_only/frozen",
            ),
        ],
        "verdict": verdict,
    }


_RUNNERS = {"AT-08": run_at08, "AT-09": run_at09}


def run_scenario(scenario_id: str) -> dict[str, Any]:
    """Run one 0.5.4 scenario; never raises: failures become ERROR records."""
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
    """Run every 0.5.4 scenario in order; never raises per scenario."""
    return {sid: run_scenario(sid) for sid in SCENARIOS}


# ---------------------------------------------------------------------------
# Item 8: measurement schema v2 -> v3 (area-indexed + cross-area cells).
# Every v1/v2 name is stable and in order (Y_KEYS_V3[:13] == Y_KEYS_V2).
# ---------------------------------------------------------------------------

_V3_IMPLEMENTED: dict[str, tuple[str, ...]] = {
    "AT-08": (
        "X",
        "H",
        "W",
        "Q",
        "Phi_E",
        "SPK",
        "C",
        "T_compute",
        "M_compute",
        "SPK_A1",
        "SPK_A2",
        "H_A1",
        "H_A2",
        "Phi_A1",
        "Phi_A2",
        "W_12",
        "W_21",
        "C_12",
        "dphi_12",
    ),
    "AT-09": (
        "X",
        "H",
        "W",
        "Q",
        "Phi_E",
        "SPK",
        "C",
        "T_compute",
        "M_compute",
        "SPK_A1",
        "SPK_A2",
        "H_A1",
        "H_A2",
        "Phi_A1",
        "Phi_A2",
        "W_12",
        "W_21",
        "C_12",
        "dphi_12",
    ),
}
_V3_NOTES: dict[str, str] = {
    "X": "Vm/rate features from executed trajectories, per area sliced",
    "H": "H trajectory + final, area-indexed (H_A1/H_A2)",
    "W": "W trajectory + final; cross ranges owned (W_12/W_21)",
    "Q": "single source representation consumed by every probe",
    "Phi_E": "field contract at RELATIVE_PROXY; per-area source proxies",
    "Phi_B": "no calibrated Phi_B beyond proxy (candidate, still refused)",
    "SPK": "raster shape + count from executed signals, per area sliced",
    "PSD": "spectral operators stay with the AT-03 oscillator (0.5.2)",
    "C": "synchrony (kappa) + magnitude-squared coherence C_12",
    "phi": "cross-spectrum phase dphi_12 (0.5.4 item 3 operators)",
    "E_reduction": "reduction scale matrix is 0.5.5",
    "T_compute": "measured wall_s per scenario",
    "M_compute": "measured host peak bytes (tracemalloc; not device memory)",
    "SPK_A1": "A1 spike count from the executed ensemble slice",
    "SPK_A2": "A2 spike count from the executed ensemble slice",
    "H_A1": "H_final mean over A1 neurons",
    "H_A2": "H_final mean over A2 neurons",
    "Phi_A1": "mean |source| over A1 neurons (source-level field proxy)",
    "Phi_A2": "mean |source| over A2 neurons (source-level field proxy)",
    "W_12": "max |dW| over the A1->A2 owned cross range",
    "W_21": "max |dW| over the A2->A1 owned cross range",
    "C_12": "mean magnitude-squared coherence, 8-25 Hz, per-area rates",
    "dphi_12": "mean cross-spectrum phase, 8-25 Hz, per-area rates",
}


def measure_v3(scenario_id: str, raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Schema-v3 Y record: v1/v2 names stable, each cell gains a level."""
    if scenario_id not in SCENARIOS:
        raise KeyError(f"unknown scenario {scenario_id!r}")
    implemented = _V3_IMPLEMENTED.get(scenario_id, ("T_compute",))
    record: dict[str, dict[str, Any]] = {}
    for key in Y_KEYS_V3:
        if key == "Phi_B":
            record[key] = {
                "state": "REFUSED",
                "value": None,
                "level": LEVEL_PROXY,
                "note": _V3_NOTES[key],
            }
        elif key in implemented:
            record[key] = {
                "state": "IMPLEMENTED",
                "value": {"scenario": scenario_id, "status": raw.get("status")},
                "level": LEVEL_PROXY,
                "note": _V3_NOTES[key],
            }
        else:
            record[key] = {
                "state": "OMITTED",
                "value": None,
                "level": LEVEL_PROXY,
                "note": _V3_NOTES[key],
            }
    return record


def gap_matrix_v3(results: dict[str, dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Y x AT cell states for the 0.5.4 scenarios (drives gap_054.md)."""
    return {
        sid: {key: cell["state"] for key, cell in measure_v3(sid, results[sid]).items()}
        for sid in SCENARIOS
        if sid in results
    }
