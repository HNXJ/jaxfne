"""V1-PFC continuous AAAB local-oddball adaptation paradigm -- first CPU smoke test.

Target milestone: 0.4.6 (see artifacts/developer/plans.json's
``v1-pfc-continuous-aaab-adaptation-paradigm`` item and
docs/tutorials/05_v1_pfc_dual_column.md). This script implements the design
scoped there: two 100-neuron canonical laminar columns (V1, PFC), both
HDP-enabled, connected V1->PFC feedforward, driven by a fixed 1000ms trial
structure repeated every trial:

    fx(0-100) - p1=A(100-200) - d1(200-300) - p2=A(300-400) - d2(400-500)
    - p3=A(500-600) - d3(600-700) - p4=B(700-800, deviant) - d4(800-900)
    - rw(900-1000)

A and B are 40Hz sinusoidal (AC) current drives targeting disjoint/overlapping
tuning-group populations within V1's L4 and L6 layers (AB responds to both,
A only to A-events, B only to B-events). HDP state (H_final/w_final) carries
over trial-to-trial via Model.with_hdp_initial_state, so adaptation is genuine
across the whole run, not reset every trial.

Three open design decisions this script resolves concretely (previously
flagged unresolved in plans.json):

1. Tuning-group identity (AB/A/B) is NOT encoded via NeuronType.name -- the
   sign-detection logic in neuronal_tensor.py checks
   ``source_neuron_type == "E"`` exactly to decide excitatory vs inhibitory,
   so renaming to e.g. "E_AB" would silently misclassify these neurons as
   inhibitory. Instead, V1's L4 and L6 are declared as single-NeuronType
   ("E", fraction=1.0) tuning-only layers, and AB/A/B membership is a
   positional post-construction tag by stable neuron_table() order (see
   point 3 below for the exact per-layer split).

2. True 40Hz AC (not a flat plateau) required extending
   ``jaxfne._signals.StimulusSchedule.to_array`` with an optional
   ``frequency_hz`` event key (backward compatible; absent = original flat
   amplitude). See that module for the change.

3. DISCOVERED GAP (not previously documented anywhere): despite each
   ``Layer`` declaring its own ``n_neurons``,
   ``neuronal_tensor.neuronal_tensor_to_configuration`` never calls
   ``Configuration.layer_fractions()``, so the constructed model silently
   falls back to jaxfne's hardcoded default layer-thickness split
   (``core._SUITE2_LAYER_FRACTIONS``: L1=10%, L2=15%, L3=20%, L4=10%,
   L5=30%, L6=15% of the area's total neuron count) instead of respecting
   the tensor's declared per-layer sizes. Verified directly: a V1 area
   declared with L1=10/L2=25/L3=15/L4=15/L5=20/L6=15 constructed as
   L1=10/L2=15/L3=20/L4=10/L5=30/L6=15 (the hardcoded default) for a
   100-neuron area regardless. This script does NOT attempt to fix that
   bridge function in the same pass as this paradigm (real fix belongs in
   neuronal_tensor.py, reviewed on its own) -- it works within the existing
   default split instead: for a 100-neuron area, L4 always gets 10 neurons
   and L6 always gets 15, for 25 total pure-E tuning slots (not the 30 the
   original spec called for). Tuning groups are split AB=9/A=8/B=8 across
   the combined 25 slots -- a documented compromise, not a silent one.

Design tradeoff (stated, not hidden): V1's L4 and L6 have PV/SST/VIP removed
entirely (pure-E tuning layers) to host the AB/A/B groups; L1, L2, L3, L5
and all of PFC keep the canonical E:PV:SST:VIP cell-type fractions (layer
sizes for both areas follow the hardcoded default split from point 3 above,
not the originally-intended canonical column proportions).

Usage: PYTHONPATH=. python3 scripts/v1_pfc_continuous_aaab_smoke_test.py [n_trials]
Default n_trials=10 (the authorized first validation step; full spec target
is 1000 trials and is NOT run by default -- pass an explicit n_trials to
opt into a longer run).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import jaxfne as jtfne
from jaxfne.core import RuntimeConfig
from jaxfne.neuronal_tensor import (
    Area,
    AreaConnection,
    Layer,
    NeuronType,
    NeuronalTensor,
    PlasticParams,
    construct_neuronal_tensor,
)

OUTPUT_DIR = Path("outputs/v1_pfc_continuous_aaab_smoke_test")

TRIAL_DURATION_MS = 1000.0
DT_MS = 0.1
BIN_MS = 100.0
DRIVE_AMPLITUDE = 5.0  # native Izhikevich-current units; moderate, not saturating
DRIVE_FREQUENCY_HZ = 40.0

# Canonical L1..L6 fractions (from jaxfne/configs/canonical-v1-column-1000n.json,
# scaled to 1/10th neuron count -- the "100-neuron canonical column" reading).
_CANONICAL_FRACTIONS = {
    "L1": {"E": 0.5, "SST": 0.15, "VIP": 0.35},
    "L2": {"E": 0.648, "PV": 0.2, "SST": 0.1, "VIP": 0.052},
    "L3": {"E": 0.8, "PV": 0.08, "SST": 0.08, "VIP": 0.04},
    "L5": {"E": 0.88, "PV": 0.06, "SST": 0.04, "VIP": 0.02},
}


def _canonical_layer(name: str, n_neurons: int) -> Layer:
    fracs = _CANONICAL_FRACTIONS[name]
    return Layer(
        name=name,
        n_neurons=n_neurons,
        neuron_types=tuple(NeuronType.make(ct, fraction=f) for ct, f in fracs.items()),
    )


# Layer sizes actually realized by neuronal_tensor_to_configuration's hardcoded
# default split (core._SUITE2_LAYER_FRACTIONS), for a 100-neuron area -- see
# module docstring point 3. Layer.n_neurons below is declared for
# documentation/intent, but construction ignores it; these are the real sizes.
_ACTUAL_LAYER_SIZES = {"L1": 10, "L2": 15, "L3": 20, "L4": 10, "L5": 30, "L6": 15}


def build_v1_area() -> Area:
    """100-neuron V1 column (canonical cell-type fractions in L1/L2/L3/L5;
    L4/L6 are pure-E tuning layers). Layer sizes follow
    ``_ACTUAL_LAYER_SIZES`` (the hardcoded default the bridge actually uses),
    not canonical proportions -- see module docstring point 3."""
    layers = [
        _canonical_layer("L1", _ACTUAL_LAYER_SIZES["L1"]),
        _canonical_layer("L2", _ACTUAL_LAYER_SIZES["L2"]),
        _canonical_layer("L3", _ACTUAL_LAYER_SIZES["L3"]),
        Layer(name="L4", n_neurons=_ACTUAL_LAYER_SIZES["L4"],
              neuron_types=(NeuronType.make("E", fraction=1.0),)),
        _canonical_layer("L5", _ACTUAL_LAYER_SIZES["L5"]),
        Layer(name="L6", n_neurons=_ACTUAL_LAYER_SIZES["L6"],
              neuron_types=(NeuronType.make("E", fraction=1.0),)),
    ]
    return Area(name="V1", layers=tuple(layers))


def build_pfc_area() -> Area:
    """100-neuron PFC column: canonical E:PV:SST:VIP cell-type fractions,
    fully generic/untuned. Layer sizes follow ``_ACTUAL_LAYER_SIZES``."""
    layers = [
        _canonical_layer("L1", _ACTUAL_LAYER_SIZES["L1"]),
        _canonical_layer("L2", _ACTUAL_LAYER_SIZES["L2"]),
        _canonical_layer("L3", _ACTUAL_LAYER_SIZES["L3"]),
        Layer(name="L4", n_neurons=_ACTUAL_LAYER_SIZES["L4"],
              neuron_types=(NeuronType.make("E", fraction=0.75),
                            NeuronType.make("PV", fraction=0.18),
                            NeuronType.make("SST", fraction=0.04),
                            NeuronType.make("VIP", fraction=0.03))),
        _canonical_layer("L5", _ACTUAL_LAYER_SIZES["L5"]),
        Layer(name="L6", n_neurons=_ACTUAL_LAYER_SIZES["L6"],
              neuron_types=(NeuronType.make("E", fraction=0.9),
                            NeuronType.make("PV", fraction=0.0533),
                            NeuronType.make("SST", fraction=0.0267),
                            NeuronType.make("VIP", fraction=0.02))),
    ]
    return Area(name="PFC", layers=tuple(layers))


def build_tensor() -> NeuronalTensor:
    v1 = build_v1_area()
    pfc = build_pfc_area()
    feedforward = [
        AreaConnection(
            source_area="V1", source_layer=layer, source_neuron_type="E",
            target_area="PFC", target_layer="L4", target_neuron_type="E",
            mechanism="AMPA",
            # PlasticParams.H defaults to 0.0, which is outside the valid HDP
            # H range (H_min=0.1, H_max=10.0) -- construct_neuronal_tensor
            # averages this into the target neurons' initial HDP state
            # (with_hdp_initial_state), so an unset H silently seeds PFC's L4
            # neurons at H0=0.0 and blows up the HDP integration immediately
            # (verified: this was the actual cause of the H_final/w_final NaN
            # seen during development, not a jaxfne-side HDP bug).
            plastic=PlasticParams(H=1.0),
        )
        for layer in ("L4", "L6")
    ]
    return NeuronalTensor(
        areas=(v1, pfc), area_connections=tuple(feedforward),
        name="v1_pfc_continuous_aaab",
    )


def tuning_group_indices(model: "jtfne.core.Model") -> dict[str, list[int]]:
    """Positional AB/A/B tagging over the combined 25 pure-E neurons in V1's
    L4 (10 neurons) + L6 (15 neurons), in stable neuron_table() order.
    Split AB=9/A=8/B=8 (25 total) -- see module docstring point 3 for why
    this isn't 10/10/10 as originally specified."""
    rows = model.neuron_table()
    layer_ids: list[int] = []
    for layer in ("L4", "L6"):
        ids = sorted(r["neuron_id"] for r in rows if r["area"] == "V1" and r["layer"] == layer)
        expected = _ACTUAL_LAYER_SIZES[layer]
        if len(ids) != expected:
            raise ValueError(f"expected {expected} pure-E neurons in V1/{layer}, got {len(ids)}")
        layer_ids.extend(ids)
    if len(layer_ids) != 25:
        raise ValueError(f"expected 25 combined tuning-layer neurons, got {len(layer_ids)}")
    return {
        "AB": layer_ids[0:9],
        "A": layer_ids[9:17],
        "B": layer_ids[17:25],
    }


def build_trial_schedule(n_neurons: int, groups: dict[str, list[int]]) -> "jtfne.core.StimulusSchedule":
    """One 1000ms fixed AAAB trial: fx-p1(A)-d1-p2(A)-d2-p3(A)-d3-p4(B,deviant)-d4-rw.

    Only the 4 stimulus windows (p1-p4) get drive events; fx/d1-d4/rw are
    left unrepresented (StimulusSchedule.to_array zero-initializes, so no
    explicit zero-amplitude marker event is needed -- unlike the
    ParadigmCondition->StimulusSchedule conversion path, this schedule is
    built directly, so there is no is_drive heuristic to route around).
    """
    a_targets = sorted(set(groups["AB"]) | set(groups["A"]))
    b_targets = sorted(set(groups["AB"]) | set(groups["B"]))
    stim_onsets = {
        "p1": (100.0, a_targets), "p2": (300.0, a_targets),
        "p3": (500.0, a_targets), "p4": (700.0, b_targets),
    }
    events = tuple(
        {
            "label": label,
            "onset_ms": onset,
            "duration_ms": BIN_MS,
            "amplitude": DRIVE_AMPLITUDE,
            "frequency_hz": DRIVE_FREQUENCY_HZ,
            "target_indices": targets,
            "is_drive_event": True,
        }
        for label, (onset, targets) in stim_onsets.items()
    )
    return jtfne.core.StimulusSchedule(events=events, n_neurons=n_neurons)


def run_smoke_test(n_trials: int = 10, seed: int = 0) -> dict:
    tensor = build_tensor()
    model = construct_neuronal_tensor(tensor, seed=seed, duration_ms=TRIAL_DURATION_MS, dt_ms=DT_MS)
    groups = tuning_group_indices(model)
    n_neurons = model.params["emitter"].n_neurons
    schedule = build_trial_schedule(n_neurons, groups)

    # NOTE (verified this session): alpha=beta=gamma=delta=C_spike=0.0 is
    # RuntimeConfig's documented "null control" for H -- it stays pinned at
    # its 1.0 equilibrium by design (confirmed: H_final/w_final are bit-
    # identical across all 10 trials of a real smoke-test run). This config
    # validates the adaptation *pipeline* end-to-end (HDP enabled, state
    # carried trial-to-trial via with_hdp_initial_state, no NaN) but shows
    # no actual adaptation *signal* yet -- wiring a genuinely-driving preset
    # (e.g. jaxfne.hdp_network.DEFAULT_HDP's fuller parameter schema, used
    # elsewhere with validated cube-law size scaling) is separate follow-up
    # work, not attempted here to avoid blind-tuning a subsystem with
    # documented open stability issues (see hdp-stability-formula-design-
    # and-validation in artifacts/developer/plans.json).
    hdp_params = {
        "K_HDP": 1.0, "tau_0_ms": 200.0, "alpha": 0.0, "beta": 0.0,
        "gamma": 0.0, "delta": 0.0, "C_spike": 0.0, "K_ctrl": 5.0,
        "barrier_c": 0.0, "barrier_d": 0.0,
    }
    runtime_cfg = RuntimeConfig(
        dtype="float32", backend="cpu", enable_hdp=True, hdp_params=hdp_params,
    )

    trial_summaries = []
    for trial in range(n_trials):
        signals = jtfne.simulate(
            model, duration_ms=TRIAL_DURATION_MS, dt_ms=DT_MS,
            seed=seed + trial, runtime=runtime_cfg, paradigm=schedule,
        )
        diag = model.last_hdp_diagnostics()
        summary = signals.summary()
        trial_summaries.append({
            "trial": trial,
            "spike_rate_hz_mean": summary.get("spike_rate_hz_mean"),
            "H_final_mean": float(np.asarray(diag["H_final"]).mean()) if diag else None,
            "w_final_mean": float(np.asarray(diag["w_final"]).mean()) if diag else None,
        })
        if diag is not None:
            model = model.with_hdp_initial_state(H0=diag["H_final"], w0=diag["w_final"])

    return {
        "n_trials": n_trials,
        "n_neurons": n_neurons,
        "tuning_group_sizes": {k: len(v) for k, v in groups.items()},
        "trial_summaries": trial_summaries,
    }


if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    result = run_smoke_test(n_trials=n_trials)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "smoke_test_receipt.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nReceipt written to {out_path}")
