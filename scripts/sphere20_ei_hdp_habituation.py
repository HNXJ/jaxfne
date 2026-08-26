"""0.4.7 practical release-condition test: long-term habituation/adaptation on a
20-neuron (15E/5PV) uniform-random-sphere network under a trial-structured
paradigm with two overlapping stimulus groups, chained across turns/trials
via HDP with TRUE full-state continuity (jaxfne._pipeline.compile_step_fn/
scan_network's DynamicState -- v, u, prev_spikes, syn_state, H, w all carried
forward) so each trial's final state seeds the next trial's initial state.
Model.with_hdp_initial_state() was tried first and found to only carry H/w,
silently resetting v/u/spikes/syn_state every call -- see the landmine note
in build_model()/run() below.

Build path (as specified): Configuration -> NeuronalTensor -> construct (Model)
-> simulate at baseline -> chained continuous-task trials -> observe Wee/Wei/
Wie/Wii, per-cell-type H, firing rates, a coarse oscillation readout, and a
stim4-response metric comparing AAAA vs AAAB trials.

Paradigm (via jtfne.general_sequential_oddball_paradigm, see build_group_a_b()/
build_trial_events()): two overlapping E-neuron stimulus groups, A (6 cells)
and B (6 cells), sharing 2 "AB" cells. A trial is 4 repetitions of
[500+-250ms random delay, then 500ms stimulus]; the stimulus in each
repetition targets group A or B per the trial's condition sequence -- "AAAA"
(habituate to A four times) or "AAAB" (habituate to A three times, then
switch to B on the last repetition -- the oddball). Comparing the shared AB
cells' response on stim4 between AAAA and AAAB tests whether habituation to
A transfers to B through the 2 shared cells, versus B's unique cells which
were never driven by A.

This is a computational scaffold / proxy diagnostic, not a physical or
biological validation claim (see artifacts/AGENTS.md truth gates).
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import jax.numpy as jnp

import jaxfne as jtfne
from jaxfne.neuronal_tensor import (
    Area,
    InterConnection,
    Layer,
    NeuronalTensor,
    NeuronType,
    PlasticParams,
    neuronal_tensor_to_configuration,
)
from jaxfne.hdp_network import DEFAULT_HDP

N_E = 15
N_I = 5
N = N_E + N_I
SEED = 0
DT_MS = 0.5
N_TURNS = 20
SPHERE_RADIUS_MM = 0.25

# Two overlapping E-neuron stimulus groups: 6 cells respond to A, 6 respond
# to B, 2 of those are shared "AB" cells driven by both. |A|=6, |B|=6,
# |A&B|=2 -> 10 distinct E cells involved (out of 15), 5 uninvolved.
GROUP_A_SIZE = 6
GROUP_B_SIZE = 6
GROUP_AB_SHARED = 2

# Trial structure: 4 repetitions of [500+-250ms random delay, 500ms stimulus].
TRIAL_N_REPS = 4
STIM_DURATION_MS = 500.0
DELAY_MEAN_MS = 500.0
DELAY_JITTER_MS = 250.0  # delay ~ Uniform(DELAY_MEAN-JITTER, DELAY_MEAN+JITTER)
STIM_AMPLITUDE = 6.0  # same amplitude for A and B (user decision -- isolates
                       # the population-overlap effect from an amplitude confound)

# Two conditions: "AAAA" (habituate to A all 4 repetitions) and "AAAB"
# (habituate to A three times, then the oddball -- switch to B on stim4).
SEQUENCE_AAAA = ("A", "A", "A", "A")
SEQUENCE_AAAB = ("A", "A", "A", "B")
ODDBALL_TURNS = (10, 15)  # trials that run AAAB instead of AAAA

# NOTE: tried switching to DEFAULT_HDP_DESYNC (gamma=0.5) so H would be
# rate-dependent -- diverged even at E_TO_PV_GAIN=1.0 (H/weights pin at their
# ceilings by turn ~5-9, rates blow up to 80-170Hz). DESYNC/AAAB's much
# weaker restoring force (K_ctrl=0.15 vs DEFAULT_HDP's 5.0) was verified for
# the V1-PFC continuous-adaptation topology at a different scale and does not
# transfer to this small, fully-recurrent 20-neuron topology at any gain
# tested. Reverted to DEFAULT_HDP: H is not rate-dependent here (gamma=0),
# but the run is bounded and stable across all 20 turns with a real
# habituation-like PV decay (see run() below).
#
# K_w_ctrl=0.0001: informed prior from the tracked hdp-k-w-ctrl-default-
# runaway-gap finding (DEFAULT_HDP's own K_w_ctrl=0.0 lets weight magnitude
# drift unbounded on long/custom-topology chained runs; the V1-PFC preset's
# 0.001 was found to over-correct -- collapses differentiation ~150x -- on a
# generic custom topology; 0.0001 was the single-seed-tested middle ground
# that stayed bounded AND kept real differentiation).
HDP_PARAMS = dict(DEFAULT_HDP)
HDP_PARAMS["K_w_ctrl"] = 0.0001

# E->PV gain multiplier (on top of the uniform w_mech=1.0 baseline used for
# EE/IE/II). Found via direct investigation: at w_mech=1.0 uniformly, PV
# stayed at ~1 Hz even when E was driven to ~60 Hz (tested via the standard
# cold-state Model.simulate() path, independent of the chained-turn carryover
# mechanism) -- PV's own lower baseline drive (3.0 vs E's 5.0) and fast-
# adapting Izhikevich params (a=0.1, d=2.0) make it chronically under-driven
# unless E->PV coupling is proportionally stronger than the other 3 blocks.
# Swept 1x-80x directly: 1x-4x had ZERO effect on PV rate (still silent after
# turn 0) despite edge weight scaling correctly (verified via edge_list
# inspection) -- this network needs an order-of-magnitude gain, not a small
# multiplier. 10x produces a genuine decaying PV response (7.8->2.8 Hz over
# 6 turns while E stays ~11 Hz flat); 20x-40x sustains PV without runaway;
# 80x trends upward turn over turn (heading toward instability) -- 10x chosen
# as the value that shows real habituation rather than either silence or
# unbounded growth.
E_TO_PV_GAIN = 10.0


def build_tensor() -> NeuronalTensor:
    """Step 1+2: config -> NeuronalTensor. One area, one flat layer, 15E/5PV,
    fully recurrent (EE/EI/IE/II) with plastic.H=1.0 explicitly set on every
    connection (footgun: PlasticParams.H defaults to 0.0, outside HDP's valid
    [0.1, 10.0] range -- leaving it default seeds H0 outside range and NaNs
    HDP integration from step 0 once enabled)."""
    layer = Layer(
        name="L",
        n_neurons=N,
        neuron_types=[
            NeuronType.make("E", fraction=N_E / N),
            NeuronType.make("PV", fraction=N_I / N),
        ],
    )

    def ic(src_type: str, tgt_type: str, mechanism: str, w_mech: float = 1.0) -> InterConnection:
        return InterConnection(
            source_layer="L",
            source_neuron_type=src_type,
            target_layer="L",
            target_neuron_type=tgt_type,
            mechanism=mechanism,
            plastic=PlasticParams(H=1.0, w_mech=w_mech),
        )

    area = Area(
        name="sphere20",
        layers=[layer],
        inter_connections=[
            ic("E", "E", "AMPA"),
            ic("E", "PV", "AMPA", w_mech=E_TO_PV_GAIN),
            ic("PV", "E", "GABA"),
            ic("PV", "PV", "GABA"),
        ],
    )
    return NeuronalTensor(areas=[area], name="sphere20_ei_hdp")


def sample_uniform_in_sphere(n: int, radius: float, seed: int) -> np.ndarray:
    """True uniform-in-volume sampling inside a sphere via rejection sampling.

    jaxfne's own `.uniform3d()` metadata / Geometry3D default distribution is
    a uniform box/cylinder, not a sphere -- this repo has no built-in sphere
    primitive, so positions are overridden post-construct() the same way
    `construct_neuronal_tensor`'s own Pose3D placement does (replace
    `model.params["positions"]` + `model.static["neuron_metadata"]`).
    """
    rng = np.random.default_rng(seed)
    pts: list[np.ndarray] = []
    while len(pts) < n:
        batch = rng.uniform(-radius, radius, size=(max(n * 2, 8), 3))
        norms = np.linalg.norm(batch, axis=1)
        keep = batch[norms <= radius]
        pts.extend(list(keep))
    return np.asarray(pts[:n], dtype=np.float64)


def override_positions_with_sphere(model: "jtfne.Model", radius: float, seed: int) -> "jtfne.Model":
    positions = jnp.asarray(sample_uniform_in_sphere(N, radius, seed), dtype=jnp.float32)
    rows = model.neuron_table()
    updated_rows = [
        dict(row, x=float(positions[i, 0]), y=float(positions[i, 1]), z=float(positions[i, 2]))
        for i, row in enumerate(rows)
    ]
    new_static = dict(model.static)
    new_static["neuron_metadata"] = updated_rows
    if "geometry" in new_static:
        new_static["geometry"] = dict(new_static["geometry"], neuron_rows=updated_rows)
    new_params = dict(model.params)
    new_params["positions"] = positions
    return dataclasses.replace(model, params=new_params, static=new_static)


NOMINAL_TRIAL_DURATION_MS = TRIAL_N_REPS * (DELAY_MEAN_MS + STIM_DURATION_MS)  # metadata only -- see below


def build_model() -> "jtfne.Model":
    """Step 3: NeuronalTensor -> Configuration bridge -> construct() -> Model."""
    tensor = build_tensor()
    # duration_ms here is cfg metadata only (manifest/provenance) -- actual
    # trial length varies turn-to-turn because of the random delays, and is
    # controlled per-turn by the length of the array handed to scan_network,
    # not by anything read from cfg.metadata at construct() time.
    cfg = neuronal_tensor_to_configuration(
        tensor, seed=SEED, duration_ms=NOMINAL_TRIAL_DURATION_MS, dt_ms=DT_MS
    )
    # LANDMINE (found during this build, not previously documented): construct()
    # ALSO generates its own default same-area dense random recurrent wiring
    # on top of the tensor bridge's explicit InterConnection edges -- without
    # this override, edge_list carries BOTH sets (double-wired network).
    # Verified via direct edge_list inspection: within_gain=0.0 zeroes exactly
    # the default component, leaving only the 4 declared InterConnection rules.
    cfg = cfg.connectivity(within_area="all_to_all_uniform_random", within_gain=0.0)

    # NOTE: cfg.hdp(...) is declared for provenance/inspection (shows up in
    # manifest()/metadata), but is NOT actually inherited by Model.simulate(sim)
    # -- only the top-level jtfne.simulate(model, ...) function inherits
    # cfg-declared runtime (verified: _construct.py's module-level simulate()
    # calls _runtime_config_from_metadata(cfg.metadata) when no explicit
    # runtime= is given; Simulation.resolved_runtime, which Model.simulate uses
    # directly, does not touch cfg.metadata at all). Since turn-chaining below
    # calls turn_model.simulate(sim, ...) (the OO method, needed for
    # with_hdp_initial_state to matter), the HDP runtime must be passed
    # explicitly via Simulation(runtime=...) at every simulate() call site --
    # see build_simulation() below.
    cfg = cfg.hdp(**HDP_PARAMS)

    model = jtfne.construct(cfg)
    model = override_positions_with_sphere(model, SPHERE_RADIUS_MM, SEED)
    return model


def build_groups(model: "jtfne.Model", seed: int) -> tuple[list[int], list[int], list[int]]:
    """Pick the two overlapping E-neuron stimulus groups: group_a (6 cells),
    group_b (6 cells), sharing group_ab (2 cells) -- group_a & group_b ==
    group_ab exactly, group_a and group_b are each otherwise disjoint from
    the other's unique cells."""
    rows = model.neuron_table()
    e_indices = [r["neuron_id"] for r in rows if r["cell_type"] == "E"]
    rng = np.random.default_rng(seed)

    shared = sorted(rng.choice(e_indices, size=GROUP_AB_SHARED, replace=False).tolist())
    remaining = [i for i in e_indices if i not in shared]
    a_only = rng.choice(remaining, size=GROUP_A_SIZE - GROUP_AB_SHARED, replace=False).tolist()
    remaining = [i for i in remaining if i not in a_only]
    b_only = rng.choice(remaining, size=GROUP_B_SIZE - GROUP_AB_SHARED, replace=False).tolist()

    group_a = sorted(shared + a_only)
    group_b = sorted(shared + b_only)
    assert sorted(set(group_a) & set(group_b)) == shared
    return group_a, group_b, shared


def build_trial_events(
    group_a: list[int], group_b: list[int], sequence: tuple[str, str, str, str], seed: int
) -> tuple[list[dict], float]:
    """Build one trial's explicit event list: TRIAL_N_REPS repetitions of
    [random delay, STIM_DURATION_MS stimulus], stimulus target group per
    `sequence`'s token ("A" -> group_a, "B" -> group_b) at each position.

    Per-event `target_indices`/`drive_amplitude` go through
    ParadigmEvent.metadata, which jaxfne._model.stimulus_schedule() reads
    directly (verified via source: `if "target_indices" in e.metadata:
    ev_dict["target_indices"] = e.metadata["target_indices"]`) -- the real,
    documented hook, not a guess.

    Delay lengths are random (see DELAY_MEAN_MS/DELAY_JITTER_MS), so the
    trial's total duration varies -- returned alongside the event list so the
    caller can size its own n_steps/drive-array per trial.
    """
    rng = np.random.default_rng(seed)
    groups = {"A": group_a, "B": group_b}
    events: list[dict] = []
    t = 0.0
    for i, token in enumerate(sequence):
        delay_ms = float(rng.uniform(DELAY_MEAN_MS - DELAY_JITTER_MS, DELAY_MEAN_MS + DELAY_JITTER_MS))
        events.append({
            "label": f"delay{i + 1}",
            "onset_ms": t,
            "duration_ms": delay_ms,
            "metadata": {"drive_amplitude": 0.0},
        })
        t += delay_ms
        events.append({
            "label": f"stim{i + 1}",
            "onset_ms": t,
            "duration_ms": STIM_DURATION_MS,
            "stimulus": token,
            "metadata": {"target_indices": groups[token], "drive_amplitude": STIM_AMPLITUDE},
        })
        t += STIM_DURATION_MS
    return events, t


def trial_paradigm(
    group_a: list[int], group_b: list[int], sequence: tuple[str, str, str, str], seed: int
) -> tuple["jtfne.Paradigm", float]:
    """Wrap one trial's explicit event list in a jtfne.general_sequential_
    oddball_paradigm (single condition, since event_windows/timings are
    randomly resampled fresh every trial rather than shared across a fixed
    condition set) -- keeps the trial declaratively structured (Paradigm/
    ParadigmCondition/ParadigmEvent) instead of a hand-rolled StimulusSchedule,
    per jaxfne-paradigm-design."""
    events, total_duration_ms = build_trial_events(group_a, group_b, sequence, seed)
    event_windows = {
        str(ev["label"]): (ev["onset_ms"], ev["onset_ms"] + ev["duration_ms"]) for ev in events
    }
    condition_name = "".join(sequence)
    paradigm = jtfne.general_sequential_oddball_paradigm(
        name="sphere20_overlapping_ab_trial",
        event_windows=event_windows,
        sequence_event_labels=[f"stim{i + 1}" for i in range(TRIAL_N_REPS)],
        conditions=[{"name": condition_name, "events": events}],
        comparison_label="stim1",
    )
    return paradigm, total_duration_ms


def trial_drive_array(paradigm: "jtfne.Paradigm", n_steps: int) -> np.ndarray:
    """Resolve a single-condition trial Paradigm to a dense (n_steps, N) drive
    array via jtfne.stimulus_schedule() -- the same free function
    Model._resolve_stimulus_schedule() calls internally for a
    ParadigmCondition -- rather than passing the ParadigmCondition straight
    to Model.simulate(), since the continuous-task loop drives jax.lax.scan
    directly (see run()'s landmine note on Model.simulate()/
    with_hdp_initial_state not giving true state continuity)."""
    condition = paradigm.conditions[0]
    schedule = jtfne.stimulus_schedule(condition.events, n_neurons=N)
    return schedule.to_array(n_steps, DT_MS)


def edge_block_weights(model: "jtfne.Model", w_final: "jnp.ndarray | None" = None) -> dict[str, float]:
    """Mean |weight| per E/I connection block (Wee/Wei/Wie/Wii) from edge_list.pre/post
    + emitter labels. Uses w_final (post-simulation carried weight) when given,
    else the edge_list's own native weight."""
    edges = model.params["edge_list"]
    labels = model.params["emitter"].labels
    w = np.asarray(w_final if w_final is not None else edges.weight)
    pre = np.asarray(edges.pre)
    post = np.asarray(edges.post)
    pre_type = np.array(["E" if labels[i] == "E" else "I" for i in pre])
    post_type = np.array(["E" if labels[i] == "E" else "I" for i in post])

    out = {}
    for src, tgt in (("E", "E"), ("E", "I"), ("I", "E"), ("I", "I")):
        mask = (pre_type == src) & (post_type == tgt)
        key = f"W{src.lower()}{tgt.lower()}"
        out[key] = float(np.mean(np.abs(w[mask]))) if mask.any() else float("nan")
    return out


def cell_type_h(model: "jtfne.Model", H_final: "jnp.ndarray") -> dict[str, float]:
    labels = model.params["emitter"].labels
    H = np.asarray(H_final)
    out = {}
    for ct in ("E", "PV"):
        mask = np.array([lbl == ct for lbl in labels])
        out[f"H_{ct}"] = float(np.mean(H[mask])) if mask.any() else float("nan")
    return out


def firing_rates(spikes: "np.ndarray", dt_ms: float, model: "jtfne.Model") -> dict[str, float]:
    spikes = np.asarray(spikes)
    labels = model.params["emitter"].labels
    out = {}
    for ct in ("E", "PV"):
        idx = [i for i, lbl in enumerate(labels) if lbl == ct]
        if idx:
            out[f"rate_{ct}_hz"] = float(np.mean(spikes[:, idx]) * (1000.0 / dt_ms))
    out["rate_all_hz"] = float(np.mean(spikes) * (1000.0 / dt_ms))
    return out


def oscillation_metric(spikes: "np.ndarray", dt_ms: float) -> dict[str, float]:
    """Coarse population-rate oscillation readout: dominant frequency + its
    power share from a simple FFT of the binned population rate, plus
    kappa_synchrony as a complementary spike-synchrony measure."""
    spikes = np.asarray(spikes)
    pop_rate = spikes.mean(axis=1)
    pop_rate = pop_rate - pop_rate.mean()
    n = pop_rate.shape[0]
    if n < 4:
        return {"dominant_freq_hz": float("nan"), "dominant_power_frac": float("nan")}
    fft = np.fft.rfft(pop_rate)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n, d=dt_ms / 1000.0)
    if len(power) > 1:
        peak_idx = 1 + int(np.argmax(power[1:]))  # skip DC
    else:
        peak_idx = 0
    total = power[1:].sum() if len(power) > 1 else 0.0
    dom_frac = float(power[peak_idx] / total) if total > 0 else float("nan")
    kappa = float(jtfne.kappa_synchrony(jnp.asarray(spikes), dt_ms=dt_ms))
    return {
        "dominant_freq_hz": float(freqs[peak_idx]),
        "dominant_power_frac": dom_frac,
        "kappa_synchrony": kappa,
    }


def stim_window_response(
    spikes: "np.ndarray", target_indices: list[int], onset_ms: float, duration_ms: float
) -> float:
    """Total spike count of `target_indices` during one stimulus window
    (onset_ms, onset_ms+duration_ms]. The direct habituation/dishabituation
    readout: comparing this at stim4 for the shared AB cells (always driven,
    every repetition, regardless of A/B) against the unique-to-the-current-
    group cells (only driven when their own group is the stim4 target) tests
    whether habituation built up over stim1-3 (to A) transfers through the
    shared cells onto a stim4 driven by B, or stays specific to A."""
    start = int(round(onset_ms / DT_MS))
    end = int(round((onset_ms + duration_ms) / DT_MS))
    return float(np.asarray(spikes)[start:end, target_indices].sum())


def run() -> list[dict[str, Any]]:
    import jax

    model = build_model()
    rows = model.neuron_table()
    cell_types = sorted(r["cell_type"] for r in rows)
    assert cell_types.count("E") == N_E and cell_types.count("PV") == N_I, (
        f"expected {N_E} E / {N_I} PV, got {cell_types.count('E')} E / {cell_types.count('PV')} PV"
    )
    print(f"[build] {N_E} E / {N_I} PV confirmed via neuron_table(); "
          f"n_edges={model.params['edge_list'].n_edges}")

    group_a, group_b, shared = build_groups(model, seed=SEED)
    a_only = [i for i in group_a if i not in shared]
    b_only = [i for i in group_b if i not in shared]
    print(f"[groups] group_a={group_a} group_b={group_b} shared(AB)={shared}")

    # Step 4: baseline simulate via the higher-level Model.simulate() path, just
    # to sanity-check finite/sane output, using one AAAA trial. HDP runtime
    # passed explicitly -- see the landmine note in build_model():
    # Model.simulate(sim) does NOT inherit cfg.hdp(...) the way top-level
    # jtfne.simulate(model, ...) does; only the top-level free function reads
    # cfg.metadata via _runtime_config_from_metadata. Footgun (jaxfne-paradigm-
    # design): simulate(paradigm=...) only recognizes StimulusSchedule/
    # ParadigmCondition, never a bare Paradigm -- pass paradigm.conditions[0],
    # not the Paradigm object itself.
    baseline_paradigm, baseline_duration_ms = trial_paradigm(group_a, group_b, SEQUENCE_AAAA, seed=SEED)
    runtime_cfg = jtfne.RuntimeConfig(enable_hdp=True, hdp_params=HDP_PARAMS)
    sim0 = jtfne.simulation(duration_ms=baseline_duration_ms, dt_ms=DT_MS, seed=SEED, runtime=runtime_cfg)
    sig0 = model.simulate(sim0, paradigm=baseline_paradigm.conditions[0])
    assert bool(jnp.all(jnp.isfinite(sig0.V_m))), "baseline V_m contains NaN/Inf"
    assert bool(jnp.all(jnp.isfinite(sig0.spikes))), "baseline spikes contain NaN/Inf"
    print(f"[baseline] trial_duration_ms={baseline_duration_ms:.1f} "
          f"rates={firing_rates(np.asarray(sig0.spikes), DT_MS, model)}")

    # Step 5: continuous task, TRUE full-state carryover across trials.
    #
    # LANDMINE (found during this build, the most consequential one): the
    # higher-level Model.simulate() + Model.with_hdp_initial_state() pair --
    # the API this script originally used -- does NOT satisfy "last state of
    # previous trial is first state of next trial". with_hdp_initial_state
    # only carries H and edge weight w forward; membrane voltage v, recovery
    # u, prev_spikes, and synaptic state syn_state are hard-reset to the
    # model's native v0/u0/zero every single call (see _model.py's
    # `init_state = {"v": emitter.v0.astype(...), "u": emitter.u0.astype(...),
    # "prev_spikes": jnp.zeros_like(...), "syn_state": jnp.zeros_like(...),
    # ...}` -- there is no v0=/u0=/prev_spikes0=/syn_state0= kwarg on
    # with_hdp_initial_state at all). Combined with an identical PRNG seed
    # every trial, an earlier version of this script (fixed-duration flat
    # pulse train) showed bit-identical firing rates/oscillation metrics
    # across all 20 trials despite H/weight visibly drifting -- a real,
    # non-obvious API footgun for anyone trying to build a genuinely
    # continuous multi-trial simulation, not a mistake specific to this script.
    #
    # The correct verified path for full v/u/prev_spikes/syn_state/H/w
    # carryover is the lower-level jtfne.compile_step_fn/jtfne.scan_network
    # pair (DynamicState = v, u, prev_spikes, syn_state, H, w), documented in
    # jaxfne-neural-tensor's "Internal pure-function layer" section as the
    # real jax.lax.scan step pattern. HDP hyperparameters are captured once
    # via closure (compile_step_fn), so it's compiled a single time and
    # reused across all trials -- only `carry`, the per-trial PRNG keys, and
    # the per-trial drive array (variable length, since each trial's random
    # delays give it a different total duration) change.
    step_fn, carry = jtfne.compile_step_fn(model, dt_ms=DT_MS, **HDP_PARAMS)
    base_key = jax.random.PRNGKey(SEED)

    history: list[dict[str, Any]] = []
    for turn in range(N_TURNS):
        is_oddball = turn in ODDBALL_TURNS
        sequence = SEQUENCE_AAAB if is_oddball else SEQUENCE_AAAA
        # Delays resampled fresh every trial (user decision) -- turn-indexed
        # seed keeps the whole run reproducible given the fixed top-level SEED.
        trial_seed = SEED + 1009 * (turn + 1)
        paradigm, duration_ms = trial_paradigm(group_a, group_b, sequence, seed=trial_seed)
        n_steps = int(round(duration_ms / DT_MS))
        drive_schedule = trial_drive_array(paradigm, n_steps)

        keys = jax.random.split(jax.random.fold_in(base_key, turn), n_steps)
        carry, outputs = jtfne.scan_network(step_fn, carry, jnp.asarray(drive_schedule), keys)
        v_trace, spikes_trace, source_trace, H_trace, w_trace = outputs

        if not bool(jnp.all(jnp.isfinite(v_trace))) or not bool(jnp.all(jnp.isfinite(spikes_trace))):
            raise RuntimeError(f"turn {turn}: non-finite v/spikes in continuous-task scan -- stopping chain")

        spikes_np = np.asarray(spikes_trace)
        stim4_event = next(ev for ev in paradigm.conditions[0].events if ev.label == "stim4")
        stim4_target_group = group_b if is_oddball else group_a
        stim4_unique = b_only if is_oddball else a_only

        row = {"turn": turn, "is_oddball": 1.0 if is_oddball else 0.0, "trial_duration_ms": duration_ms}
        row.update(firing_rates(spikes_np, DT_MS, model))
        # compile_step_fn returns a ContinuationState wrapper whose kernel
        # carry (the six-field DynamicState) lives under `.dynamic` -- the
        # suite predated the wrapper and accessed carry.H/carry.w directly.
        row.update(cell_type_h(model, carry.dynamic.H))
        row.update(edge_block_weights(model, carry.dynamic.w))
        row.update(oscillation_metric(spikes_np, DT_MS))
        row["stim4_shared_response"] = stim_window_response(
            spikes_np, shared, stim4_event.onset_ms, stim4_event.duration_ms
        )
        row["stim4_unique_response"] = stim_window_response(
            spikes_np, stim4_unique, stim4_event.onset_ms, stim4_event.duration_ms
        )
        row["stim4_group_response"] = stim_window_response(
            spikes_np, stim4_target_group, stim4_event.onset_ms, stim4_event.duration_ms
        )
        history.append(row)
        tag = "AAAB" if is_oddball else "AAAA"
        print(f"[trial {turn:2d} {tag}] " + " ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "turn"))

    return history


if __name__ == "__main__":
    run()
