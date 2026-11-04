"""Izhikevich 2003 E1 library path on the delayed-edge route (public API only).

A connection rule's ``delay_ms`` is realized at construction to ``round(delay_ms / dt_ms)``
steps, and the edge_list backend applies it. The same network is run with and without the
delay: the target's first spike must move by exactly that many steps, and the presynaptic
spikes before that first spike must be the same in both runs.
"""

from __future__ import annotations

import numpy as np

import jaxfne as jtfne

DT_MS = 0.5
N_STEPS = 200  # T <= 200 steps
N_E = 2  # E cells first, then I
N_I = 2  # N <= 40
PRE = 0  # an E cell, driven to spike
TGT = N_E  # an I cell with no drive: it spikes only from E input
DELAY_MS = 2.5  # 5 steps at dt 0.5 ms


def _spikes(delay_ms):
    rule = {} if delay_ms is None else {"delay_ms": delay_ms}
    cfg = (
        jtfne.configuration()
        .network(name="izh_e1_delay", kind="balanced_ei_population", n=N_E + N_I,
                 cell_types={"E": 0.5, "I": 0.5})
        .connections(name="e_to_i", source={"cell_type": "E"}, target={"cell_type": "I"},
                     probability=1.0, weight=8.0, sign="excitatory", **rule)
        .cell_params({"cell_type": "E"}, {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0,
                                          "v0": -65.0, "u0": -13.0, "drive": 15.0})
        .cell_params({"cell_type": "I"}, {"a": 0.1, "b": 0.2, "c": -65.0, "d": 2.0,
                                          "v0": -65.0, "u0": -13.0, "drive": 0.0})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy")
        .probe(name="spike_probe", modes=["spikes"])
        .runtime(random_v0=False, dt_ms=DT_MS)
    )
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(
        duration_ms=N_STEPS * DT_MS,
        dt_ms=DT_MS,
        seed=0,
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list", dtype="float32"),
    )
    spikes = np.asarray(model.simulate(sim).spikes)
    assert spikes.shape == (N_STEPS, N_E + N_I)
    return spikes


def test_delayed_edge_moves_first_target_spike_by_round_delay_steps():
    control = _spikes(None)
    delayed = _spikes(DELAY_MS)
    pre = np.flatnonzero(control[:, PRE])
    assert pre.size > 0, "fixture must spike: the presynaptic E cell has no spike"
    tgt_control = np.flatnonzero(control[:, TGT])
    tgt_delayed = np.flatnonzero(delayed[:, TGT])
    assert tgt_control.size > 0 and tgt_delayed.size > 0, (
        "fixture must spike: the target I cell has no spike"
    )
    assert tgt_control[0] > pre[0]
    # The target feeds E back after its first spike, so E is only comparable before it.
    before = int(tgt_control[0])
    np.testing.assert_array_equal(
        np.flatnonzero(delayed[:before, PRE]), pre[pre < before],
        err_msg="presynaptic train differs before the target's first spike",
    )
    expected = round(DELAY_MS / DT_MS)
    shift = int(tgt_delayed[0] - tgt_control[0])
    assert shift == expected, (
        f"target first spike moved {shift} steps; expected round({DELAY_MS}/{DT_MS}) = {expected}"
    )
