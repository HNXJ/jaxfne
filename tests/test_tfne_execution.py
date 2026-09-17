"""TFNE-EXEC-01: end-to-end proof that a TFNE specification executes.

`tests/test_tfne_algebra.py` covers the specification layer and touches no
simulation kernel. This file covers the other half of the claim made by
`jaxfne/tfne.py` and the doctrine pipeline table -- that TFNE output runs in
the existing JaxFNE kernels:

    TFNE source -> NF -> to_neuronal_tensor -> construct -> simulate

It asserts what transfers and, where something does not, pins the divergence
rather than leaving it to be discovered again.
"""

import numpy as np
import pytest

import jaxfne
from jaxfne.neuronal_tensor import RuntimeConfiguration
from jaxfne.tfne import parse, realize, resolve, to_neuronal_tensor

SPEC = """
O[ff] := [direction = >; mechanism = AMPA; probability = 1.0; weight = 0.5];
L4 := [C = {E, PV}; P = {E: 0.8, PV: 0.2}; N = 10; model = izhikevich];
L2 := [C = {E}; N = 5; model = izhikevich];
V1 := L4 O[ff] L2;
x : V1 : y
"""

DURATION_MS = 20.0
DT_MS = 0.1


@pytest.fixture(scope="module")
def pipeline():
    """One pass of the full chain, reused by every assertion below."""
    program = parse(SPEC)
    explicit = resolve(program)
    realization = realize(explicit, program)
    tensor = to_neuronal_tensor(explicit)
    model = jaxfne.construct(
        tensor, RuntimeConfiguration(duration_ms=DURATION_MS, dt_ms=DT_MS,
                                     seed=0))
    signals = jaxfne.simulate(model, duration_ms=DURATION_MS, dt_ms=DT_MS,
                              seed=0)
    return realization, tensor, model, signals


def test_tfne_source_reaches_kernel_execution(pipeline):
    """The chain runs and the kernel integrates: V_m varies and stays finite."""
    realization, _, _, signals = pipeline
    n_neurons = realization.s["n_neurons"]
    n_steps = int(round(DURATION_MS / DT_MS))

    v = np.asarray(signals.V_m)
    spikes = np.asarray(signals.spikes)
    assert v.shape == (n_steps, n_neurons)
    assert spikes.shape == (n_steps, n_neurons)
    assert np.isfinite(v).all()
    # A constant trace would mean the arrays were allocated but never
    # integrated; a varying one is evidence the kernel actually ran.
    assert float(v.std()) > 0.0
    assert set(np.unique(spikes).tolist()) <= {0.0, 1.0}


def test_executed_topology_matches_the_tfne_realization(pipeline):
    """Edge count and its cell-type split survive the bridge exactly."""
    realization, _, model, _ = pipeline
    summary = model.connectivity_summary()
    assert summary["n_edges"] == realization.s["n_edges"] == 50

    edges = model.edge_table()
    by_pre_type = {}
    for e in edges:
        key = (e["pre_layer"], e["pre_cell_type"], e["post_layer"],
               e["post_cell_type"])
        by_pre_type[key] = by_pre_type.get(key, 0) + 1
    # 8 E and 2 PV in L4 (P = 0.8/0.2 of N = 10), each onto all 5 of L2.E
    assert by_pre_type == {("L4", "E", "L2", "E"): 40,
                           ("L4", "PV", "L2", "E"): 10}


def test_index_map_addresses_the_simulated_neuron_axis(pipeline):
    """I indexes the columns the kernel actually produced."""
    realization, _, _, signals = pipeline
    v = np.asarray(signals.V_m)
    l4 = realization.path_to_slice("V1.L4")
    l2 = realization.path_to_slice("V1.L2")
    assert l4 == (0, 10)
    assert l2 == (10, 15)
    # the two slices partition the simulated axis, so a TFNE path selects the
    # right columns of the result rather than merely a valid range
    assert l4[1] == l2[0]
    assert l2[1] == v.shape[1]
    assert np.isfinite(v[:, slice(*l4)]).all()
    assert np.isfinite(v[:, slice(*l2)]).all()


def test_mechanism_identity_survives_the_bridge(pipeline):
    """The rule's declared mechanism reaches the executed edges."""
    _, _, model, _ = pipeline
    receptors = {e["receptor_type"] for e in model.edge_table()}
    assert all(r.startswith("AMPA") for r in receptors), receptors


def test_declared_rule_weight_does_not_reach_the_executed_model(pipeline):
    """Pinned divergence: `weight` is dropped between the two compilations.

    `realize()` and `to_neuronal_tensor()` are two independent compilations of
    one source. The first carries the rule's declared `weight`; the second
    builds `InterConnection`s without it, so construction applies its own
    scaling and sign convention instead. Topology and identity transfer,
    connection parameters do not.

    This test records the current behaviour so the gap cannot close or widen
    silently. It is not an endorsement of it.
    """
    realization, tensor, model, _ = pipeline

    declared = sorted({round(float(w), 6)
                       for w in np.asarray(realization.s["edge_weight"])})
    assert declared == [0.5]

    executed = sorted({round(float(e["weight"]), 6)
                       for e in model.edge_table()})
    assert 0.5 not in executed
    # the bridge leaves w_mech at its default rather than carrying 0.5
    w_mech = {ic.plastic.w_mech
              for area in tensor.areas for ic in area.inter_connections}
    assert w_mech == {1.0}
