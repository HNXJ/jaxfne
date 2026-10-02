"""0.5.4 item 2 — hierarchical <-> flattened identity.

A nested program (areas composed of layers, composed with a cross rule)
and a flat program (leaf pairs composed directly, same leaf multiset in
the same realized order, same rules and frontiers) give the same realized
values and bit-identical runs. This closes agent-native step 2
(canonical TFNE -> JaxFNE compilation and `I`) for composed models: the
hierarchy is a scoping device, not a numerical one.
"""

import numpy as np

import jaxfne
from jaxfne.tfne import parse, realize, resolve, to_configuration

HIER = """
O[ff] := [mechanism = AMPA; weight = 0.5; $L.out>$R.in];
out[V1] := [L4];
in[V2] := [L4];
L4 := [C = {E}; N = 4];
L5 := [C = {E}; N = 2];
V1 := L4 O L5;
V2 := L4 O L5;
V := V1 O[ff] V2;
x : V : y
"""

FLAT = """
O[ff] := [mechanism = AMPA; weight = 0.5; $L.out>$R.in];
out[A] := [X1];
in[B] := [X3];
X1 := [C = {E}; N = 4];
X2 := [C = {E}; N = 2];
X3 := [C = {E}; N = 4];
X4 := [C = {E}; N = 2];
A := X1 O X2;
B := X3 O X4;
C := A O[ff] B;
x : C : y
"""


def _run(prog):
    program = parse(prog)
    realized = realize(resolve(program), program, seed=0)
    model = jaxfne.construct(
        to_configuration(realized, duration_ms=5.0, dt_ms=0.5).runtime(
            recurrent_backend="edge_list"
        )
    )
    sig = jaxfne.simulate(model, duration_ms=5.0, dt_ms=0.5, seed=3)
    return realized, model, sig


def _suffix(path):
    return path.split(".")[-2] + "/" + path.split(".")[-1]


def test_hierarchical_flat_same_realized_values():
    rh, _, _ = _run(HIER)
    rf, _, _ = _run(FLAT)
    assert rh.s["n_neurons"] == rf.s["n_neurons"] == 12
    # same leaf multiset in the same order, modulo the area prefix:
    # V.V1.L4 ~ C.A.X1, V.V1.L5 ~ C.A.X2, V.V2.L4 ~ C.B.X3, V.V2.L5 ~ C.B.X4.
    kinds_h = [p.split(".")[-1] for p in rh.s["neuron_paths"]]
    assert kinds_h == ["L4"] * 4 + ["L5"] * 2 + ["L4"] * 4 + ["L5"] * 2
    kinds_f = [p.split(".")[-1] for p in rf.s["neuron_paths"]]
    assert kinds_f == ["X1"] * 4 + ["X2"] * 2 + ["X3"] * 4 + ["X4"] * 2
    for key in ("edge_pre", "edge_post", "edge_weight", "edge_mechanism"):
        assert np.array_equal(np.asarray(rh.s[key]), np.asarray(rf.s[key])), key


def test_hierarchical_flat_runs_bit_identical():
    _, mh, sh = _run(HIER)
    _, mf, sf = _run(FLAT)
    el_h, el_f = mh.params["edge_list"], mf.params["edge_list"]
    assert np.array_equal(np.asarray(el_h.pre), np.asarray(el_f.pre))
    assert np.array_equal(np.asarray(el_h.post), np.asarray(el_f.post))
    assert np.array_equal(np.asarray(el_h.weight), np.asarray(el_f.weight))
    assert np.array_equal(np.asarray(sh.V_m), np.asarray(sf.V_m))
    assert np.array_equal(np.asarray(sh.spikes), np.asarray(sf.spikes))
