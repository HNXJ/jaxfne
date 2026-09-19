"""TFNE algebra adoption tests: adversarial corpus for the specification layer.

Every case verifies TFNE -> canonical explicit realization -> flattened
JaxFNE representation, per the algebra project source
(artifacts/project_sources/7_tfne_algebra.md). The tests target the
specification-time compiler (jaxfne.tfne) and its reuse of the validated
connectivity scaffold; no simulation kernels are touched.
"""

import json

import numpy as np
import pytest

from jaxfne import tfne
from jaxfne.tfne import (TFNEError, flatten, normalize, parse, realization_summary,
                          realize, resolve, to_neuronal_tensor)


def _realize(text, seed=None):
    program = parse(text)
    explicit = resolve(program)
    return program, explicit, realize(explicit, program, seed=seed)


def _replay_ok(text):
    program = parse(text)
    first = normalize(program)
    assert normalize(parse(first)) == first
    return first


# --------------------------------------------------------------------------- #
# 1. One cell (degenerate validity)
# --------------------------------------------------------------------------- #

def test_one_cell():
    text = "Cell := [C = {pyr}; N = 1; model = hh]; x : Cell : y"
    norm = _replay_ok(text)
    program, explicit, r = _realize(text)
    assert r.s["n_neurons"] == 1
    assert r.s["n_edges"] == 0
    assert r.s["counts"] == {"Cell": {"pyr": 1}}
    assert r.s["models"] == {"Cell": "hh"}
    assert r.path_to_slice("Cell") == (0, 1)
    assert r.slice_to_path(0) == "Cell"
    assert r.h0["v"].shape == (1,)
    assert r.h0["H"].shape == (1, 1)
    tensor = to_neuronal_tensor(explicit)
    assert len(tensor.areas) == 1
    assert tensor.areas[0].layers[0].n_neurons == 1
    assert "Cell" in norm


# --------------------------------------------------------------------------- #
# 2. One nonlaminar nucleus (no layers required)
# --------------------------------------------------------------------------- #

NUCLEUS = """
Nucleus := [C = {E, PV}; P = {E: 0.7, PV: 0.3}; N = 10; model = izhikevich];
x : Nucleus : y
"""


def test_nonlaminar_nucleus():
    _replay_ok(NUCLEUS)
    _, explicit, r = _realize(NUCLEUS)
    assert r.s["n_neurons"] == 10
    assert r.s["counts"] == {"Nucleus": {"E": 7, "PV": 3}}
    assert r.path_to_slice("Nucleus.E") == (0, 7)
    assert r.path_to_slice("Nucleus.PV") == (7, 10)
    tensor = to_neuronal_tensor(explicit)
    assert len(tensor.areas) == 1
    assert tensor.areas[0].name == "Nucleus"


# --------------------------------------------------------------------------- #
# 3. Six-layer cortical area
# --------------------------------------------------------------------------- #

CORTEX = """
O[ff] := [direction = >; mechanism = AMPA; probability = 1.0; weight = 0.4];
L1 := [C = {E}; N = 2; model = izhikevich];
L2 := [C = {E, PV}; P = {E: 0.8, PV: 0.2}; N = 5; model = izhikevich];
L3 := [C = {E, PV}; P = {E: 0.8, PV: 0.2}; N = 5; model = izhikevich];
L4 := [C = {E, PV}; P = {E: 0.8, PV: 0.2}; N = 5; model = izhikevich];
L5 := [C = {E, PV}; P = {E: 0.7, PV: 0.3}; N = 10; model = izhikevich];
L6 := [C = {E, PV}; P = {E: 0.7, PV: 0.3}; N = 10; model = izhikevich];
V1 := L1 O[ff] L2 O[ff] L3 O[ff] L4 O[ff] L5 O[ff] L6;
x : V1 : y
"""


def test_six_layer_cortical_area():
    _replay_ok(CORTEX)
    _, explicit, r = _realize(CORTEX)
    assert r.s["n_neurons"] == 2 + 5 + 5 + 5 + 10 + 10
    total = sum(r.s["counts"][f"V1.L{i}"][c]
                for i in (1, 2, 3, 4, 5, 6)
                for c in r.s["counts"][f"V1.L{i}"])
    assert total == r.s["n_neurons"]
    assert len(r.explicit.relations) == 5
    tensor = to_neuronal_tensor(explicit)
    assert [a.name for a in tensor.areas] == ["V1"]
    assert sorted(layer.name for layer in tensor.areas[0].layers) == [
        "L1", "L2", "L3", "L4", "L5", "L6"]
    assert len(tensor.areas[0].inter_connections) > 0
    mechs = {c.mechanism for c in tensor.areas[0].inter_connections}
    assert mechs == {"AMPA"}


# --------------------------------------------------------------------------- #
# 4-6. Serial composition and group sensitivity
# --------------------------------------------------------------------------- #

FLAT = "A := [C = {cell}; N = 2]; B := [C = {cell}; N = 2]; C := [C = {cell}; N = 2]; x : A O B O C : y"
LEFT = "A := [C = {cell}; N = 2]; B := [C = {cell}; N = 2]; C := [C = {cell}; N = 2]; x : {A O B} O C : y"
RIGHT = "A := [C = {cell}; N = 2]; B := [C = {cell}; N = 2]; C := [C = {cell}; N = 2]; x : A O {B O C} : y"


def test_flat_serial_bare_composition_has_no_edges():
    _replay_ok(FLAT)
    _, _, r = _realize(FLAT)
    assert r.s["n_neurons"] == 6
    assert r.s["n_edges"] == 0  # bare O is structural only
    assert r.path_to_slice("A") == (0, 2)


def test_group_sensitivity_paths_differ_edges_match():
    for text in (LEFT, RIGHT):
        _replay_ok(text)
    _, _, flat = _realize(FLAT)
    _, _, left = _realize(LEFT)
    _, _, right = _realize(RIGHT)
    assert flat.s["n_edges"] == left.s["n_edges"] == right.s["n_edges"] == 0
    flat_paths = set(flat.I["object_slices"])
    left_paths = set(left.I["object_slices"])
    right_paths = set(right.I["object_slices"])
    assert flat_paths == {"A", "B", "C"}
    assert left_paths != flat_paths and right_paths != flat_paths
    assert left_paths != right_paths
    # boundaries preserved: grouped members nest under a g-segment
    assert "g0.A" in left_paths and "g0.B" in left_paths
    assert "g0.B" in right_paths and "g0.C" in right_paths


# --------------------------------------------------------------------------- #
# 7-9. Nested O/X composition, cross-connections, multiple rules
# --------------------------------------------------------------------------- #

NESTED = """
O[ff] := [direction = >; mechanism = AMPA; probability = 1.0; weight = 0.5];
O[fb] := [direction = <; mechanism = GABA; probability = 1.0; weight = 0.3];
X[lat] := [direction = <>; mechanism = AMPA; probability = 0.5; weight = 0.2];
A := [C = {E}; N = 4];
B := [C = {E}; N = 4];
C := [C = {E}; N = 4];
D := [C = {E}; N = 4];
x : {A O[ff] B} X[lat] {C O[fb] D} : y
"""


def test_nested_ox_composition_multiple_rules():
    _replay_ok(NESTED)
    _, explicit, r = _realize(NESTED)
    assert r.s["n_neurons"] == 16
    # ff, fb, and the two directed halves of the bidirectional lateral rule
    assert len(explicit.relations) == 4
    kinds = sorted(rel.rule for rel in explicit.relations)
    assert kinds == ["fb", "ff", "lat", "lat"]
    total_edges = sum(
        stop - start for start, stop in r.I["rule_slices"].values())
    assert total_edges == r.s["n_edges"] > 0
    mechs = {m["name"] for m in r.s["mechanism_table"]}
    assert {"AMPA", "GABA"} <= mechs
    # every edge maps back to its originating relation/rule
    for e in range(r.s["n_edges"]):
        key = r.edge_to_rule(e)
        origin = r.relation_origin(key)
        assert origin["rule"] in ("ff", "fb", "lat")
    # bidirectional lateral rule contributes both directions on the same pair
    lat_keys = [k for k in r.I["rule_slices"] if "[lat]" in k]
    assert len(lat_keys) == 2
    assert r.relation_group(lat_keys[0]) == sorted(lat_keys)


def test_explicit_projections_and_bidirectional():
    text = """
    A := [C = {cell}; N = 3];
    B := [C = {cell}; N = 3];
    x : A > B : y
    """
    _, _, fwd = _realize(text)
    assert fwd.s["n_edges"] == 9
    text2 = """
    A := [C = {cell}; N = 3];
    B := [C = {cell}; N = 3];
    x : A <> B : y
    """
    _, _, both = _realize(text2)
    assert both.s["n_edges"] == 18
    text3 = """
    A := [C = {cell}; N = 3];
    B := [C = {cell}; N = 3];
    x : A < B : y
    """
    _, _, back = _realize(text3)
    assert back.s["n_edges"] == 9
    fwd_pairs = sorted(zip(np.asarray(fwd.s["edge_pre"]),
                           np.asarray(fwd.s["edge_post"])))
    back_pairs = sorted(zip(np.asarray(back.s["edge_pre"]),
                            np.asarray(back.s["edge_post"])))
    assert sorted((b, a) for a, b in fwd_pairs) == back_pairs


# --------------------------------------------------------------------------- #
# 9b. S10 ordered adjacency
# --------------------------------------------------------------------------- #

FF_RULE = ("O[ff] := [direction = >; mechanism = AMPA; probability = 1.0; "
           "weight = 0.4];")
FB_RULE = ("O[fb] := [direction = >; mechanism = GABA; probability = 1.0; "
           "weight = 0.3];")
ABC = ("A := [C = {cell}; N = 2]; B := [C = {cell}; N = 2]; "
       "C := [C = {cell}; N = 2];")


def _projection_identities(explicit, r):
    """Realized edges reduced to (pre_leaf, post_leaf) scope pairs."""
    leaves = [p for p in explicit.order if explicit.nodes[p].kind == "object"]
    slices = {p: r.path_to_slice(p) for p in leaves}

    def owner(i):
        for path, (lo, hi) in slices.items():
            if lo <= i < hi:
                return path
        raise AssertionError(f"neuron {i} belongs to no leaf")

    return sorted({(owner(int(a)), owner(int(b)))
                   for a, b in zip(np.asarray(r.s["edge_pre"]),
                                   np.asarray(r.s["edge_post"]))})


def test_ordered_chain_binds_only_its_own_adjacency():
    """S10: A O[k] B O[k] C generates k(A,B) and k(B,C), never A>C."""
    _, explicit, r = _realize(f"{FF_RULE} {ABC} x : A O[ff] B O[ff] C : y")
    assert len(explicit.relations) == 2
    assert _projection_identities(explicit, r) == [("A", "B"), ("B", "C")]
    assert r.s["n_edges"] == 8


def test_mixed_rule_chain_keeps_each_rule_on_its_adjacency():
    """S10: A O[k] B O[j] C normalizes to (A,k,B,j,C)."""
    _, explicit, r = _realize(
        f"{FF_RULE} {FB_RULE} {ABC} x : A O[ff] B O[fb] C : y")
    assert [rel.rule for rel in explicit.relations] == ["ff", "fb"]
    assert [(rel.pre_scopes, rel.post_scopes) for rel in explicit.relations] \
        == [(("A",), ("B",)), (("B",), ("C",))]
    assert _projection_identities(explicit, r) == [("A", "B"), ("B", "C")]


def test_six_layer_chain_is_five_adjacencies_not_fifteen():
    """A six-layer column is a chain, not all-to-all onto every later layer."""
    defs = " ".join(f"L{i} := [C = {{cell}}; N = 2];" for i in range(1, 7))
    body = " O[ff] ".join(f"L{i}" for i in range(1, 7))
    _, explicit, r = _realize(f"{FF_RULE} {defs} x : {body} : y")
    assert len(explicit.relations) == 5
    assert _projection_identities(explicit, r) == \
        [(f"L{i}", f"L{i + 1}") for i in range(1, 6)]
    assert r.s["n_edges"] == 20          # 5 adjacencies x 2x2, not 15 x 4


def test_braced_composite_composes_through_its_out_frontier():
    """S8/S9: {A O B} O C binds out({A O B}) = B, not every member."""
    _, explicit, r = _realize(
        f"{FF_RULE} {ABC} x : {{A O[ff] B}} O[ff] C : y")
    assert len(explicit.relations) == 2
    outer = explicit.relations[1]
    # the composite reaches C through B alone; g0 as a whole would drag in A
    assert outer.pre_scopes == ("g0.B",)
    assert outer.post_scopes == ("C",)
    assert _projection_identities(explicit, r) == \
        [("g0.A", "g0.B"), ("g0.B", "C")]
    assert r.s["n_edges"] == 8


# --------------------------------------------------------------------------- #
# 10. Replication A^n
# --------------------------------------------------------------------------- #

def test_replication_creates_indexed_instances_without_edges():
    text = "E := [C = {cell}; N = 2]; x : E^4 : y"
    _replay_ok(text)
    _, _, r = _realize(text)
    assert r.s["n_neurons"] == 8
    assert r.s["n_edges"] == 0
    # S6: A^n = {A.1 ... A.n}. Indices are 1-based, so E.1 is the first
    # instance and E.0 names nothing.
    assert set(r.I["object_slices"]) == {f"E.{i}" for i in range(1, 5)}
    assert r.path_to_slice("E.1") == (0, 2)
    assert r.path_to_slice("E.2") == (2, 4)
    assert r.path_to_slice("E.4") == (6, 8)
    with pytest.raises(TFNEError):
        r.path_to_slice("E.0")


# --------------------------------------------------------------------------- #
# 11. N/P exact realization
# --------------------------------------------------------------------------- #

def test_proportion_normalization_and_exact_counts():
    text = ("Pop := [C = {E, PV}; P = {E: 0.8, PV: 0.2}; N = 10]; "
            "x : Pop : y")
    _, _, r = _realize(text)
    assert r.s["counts"] == {"Pop": {"E": 8, "PV": 2}}
    # largest remainder, declaration-order tiebreak: 3 * 0.5/0.5 -> 2/1
    text2 = ("Pop := [C = {E, PV}; P = {E: 0.5, PV: 0.5}; N = 3]; "
             "x : Pop : y")
    _, _, r2 = _realize(text2)
    assert r2.s["counts"] == {"Pop": {"E": 2, "PV": 1}}
    assert sum(r2.s["counts"]["Pop"].values()) == 3
    # exact N map form
    text3 = ("Pop := [C = {E, PV}; N = {E: 6, PV: 4}]; x : Pop : y")
    _, _, r3 = _realize(text3)
    assert r3.s["counts"] == {"Pop": {"E": 6, "PV": 4}}


def test_proportion_violations_raise():
    with pytest.raises(TFNEError):
        _realize("Pop := [C = {E, PV}; P = {E: 0.9, PV: 0.2}; N = 10]; "
                 "x : Pop : y")
    with pytest.raises(TFNEError):
        _realize("Pop := [C = {E, PV}; P = {E: 1.2, PV: -0.2}; N = 10]; "
                 "x : Pop : y")
    with pytest.raises(TFNEError):
        _realize("Pop := [C = {E, PV}; N = 10]; x : Pop : y")


# --------------------------------------------------------------------------- #
# 12. G geometry
# --------------------------------------------------------------------------- #

def test_geometry_compiles_to_static_state():
    text = ("Pop := [C = {cell}; N = 4; "
            "G = [distribution = uniform_random; x0 = 0.0; x1 = 1.0]]; "
            "x : Pop : y")
    _replay_ok(text)
    _, explicit, r = _realize(text)
    assert r.s["geometry"]["Pop"]["x0"] == 0.0
    assert r.s["geometry"]["Pop"]["x1"] == 1.0
    tensor = to_neuronal_tensor(explicit)
    geo = tensor.areas[0].layers[0].geometry
    assert tuple(geo.x_range) == (0.0, 1.0)


# --------------------------------------------------------------------------- #
# 13. Typed x/y boundaries
# --------------------------------------------------------------------------- #

def test_typed_boundaries_preserved():
    text = "Cell := [C = {cell}; N = 1]; x[retina] : Cell : y[choice]"
    _replay_ok(text)
    _, _, r = _realize(text)
    assert r.s["boundaries"] == {"x": "x", "x_type": "retina",
                                 "y": "y", "y_type": "choice"}


# --------------------------------------------------------------------------- #
# 14. Shared biological type, different dynamical models
# --------------------------------------------------------------------------- #

def test_shared_type_distinct_models():
    text = """
    E1 := [C = {pyr}; N = 3; model = hh];
    E2 := [C = {pyr}; N = 3; model = lif];
    x : E1 O E2 : y
    """
    _replay_ok(text)
    _, _, r = _realize(text)
    assert r.s["models"] == {"E1": "hh", "E2": "lif"}
    assert (r.s["counts"]["E1"] == r.s["counts"]["E2"] == {"pyr": 3})


# --------------------------------------------------------------------------- #
# 15. Mutable/plastic state separation
# --------------------------------------------------------------------------- #

def test_plastic_rule_declared_state_param_separation():
    text = """
    O[stdp] := [direction = >; mechanism = AMPA; probability = 1.0;
                weight = 0.5; plasticity = hdp];
    A := [C = {cell}; N = 4];
    B := [C = {cell}; N = 4];
    x : A O[stdp] B : y
    """
    _replay_ok(text)
    _, _, r = _realize(text)
    assert r.s["n_edges"] == 16
    key = next(iter(r.I["rule_slices"]))
    assert r.relation_origin(key)["params"]["plasticity"] == "hdp"
    # state (mutable, in h0) vs parameter (static, in s) stay distinct
    assert r.h0["w"].shape == (16,)
    assert np.array_equal(np.asarray(r.h0["w"], dtype=np.float64),
                          np.asarray(r.s["edge_weight"]))
    assert r.h0["H"].shape == (4 + 4, 1)


# --------------------------------------------------------------------------- #
# 16. Flatten -> inspect round trip
# --------------------------------------------------------------------------- #

def test_flatten_inspect_round_trip():
    _, _, r = _realize(NESTED)
    # every neuron maps to a path and back into a containing slice
    for nid in range(r.s["n_neurons"]):
        path = r.slice_to_path(nid)
        start, stop = r.path_to_slice(path)
        assert start <= nid < stop
        ids = r.neuron_ids_in_scope(path)
        assert nid in ids
    # every edge maps to its rule and back into the rule's range
    for e in range(r.s["n_edges"]):
        key = r.edge_to_rule(e)
        start, stop = r.rule_to_edges(key)
        assert start <= e < stop
        origin = r.relation_origin(key)
        assert origin["pre_scopes"] and origin["post_scopes"]
    # TFNE parameter/state identity reaches the flat representation
    for path, counts in r.s["counts"].items():
        start, stop = r.path_to_slice(path)
        assert stop - start == sum(counts.values())
    with pytest.raises(TFNEError):
        r.path_to_slice("Nope")
    with pytest.raises(TFNEError):
        r.slice_to_path(r.s["n_neurons"])


# --------------------------------------------------------------------------- #
# 17. Deterministic normalization and replay
# --------------------------------------------------------------------------- #

def test_deterministic_normalization_and_replay():
    first = _replay_ok(NESTED)
    _, _, r1 = _realize(NESTED)
    _, _, r2 = _realize(first)  # replay from canonical form
    assert r1.s["digest"] == r2.s["digest"]
    s1 = realization_summary(r1)
    s2 = realization_summary(r2)
    assert s1["realization_sha256"] == s2["realization_sha256"]
    # scale/whitespace-insensitive: same model, shuffled definitions
    shuffled = """
    D := [C = {E}; N = 4]; C := [C = {E}; N = 4];
    B := [C = {E}; N = 4]; A := [C = {E}; N = 4];
    X[lat] := [direction = <>; mechanism = AMPA; probability = 0.5; weight = 0.2];
    O[fb] := [direction = <; mechanism = GABA; probability = 1.0; weight = 0.3];
    O[ff] := [direction = >; mechanism = AMPA; probability = 1.0; weight = 0.5];
    x:{A O[ff] B}X[lat]{C O[fb] D}:y
    """
    _, _, r3 = _realize(shuffled)
    assert r3.s["digest"] == r1.s["digest"]
    assert (realization_summary(r3)["realization_sha256"]
            == s1["realization_sha256"])


def test_explicit_seed_replay():
    _, _, r1 = _realize(NESTED, seed=7)
    _, _, r2 = _realize(NESTED, seed=7)
    assert r1.s["seed"] == r2.s["seed"] == 7
    assert np.array_equal(np.asarray(r1.s["edge_pre"]),
                          np.asarray(r2.s["edge_pre"]))


# --------------------------------------------------------------------------- #
# Equivalent construction paths
# --------------------------------------------------------------------------- #

def test_aliased_and_direct_construction_agree():
    direct = """
    O[ff] := [direction = >; mechanism = AMPA; probability = 1.0; weight = 0.5];
    A := [C = {cell}; N = 3];
    B := [C = {cell}; N = 3];
    x : A O[ff] B : y
    """
    aliased = """
    O[ff] := [direction = >; mechanism = AMPA; probability = 1.0; weight = 0.5];
    A := [C = {cell}; N = 3];
    B := [C = {cell}; N = 3];
    S := A O[ff] B;
    x : S : y
    """
    _, _, r_direct = _realize(direct)
    _, _, r_aliased = _realize(aliased)
    # same realized edge signature...
    assert (r_direct.s["n_neurons"], r_direct.s["n_edges"]) == (
        r_aliased.s["n_neurons"], r_aliased.s["n_edges"])
    assert sorted(float(w) for w in r_direct.s["edge_weight"]) == sorted(
        float(w) for w in r_aliased.s["edge_weight"])
    # ...with hierarchical identity recording the factoring path
    assert set(r_direct.I["object_slices"]) == {"A", "B"}
    assert set(r_aliased.I["object_slices"]) == {"S.A", "S.B"}
    _, explicit_aliased, _ = _realize(aliased)
    assert explicit_aliased.nodes["S"].kind == "composite"


def test_specialization_inherits_base():
    text = """
    CTX := [C = {E, PV}; P = {E: 0.8, PV: 0.2}; N = 10; model = izhikevich];
    V1 := CTX[v1];
    V2 := CTX[v2];
    x : V1 O V2 : y
    """
    _replay_ok(text)
    _, explicit, r = _realize(text)
    assert r.s["counts"]["V1"] == r.s["counts"]["V2"] == {"E": 8, "PV": 2}
    assert explicit.nodes["V1"].special == "CTX[v1]"
    assert explicit.nodes["V2"].special == "CTX[v2]"


# --------------------------------------------------------------------------- #
# Exclusions, reserved names, rule validation
# --------------------------------------------------------------------------- #

def _program_with_exclusion(text, left, right, direction="!>"):
    """Attach an exclusion to ``text``'s system body.

    The grammar has no statement separator inside a composite yet (S14
    atomicity, queued), so a generating projection and an exclusion over it
    cannot be written in one source body. The exclusion is attached directly.
    """
    program = parse(text)
    excl = tfne.Exclude(direction=direction, left=tfne.Ref((left,)),
                        right=tfne.Ref((right,)))
    return tfne.Program(
        defs=program.defs, rules=program.rules,
        system=tfne.System(x="x", x_type=None,
                           body=tfne.Ordered(left=program.system.body,
                                             right=excl),
                           y="y", y_type=None))


def test_exclusion_subtracts_the_projection_it_names():
    """S14: G = G_0 minus E_-. An exclusion removes, it does not veto."""
    text = """
    A := [C = {cell}; N = 2];
    B := [C = {cell}; N = 2];
    x : A > B : y
    """
    _, _, before = _realize(text)
    assert before.s["n_edges"] == 4
    program = _program_with_exclusion(text, "A", "B")
    after = realize(resolve(program), program)
    assert after.s["n_edges"] == 0


def test_unmatched_exclusion_is_invalid():
    """S14: an exclusion matching no generated projection is E_EXCLUSION_UNKNOWN."""
    text = """
    A := [C = {cell}; N = 2];
    B := [C = {cell}; N = 2];
    x : {A O B} : y
    """
    _, _, r = _realize(text)
    assert r.s["n_edges"] == 0          # bare O generates nothing to exclude
    program = _program_with_exclusion(text, "A", "B")
    with pytest.raises(TFNEError, match="E_EXCLUSION_UNKNOWN"):
        realize(resolve(program), program)


def test_exclusion_subtracts_one_route_and_leaves_the_rest():
    """S14 subtraction is per projection identity, not per relation."""
    import dataclasses

    text = """
    A := [C = {cell}; N = 2];
    B := [C = {cell}; N = 2];
    C := [C = {cell}; N = 2];
    x : A O B O C : y
    """
    program = _program_with_exclusion(text, "A", "C")
    explicit = resolve(program)
    assert explicit.relations == ()     # bare O generates no projections
    # One relation carrying two routes, A>C and B>C, so that excluding A!>C
    # must keep B>C. Injected directly: no surface form produces a
    # multi-route relation independently of the queued S10 change.
    wide = tfne.RelationRecord(
        key="r0:direct[-]:A B>C", form="rule", kind="O", rule=None,
        direction=">", pre_label="A B", post_label="C",
        pre_scopes=("A", "B"), post_scopes=("C",))
    explicit = dataclasses.replace(explicit, relations=(wide,))
    r = realize(explicit, program)
    lo_a, hi_a = r.path_to_slice("A")
    lo_b, hi_b = r.path_to_slice("B")
    pre = {int(v) for v in r.s["edge_pre"]}
    assert r.s["n_edges"] == 4
    assert pre & set(range(lo_a, hi_a)) == set()      # A>C subtracted
    assert pre & set(range(lo_b, hi_b)) == set(range(lo_b, hi_b))  # B>C kept


def test_reserved_names_and_rule_validation():
    with pytest.raises(TFNEError):
        parse("H := [C = {cell}; N = 1]; x : H : y")
    with pytest.raises(TFNEError):
        parse("h := [C = {cell}; N = 1]; x : h : y")
    with pytest.raises(TFNEError):
        _realize("O[bad] := [mechanism = AMPA]; "
                 "A := [C = {cell}; N = 2]; B := [C = {cell}; N = 2]; "
                 "x : A O[bad] B : y")
    with pytest.raises(TFNEError):
        _realize("A := [C = {cell}; N = 2]; B := [C = {cell}; N = 2]; "
                 "x : A O[nope] B : y")
    with pytest.raises(TFNEError):
        parse("A := [C = {cell}; N = 2]; x : A O B : y; x : A : y")


def test_selection_addressing_and_layer_types():
    text = """
    O[ff] := [direction = >; mechanism = AMPA; probability = 1.0; weight = 1.0];
    L4 := [C = {E, PV}; P = {E: 0.5, PV: 0.5}; N = 4];
    L23 := [C = {E, PV}; P = {E: 0.5, PV: 0.5}; N = 4];
    V1 := L4 O L23;
    x : V1.L4[E] > V1.L23[E] : y
    """
    _replay_ok(text)
    _, _, r = _realize(text)
    # E->E across layers only: 2 pre x 2 post
    assert r.s["n_edges"] == 4
    pre = {int(v) for v in r.s["edge_pre"]}
    assert pre == set(range(*r.path_to_slice("V1.L4.E")))


# --------------------------------------------------------------------------- #
# NeuronalTensor bridge round trip
# --------------------------------------------------------------------------- #

def test_tensor_bridge_json_round_trip(tmp_path):
    _, explicit, _ = _realize(CORTEX)
    tensor = to_neuronal_tensor(explicit)
    payload = tensor.to_dict()
    blob = json.dumps(payload, sort_keys=True, default=str)
    assert json.loads(blob)["name"] == "tfne"
    import jaxfne.neuronal_tensor as nt_mod
    assert hasattr(nt_mod, "NeuronalTensor")
    # fractions within each layer sum to one
    for area in tensor.areas:
        for layer in area.layers:
            assert abs(sum(t.fraction for t in layer.neuron_types) - 1.0) < 1e-9
    # provenance carries the TFNE digest
    assert payload["provenance"]["tfne_digest"] == explicit.digest


def test_flatten_one_call_json_safe_summary():
    r = flatten("Cell := [C = {pyr}; N = 1; model = hh]; x : Cell : y")
    summary = realization_summary(r)
    json.dumps(summary)
    assert summary["n_neurons"] == 1
    assert summary["value_tag"] == "relative"


# --------------------------------------------------------------------------- #
# 9c. S20 canonical ordering
# --------------------------------------------------------------------------- #

def test_declaration_order_does_not_determine_realization_indexing():
    """S20: the same nervous system indexes identically either way round.

    Two specs differing only in the order the layers appear in the composite
    body. The projections still follow the body (S10), but the neuron axis
    does not.
    """
    forward = """
    L1  := [C = {E}; N = 1];
    L2  := [C = {E}; N = 2];
    L10 := [C = {E}; N = 3];
    V := L1 O L2 O L10;
    x : V : y
    """
    reverse = """
    L10 := [C = {E}; N = 3];
    L2  := [C = {E}; N = 2];
    L1  := [C = {E}; N = 1];
    V := L10 O L2 O L1;
    x : V : y
    """
    _, _, a = _realize(forward)
    _, _, b = _realize(reverse)
    assert a.I["neuron_paths"] == b.I["neuron_paths"]
    assert a.I["neuron_paths"] == ["V.L1", "V.L2", "V.L2",
                                   "V.L10", "V.L10", "V.L10"]


def test_typed_natural_order_beats_lexical_order():
    """S20: `L1 < L2 < L10`, not the lexical `L1 < L10 < L2`."""
    text = """
    L1  := [C = {E}; N = 1];
    L2  := [C = {E}; N = 1];
    L10 := [C = {E}; N = 1];
    V := L1 O L2 O L10;
    x : V : y
    """
    _, _, r = _realize(text)
    assert r.path_to_slice("V.L1") == (0, 1)
    assert r.path_to_slice("V.L2") == (1, 2)
    assert r.path_to_slice("V.L10") == (2, 3)


def test_replication_indices_order_numerically():
    """S20: `SEG.2 < SEG.10`, so replicas do not sort lexically."""
    _, _, r = _realize("SEG := [C = {cell}; N = 1]; x : SEG^12 : y")
    replicas = [p for p in r.I["neuron_paths"] if p.startswith("SEG.")]
    assert replicas == [f"SEG.{i}" for i in range(1, 13)]
    # the lexical order would have put SEG.10 immediately after SEG.1
    assert r.path_to_slice("SEG.2") < r.path_to_slice("SEG.10")


def test_natural_order_is_hierarchical_parent_before_child():
    """A parent's key is a proper prefix of its children's, so it sorts first."""
    from jaxfne.tfne import _natural_path_key

    assert _natural_path_key("V") < _natural_path_key("V.L1")
    assert _natural_path_key("V.L2") < _natural_path_key("V.L10")
    assert _natural_path_key("A.9") < _natural_path_key("A.10")
    # text runs still compare as text
    assert _natural_path_key("V1.L4") < _natural_path_key("V2.L1")


# --------------------------------------------------------------------------- #
# 9d. S20 explicit order declarations
# --------------------------------------------------------------------------- #

_ORDER_BASE = """
L1 := [C = {E}; N = 1];
L2 := [C = {E}; N = 2];
L4 := [C = {E}; N = 3];
V := L4 O L2 O L1;
x : V : y
"""


def test_explicit_order_overrides_natural_order():
    """S20: only `order[A] := [...]` overrides the natural default."""
    _, _, natural = _realize(_ORDER_BASE)
    assert natural.I["neuron_paths"] == ["V.L1", "V.L2", "V.L2",
                                         "V.L4", "V.L4", "V.L4"]
    _, _, declared = _realize("order[V] := [L4, L2, L1];\n" + _ORDER_BASE)
    assert declared.I["neuron_paths"] == ["V.L4", "V.L4", "V.L4",
                                          "V.L2", "V.L2", "V.L1"]


def test_same_order_declaration_survives_reserialization():
    """Serialization is not semantics: same structure + same order, same index.

    The two programs below differ in statement order and in how the composite
    body is written. Neither may reach the neuron axis; only `order[V]` does.
    """
    a = """
    L1 := [C = {E}; N = 1];
    L2 := [C = {E}; N = 2];
    L4 := [C = {E}; N = 3];
    order[V] := [L4, L2, L1];
    V := L4 O L2 O L1;
    x : V : y
    """
    b = """
    order[V] := [L4, L2, L1];
    L4 := [C = {E}; N = 3];
    L1 := [C = {E}; N = 1];
    L2 := [C = {E}; N = 2];
    V := L1 O L2 O L4;
    x : V : y
    """
    _, _, ra = _realize(a)
    _, _, rb = _realize(b)
    assert ra.I["neuron_paths"] == rb.I["neuron_paths"]
    assert ra.I["neuron_paths"] == ["V.L4", "V.L4", "V.L4",
                                    "V.L2", "V.L2", "V.L1"]


_NESTED = """
P := [C = {E}; N = 1];
Q := [C = {E}; N = 2];
A := P O Q;
B := P O Q;
W := A O B;
x : W : y
"""


def test_nested_objects_order_independently():
    """S20 locality: a declaration orders its own members and nothing else."""
    _, _, natural = _realize(_NESTED)
    assert natural.I["neuron_paths"] == ["W.A.P", "W.A.Q", "W.A.Q",
                                         "W.B.P", "W.B.Q", "W.B.Q"]
    _, _, local = _realize("order[W.A] := [Q, P];\n" + _NESTED)
    # A is reordered; B keeps the natural default
    assert local.I["neuron_paths"] == ["W.A.Q", "W.A.Q", "W.A.P",
                                       "W.B.P", "W.B.Q", "W.B.Q"]


@pytest.mark.parametrize("decl,code", [
    ("order[V] := [L4, L2];",           "E_ORDER_INCOMPLETE"),
    ("order[V] := [L4, L4, L2, L1];",   "E_ORDER_DUPLICATE_MEMBER"),
    ("order[V] := [L4, L2, L1, L9];",   "E_ORDER_MEMBER_UNKNOWN"),
    ("order[Z] := [L4];",               "E_ORDER_SCOPE_UNKNOWN"),
    ("order[V] := [L4.1, L2, L1];",     "E_ORDER_NOT_IMMEDIATE"),
])
def test_malformed_order_declarations_fail_closed(decl, code):
    """A declaration that cannot be honoured exactly is refused, not patched.

    Ordering a partial or contradictory declaration would silently index some
    members by declaration and the rest by another rule.
    """
    with pytest.raises(TFNEError, match=code):
        _realize(decl + "\n" + _ORDER_BASE)


def test_ambiguous_order_scope_is_refused():
    """A bare name that matches two nested objects must not be guessed."""
    with pytest.raises(TFNEError, match="E_ORDER_SCOPE_AMBIGUOUS"):
        _realize("order[P] := [E];\n" + _NESTED)


def test_duplicate_order_declaration_for_one_scope_is_refused():
    with pytest.raises(TFNEError, match="E_ORDER_DUPLICATE"):
        _realize("order[V] := [L4, L2, L1];\n"
                 "order[V] := [L1, L2, L4];\n" + _ORDER_BASE)


def test_order_declaration_survives_normalization_replay():
    """Canonical form must carry the declaration, or digest/provenance collide."""
    text = "order[V] := [L4, L2, L1];\n" + _ORDER_BASE
    program = parse(text)
    first = normalize(program)
    assert "order[V]" in first
    assert normalize(parse(first)) == first
    _, _, replayed = _realize(first)
    assert replayed.I["neuron_paths"] == ["V.L4", "V.L4", "V.L4",
                                          "V.L2", "V.L2", "V.L1"]


def test_different_orders_give_different_digests():
    """Two nervous systems differing only in order must not share a digest."""
    from jaxfne.tfne import spec_hash
    natural = parse(_ORDER_BASE)
    declared = parse("order[V] := [L4, L2, L1];\n" + _ORDER_BASE)
    assert spec_hash(natural) != spec_hash(declared)


def test_replica_instances_order_independently():
    """S20 replication: replicas are immediate members and can be reordered."""
    base = "SEG := [C = {E}; N = 1];\nV := SEG^3;\nx : V : y\n"
    _, _, natural = _realize(base)
    assert natural.I["neuron_paths"] == ["V.SEG.1", "V.SEG.2", "V.SEG.3"]
    _, _, declared = _realize("order[V] := [SEG.3, SEG.2, SEG.1];\n" + base)
    assert declared.I["neuron_paths"] == ["V.SEG.3", "V.SEG.2", "V.SEG.1"]


# --------------------------------------------------------------------------- #
# 9e. S6.1 replication with relation (A^{nX} / A^{nO})
# --------------------------------------------------------------------------- #

_REPL_REL_BASE = """
O[k] := [direction = >; mechanism = AMPA; probability = 1.0; weight = 0.5];
SEG := [C = {E}; N = 1];
Q := [C = {E}; N = 2];
%s
x : V : y
"""


def test_replicate_with_X_matches_bare_replication_frontiers():
    """`SEG^{3X}` exposes every instance, exactly like `SEG^3`."""
    _, _, bare = _realize(_REPL_REL_BASE % "V := SEG^3 O[k] Q;")
    _, _, joined = _realize(_REPL_REL_BASE % "V := SEG^{3X} O[k] Q;")
    assert bare.s["n_edges"] == joined.s["n_edges"] == 6
    assert (bare.I["rule_origins"]["r0:O[k]:SEG^3>Q"]["pre_scopes"]
            == joined.I["rule_origins"]["r0:O[k]:SEG^{3X}>Q"]["pre_scopes"]
            == ["V.SEG.1", "V.SEG.2", "V.SEG.3"])


def test_replicate_with_O_restricts_adjacency_to_chain_ends():
    """`SEG^{3O} O[k] Q` binds only the last instance to Q (2 edges, not 6)."""
    _, _, chained = _realize(_REPL_REL_BASE % "V := SEG^{3O} O[k] Q;")
    assert chained.s["n_edges"] == 2
    assert (chained.I["rule_origins"]["r0:O[k]:SEG^{3O}>Q"]["pre_scopes"]
            == ["V.SEG.3"])


def test_replicate_with_O_binds_incoming_to_chain_head():
    """`P O[k] SEG^{3O}` reaches only the first instance."""
    text = """
    O[k] := [direction = >; mechanism = AMPA; probability = 1.0; weight = 0.5];
    SEG := [C = {E}; N = 1];
    P := [C = {E}; N = 2];
    V := P O[k] SEG^{3O};
    x : V : y
    """
    _, _, r = _realize(text)
    assert r.s["n_edges"] == 2
    origins = r.I["rule_origins"]
    key = next(k for k in origins if k.startswith("r0:O[k]:P>"))
    assert origins[key]["post_scopes"] == ["V.SEG.1"]


def test_replicate_relation_survives_normalization_replay():
    """The braced relation form is canonical: replay preserves it exactly."""
    for form in ("SEG^{3X}", "SEG^{3O}", "SEG^{3}", "SEG^3"):
        text = f"SEG := [C = {{E}}; N = 1];\nV := {form};\nx : V : y\n"
        first = normalize(parse(text))
        assert normalize(parse(first)) == first
    assert "SEG^{3X}" in normalize(parse("V := SEG^{3X};\nQ := [C = {E}; N = 1];\nx : V : y\n"))


def test_bare_replication_followed_by_cross_is_not_a_relation():
    """`SEG^2 X Q` stays replication-then-composition; braces are required."""
    _, _, r = _realize(_REPL_REL_BASE % "V := SEG^2 X Q;")
    assert r.I["neuron_paths"] == ["V.Q", "V.Q", "V.SEG.1", "V.SEG.2"]


@pytest.mark.parametrize("form", [
    "V := SEG^{3Y};",
    "V := SEG^{X};",
    "V := SEG^{3};",
])
def test_malformed_replicate_relations_fail_closed(form):
    """Unknown relation letters and missing counts are refused, not guessed."""
    if form == "V := SEG^{3};":
        # Braced plain count is legal and identical to SEG^3.
        _, _, bare = _realize(_REPL_REL_BASE % "V := SEG^3;")
        _, _, braced = _realize(_REPL_REL_BASE % form)
        assert (bare.I["neuron_paths"] == braced.I["neuron_paths"]
                == ["V.SEG.1", "V.SEG.2", "V.SEG.3"])
    else:
        with pytest.raises(TFNEError):
            _realize(_REPL_REL_BASE % form)


# --------------------------------------------------------------------------- #
# 9f. S9 declared composition frontiers (in[A] / out[A])
# --------------------------------------------------------------------------- #

_FRONTIER_RULES = """
O[k] := [direction = >; mechanism = AMPA; probability = 1.0; weight = 0.5];
O[j] := [direction = >; mechanism = AMPA; probability = 1.0; weight = 0.5];
"""

_FRONTIER_DEFS = """
A := [C = {E}; N = 1];
B := [C = {E}; N = 2];
C := [C = {E}; N = 1];
P := [C = {E}; N = 1];
"""


def _frontier_realize(body, declarations=""):
    return _realize(declarations + _FRONTIER_RULES + _FRONTIER_DEFS + body)


def test_derived_chain_binds_head_and_tail():
    """Derived default: `A O[k] B O[k] C` binds A->B and B->C only."""
    _, _, r = _frontier_realize("V := A O[k] B O[k] C;\nx : V : y\n")
    assert r.s["n_edges"] == 4  # 1x2 + 2x1; never A->C
    origins = r.I["rule_origins"]
    assert origins["r0:O[k]:A>B"]["pre_scopes"] == ["V.A"]
    assert origins["r1:O[k]:A O[k] B>C"]["post_scopes"] == ["V.C"]


def test_braced_and_unbraced_chains_coincide_in_edge_set_derived():
    """S8: with derived frontiers only, `{A O B} O C` matches `A O B O C`."""
    _, _, braced = _frontier_realize(
        "V := {A O[k] B} O[k] C;\nx : V : y\n")
    _, _, flat = _frontier_realize("V := A O[k] B O[k] C;\nx : V : y\n")
    assert braced.s["n_edges"] == flat.s["n_edges"] == 4


def test_declared_in_restricts_incoming_binding_to_named_members():
    """`in[M] := [B]` makes `P O[j] M` bind P->B, not P->A."""
    _, _, declared = _frontier_realize(
        "M := A O[k] B;\nV := P O[j] M;\nx : V : y\n",
        "in[M] := [B];\n")
    assert declared.s["n_edges"] == 4  # 2 (A->B) + 2 (P->B)
    origins = declared.I["rule_origins"]
    assert origins["r1:O[j]:P>M"]["post_scopes"] == ["V.M.B"]
    # The inner rule still binds by name: S10 untouched by the declaration.
    assert origins["r0:O[k]:A>B"]["pre_scopes"] == ["V.M.A"]


def test_declared_out_restricts_outgoing_binding():
    """`out[M] := [A]` makes `M O[j] C` bind A->C, not B->C."""
    _, _, declared = _frontier_realize(
        "M := A O[k] B;\nV := M O[j] C;\nx : V : y\n",
        "out[M] := [A];\n")
    origins = declared.I["rule_origins"]
    assert origins["r1:O[j]:M>C"]["pre_scopes"] == ["V.M.A"]
    assert declared.s["n_edges"] == 3  # 2 (A->B) + 1 (A->C)


def test_declared_in_and_out_together():
    """Both sides declare independently; each overrides only its own side."""
    _, _, declared = _frontier_realize(
        "M := A O[k] B;\nV := P O[j] M O[j] C;\nx : V : y\n",
        "in[M] := [B];\nout[M] := [A];\n")
    origins = declared.I["rule_origins"]
    assert origins["r1:O[j]:P>M"]["post_scopes"] == ["V.M.B"]
    assert origins["r2:O[j]:P O[j] M>C"]["pre_scopes"] == ["V.M.A"]


def test_declared_frontier_makes_braced_differ_from_flat():
    """S8's load-bearing case: only a declaration changes the edge set."""
    _, _, flat = _frontier_realize(
        "M := A O[k] B;\nV := P O[j] M;\nx : V : y\n")
    _, _, declared = _frontier_realize(
        "M := A O[k] B;\nV := P O[j] M;\nx : V : y\n",
        "in[M] := [B];\n")
    assert flat.s["n_edges"] == 3
    assert declared.s["n_edges"] == 4


def test_nested_declarations_own_only_their_interface():
    """Derived frontiers compose through declared ones; declarations
    override per scope: with only `in[A] := [P]`, W's derived in-interface
    narrows to P; adding `in[W] := [A]` reopens W to A's whole subtree."""
    body = (_FRONTIER_RULES
            + "P := [C = {E}; N = 1];\nQ := [C = {E}; N = 2];\n"
            + "D := P O[k] Q;\nE := P O[k] Q;\n"
            + "W := D O[k] E;\nV := C O[j] W;\nx : V : y\n"
            + "C := [C = {E}; N = 1];\n")
    _, _, narrowed = _realize("in[D] := [P];\n" + body)
    outer_n = next(v for k, v in narrowed.I["rule_origins"].items()
                   if ">W" in k)
    assert outer_n["post_scopes"] == ["V.W.D.P"]
    _, _, reopened = _realize("in[W] := [D];\nin[D] := [P];\n" + body)
    origins = reopened.I["rule_origins"]
    outer = next(v for k, v in origins.items() if ">W" in k)
    # W's interface is D whole (P and Q subtrees); D's own stays P.
    assert outer["post_scopes"] == ["V.W.D"]
    inner_d = origins["r0:O[k]:P>Q"]
    assert inner_d["pre_scopes"] == ["V.W.D.P"]


def test_group_and_nested_mixed_composition_bind_endpoints():
    """`{A X B} O C`, `A O {B X C}` and nesting bind frontier endpoints."""
    _, _, r = _frontier_realize(
        "V := {A X[k] B} O[k] C;\nx : V : y\n")
    origins = r.I["rule_origins"]
    outer = next(v for k, v in origins.items() if k.endswith(">C"))
    assert sorted(outer["pre_scopes"]) == ["V.g0.A", "V.g0.B"]
    _, _, r2 = _frontier_realize("V := A O[k] {B X[k] C};\nx : V : y\n")
    origins2 = r2.I["rule_origins"]
    first = next(v for k, v in origins2.items()
                 if k.endswith(">{B X[k] C}"))
    assert first["post_scopes"] == ["V.g0.B", "V.g0.C"]


def test_declared_frontier_names_replica_members():
    """Replica remainders are immediate members: `out[V] := [SEG.3]`."""
    body = "SEG := [C = {E}; N = 1];\nQ := [C = {E}; N = 1];\nV := SEG^3;\nW := V O[k] Q;\nx : W : y\n"
    _, _, declared = _realize(
        "out[V] := [SEG.3];\n" + _FRONTIER_RULES + body)
    origins = declared.I["rule_origins"]
    assert origins["r0:O[k]:V>Q"]["pre_scopes"] == ["W.V.SEG.3"]
    assert declared.s["n_edges"] == 1


def test_declared_frontier_applies_under_replicate_join():
    """`SEG^{3O}` chain tail is the only member an out-declaration can name
    beyond the derived tail — declaring it changes nothing, pinning that
    derived and declared tails agree."""
    body = "SEG := [C = {E}; N = 1];\nQ := [C = {E}; N = 1];\nV := SEG^{3O};\nW := V O[k] Q;\nx : W : y\n"
    _, _, derived = _realize(_FRONTIER_RULES + body)
    _, _, declared = _realize(
        "out[V] := [SEG.3];\n" + _FRONTIER_RULES + body)
    assert derived.s["n_edges"] == declared.s["n_edges"] == 1
    assert (derived.I["rule_origins"]["r0:O[k]:V>Q"]["pre_scopes"]
            == declared.I["rule_origins"]["r0:O[k]:V>Q"]["pre_scopes"]
            == ["W.V.SEG.3"])


@pytest.mark.parametrize("decl,body", [
    ("in[M] := [Z];", "M := A O[k] B;\nV := P O[j] M;\nx : V : y\n"),
    ("in[M] := [A.B];", "M := A O[k] B;\nV := P O[j] M;\nx : V : y\n"),
    ("in[M] := [A, A];", "M := A O[k] B;\nV := P O[j] M;\nx : V : y\n"),
    ("in[M] := [B];\nin[M] := [A];",
     "M := A O[k] B;\nV := P O[j] M;\nx : V : y\n"),
    ("in[Z] := [A];", "M := A O[k] B;\nV := P O[j] M;\nx : V : y\n"),
    ("in[A] := [P];", "M := A O[k] B;\nV := P O[j] M;\nx : V : y\n"),
    ("in[M] := [];", "M := A O[k] B;\nV := P O[j] M;\nx : V : y\n"),
])
def test_malformed_frontier_declarations_fail_closed(decl, body):
    """Unknown, nonlocal, duplicated, empty, or rescoped declarations are
    refused as E_FRONTIER_UNRESOLVED — never partially applied."""
    with pytest.raises(TFNEError, match="E_FRONTIER_UNRESOLVED"):
        _frontier_realize(body, decl + "\n")


def test_ambiguous_frontier_scope_is_refused():
    """A bare name matching two nested objects must not be guessed."""
    body = ("P := [C = {E}; N = 1];\nQ := [C = {E}; N = 2];\n"
            "A := P O[k] Q;\nB := P O[k] Q;\n"
            "M := A O[k] B;\nN := A O[k] Q;\n"
            "V := M O[j] N;\nx : V : y\n")
    with pytest.raises(TFNEError, match="E_FRONTIER_UNRESOLVED"):
        _realize("in[A] := [P];\n" + _FRONTIER_RULES + body)


def test_frontier_declaration_survives_normalization_replay():
    """Frontier metadata is realization-affecting: NF carries it exactly."""
    text = ("in[M] := [B];\nout[M] := [A];\n"
            "M := A O[k] B;\nV := P O[j] M;\nx : V : y\n")
    first = normalize(parse(_FRONTIER_RULES + _FRONTIER_DEFS + text))
    assert "in[M] := [B]" in first and "out[M] := [A]" in first
    assert normalize(parse(first)) == first


def test_frontier_semantics_drive_digest():
    """Different interfaces are different models: digests must differ."""
    from jaxfne.tfne import spec_hash
    base = _FRONTIER_RULES + _FRONTIER_DEFS + "M := A O[k] B;\nV := P O[j] M;\nx : V : y\n"
    assert (spec_hash(parse(base))
            != spec_hash(parse("in[M] := [B];\n" + base))
            != spec_hash(parse("in[M] := [A];\n" + base)))


def test_equivalent_serialization_keeps_frontier_semantics():
    """Statement order and composite-body spelling never reach the axis."""
    a = ("in[M] := [B];\nM := A O[k] B;\nV := P O[j] M;\nx : V : y\n")
    b = ("M := B O[k] A;\nV := P O[j] M;\nin[M] := [B];\nx : V : y\n")
    _, _, ra = _frontier_realize(a)
    _, _, rb = _frontier_realize(b)
    assert ra.I["neuron_paths"] == rb.I["neuron_paths"]
    assert ra.s["n_edges"] == rb.s["n_edges"] == 4
