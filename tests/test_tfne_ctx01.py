"""CTX-01: one integrated cortical model across the conformant algebra.

A single compact TFNE source jointly exercises named reusable area
definitions, layers with N/P allocation, O/X relations with typed rule
bodies, declared in/out frontiers, mechanisms with kinetics, replication,
order[A], and contextual atomicity — verified by semantic class across
TFNE -> JDNA completion -> JaxFNE Model -> executed model. Asymmetric
values throughout so accidental defaults cannot pass.

PARAM-04 (geometry) is repaired here: declared relative sub-ranges reach
executed positions and outside-[0,1] ranges refuse. PARAM-02 (delay) is
repaired here as well: a declared delay transfers to executed delay_steps
(its refusal half lives in `test_tfne_delay_transfer.py`).
"""

import numpy as np
import pytest

import jaxfne
from jaxfne.jdna.completion import (
    ORIGIN_DECLARED as JDNA_DECLARED,
    ORIGIN_DEFAULT as JDNA_DEFAULT,
    ORIGIN_SAMPLED as JDNA_SAMPLED,
    complete_tfne,
)
from jaxfne.tfne import (
    TFNEError,
    normalize,
    parse,
    realize,
    resolve,
    spec_hash,
    to_configuration,
    to_neuronal_tensor,
)

DT_MS = 0.1
DURATION_MS = 5.0

CTX = """
O[both] := [mechanism = AMPA; weight = 0.625; plasticity = stdp; $L.out>$R.in [mech=AMPA]; {L4,L23}<L5 [mech=GABA_A]; {L23}<>{L23} [mech=AMPA]];
O[aux] := [direction = >; mechanism = AMPA; weight = 0.375];
out[V1] := [L23];
in[V2] := [L4];
order[V1] := [L5, L23, L4, SEG.2, SEG.1];
L4 := [C = {E}; N = 2; G = [z0 = 0.0; z1 = 0.75]];
L23 := [C = {E, PV}; P = {E: 0.667, PV: 0.333}; N = 3];
L5 := [C = {E}; N = 1];
SEG := [C = {E}; N = 1];
V1 := L4 O L23 O L5 O SEG^2;
V2 := L4 O L23 O L5;
AUX := {L5 O[aux] L4; Z.Q > L4};
V := AUX O V1 O[both] V2;
x : V : y
"""


def _ctx(seed=None):
    program = parse(CTX)
    explicit = resolve(program)
    return program, explicit, realize(explicit, program, seed=seed)


def test_source_normalizes_with_digest_stability():
    """The compact source is canonical: replay-stable, digest-pinned
    against silent reserialization drift."""
    first = normalize(parse(CTX))
    assert normalize(parse(first)) == first
    for token in (
        "order[V1]",
        "out[V1]",
        "in[V2]",
        "$L.out",
        "{L4, L23} < L5",
        "SEG^2",
        "plasticity",
    ):
        assert token in first
    assert spec_hash(parse(CTX)) == spec_hash(parse(first))


def test_object_identity_and_slices():
    """Named reusable definitions address every realized neuron."""
    _, _, r = _ctx()
    assert r.s["n_neurons"] == 17
    assert r.path_to_slice("V.V1.L23") == (4, 7)
    assert r.path_to_slice("V.V1.SEG.1") == (10, 11)
    assert r.path_to_slice("V.AUX.g0.L4") == (0, 2)
    assert r.slice_to_path(0) == "V.AUX.g0.L4"
    assert r.slice_to_path(16) == "V.V2.L23"


def test_cardinality_and_proportions():
    """N/P allocation is exact and asymmetric: L23 holds 2E+1PV."""
    _, _, r = _ctx()
    assert r.s["counts"]["V.V1.L23"] == {"E": 2, "PV": 1}
    assert r.s["counts"]["V.V2.L23"] == {"E": 2, "PV": 1}
    assert sum(r.s["counts"]["V.V1.L23"].values()) == 3


def test_ordering_override_with_replicas():
    """order[V1] lays out L5, L23, L4, SEG.2, SEG.1; V2 keeps the
    natural default L4 < L5 < L23."""
    _, _, r = _ctx()
    assert r.I["neuron_paths"][:11] == [
        "V.AUX.g0.L4",
        "V.AUX.g0.L4",
        "V.AUX.g0.L5",
        "V.V1.L5",
        "V.V1.L23",
        "V.V1.L23",
        "V.V1.L23",
        "V.V1.L4",
        "V.V1.L4",
        "V.V1.SEG.2",
        "V.V1.SEG.1",
    ]
    assert r.I["neuron_paths"][11:] == [
        "V.V2.L4",
        "V.V2.L4",
        "V.V2.L5",
        "V.V2.L23",
        "V.V2.L23",
        "V.V2.L23",
    ]
    assert r.path_to_slice("V.V2.L4") == (11, 13)


def test_topology_by_statement_identities():
    """31 edges from named statements, not counts: 6 + 5 + 9 + 9 + 2."""
    _, _, r = _ctx()
    origins = r.I["rule_origins"]
    assert origins["r1:O[both]:$L.out>$R.in"]["pre_scopes"] == ["V.V1.L23"]
    assert origins["r1:O[both]:$L.out>$R.in"]["post_scopes"] == ["V.V2.L4"]
    assert origins["r2:O[both]:{L4, L23}<L5"]["pre_scopes"] == ["V.V1.L4", "V.V1.L23"]
    assert origins["r4:O[both]:{L23}<{L23}"]["pre_scopes"] == ["V.V1.L23"]
    assert r.s["n_edges"] == 31


def test_frontiers_own_only_their_interface():
    """r1 binds through declared out/in; the inner names are untouched."""
    _, _, r = _ctx()
    origins = r.I["rule_origins"]
    assert origins["r1:O[both]:$L.out>$R.in"]["pre_scopes"] == ["V.V1.L23"]
    assert origins["r0:O[aux]:L5>L4"]["pre_scopes"] == ["V.AUX.g0.L5"]


def test_mechanism_identity_and_kinetics_executed():
    """Both mechanisms transfer by identity and execute at canonical
    kinetics — asymmetrically, so no uniform placeholder can pass."""
    _, explicit, r = _ctx()
    model = jaxfne.construct(to_configuration(r, duration_ms=DURATION_MS, dt_ms=DT_MS))
    mechs = {c["mechanism"] for c in r.s["connection_table"]}
    assert mechs == {"AMPA", "GABA_A"}
    taus = {round(float(e["tau_ms"]), 6) for e in model.edge_table()}
    assert taus == {2.0, 5.0}
    tensor = to_neuronal_tensor(explicit)
    tensor_mechs = set()
    for area in tensor.areas:
        tensor_mechs.update(c.mechanism for c in area.inter_connections)
        tensor_mechs.update(c.mechanism for c in tensor.area_connections)
    assert {"AMPA", "GABA_A"} <= tensor_mechs


def test_fixed_weights_are_non_default():
    """0.625 and 0.375 are asymmetric non-defaults: a fallback to 1.0,
    0.5, or the old 0.353553 substitution is visible."""
    _, _, r = _ctx()
    weights = {round(float(w), 6) for w in np.asarray(r.s["edge_weight"])}
    assert weights == {0.625, 0.375}


def test_developmental_provenance_per_leaf():
    """Every leaf answers which stage chose its geometry: TFNE-declared
    bounds, JDNA defaults elsewhere, sampled positions, explicit K_D."""
    _, _, r = _ctx(seed=11)
    out = complete_tfne(r, seed=7)
    assert out["domain"] == "K_D" and out["seed"] == 7
    assert len(out["positions"]) == 10
    assert out["origins"]["V.V1.L4"]["z"] == JDNA_DECLARED
    assert out["origins"]["V.V1.SEG.1"]["z"] == JDNA_DEFAULT
    assert out["origins"]["V.V1.L4"]["positions"] == JDNA_SAMPLED
    pos = np.asarray(out["positions"]["V.V1.L4"])
    assert pos.shape == (2, 3)
    assert float(pos[:, 2].min()) >= 0.0 and float(pos[:, 2].max()) <= 0.75
    again = complete_tfne(r, seed=7)["positions"]
    for leaf in again:
        assert bool((np.asarray(again[leaf]) == np.asarray(out["positions"][leaf])).all())


def test_geometry_outside_unit_interval_refused():
    """PARAM-04 repair complement: the old inert ranges refuse at resolve.

    The fixture declares a relative sub-range (z1 = 0.75); the pre-repair
    inert values it replaced (z1 = 4.0, and 40.0) are outside [0,1] and
    now fail closed with `E_GEOMETRY_OUT_OF_RANGE` instead of executing.
    Executed-sub-range coverage lives in
    `test_declared_geometry_reaches_the_executed_positions`.
    """
    for bad_z1 in ("4.0", "40.0"):
        other = CTX.replace("z1 = 0.75", f"z1 = {bad_z1}")
        with pytest.raises(TFNEError, match="E_GEOMETRY_OUT_OF_RANGE"):
            resolve(parse(other))


def test_plastic_provenance_without_execution_effect():
    """`plasticity = stdp` is recorded identity, not an edge parameter:
    31 edges with or without it (compare against the stripped spec)."""
    _, _, r = _ctx()
    key = next(k for k in r.I["rule_origins"] if "both" in k)
    assert r.relation_origin(key)["params"].get("plasticity") == "stdp"
    stripped = CTX.replace("; plasticity = stdp;", ";")
    prog = parse(stripped)
    r0 = realize(resolve(prog), prog)
    assert r0.s["n_edges"] == r.s["n_edges"] == 31


def test_atomic_coexistence_leaves_no_trace():
    """The dropped `Z.Q > L4` statement creates no nodes, edges, or
    relations inside the integrated model."""
    _, _, r = _ctx()
    assert "Z" not in str(r.I["neuron_paths"])
    assert list(r.I["rule_origins"]) == [
        "r0:O[aux]:L5>L4",
        "r1:O[both]:$L.out>$R.in",
        "r2:O[both]:{L4, L23}<L5",
        "r3:O[both]:{L23}>{L23}",
        "r4:O[both]:{L23}<{L23}",
    ]


def test_constructed_model_matches_realization():
    """TFNE -> JDNA -> Model -> executed: the neuron table follows the
    realization axis and every executed edge was realized."""
    _, _, r = _ctx()
    model = jaxfne.construct(to_configuration(r, duration_ms=DURATION_MS, dt_ms=DT_MS))
    executed = [(row["area"], row["layer"]) for row in model.neuron_table()]
    realized = [(p.split(".")[0], p.split(".")[1]) for p in r.I["neuron_paths"]]
    assert executed == realized
    realized_pairs = {
        (int(a), int(b)) for a, b in zip(np.asarray(r.s["edge_pre"]), np.asarray(r.s["edge_post"]))
    }
    for e in model.edge_table():
        assert (int(e["pre"]), int(e["post"])) in realized_pairs


def test_simulation_evidence_is_finite():
    """The integrated model simulates: finite V_m and spikes."""
    _, _, r = _ctx()
    model = jaxfne.construct(to_configuration(r, duration_ms=DURATION_MS, dt_ms=DT_MS))
    signals = jaxfne.simulate(model, duration_ms=DURATION_MS, dt_ms=DT_MS, seed=0)
    v = np.asarray(signals.V_m)
    assert v.shape[1] == r.s["n_neurons"] == 17
    assert bool(np.isfinite(v).all())


def test_refused_quantities_fail_closed():
    """Unresolvable mechanisms refuse; declared delay transfers (PARAM-02).

    GABA at a statement fails closed with its own code; delay on the rule
    is carried end to end (2.0 ms at dt 0.1 -> 20 steps on the executed
    edges of that rule's projection).
    """
    with pytest.raises(TFNEError, match="E_MECHANISM_UNRESOLVED"):
        bad = CTX.replace("$L.out>$R.in [mech=AMPA]", "$L.out>$R.in [mech=GABA]")
        prog = parse(bad)
        to_configuration(realize(resolve(prog), prog), duration_ms=DURATION_MS, dt_ms=DT_MS)
    from jaxfne.emitters import resolve_edge_delay_steps

    bad = CTX.replace("weight = 0.625", "weight = 0.625; delay = 2.0")
    prog = parse(bad)
    r = realize(resolve(prog), prog)
    model = jaxfne.construct(to_configuration(r, duration_ms=DURATION_MS, dt_ms=DT_MS))
    assert int(np.asarray(resolve_edge_delay_steps(model.params["edge_list"])).max()) == 20
