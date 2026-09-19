"""TFNE-PARAM-01: declared parameters must survive into execution.

Topology identity is not parameter identity. Equal edge counts hid a total
substitution of the declared weight: three specifications declaring 0.5, 0.25
and 0.125 all executed at 0.353553, and a declared probability of 0.5 realized
8 edges while executing 16.

Every case here compares realized and executed edges as multisets of
``(pre, post, weight, mechanism)`` rather than by count, so a substituted
parameter cannot pass. Values are asymmetric and non-default so a swap or a
fallback default is visible.
"""

from collections import Counter

import numpy as np
import pytest

import jaxfne
from jaxfne.tfne import (TFNEError, parse, realize, resolve, to_configuration,
                         to_neuronal_tensor)

DT_MS = 0.1
DURATION_MS = 5.0


def _realized_edges(r):
    """Realized edges as a multiset of (pre, post, weight, mechanism)."""
    mech_names = [m["name"] for m in r.s["mechanism_table"]]
    return Counter(
        (int(pre), int(post), round(float(w), 6), mech_names[int(m)])
        for pre, post, w, m in zip(np.asarray(r.s["edge_pre"]),
                                   np.asarray(r.s["edge_post"]),
                                   np.asarray(r.s["edge_weight"]),
                                   np.asarray(r.s["edge_mechanism"]))
    )


def _executed_edges(model):
    """Executed edges in the same shape. Mechanism is the receptor's kind."""
    return Counter(
        (int(e["pre"]), int(e["post"]), round(float(e["weight"]), 6),
         str(e["receptor_type"]).split("__")[0])
        for e in model.edge_table()
    )


def equivalence(spec):
    """Realize a spec, execute it, and require edge-for-edge equivalence."""
    program = parse(spec)
    explicit = resolve(program)
    r = realize(explicit, program)
    model = jaxfne.construct(
        to_configuration(r, duration_ms=DURATION_MS, dt_ms=DT_MS))
    realized, executed = _realized_edges(r), _executed_edges(model)
    assert executed == realized, (
        "realized and executed edges differ\n"
        f"  realized only: {sorted((realized - executed).elements())[:8]}\n"
        f"  executed only: {sorted((executed - realized).elements())[:8]}"
    )
    return r, model, realized


AB = "A := [C = {E}; N = 4]; B := [C = {E}; N = 4];"
ABC = AB + " C := [C = {E}; N = 4];"
ABCD = ABC + " D := [C = {E}; N = 4];"


# --------------------------------------------------------------------------- #
# a-d: one parameter at a time
# --------------------------------------------------------------------------- #

def test_a_weight_only():
    r, _, edges = equivalence(
        "O[k] := [direction = >; mechanism = AMPA; weight = 0.375];\n"
        f"{AB}\nx : A O[k] B : y\n")
    assert {w for _, _, w, _ in edges} == {0.375}


def test_b_probability_only():
    """A declared probability must thin the executed edges, not just the realized."""
    r, model, edges = equivalence(
        "O[k] := [direction = >; mechanism = AMPA; probability = 0.5];\n"
        f"{AB}\nx : A O[k] B : y\n")
    assert 0 < r.s["n_edges"] < 16
    assert len(model.edge_table()) == r.s["n_edges"]


def test_c_delay_is_refused_not_dropped():
    """No execution path carries delay, so declaring it must fail loudly."""
    spec = ("O[k] := [direction = >; mechanism = AMPA; delay = 2.0];\n"
            f"{AB}\nx : A O[k] B : y\n")
    program = parse(spec)
    explicit = resolve(program)
    with pytest.raises(TFNEError, match="E_PARAM_UNSUPPORTED"):
        realize(explicit, program)


def test_c2_plasticity_is_preserved_as_provenance_not_dropped():
    """`plasticity` is a declared rule identity, not a connection parameter.

    It names a rule for the separate registrable HDP surface, so unlike
    `delay` it is not refused -- but it must not vanish either. It stays
    readable on the relation it was declared on, and it does not perturb the
    realized/executed edges it accompanies.
    """
    spec = ("O[k] := [direction = >; mechanism = AMPA; weight = 0.375;\n"
            "         plasticity = stdp];\n"
            f"{AB}\nx : A O[k] B : y\n")
    r, _, edges = equivalence(spec)
    key = next(iter(r.I["rule_slices"]))
    assert r.relation_origin(key)["params"]["plasticity"] == "stdp"
    assert {w for _, _, w, _ in edges} == {0.375}


def test_d_mechanism_and_weight():
    # GABA_A, not GABA: bare GABA is ambiguous between GABA_A (5ms) and
    # GABA_B (150ms), so the vocabulary refuses it rather than aliasing.
    # These fixtures test inhibitory-mechanism identity transfer, which
    # the canonical inhibitory receptor preserves.
    _, _, edges = equivalence(
        "O[k] := [direction = >; mechanism = GABA_A; weight = 0.625];\n"
        f"{AB}\nx : A O[k] B : y\n")
    assert {m for _, _, _, m in edges} == {"GABA_A"}
    assert {w for _, _, w, _ in edges} == {0.625}


# --------------------------------------------------------------------------- #
# e-h: parameters that must not leak between sites
# --------------------------------------------------------------------------- #

def test_e_adjacent_relations_keep_distinct_weights():
    """Two adjacencies in one chain, deliberately unequal weights."""
    r, _, edges = equivalence(
        "O[k] := [direction = >; mechanism = AMPA; weight = 0.125];\n"
        "O[j] := [direction = >; mechanism = AMPA; weight = 0.875];\n"
        f"{ABC}\nx : A O[k] B O[j] C : y\n")
    lo, hi = r.path_to_slice("A")
    a_ids = set(range(lo, hi))
    from_a = {w for pre, _, w, _ in edges if pre in a_ids}
    from_b = {w for pre, _, w, _ in edges if pre not in a_ids}
    assert from_a == {0.125}, from_a
    assert from_b == {0.875}, from_b


def test_f_ordered_and_cross_rules_keep_distinct_parameters():
    _, _, edges = equivalence(
        "O[k] := [direction = >; mechanism = AMPA; weight = 0.25];\n"
        "X[j] := [direction = >; mechanism = GABA_A; weight = 0.75];\n"
        f"{ABCD}\nx : {{A O[k] B}} X[j] {{C O[k] D}} : y\n")
    by_mech = {}
    for _, _, w, m in edges:
        by_mech.setdefault(m, set()).add(w)
    assert by_mech == {"AMPA": {0.25}, "GABA_A": {0.75}}, by_mech


def test_g_shared_rule_reused_at_several_sites():
    """One rule at three adjacencies: every site gets the declared value."""
    r, _, edges = equivalence(
        "O[k] := [direction = >; mechanism = AMPA; weight = 0.4375];\n"
        f"{ABCD}\nx : A O[k] B O[k] C O[k] D : y\n")
    assert len(r.explicit.relations) == 3
    assert {w for _, _, w, _ in edges} == {0.4375}
    assert r.s["n_edges"] == 48          # 3 adjacencies x 4x4


def test_h_no_per_site_override_in_the_grammar():
    """Per-site parameter override is not expressible, so it cannot diverge.

    `tfne/2` binds parameters to the rule, and `A O[k][weight=...] B` is not
    grammar. Recorded so that if an override is added later, the equivalence
    helper above is applied to it rather than the feature landing untested.
    """
    spec = ("O[k] := [direction = >; mechanism = AMPA; weight = 0.25];\n"
            f"{AB}\nx : A O[k][weight = 0.5] B : y\n")
    with pytest.raises(TFNEError):
        resolve(parse(spec))


# --------------------------------------------------------------------------- #
# Invariants the equivalence rests on
# --------------------------------------------------------------------------- #

def test_executed_neuron_order_matches_the_realization():
    """Specs address neurons by realized id, so the orders must agree.

    If this drifts, `to_configuration` would silently wire the wrong neurons
    while every edge count still matched.
    """
    spec = ("O[k] := [direction = >; mechanism = AMPA; weight = 0.5];\n"
            "L4 := [C = {E, PV}; P = {E: 0.8, PV: 0.2}; N = 10];\n"
            "L2 := [C = {E, PV}; P = {E: 0.6, PV: 0.4}; N = 5];\n"
            "V1 := L4 O[k] L2;\nx : V1 : y\n")
    program = parse(spec)
    explicit = resolve(program)
    r = realize(explicit, program)
    model = jaxfne.construct(
        to_configuration(r, duration_ms=DURATION_MS, dt_ms=DT_MS))

    types = [None] * r.s["n_neurons"]
    for key, (lo, hi) in r.I["member_slices"].items():
        for i in range(lo, hi):
            types[i] = key.split(".")[-1]
    realized_seq = [
        (p.split(".")[0], p.split(".")[1] if "." in p else p, t)
        for p, t in zip(r.I["neuron_paths"], types)
    ]
    executed_seq = [(row["area"], row["layer"], row["cell_type"])
                    for row in model.neuron_table()]
    assert executed_seq == realized_seq


def test_tensor_bridge_still_cannot_carry_parameters():
    """Why execution goes through the realized specs and not the tensor.

    `InterConnection`/`AreaConnection` have no field for a declared weight,
    probability or delay. Pinned so that if those structures gain such
    fields, this route is reconsidered rather than left as the workaround.
    """
    spec = ("O[k] := [direction = >; mechanism = AMPA; weight = 0.375];\n"
            f"{AB}\nx : A O[k] B : y\n")
    tensor = to_neuronal_tensor(resolve(parse(spec)))
    conns = [c for area in tensor.areas for c in area.inter_connections]
    conns += list(tensor.area_connections)
    assert conns, "expected at least one bridged connection"
    for c in conns:
        assert not hasattr(c, "probability")
        assert not hasattr(c, "delay_ms")
        assert c.plastic.w_mech == 1.0      # declared 0.375 cannot land here


def test_kernel_resolved_weights_equal_the_realized_weights():
    """The equivalence above reads `edge_table()`; this reads the kernel.

    `edge_table()` could in principle report declared metadata rather than
    what the integrator uses. Every `simulate_*` in `jaxfne.emitters`
    resolves synaptic weight through `_resolved_edge_weight(edges, dtype,
    params)`; this calls that same function on the constructed model and
    compares it to `s["edge_weight"]`, so the claim rests on the kernel's own
    resolver rather than on a reporting surface.
    """
    from jaxfne.emitters import _resolved_edge_weight

    spec = ("O[k] := [direction = >; mechanism = AMPA; weight = 0.3125];\n"
            f"{AB}\nx : A O[k] B : y\n")
    program = parse(spec)
    explicit = resolve(program)
    r = realize(explicit, program)
    model = jaxfne.construct(
        to_configuration(r, duration_ms=DURATION_MS, dt_ms=DT_MS))

    edges = model.params["edge_list"]
    kernel_w = np.asarray(_resolved_edge_weight(
        edges, edges.weight.dtype, model.params["emitter"]))
    realized_w = np.asarray(r.s["edge_weight"])
    assert kernel_w.shape == realized_w.shape
    assert np.allclose(np.sort(kernel_w), np.sort(realized_w), atol=1e-6), (
        f"kernel {sorted(set(kernel_w.round(6).tolist()))} != "
        f"realized {sorted(set(realized_w.round(6).tolist()))}")
    assert np.allclose(kernel_w, 0.3125, atol=1e-6)


@pytest.mark.parametrize("weight,other", [("0.5", "8.0")])
def test_declared_weight_changes_the_integrated_trajectory(weight, other):
    """Stored is not consumed: the value must alter the dynamics.

    Equal arrays would still pass every check above if the kernel read the
    weight into a buffer it never used. Two specs differing only in declared
    weight must therefore produce different membrane trajectories.
    """
    spec = (
        "O[k] := [direction = >; mechanism = AMPA; probability = 1.0;\n"
        "         weight = %s];\n"
        "A := [C = {E}; N = 20; model = izhikevich];\n"
        "B := [C = {E}; N = 20; model = izhikevich];\n"
        "x : A O[k] B : y\n"
    )

    def trace(w):
        program = parse(spec % w)
        explicit = resolve(program)
        r = realize(explicit, program)
        model = jaxfne.construct(
            to_configuration(r, duration_ms=40.0, dt_ms=DT_MS))
        signals = jaxfne.simulate(model, duration_ms=40.0, dt_ms=DT_MS, seed=0)
        return np.asarray(signals.V_m)

    low, high = trace(weight), trace(other)
    assert low.shape == high.shape
    assert np.isfinite(low).all() and np.isfinite(high).all()
    assert not np.allclose(low, high), (
        f"declared weight {weight} and {other} integrated identically; the "
        "kernel is not consuming the realized weight")


def test_synaptic_tau_does_not_depend_on_the_integration_timestep():
    """Kinetics must not be coupled to `dt`.

    `to_configuration` replaces the circuit's mechanism declarations, so it
    has to supply a `tau_ms`. TFNE declares none -- the realized mechanism
    table carries `tau_ms: None` and `declared_not_simulated` -- so the value
    is inherited from the structural bridge, which derives it from the
    connection's `static.dT_ms`.

    An earlier revision of `to_configuration` used `dt_ms` instead. It agreed
    with the bridge at the default 0.1 by coincidence and diverged elsewhere
    (0.025 against 0.1), which would make a refined timestep silently change
    the synapse model rather than integrate the same one more accurately.
    """
    from jaxfne.neuronal_tensor import neuronal_tensor_to_configuration

    spec = ("O[k] := [direction = >; mechanism = AMPA; probability = 1.0;\n"
            "         weight = 0.5];\n"
            f"{AB}\nx : A O[k] B : y\n")
    program = parse(spec)
    explicit = resolve(program)
    r = realize(explicit, program)

    taus = {}
    for dt in (0.1, 0.025, 0.5):
        model = jaxfne.construct(
            to_configuration(r, duration_ms=5.0, dt_ms=dt))
        bridge = jaxfne.construct(neuronal_tensor_to_configuration(
            to_neuronal_tensor(explicit), seed=int(r.s["seed"]),
            duration_ms=5.0, dt_ms=dt))
        taus[dt] = {round(float(e["tau_ms"]), 6) for e in model.edge_table()}
        bridge_taus = {round(float(e["tau_ms"]), 6)
                       for e in bridge.edge_table()}
        assert taus[dt] == bridge_taus, (
            f"dt={dt}: to_configuration tau {sorted(taus[dt])} != "
            f"bridge tau {sorted(bridge_taus)}")

    assert len({frozenset(v) for v in taus.values()}) == 1, (
        f"synaptic tau varies with the integration timestep: {taus}")


def test_declared_geometry_does_not_reach_the_executed_positions():
    """TFNE-PARAM-04, pinned: geometry is realized but not executed.

    Geometry is the one semantic class where a declared value still does not
    survive. `G = [z0 = ...; z1 = ...]` is recorded faithfully in
    `s["geometry"]`, but at equal seed the executed positions are
    bit-identical whatever range is declared -- (0,1), (10,20) and (-5,-4)
    all sample the same z.
    A `value_tag="relative"` normalization would still let the declared extent
    matter; identical output means the declaration is inert.

    This matters because geometry is what field observables are computed
    against, so an LFP-style claim would rest on coordinates the
    specification did not choose.

    Asserted as the *current* state, not as correct. When PARAM-04 is
    repaired this test fails and must be inverted to require the declared
    range -- that is the point of pinning it rather than describing it.
    """
    def executed_z(z0, z1):
        spec = (
            "O[k] := [direction = >; mechanism = AMPA; probability = 1.0;\n"
            "         weight = 0.5];\n"
            f"A := [C = {{E}}; N = 60; model = izhikevich; "
            f"G = [z0 = {z0}; z1 = {z1}]];\n"
            "x : A : y\n"
        )
        program = parse(spec)
        explicit = resolve(program)
        # Pin the seed. It otherwise defaults to the normalization digest,
        # which changes with the source text, so the declared range and the
        # RNG stream would both vary and neither could be isolated.
        r = realize(explicit, program, seed=20260918)
        model = jaxfne.construct(
            to_configuration(r, duration_ms=5.0, dt_ms=DT_MS))
        return r, np.asarray(model.params["positions"])[:, 2]

    r_unit, z_unit = executed_z(0.0, 1.0)
    r_far, z_far = executed_z(10.0, 20.0)

    # realization keeps the declaration
    assert r_unit.s["geometry"]["A"] == {"z0": 0.0, "z1": 1.0}
    assert r_far.s["geometry"]["A"] == {"z0": 10.0, "z1": 20.0}

    # execution ignores it entirely
    assert np.allclose(z_unit, z_far), (
        "declared geometry now reaches the executed positions -- TFNE-PARAM-04 "
        "appears repaired; invert this test to require the declared range")
    assert z_far.max() <= 1.0 + 1e-6, (
        f"executed z max {z_far.max()} left the unit interval; the "
        "normalization assumption behind this pin has changed")


# --------------------------------------------------------------------------- #
# S20: an explicit ordering override must not disturb execution identity
# --------------------------------------------------------------------------- #

_ORDER_SPEC = """
O[k] := [direction = >; mechanism = AMPA; probability = 1.0; weight = 0.5];
A := [C = {E}; N = 2];
B := [C = {E}; N = 3];
V := A O[k] B;
x : V : y
"""


def test_order_override_keeps_realization_and_execution_aligned():
    """S20 must not break the invariant `to_configuration` depends on.

    Specs address neurons by realized id, so if an ordering override moved the
    realization axis without moving the constructed neuron table, every count
    would still match while the wrong neurons were wired.
    """
    spec = "order[V] := [B, A];\n" + _ORDER_SPEC
    program = parse(spec)
    explicit = resolve(program)
    r = realize(explicit, program)
    model = jaxfne.construct(
        to_configuration(r, duration_ms=DURATION_MS, dt_ms=DT_MS))

    # B first, against the natural order and against the composition order
    assert r.path_to_slice("V.B") == (0, 3)
    assert r.path_to_slice("V.A") == (3, 5)

    executed = [(row["area"], row["layer"]) for row in model.neuron_table()]
    realized = [(p.split(".")[0], p.split(".")[1]) for p in r.I["neuron_paths"]]
    assert executed == realized


def test_order_override_preserves_edge_identity():
    """The projection still runs A -> B after B is moved to the front.

    S20 reorders indexing; S10 binds projections by name. Reordering must
    therefore relabel the edges, not redirect them.
    """
    r, model, edges = equivalence("order[V] := [B, A];\n" + _ORDER_SPEC)

    a_lo, a_hi = r.path_to_slice("V.A")
    b_lo, b_hi = r.path_to_slice("V.B")
    assert (b_lo, b_hi) == (0, 3) and (a_lo, a_hi) == (3, 5)

    assert r.s["n_edges"] == 6                      # 2 x 3, unchanged
    for pre, post, _, _ in edges:
        assert a_lo <= pre < a_hi, f"pre {pre} is not in A {a_lo, a_hi}"
        assert b_lo <= post < b_hi, f"post {post} is not in B {b_lo, b_hi}"


def test_order_override_does_not_change_edge_count_or_weights():
    """Same nervous system either way round; only the labelling moves."""
    _, _, natural = equivalence(_ORDER_SPEC)
    _, _, declared = equivalence("order[V] := [B, A];\n" + _ORDER_SPEC)
    assert len(natural) == len(declared) == 6
    assert {w for _, _, w, _ in natural} == {w for _, _, w, _ in declared} == {0.5}


# --------------------------------------------------------------------------- #
# TFNE-PARAM-03: declared mechanism kinetics must reach execution
# --------------------------------------------------------------------------- #

def _executed_taus(spec):
    program = parse(spec)
    explicit = resolve(program)
    r = realize(explicit, program)
    model = jaxfne.construct(
        to_configuration(r, duration_ms=DURATION_MS, dt_ms=DT_MS))
    return {round(float(e["tau_ms"]), 6) for e in model.edge_table()}


def test_canonical_mechanism_kinetics_reach_execution():
    """Each canonical receptor executes at its canonical tau — the value
    hand-written callers copy into dT_ms, now resolved, not inherited."""
    from jaxfne.emitters import standard_receptor_specs
    table = standard_receptor_specs()
    for mech in ("AMPA", "GABA_A", "NMDA", "GABA_B"):
        spec = (f"O[k] := [direction = >; mechanism = {mech}; weight = 0.5];\n"
                f"{AB}\nx : A O[k] B : y\n")
        assert _executed_taus(spec) == {round(table[mech].tau_ms, 6)}, mech


def test_custom_mechanism_tau_reaches_execution():
    """A sufficient custom definition (name + finite positive tau) runs
    at the declared kinetics."""
    spec = ("O[k] := [direction = >; mechanism = FOO; tau_ms = 3.5;\n"
            "         weight = 0.5];\n"
            f"{AB}\nx : A O[k] B : y\n")
    assert _executed_taus(spec) == {3.5}


def test_asymmetric_mechanisms_keep_distinct_taus():
    """All mechanisms move together: AMPA and GABA_A execute at 2.0 and
    5.0 in one model — no uniform placeholder, no E/I distortion."""
    spec = ("O[k] := [direction = >; mechanism = AMPA; weight = 0.5];\n"
            "O[j] := [direction = >; mechanism = GABA_A; weight = 0.5];\n"
            f"{ABC}\nx : A O[k] B O[j] C : y\n")
    assert _executed_taus(spec) == {2.0, 5.0}


def test_tfne_matches_handwritten_canonical_construction():
    """A TFNE AMPA spec executes at exactly the tau the hand-written
    canonical practice constructs (dT_ms = AMPA_TAU_MS = 2.0)."""
    tfne_taus = _executed_taus(
        "O[k] := [direction = >; mechanism = AMPA; weight = 0.5];\n"
        f"{AB}\nx : A O[k] B : y\n")
    cfg = (jaxfne.Configuration().areas(["V"]).column("V", layers=["L"], n=4)
           .cell_types({"E": 1.0}).set_emitter("izhikevich", "cortical_eig")
           .probes(["spikes"]).field(domain="laminar_column",
                                      conductivity="proxy").runtime(seed=0)
           .mechanisms(name="ampa__dt2__0", kind="AMPA", params={"tau_ms": 2.0}))
    cfg = cfg.connections(
        name="ee", source={"area": "V", "layer": "L", "cell_type": "E"},
        target={"area": "V", "layer": "L", "cell_type": "E"},
        probability=1.0, weight=0.5, sign="excitatory",
        mechanism="ampa__dt2__0")
    hand = {round(float(e["tau_ms"]), 6)
            for e in jaxfne.construct(cfg).edge_table()}
    assert tfne_taus == hand == {2.0}


def test_unresolved_mechanism_refused_at_execution_not_realization():
    """Refusal lands where invention would: realize() still carries the
    identity (PARAM-01), but execution fails closed."""
    spec = ("O[k] := [direction = >; mechanism = GABA; weight = 0.5];\n"
            f"{AB}\nx : A O[k] B : y\n")
    program = parse(spec)
    explicit = resolve(program)
    r = realize(explicit, program)  # identity transfers; kinetics refused
    assert r.s["n_edges"] == 16
    with pytest.raises(TFNEError, match="E_MECHANISM_UNRESOLVED"):
        to_configuration(r, duration_ms=DURATION_MS, dt_ms=DT_MS)
