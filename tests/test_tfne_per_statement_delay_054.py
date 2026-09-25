"""0.5.4 item 1c — per-statement delay in S12 rule bodies.

Each inter-area projection in a rule body declares its own delay
(`[delay=MS]`, alone or with `[mech=...]` in either order), overriding
the rule default for that statement's projections only. Delays are ms at
the grammar level and realize to steps at construction (0.5.2 rule).
"""

import numpy as np
import pytest

import jaxfne
from jaxfne.tfne import (
    TFNEError,
    _emit_rulestmt,
    normalize,
    parse,
    realize,
    resolve,
    to_configuration,
    to_neuronal_tensor,
)

DT_MS = 0.5

PROG = """
O[ff] := [mechanism = AMPA; weight = 0.5; delay = 1.0; {L4}>{L4} [delay=2.0]; {L4}>L5 [mech=AMPA, delay=4.0]; {L5}>L4 [delay=3.0, mech=AMPA]];
L4 := [C = {E}; N = 2];
L5 := [C = {E}; N = 1];
V1 := L4 O L5;
V2 := L4 O L5;
V := V1 O[ff] V2;
x : V : y
"""


def _stmts():
    return parse(PROG).rules["ff"].body


def test_statement_delay_parses():
    d0, d1, d2 = _stmts()
    assert d0.delay == 2.0 and d0.mechanism is None
    assert (d1.mechanism, d1.delay) == ("AMPA", 4.0)
    assert (d2.mechanism, d2.delay) == ("AMPA", 3.0)


def test_statement_delay_refusals():
    for bad in (
        "[delay=-1.0]",
        "[delay=abc]",
        "[delay=2.0, delay=3.0]",
        "[mech=AMPA, mech=GABA_A]",
        "[speed=2.0]",
    ):
        with pytest.raises(TFNEError):
            parse(
                "O[r] := [mechanism = AMPA; $L>$R " + bad + "];\n"
                "L := [C = {E}; N = 1];\n"
                "A := L; B := L;\nV := A O[r] B;\nx : V : y\n"
            )


def test_statement_delay_overrides_rule_default():
    explicit = resolve(parse(PROG))
    delays = {(r.pre_label, r.post_label): r.delay for r in explicit.relations}
    assert delays[("{L4}", "{L4}")] == 2.0
    assert delays[("{L4}", "L5")] == 4.0
    assert delays[("{L5}", "L4")] == 3.0
    # a silent statement keeps the rule default (None = default at this stage):
    prog2 = PROG.replace("{L5}>L4 [delay=3.0, mech=AMPA]", "{L5}>L4 [mech=AMPA]")
    explicit2 = resolve(parse(prog2))
    delays2 = {(r.pre_label, r.post_label): r.delay for r in explicit2.relations}
    assert delays2[("{L5}", "L4")] is None


def test_statement_delay_reaches_tensor_and_steps():
    program = parse(PROG)
    explicit = resolve(program)
    realized = realize(explicit, program, seed=0)
    tensor = to_neuronal_tensor(explicit)
    area_delays = sorted(
        {c.delay_ms for c in tensor.area_connections}
        | {c.delay_ms for a in tensor.areas for c in a.inter_connections}
    )
    assert area_delays == [2.0, 3.0, 4.0]
    model = jaxfne.construct(
        to_configuration(realized, duration_ms=5.0, dt_ms=DT_MS).runtime(
            recurrent_backend="edge_list"
        )
    )
    steps = sorted(set(np.asarray(model.params["edge_list"].delay_steps).tolist()))
    assert steps == [4, 6, 8]


def test_meta_statement_delay_reaches_tensor():
    prog = """
O[m] := [mechanism = AMPA; weight = 0.5; $L.out>$R.in [delay=5.0]];
out[UP] := [L4];
in[DN] := [L5];
L4 := [C = {E}; N = 2];
L5 := [C = {E}; N = 1];
UP := L4 O L5;
DN := L4 O L5;
V := UP O[m] DN;
x : V : y
"""
    explicit = resolve(parse(prog))
    tensor = to_neuronal_tensor(explicit)
    delays = {c.delay_ms for c in tensor.area_connections} | {
        c.delay_ms for a in tensor.areas for c in a.inter_connections
    }
    assert delays == {5.0}


def test_statement_emit_roundtrip_stable():
    first = normalize(parse(PROG))
    assert normalize(parse(first)) == first
    assert _emit_rulestmt(_stmts()[1]) == "{L4} > L5 [mech=AMPA, delay=4.0]"
