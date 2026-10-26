"""Property guards for definition-map rows bound to public symbols (lane L10).

Each test reads its row's Symbol from docs/theory/doctrine/definition_map.md and
checks the definition's property through that object. A wrong Symbol in the map
fails the matching test: the call or the assertion breaks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_definition_map import _parse_rows, _resolve_symbol  # noqa: E402

from jaxfne.tfne import TFNEFrontierUnresolved, flatten, parse  # noqa: E402

_ROWS = {r["Id"]: r for r in _parse_rows()}


def _mapped(row_id: str):
    return _resolve_symbol(_ROWS[row_id]["Symbol"])


_OUT = "O[k] := [direction = >; mechanism = AMPA; probability = 1.0; weight = 0.5];\n"


def test_d9_counts_come_from_the_mapped_expansion():
    """D9: N[A] is allocated by largest remainder with the sum equal to N."""
    resolve = _mapped("D9")
    counts = resolve(parse("Pop := [C = {E, PV}; P = {E: 0.8, PV: 0.2}; N = 10]; x : Pop : y")).nodes["Pop"].counts
    assert dict(counts) == {"E": 8, "PV": 2}
    tie = resolve(parse("Pop := [C = {E, PV}; P = {E: 0.5, PV: 0.5}; N = 3]; x : Pop : y")).nodes["Pop"].counts
    assert dict(tie) == {"E": 2, "PV": 1}  # declaration-order tie-break
    assert sum(tie.values()) == 3


def test_d15_frontier_members_are_validated_by_the_mapped_expansion():
    """D15: a declared in-frontier must name an immediate member of its scope."""
    resolve = _mapped("D15")
    defs = ("A := [C = {E}; N = 1];\nB := [C = {E}; N = 2];\nP := [C = {E}; N = 1];\n"
            "M := A O[k] B;\n")
    rules = _OUT + "O[j] := [direction = >; mechanism = AMPA; probability = 1.0; weight = 0.5];\n"
    explicit = resolve(parse(rules + defs + "in[M] := [B];\nV := P O[j] M;\nx : V : y\n"))
    assert any(rel.post_scopes == ("V.M.B",) for rel in explicit.relations)
    with pytest.raises(TFNEFrontierUnresolved, match="names 'Z', which is not a member of 'V.M'"):
        resolve(parse(rules + defs + "in[M] := [Z];\nV := P O[j] M;\nx : V : y\n"))


def test_d16_paths_come_out_in_natural_order():
    """D16: the canonical order sorts SEG.2 before SEG.10, not lexically."""
    resolve = _mapped("D16")
    explicit = resolve(parse("SEG := [C = {cell}; N = 1]; x : SEG^12 : y"))
    seg = [p for p in explicit.order if p.startswith("SEG.")]
    assert seg == [f"SEG.{i}" for i in range(1, 13)]


def test_d20_declared_geometry_is_a_fraction_of_the_block():
    """D20: a declared sub-range is a fraction in [0, 1] of its sampled block."""
    to_configuration = _mapped("D20")
    text = (_OUT + "A := [C = {E}; N = 60; model = izhikevich; G = [z0 = 0.2; z1 = 0.5]];\n"
            "x : A : y\n")
    cfg = to_configuration(flatten(text), duration_ms=5.0, dt_ms=0.1)
    domains = cfg.metadata["tfne_geometry"]["domains"]
    blocks = [block for layers in domains.values() for block in layers.values()]
    assert blocks, "fixture must declare a non-full geometry range"
    for block in blocks:
        for lo, hi in block.values():
            assert 0.0 <= lo < hi <= 1.0
    assert blocks == [{"z": [0.2, 0.5]}]


def test_d21_declared_weight_reaches_the_kernel_resolver():
    """D21: the kernel resolver returns the declared weight, signed by the presynaptic cell."""
    import jaxfne
    from jaxfne.tfne import realize, resolve, to_configuration

    resolve_weight = _mapped("D21")
    ab = "A := [C = {E}; N = 4]; B := [C = {E}; N = 4];"
    program = parse(f"O[k] := [direction = >; mechanism = AMPA; weight = 0.3125];\n{ab}\nx : A O[k] B : y\n")
    realization = realize(resolve(program), program)
    model = jaxfne.construct(to_configuration(realization, duration_ms=5.0, dt_ms=0.1))
    edges, emitter = model.params["edge_list"], model.params["emitter"]
    assert int(edges.n_edges) > 0, "fixture must have edges"
    w = np.asarray(resolve_weight(edges, edges.weight.dtype, presynaptic_sign=emitter.sign))
    assert np.allclose(np.sort(w), np.sort(np.asarray(realization.s["edge_weight"])), atol=1e-6)
    assert np.allclose(w, 0.3125, atol=1e-6)


def test_d34_gain_map_is_one_at_neutral_h_and_affine_otherwise():
    """D34: G_H(H; beta_H) = 1 + beta_H (H - 1); exactly 1 for family f0 or beta_H = 0."""
    import jax.numpy as jnp

    gain = _mapped("D34")
    H = jnp.asarray([0.8, 1.0, 1.2], dtype=jnp.float32)
    beta = jnp.asarray(0.5, dtype=jnp.float32)
    assert np.allclose(gain("f1", H, beta, jdtype=jnp.float32), [0.9, 1.0, 1.1], rtol=1e-6)
    assert np.array_equal(gain("f1", H, jnp.asarray(0.0, dtype=jnp.float32), jdtype=jnp.float32), np.ones(3))
    assert np.array_equal(gain("f0", H, beta, jdtype=jnp.float32), np.ones(3))
    assert np.isnan(np.asarray(gain("f1", jnp.asarray([-2.0], dtype=jnp.float32), beta, jdtype=jnp.float32))).all()


def test_d22_declared_delay_realizes_to_round_delay_over_dt():
    """D22: delay_steps = round(delay_ms / dt_ms); 0.3 ms is 3 steps at 0.1 ms, 6 at 0.05 ms."""
    to_configuration = _mapped("D22")
    ab = "A := [C = {E}; N = 4]; B := [C = {E}; N = 4];"
    realization = flatten(f"O[k] := [direction = >; mechanism = AMPA; delay = 0.3];\n{ab}\nx : A O[k] B : y\n")
    for dt_ms, expected in ((0.1, 3), (0.05, 6)):
        cfg = to_configuration(realization, duration_ms=5.0, dt_ms=dt_ms)
        steps = cfg.metadata["tfne_delay"]["realized_steps"]
        assert list(steps.values()) == [expected], steps


def test_d30_population_edge_weight_is_sign_times_theta0():
    """D30: w_eff = sign(w_baseline) * theta_0 on the E->I channel (magnitude-replacing)."""
    from test_hdp_population_restoring import _mcc3_model, _population_hdp_params

    kernel = _mapped("D30")
    model, mei_mask, e_mask = _mcc3_model()
    hp = _population_hdp_params(mei_mask, e_mask, r0_e=10.9, r0_i=9.14)
    hp["controller_theta_S_init"] = (2.0, 1.0)
    _, _, _, diag = kernel(
        params=model.params["emitter"], edges=model.params["edge_list"],
        n_steps=200, dt_ms=0.1, key=jax.random.PRNGKey(0), **hp,
    )
    mask = np.asarray(hp["m_ei_edge_mask"], dtype=bool)
    assert mask.any(), "fixture must have E->I edges"
    w_base = np.asarray(model.params["edge_list"].weight)[mask]
    w_final = np.asarray(diag["w_final"])[mask]
    theta0 = float(np.asarray(diag["theta_S_final"])[0])
    assert np.array_equal(np.sign(w_final), np.sign(w_base))
    assert np.allclose(np.abs(w_final), theta0, rtol=1e-6)
