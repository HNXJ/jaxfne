"""JDNA developmental completion: defaults table, K_D sampling, origins.

Every test targets the TFNE->JDNA boundary doctrine
(`docs/doctrine/tfne_jdna_boundary.md`, source 7 S29.1): TFNE constrains,
JDNA completes under D + K_D, every value records its origin. The TFNE
bridge itself is never touched here — completion is the JDNA-side path.
"""

import jax.numpy as jnp
import pytest

from jaxfne.jdna.completion import (
    ORIGIN_DECLARED,
    ORIGIN_DERIVED,
    ORIGIN_DEFAULT,
    ORIGIN_SAMPLED,
    complete_tfne,
    realize_geometry,
    resolve,
)
from jaxfne.tfne import parse, realize, resolve as tfne_resolve


# --------------------------------------------------------------------------- #
# resolve(): the five arms of the defaults table
# --------------------------------------------------------------------------- #

def test_declared_value_always_wins():
    assert resolve("geometry_distribution", "uniform") == (
        "uniform", ORIGIN_DECLARED)


def test_canonical_default_applies_with_origin():
    value, origin = resolve("geometry_distribution")
    assert value == "uniform" and origin == ORIGIN_DEFAULT
    value, origin = resolve("geometry_domain")
    assert value == ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    assert origin == ORIGIN_DEFAULT


def test_derivation_is_deterministic_and_marked():
    """cell_allocation derives from N/P only: no default, no sampling."""
    value, origin = resolve("cell_allocation", derive=lambda: {"E": 2})
    assert value == {"E": 2} and origin == ORIGIN_DERIVED
    with pytest.raises(ValueError, match="must be derived"):
        resolve("cell_allocation")


def test_required_quantity_is_refused_not_invented():
    """mechanism_tau_ms has no canonical default: kinetics stay rejected."""
    with pytest.raises(ValueError, match="required but has no canonical"):
        resolve("mechanism_tau_ms")
    with pytest.raises(ValueError, match="required but has no canonical"):
        resolve("mechanism_tau_ms", derive=lambda: 2.0)


def test_unknown_quantity_is_refused_even_when_declared():
    """A misspelled quantity must not pass as completed while ignored."""
    with pytest.raises(ValueError, match="unknown completable quantity"):
        resolve("geomtry_domain", ((0.0, 1.0),) * 3)
    with pytest.raises(ValueError, match="unknown completable quantity"):
        resolve("nope")


# --------------------------------------------------------------------------- #
# realize_geometry(): declared obeyed, omitted defaulted, partial refused
# --------------------------------------------------------------------------- #

def test_declared_domain_is_obeyed():
    pos, origins = realize_geometry({"z0": 0.0, "z1": 5.0}, 32, seed=0)
    assert pos.shape == (32, 3) and pos.dtype == jnp.float32
    assert float(pos[:, 2].min()) >= 0.0 and float(pos[:, 2].max()) <= 5.0
    assert origins["z"] == ORIGIN_DECLARED
    assert origins["x"] == ORIGIN_DEFAULT  # unit fallback, recorded
    assert origins["positions"] == ORIGIN_SAMPLED


def test_omitted_geometry_falls_back_to_unit_cube():
    pos, origins = realize_geometry({}, 16, seed=0)
    assert pos.shape == (16, 3)
    assert float(pos.min()) >= 0.0 and float(pos.max()) <= 1.0
    assert origins["x"] == origins["y"] == origins["z"] == ORIGIN_DEFAULT


def test_geometry_is_deterministic_in_seed_and_varies_across_seeds():
    a, _ = realize_geometry({"z0": 0.0, "z1": 5.0}, 8, seed=0)
    b, _ = realize_geometry({"z0": 0.0, "z1": 5.0}, 8, seed=0)
    c, _ = realize_geometry({"z0": 0.0, "z1": 5.0}, 8, seed=1)
    assert bool((a == b).all())  # same K_D -> same completion
    assert bool((a != c).any())  # different K_D -> different phenotype


def test_empty_population_realizes_empty_positions():
    pos, origins = realize_geometry({"z0": 0.0, "z1": 5.0}, 0, seed=0)
    assert pos.shape == (0, 3)
    assert origins["positions"] == ORIGIN_SAMPLED


def test_partial_domain_and_bad_distribution_fail_closed():
    with pytest.raises(ValueError, match="partial z domain"):
        realize_geometry({"z0": 1.0}, 4, seed=0)
    with pytest.raises(ValueError, match="unsupported geometry distribution"):
        realize_geometry({"distribution": "cortical_sheet"}, 4, seed=0)
    with pytest.raises(ValueError, match="degenerate"):
        realize_geometry({"z0": 2.0, "z1": 2.0}, 4, seed=0)


# --------------------------------------------------------------------------- #
# complete_tfne(): per-leaf completion under K_D, bridge untouched
# --------------------------------------------------------------------------- #

_TFNE_GEO = """
A := [C = {E}; N = 2; G = [z0 = 0.0; z1 = 4.0]];
B := [C = {E}; N = 3];
V := A O B;
x : V : y
"""


def test_complete_tfne_covers_every_leaf_with_origins():
    prog = parse(_TFNE_GEO)
    r = realize(tfne_resolve(prog), prog, seed=0)
    before = dict(r.s["geometry"])
    out = complete_tfne(r, seed=7)
    assert out["domain"] == "K_D" and out["seed"] == 7
    assert sorted(out["positions"]) == ["V.A", "V.B"]
    assert out["positions"]["V.A"].shape == (2, 3)
    assert out["positions"]["V.B"].shape == (3, 3)
    assert out["origins"]["V.A"]["z"] == ORIGIN_DECLARED
    assert out["origins"]["V.B"]["z"] == ORIGIN_DEFAULT
    # The TFNE bridge is untouched: declarations pass through unchanged.
    assert r.s["geometry"] == before


def test_complete_tfne_is_deterministic_in_seed():
    prog = parse(_TFNE_GEO)
    r = realize(tfne_resolve(prog), prog, seed=0)
    first = complete_tfne(r, seed=7)["positions"]
    second = complete_tfne(r, seed=7)["positions"]
    for leaf in first:
        assert bool((first[leaf] == second[leaf]).all())


# --------------------------------------------------------------------------- #
# develop(): value origins ride in provenance, phenotype unchanged
# --------------------------------------------------------------------------- #

def test_develop_records_value_origins():
    import jaxfne as jtfne
    g = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")
    t = jtfne.develop(g, seed=0)
    origins = t.provenance["value_origins"]
    layer = origins["areas.V1.layers.L4"]
    assert layer["counts"] == ORIGIN_SAMPLED  # jitter active by default
    assert layer["geometry"]["z_range"] == ORIGIN_DECLARED
    assert set(layer["geometry"]) == {
        "distribution", "x_range", "y_range", "z_range"}
    assert origins["areas.V1.pose"]["plane"] == ORIGIN_DECLARED


def test_develop_origins_follow_declared_vs_defaulted():
    """A layer with no declared geometry gets JDNA-default on every field."""
    from dataclasses import replace
    import jaxfne as jtfne
    g = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")
    area = g.areas[0]
    bare = replace(area.layers[0], geometry={})
    layers = (bare,) + tuple(area.layers[1:])
    g2 = replace(g, areas=(replace(area, layers=layers),))
    params = dict(g2.development_parameters)
    params["fraction_jitter_sigma"] = 0.0
    g2 = replace(g2, development_parameters=params)
    t = jtfne.develop(g2, seed=0)
    layer = t.provenance["value_origins"][
        f"areas.{area.name}.layers.{bare.name}"]
    assert layer["counts"] == ORIGIN_DERIVED  # jitter disabled
    assert layer["geometry"]["distribution"] == ORIGIN_DEFAULT
    assert layer["geometry"]["x_range"] == ORIGIN_DEFAULT
    assert layer["geometry"]["z_range"] == ORIGIN_DECLARED  # depth band


# --------------------------------------------------------------------------- #
# TFNE -> JDNA preservation across declared frontiers (TFNE2-04)
# --------------------------------------------------------------------------- #

_FRONTIER_SPEC = """
O[k] := [direction = >; mechanism = AMPA; probability = 1.0; weight = 0.5];
O[j] := [direction = >; mechanism = AMPA; probability = 1.0; weight = 0.5];
A := [C = {E}; N = 1];
B := [C = {E}; N = 2];
P := [C = {E}; N = 1];
M := A O[k] B;
V := P O[j] M;
x : V : y
"""


def test_declared_frontier_reaches_jdna_completion_unchanged():
    """JDNA completes leaves, never reinterprets relations: a declared
    frontier changes edge identities upstream while per-leaf positions
    and origins under the same K_D are identical."""
    prog = parse(_FRONTIER_SPEC)
    derived = realize(tfne_resolve(prog), prog, seed=0)
    prog_d = parse("in[M] := [B];\n" + _FRONTIER_SPEC)
    declared = realize(tfne_resolve(prog_d), prog_d, seed=0)
    # The frontier had an effect upstream: different edge identities.
    assert derived.s["n_edges"] == 3
    assert declared.s["n_edges"] == 4
    assert (derived.I["rule_origins"]["r1:O[j]:P>M"]["post_scopes"]
            == ["V.M.A"])
    assert (declared.I["rule_origins"]["r1:O[j]:P>M"]["post_scopes"]
            == ["V.M.B"])
    # JDNA sees the same leaves, counts and declarations either way.
    out_derived = complete_tfne(derived, seed=7)
    out_declared = complete_tfne(declared, seed=7)
    assert sorted(out_derived["positions"]) == sorted(
        out_declared["positions"])
    for leaf in out_derived["positions"]:
        assert bool((out_derived["positions"][leaf]
                     == out_declared["positions"][leaf]).all())
        assert (out_derived["origins"][leaf]
                == out_declared["origins"][leaf])
