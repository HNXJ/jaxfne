import json
import math
import jax
import jax.numpy as jnp
import pytest

import jaxfne as jtfne
from jaxfne.core import LaminarPopulation, LaminarSourceGeometry


def _two_pop_geometry():
    pops = [
        jtfne.LaminarPopulation(name="E_L4", cell_type="excitatory", layer="L4",
                                depth_min=0.4, depth_max=0.6, n_units=3),
        jtfne.LaminarPopulation(name="PV_L4", cell_type="pv", layer="L4",
                                depth_min=0.4, depth_max=0.6, n_units=2),
    ]
    return jtfne.laminar_source_geometry(pops)


def _cfg(n=5):
    return (
        jtfne.configuration()
        .network(n=n)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", n_contacts=16)
    )


# Test A — population construction and JSON-safe to_dict
def test_laminar_population_construction_and_json_safe():
    pop = jtfne.LaminarPopulation(
        name="E_L23", cell_type="excitatory", layer="L2/3",
        depth_min=0.1, depth_max=0.4, n_units=4,
    )
    assert pop.name == "E_L23"
    assert pop.n_units == 4
    assert pop.physical_amplitude_claim_allowed is False
    assert pop.claim_level == "computational_scaffold"

    d = pop.to_dict()
    assert d["physical_amplitude_claim_allowed"] is False
    assert d["source_calibration_status"] == "uncalibrated_izhikevich_native_current"
    json.dumps(d, allow_nan=False)


# Test B — invalid population validation
def test_laminar_population_validate_bad_depth_and_zero_units():
    bad_depth = LaminarPopulation(
        name="bad", cell_type="excitatory", layer="L4",
        depth_min=0.8, depth_max=0.2, n_units=2,
    )
    v = bad_depth.validate()
    assert not v["valid"]
    assert "depth_range_invalid" in v["issues"]

    zero_units = LaminarPopulation(
        name="empty", cell_type="excitatory", layer="L4",
        depth_min=0.1, depth_max=0.5, n_units=0,
    )
    v2 = zero_units.validate()
    assert not v2["valid"]
    assert "n_units_must_be_positive" in v2["issues"]

    # laminar_source_geometry raises on invalid pop
    with pytest.raises(ValueError, match="Invalid LaminarPopulation"):
        jtfne.laminar_source_geometry([bad_depth])


# Test C — source geometry construction and n_units_total
def test_laminar_source_geometry_construction_and_n_units_total():
    geom = _two_pop_geometry()
    assert isinstance(geom, LaminarSourceGeometry)
    assert geom.n_units_total == 5
    assert len(geom.populations) == 2
    assert geom.physical_amplitude_claim_allowed is False
    assert geom.claim_level == "computational_scaffold"

    d = geom.to_dict()
    assert d["n_units_total"] == 5
    assert d["physical_amplitude_claim_allowed"] is False
    json.dumps(d, allow_nan=False)


# Test D — positions_array shape, dtype, and deterministic depths
def test_positions_array_shape_dtype_deterministic():
    geom = _two_pop_geometry()
    arr = geom.positions_array(dtype="float32")

    assert arr.shape == (5, 3)
    assert arr.dtype == jnp.float32

    # x and y columns must be zero (proxy geometry)
    assert jnp.all(arr[:, 0] == 0.0)
    assert jnp.all(arr[:, 1] == 0.0)

    # first pop (E_L4, 3 units): linspace(0.4, 0.6, 3)
    expected_z_e = jnp.array([0.4, 0.5, 0.6], dtype=jnp.float32)
    assert jnp.allclose(arr[:3, 2], expected_z_e, atol=1e-6)

    # second pop (PV_L4, 2 units): linspace(0.4, 0.6, 2)
    expected_z_pv = jnp.array([0.4, 0.6], dtype=jnp.float32)
    assert jnp.allclose(arr[3:, 2], expected_z_pv, atol=1e-6)

    # determinism: same output on second call
    arr2 = geom.positions_array(dtype="float32")
    assert jnp.array_equal(arr, arr2)


# Test E — construct with geometry succeeds and simulate runs
def test_construct_with_geometry_and_simulate():
    geom = _two_pop_geometry()
    cfg = _cfg(n=5)
    model = jtfne.construct(cfg, geometry=geom)

    assert "geometry" in model.static
    assert model.static["geometry"]["n_units_total"] == 5

    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0)
    signals = model.simulate(sim)
    assert signals.V_m.shape == (100, 5)


# Test F — construct without geometry is backward compatible
def test_construct_without_geometry_backward_compatible():
    cfg = _cfg(n=5)
    model = jtfne.construct(cfg)
    assert "geometry" not in model.static

    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0)
    signals = model.simulate(sim)
    assert signals.V_m.shape == (100, 5)


# Test G — manifest includes geometry only when provided
def test_manifest_includes_geometry_only_when_provided():
    cfg = _cfg(n=5)
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0)

    # with geometry
    geom = _two_pop_geometry()
    model_with = jtfne.construct(cfg, geometry=geom)
    sig_with = model_with.simulate(sim)
    manifest_with = model_with.manifest(sig_with)
    assert "source_geometry" in manifest_with
    assert manifest_with["source_geometry"]["n_units_total"] == 5

    # without geometry
    model_without = jtfne.construct(cfg)
    sig_without = model_without.simulate(sim)
    manifest_without = model_without.manifest(sig_without)
    assert "source_geometry" not in manifest_without

    # manifests are JSON safe
    json.dumps(manifest_with, allow_nan=False)
    json.dumps(manifest_without, allow_nan=False)


# Test H — truth gates unchanged with geometry
def test_truth_gates_unchanged_with_geometry():
    geom = _two_pop_geometry()
    cfg = _cfg(n=5)
    model = jtfne.construct(cfg, geometry=geom)
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1)
    signals = model.simulate(sim)

    manifest = model.manifest(signals)
    assert manifest["truth_mode"] == "truth_safe_unverified"
    assert manifest["claim_level"] == "computational_scaffold"
    labels = manifest.get("v005_claim_labels", {})
    assert labels.get("physical_amplitude_claim_allowed", False) is False
    assert labels.get("empirical_validation_status", "not_empirically_validated") == "not_empirically_validated"
    assert labels.get("mechanism_claim_status", "not_claimed") == "not_claimed"

    # geometry itself preserves truth gates
    geom_dict = geom.to_dict()
    assert geom_dict["physical_amplitude_claim_allowed"] is False
    assert geom_dict["source_calibration_status"] == "uncalibrated_izhikevich_native_current"
    assert geom_dict["claim_level"] == "computational_scaffold"


# Test I — geometry_n_units_total_mismatch raises ValueError
def test_geometry_n_units_total_mismatch_raises():
    # geometry has 5 units, cfg has n=8 -> mismatch
    geom = _two_pop_geometry()  # n_units_total = 5
    cfg = _cfg(n=8)

    with pytest.raises(ValueError, match="geometry_n_units_total_mismatch"):
        jtfne.construct(cfg, geometry=geom)
