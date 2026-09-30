"""P-023: time constants are required, never defaulted; sign by mechanism name."""

import math

import jax.numpy as jnp
import pytest

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt
from jaxfne.jdna import develop, pseudogenome_from_dict
from jaxfne.tfne import parse, resolve, to_neuronal_tensor


def _layer(n=4):
    return nt.Layer(
        name="L4",
        n_neurons=n,
        neuron_types=[nt.NeuronType.make("E"), nt.NeuronType.make("PV")],
    )


def _tensor(source, target, mechanism, static):
    return nt.NeuronalTensor(
        areas=[
            nt.Area(
                name="V1",
                layers=[_layer()],
                inter_connections=[
                    nt.InterConnection(
                        "L4", source, "L4", target, mechanism, static=static
                    )
                ],
            )
        ]
    )


def _construct(tensor):
    return jtfne.construct(
        tensor, jtfne.RuntimeConfiguration(seed=0, duration_ms=2.0, dt_ms=0.5)
    )


def test_missing_tau_is_refused():
    """A connection with undeclared StaticParams.dT_ms is refused (P-023)."""
    tensor = _tensor("E", "PV", "AMPA", nt.StaticParams())
    assert tensor.areas[0].inter_connections[0].static.dT_ms is None
    with pytest.raises(ValueError, match="P-023"):
        _construct(tensor)


def test_e_source_with_inhibitory_mechanism_is_refused():
    """E -> PV GABA_A is refused even with a declared tau."""
    tensor = _tensor("E", "PV", "GABA_A", nt.StaticParams(dT_ms=5.0))
    with pytest.raises(ValueError, match="P-023"):
        _construct(tensor)


def test_non_e_source_with_excitatory_mechanism_is_refused():
    """PV -> E AMPA is refused even with a declared tau."""
    tensor = _tensor("PV", "E", "AMPA", nt.StaticParams(dT_ms=2.0))
    with pytest.raises(ValueError, match="P-023"):
        _construct(tensor)


def test_unknown_mechanism_name_is_refused():
    """A mechanism outside {AMPA, NMDA, GABA_A, GABA_B} is refused."""
    tensor = _tensor("E", "PV", "FOO", nt.StaticParams(dT_ms=3.0))
    with pytest.raises(ValueError, match="P-023"):
        _construct(tensor)


def test_genome_mechanism_without_tau_entry_is_refused():
    """develop() refuses a connection whose mechanism has no table entry."""
    genome = pseudogenome_from_dict({
        "name": "g-no-tau",
        "areas": [{
            "name": "V1",
            "layers": [{
                "name": "L4", "n_neurons": 8, "depth_band": [0.0, 1.0],
                "cell_type_fractions": {"E": 0.5, "PV": 0.5},
            }],
            "inter_connections": [{
                "source_layer": "L4", "source_neuron_type": "E",
                "target_layer": "L4", "target_neuron_type": "PV",
                "mechanism": "AMPA",
            }],
        }],
        "development_parameters": {"fraction_jitter_sigma": 0.0},
    })
    with pytest.raises(ValueError, match="P-023"):
        develop(genome, seed=0)


def test_monotonic_cable_synapse_still_builds():
    """The AreaConnection default stays exempt: declared tau + source sign."""
    layers = [
        nt.Layer(
            name="L4", n_neurons=2,
            neuron_types=[nt.NeuronType.make("E", fraction=1.0)],
        )
    ]
    tensor = nt.NeuronalTensor(
        areas=[nt.Area(name="V1", layers=layers), nt.Area(name="V4", layers=layers)],
        area_connections=[
            nt.AreaConnection(
                "V1", "L4", "E", "V4", "L4", "E",
                static=nt.StaticParams(dT_ms=3.0),
            )
        ],
    )
    assert tensor.area_connections[0].mechanism == "monotonic_cable_synapse"
    model = _construct(tensor)
    edges = model.edge_table()
    assert {round(float(e["tau_ms"]), 6) for e in edges} == {3.0}
    assert all(float(e["weight"]) > 0 for e in edges)  # E source -> excitatory


def test_canonical_genome_develops_constructs_simulates_with_declared_taus():
    """The canonical genome's table reaches the realized mechanisms exactly."""
    genome = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")
    table = dict(genome.mechanism_tau_ms)
    assert table == {"AMPA": 2.0, "GABA_A": 5.0}
    tensor = develop(genome, seed=0)
    for area in tensor.areas:
        for conn in area.inter_connections:
            assert conn.static.dT_ms == table[conn.mechanism]
    model = jtfne.construct(
        tensor, jtfne.RuntimeConfiguration(seed=0, duration_ms=20.0, dt_ms=0.5)
    )
    assert {round(float(e["tau_ms"]), 6) for e in model.edge_table()} == {2.0, 5.0}
    signals = jtfne.simulate(model, duration_ms=20.0, dt_ms=0.5, seed=0)
    assert bool(jnp.all(jnp.isfinite(signals.spikes)))


@pytest.mark.parametrize(
    "bad_tau",
    [0, 0.0, -2.0, -0.5, math.nan, math.inf, -math.inf, True, False, "2.0", "fast"],
    ids=["zero-int", "zero", "negative", "negative-frac", "nan", "inf", "neg-inf",
         "bool-true", "bool-false", "numeric-string", "non-numeric-string"],
)
def test_inadmissible_tau_is_refused(bad_tau):
    """Only a finite number > 0 wires: 0/negative/NaN/inf/bool/string are refused (P-023).

    Bools and numeric strings are refused explicitly (no silent coercion),
    matching the genome-side mechanism_tau_ms checks."""
    tensor = _tensor("E", "PV", "AMPA", nt.StaticParams(dT_ms=bad_tau))
    with pytest.raises(ValueError, match="P-023"):
        _construct(tensor)


def test_forged_tfne_provenance_does_not_qualify_for_carve_out():
    """A hand-built tensor with provenance={'tfne_digest': ...} is still refused.

    Provenance is a free constructor kwarg, so it cannot authorize the
    TFNE carve-out; only tensors minted by jaxfne.tfne.to_neuronal_tensor
    qualify. E -> PV GABA_A would pass only under the carve-out."""
    tensor = _tensor("E", "PV", "GABA_A", nt.StaticParams(dT_ms=5.0))
    tensor.provenance = {"tfne_digest": "forged"}
    assert tensor.provenance == {"tfne_digest": "forged"}
    assert not nt._is_tfne_minted(tensor)
    with pytest.raises(ValueError, match="P-023"):
        _construct(tensor)


def test_genuine_tfne_tensor_passes_with_custom_mechanism_and_polarity_sign():
    """A tensor minted by to_neuronal_tensor skips the declaration vocabulary.

    The program declares a custom mechanism (FOO, tau_ms 3.5 — unknown to the
    {AMPA, NMDA, GABA_A, GABA_B} vocabulary) on a PV -> E projection (a
    polarity-signed pair the declaration vocabulary would refuse as a
    non-E source with an excitatory-named mechanism). Both ride
    neuronal_tensor_to_configuration on TFNE's own kinetics + weight
    polarity (owner ruling 2026-09-30)."""
    program = parse(
        "O[k] := [direction = >; mechanism = FOO; tau_ms = 3.5; weight = 0.5];\n"
        "A := [C = {PV}; N = 4]; B := [C = {E}; N = 4];\n"
        "x : A O[k] B : y\n"
    )
    tensor = to_neuronal_tensor(resolve(program))
    assert nt._is_tfne_minted(tensor)
    mechs = {c.mechanism for c in tensor.area_connections}
    assert mechs == {"FOO"}
    pair = tensor.area_connections[0]
    assert (pair.source_neuron_type, pair.target_neuron_type) == ("PV", "E")
    cfg = nt.neuronal_tensor_to_configuration(
        tensor, seed=0, duration_ms=2.0, dt_ms=0.5
    )
    kinds = {m["kind"] for m in cfg.metadata["circuit"]["mechanisms"]}
    assert kinds == {"FOO"}
    taus = {m["params"]["tau_ms"] for m in cfg.metadata["circuit"]["mechanisms"]}
    assert taus == {3.5}


def test_reloaded_tfne_tensor_loses_carve_out_status(tmp_path):
    """Save strips provenance and load builds a new object: no mint status.

    A genuine TFNE tensor, round-tripped through save/load, faces the
    declaration vocabulary like any hand-built tensor."""
    program = parse(
        "O[k] := [direction = >; mechanism = AMPA; weight = 0.5];\n"
        "A := [C = {E}; N = 4]; B := [C = {E}; N = 4];\n"
        "x : A O[k] B : y\n"
    )
    tensor = to_neuronal_tensor(resolve(program))
    assert nt._is_tfne_minted(tensor)
    path = nt.save_neuronal_tensor(tensor, tmp_path / "tfne.json")
    reloaded = nt.load(path)
    assert not nt._is_tfne_minted(reloaded)
