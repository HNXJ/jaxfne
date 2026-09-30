"""Executable JDNA scenarios J0-J5 (campaign block K) and full-flow integration.

    J0: PseudoGenome -> NeuronalTensor
    J1: G -> NeuronalTensor -> Model -> Signals
    J2: G -> N -> M -> X -> S -> Q -> Phi -> Y   (full TFNE observation chain)
    J3: JDNA + declared RBS relationship
    J4: phenotype ensemble (G, K_D 1:n) -> {N_i}
    J5: G -> N -> M -> Y -> Objective -> AGSDR -> Theta_R'
"""
from __future__ import annotations

import math
import pathlib

import jax.numpy as jnp
import pytest

import jaxfne as jtfne
from jaxfne.jdna import develop, load_canonical_pseudogenome, pseudogenome_from_dict
from jaxfne.neuronal_tensor import NeuronalTensor

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _model(seed_d=0, seed_s=1, duration_ms=200.0):
    genome = load_canonical_pseudogenome("canonical-v1-column-1000n")
    tensor = develop(genome, seed=seed_d)
    model = jtfne.construct(
        tensor,
        jtfne.RuntimeConfiguration(seed=seed_s, duration_ms=duration_ms, dt_ms=0.5),
    )
    return tensor, model


class TestJ0:
    def test_genome_to_phenotype(self):
        genome = load_canonical_pseudogenome("canonical-v1-column-1000n")
        tensor = develop(genome, seed=0)
        assert isinstance(tensor, NeuronalTensor)
        assert tensor.name == "canonical-v1-column-1000n"
        assert tensor.provenance["development_seed"] == 0


class TestJ1:
    def test_development_to_simulation(self):
        tensor, model = _model()
        signals = jtfne.simulate(model)
        vm = signals.get("vm")
        assert vm is not None
        assert vm.shape[-1] == 1000
        assert bool((vm != vm).sum() == 0)  # no NaNs


class TestJ2:
    def test_full_observation_chain(self):
        """G -> N -> M -> X -> S -> Q -> Phi -> Y using the ordinary operators."""
        tensor, model = _model(duration_ms=200.0)
        signals = jtfne.simulate(model)
        vm = signals.get("vm")  # X: raw membrane trajectories
        assert vm is not None
        positions = model.params["positions"]  # declared geometry realization
        source, meta = jtfne.construct_source_tensor(total_membrane_current=vm)  # S -> Q
        assert meta["mode"] == "total_membrane_current_proxy"
        field = jtfne.project_laminar_sources(source, positions, n_contacts=16)  # F -> Phi
        assert field is not None
        phi = field.phi_e_proxy
        assert phi.shape[0] == vm.shape[0]  # time axis preserved
        assert phi.shape[1] == 16  # n_contacts

    def test_analysis_distinct_from_objective(self):
        """Analysis (rate metric) is not an optimization loss; both usable."""
        _, model = _model(duration_ms=200.0)
        signals = jtfne.simulate(model)
        spk = signals.get("spk")
        rate = float(spk.sum()) / 200.0
        assert rate >= 0.0
        # The objective path is separate machinery:
        obj = jtfne.rate_targets(groups={"all": range(1000)}, targets_hz={"all": 5.0})
        report = model.evaluate(signals, objective=obj)
        assert report is not None


class TestJ3:
    def test_declared_rbs_relationship_documented(self):
        """JDNA + RBS: development may use RBS-like coordinates; the canonical
        genome declares none, and no developmental state crosses into runtime."""
        guide = (REPO_ROOT / "docs/guides/jdna.md").read_text(encoding="utf-8")
        assert "H_D" in guide
        assert "developmental state" in guide
        assert "No developmental state crosses" in guide
        genome = load_canonical_pseudogenome("canonical-v1-column-1000n")
        tensor = develop(genome, seed=0)
        # No H-like developmental state leaks into the phenotype:
        blob = str(tensor.to_dict())
        assert "hdp_initial_H" not in blob
        assert '"H"' not in blob
        # Runtime RBS (h_state) arises from the ordinary construct path:
        _, model = _model()
        assert "hdp_initial_H" in model.params


class TestJ4:
    def test_ensemble(self):
        genome = load_canonical_pseudogenome("canonical-v1-column-1000n")
        tensors = [develop(genome, seed=s) for s in range(4)]
        hashes = {t.provenance["phenotype_sha256"] for t in tensors}
        assert len(hashes) >= 2
        for t in tensors:
            total = sum(l.n_neurons for a in t.areas for l in a.layers)
            assert total == 1000


class TestJ5:
    def test_agsdr_optimization_of_runtime_parameter(self):
        """G -> N -> M -> Y -> Objective -> AGSDR -> Theta_R' (runtime params only)."""
        genome = load_canonical_pseudogenome("canonical-v1-column-1000n")
        tensor = develop(genome, seed=0)
        model = jtfne.construct(
            tensor,
            jtfne.RuntimeConfiguration(seed=1, duration_ms=200.0, dt_ms=0.5),
        )
        objective = jtfne.rate_synchrony_targets()
        optimizer = jtfne.agsdr(
            parameters={"drive_gain": (0.5, 2.0)},
            generations=3,
            population_size=4,
            seed=42,
        )
        result = model.tune(objectives=objective, optimizer=optimizer)
        assert result.best_score is not None
        assert result.best_parameters
        assert "drive_gain" in result.best_parameters
        # AGSDR optimizes runtime/model parameters; no genome claim is made:
        assert "genome" not in result.best_parameters


class TestFullFlow:
    def test_hard_boundary(self):
        """After develop() returns, only ordinary jaxfne execution is used."""
        tensor, model = _model()
        signals = jtfne.simulate(model)
        assert isinstance(signals, jtfne.Signals)
        assert model.__class__ is jtfne.Model

    def test_canonical_usage_snippet(self):
        """Block F target usage verbatim."""
        genome = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")
        tensor = jtfne.develop(genome, seed=0)
        model = jtfne.construct(
            tensor,
            jtfne.RuntimeConfiguration(seed=1, duration_ms=1000.0, dt_ms=0.5),
        )
        signals = jtfne.simulate(model)
        assert signals.get("vm").shape[-1] == 1000


def _tiny_deterministic_genome():
    """Smallest deterministic genome: one area, one layer, E/PV, zero jitter."""
    return pseudogenome_from_dict({
        "name": "g-tiny",
        "areas": [{
            "name": "V1",
            "layers": [{
                "name": "L4", "n_neurons": 20, "depth_band": [0.0, 1.0],
                "cell_type_fractions": {"E": 0.5, "PV": 0.5},
            }],
            "inter_connections": [{
                "source_layer": "L4", "source_neuron_type": "E",
                "target_layer": "L4", "target_neuron_type": "PV",
                "mechanism": "GABA_A",
            }],
        }],
        "development_parameters": {"fraction_jitter_sigma": 0.0},
    })


def test_jdna_developed_tensor_matches_hand_spelled_configuration():
    """JDNA entrance lowers like the tensor entrance: `develop(G, seed=0)`
    followed by the ordinary `construct -> simulate` path must give the same
    Signals as the same circuit spelled by hand as a Configuration.

    Entrance A goes through `jtfne.construct(develop(genome, seed=0),
    runtime)` (which lowers via the bridge internally). Entrance B below
    spells the SAME values literally without calling `develop`,
    `neuronal_tensor_to_configuration`, or anything that calls them.

    Literal provenance (read from `neuronal_tensor_to_configuration` +
    `_wire_connection` + `_connection_edge_weight`, not executed): the genome
    develops to V1 = 20 neurons in L4 with E/PV = 0.5/0.5 (exact, sigma 0, so
    the bridge's declared-fraction path records 0.5/0.5 per area/layer and
    globally); one InterConnection E -> PV with mechanism GABA_A, whose
    StaticParams default (dT_ms=0.1, no g_mech, no reversal) wires mechanism
    "GABA_A__dt0.1__0" (kind + `:g`-formatted dT_ms + dedup index 0) with
    params tau_ms=0.1, rule "interconn_V1_0", full-bipartite probability 1.0,
    weight |1.0 * 1.0| / sqrt(20), sign from the E source -> excitatory, no
    delay (InterConnection.delay_ms defaults to None); connectivity_mode
    "explicit" (passing inter_connections=[...] marks the area explicit, so
    the compiled graph holds only the declared rule). Omitted as inert:
    `tensor_identity` (a provenance hash nothing in construct/simulate
    reads), the H overlay (stored but inert with HDP disabled), and declared
    geometry (positions differ by entrance and move only the field proxies,
    so field is not compared here).

    D=200 ms spikes (38 spikes), so the comparison is not vacuous."""
    genome = _tiny_deterministic_genome()
    assert develop(genome, seed=0).to_dict() == develop(genome, seed=0).to_dict()

    D = 200.0
    tensor = develop(genome, seed=0)
    model_a = jtfne.construct(
        tensor,
        jtfne.RuntimeConfiguration(seed=0, duration_ms=D, dt_ms=0.5),
    )
    sig_a = jtfne.simulate(model_a, duration_ms=D, dt_ms=0.5, seed=0)

    cfg_b = (
        jtfne.Configuration()
        .runtime(seed=0, duration_ms=D, dt_ms=0.5, dtype="float32")
        .update_metadata(connectivity_mode="explicit")
        .population(20, neurons={"L4": 20}, name="V1", layers=["L4"])
        .area_layer_cell_types("V1", {"L4": {"E": 0.5, "PV": 0.5}})
        .cell_types({"E": 0.5, "PV": 0.5})
        .mechanisms(
            name="GABA_A__dt0.1__0", kind="GABA_A",
            params={"tau_ms": 0.1},
        )
        .connections(
            name="interconn_V1_0",
            source={"area": "V1", "layer": "L4", "cell_type": "E"},
            target={"area": "V1", "layer": "L4", "cell_type": "PV"},
            probability=1.0,
            weight=1.0 / math.sqrt(20),
            sign="excitatory",
            mechanism="GABA_A__dt0.1__0",
        )
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m"], n_contacts=16)
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    )
    model_b = jtfne.construct(cfg_b)
    assert model_b.cfg.metadata.get("recurrent_backend") == "edge_list"
    sig_b = jtfne.simulate(model_b, duration_ms=D, dt_ms=0.5, seed=0)
    assert int(sig_a.spikes.sum()) > 0, "fixture must spike for a non-vacuous comparison"
    assert bool(jnp.array_equal(sig_a.spikes, sig_b.spikes))
    assert bool(jnp.array_equal(sig_a.V_m, sig_b.V_m))
    assert bool(jnp.array_equal(sig_a.sources, sig_b.sources))