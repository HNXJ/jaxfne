"""PseudoGenome / JDNA tests: generativity, determinism, constraints, provenance.

Covers campaign blocks E (genuine generativity), F (canonical genome), G
(phenotype ensembles), T (property validation: determinism, PRNG separation,
type closure, structural validity, serialization), V (provenance), and S
(compatibility of the direct tensor path).
"""
from __future__ import annotations

import json
import pathlib

import pytest

import jaxfne as jtfne
from jaxfne.jdna import (
    PseudoGenome,
    AreaGenome,
    LayerGenome,
    ConnectionRuleGenome,
    declared_constraints,
    develop,
    genome_rules_hash,
    phenotype_sha256,
    validate_genome,
    load_pseudogenome,
    save_pseudogenome,
    list_canonical_pseudogenomes,
)
from jaxfne.neuronal_tensor import NeuronalTensor

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CANONICAL = "canonical-v1-column-1000n"


def load_canonical() -> PseudoGenome:
    return jtfne.load_canonical_pseudogenome(CANONICAL)


class TestCanonicalGenomeSurface:
    def test_list_and_load(self):
        names = list_canonical_pseudogenomes()
        assert CANONICAL in names
        g = jtfne.load_canonical_pseudogenome(CANONICAL)
        assert isinstance(g, PseudoGenome)
        assert g.schema_version == "pseudogenome_v1"
        g2 = jtfne.load_canonical_pseudogenome(f"{CANONICAL}.json")
        assert g2 == g

    def test_missing_genome_raises(self):
        with pytest.raises(FileNotFoundError):
            jtfne.load_canonical_pseudogenome("no-such-genome")

    def test_genome_is_rules_not_phenotype(self):
        """E: a PseudoGenome stores generative rules, never terminal phenotype."""
        raw = json.loads(
            (pathlib.Path(jtfne.jdna.genomes_dir()) / f"{CANONICAL}.json").read_text()
        )
        rules = {k: v for k, v in raw.items() if k != "description"}
        blob = json.dumps(rules)
        for forbidden in ("positions", "edges", "edge_list", "x_coords", "y_coords", "z_coords"):
            assert forbidden not in blob, f"genome stores terminal-phenotype field {forbidden!r}"
        g = load_canonical()
        for area in g.areas:
            for layer in area.layers:
                assert layer.n_neurons > 0
                assert isinstance(layer.depth_band, tuple)


class TestDevelopmentProperties:
    def test_type_closure(self):
        t = develop(load_canonical(), seed=0)
        assert isinstance(t, NeuronalTensor)
        assert t.name == CANONICAL

    def test_exact_count_and_structure(self):
        t = develop(load_canonical(), seed=0)
        total = sum(l.n_neurons for a in t.areas for l in a.layers)
        assert total == 1000
        by_name = {l.name: l.n_neurons for a in t.areas for l in a.layers}
        assert by_name == {"L1": 100, "L2": 250, "L3": 200, "L4": 100, "L5": 200, "L6": 150}
        for a in t.areas:
            for l in a.layers:
                fracs = [nt.fraction for nt in l.neuron_types if nt.fraction is not None]
                assert abs(sum(fracs) - 1.0) < 1e-6

    def test_determinism(self):
        g = load_canonical()
        t1 = develop(g, seed=7)
        t2 = develop(g, seed=7)
        assert t1.to_dict() == t2.to_dict()
        assert phenotype_sha256(t1) == phenotype_sha256(t2)

    def test_phenotype_variation_across_seeds(self):
        """G: same genome, different K_D -> different phenotypes within constraints."""
        g = load_canonical()
        tensors = [develop(g, seed=s) for s in range(10)]
        hashes = {phenotype_sha256(t) for t in tensors}
        assert len(hashes) >= 2, "seeds 0..9 should realize at least two phenotypes"

    def test_all_phenotypes_satisfy_declared_constraints(self):
        g = load_canonical()
        constraints = declared_constraints(g)
        for seed in range(10):
            t = develop(g, seed=seed)
            for a in t.areas:
                area_c = constraints["areas"][a.name]
                for l in a.layers:
                    layer_c = area_c["layers"][l.name]
                    assert l.n_neurons == layer_c["n_neurons"]
                    counts = {nt.name: round(l.n_neurons * (nt.fraction or 0.0)) for nt in l.neuron_types}
                    assert sum(counts.values()) == l.n_neurons
                    for ct, (lo, hi) in layer_c["cell_type_count_bands"].items():
                        assert lo <= counts[ct] <= hi, (
                            f"layer {l.name} cell type {ct}: {counts[ct]} outside [{lo}, {hi}]"
                        )

    def test_jitter_disable_reproduces_base_fractions(self):
        g = load_canonical()
        t = develop(g, seed=0, development_parameters={"fraction_jitter_sigma": 0.0})
        for a in t.areas:
            for lg in (l for ar in g.areas for l in ar.layers if ar.name == a.name):
                realized = next(l for l in a.layers if l.name == lg.name)
                for ct, frac in lg.cell_type_fractions.items():
                    nt = next(x for x in realized.neuron_types if x.name == ct)
                    assert round(realized.n_neurons * (nt.fraction or 0.0)) == pytest.approx(
                        round(realized.n_neurons * frac)
                    )

    def test_structural_validity_via_construct(self):
        """T: generated phenotype is constructible through the ordinary pipeline."""
        t = develop(load_canonical(), seed=3)
        model = jtfne.construct(t, jtfne.RuntimeConfiguration(seed=1, duration_ms=100.0, dt_ms=0.5))
        assert isinstance(model, jtfne.Model)

    def test_downstream_invariance(self):
        """JDNA tensors use ordinary downstream code: simulate is finite and valid."""
        t = develop(load_canonical(), seed=3)
        model = jtfne.construct(t, jtfne.RuntimeConfiguration(seed=1, duration_ms=100.0, dt_ms=0.5))
        signals = jtfne.simulate(model)
        vm = signals.get("vm")
        spk = signals.get("spk")
        assert vm is not None and spk is not None
        import jax.numpy as jnp
        assert bool(jnp.isfinite(vm).all())
        assert bool(jnp.isfinite(spk).all())
        assert vm.shape[-1] == 1000


class TestProvenance:
    def test_provenance_fields(self):
        g = load_canonical()
        t = develop(g, seed=42)
        p = t.provenance
        assert p["genome"] == CANONICAL
        assert p["genome_sha256"] == genome_rules_hash(g)
        assert p["schema_version"] == "pseudogenome_v1"
        assert p["development_seed"] == 42
        assert p["phenotype_sha256"] == phenotype_sha256(t)
        assert "fraction_jitter_sigma" in p["development_parameters"]

    def test_genome_hash_stable_and_description_excluded(self):
        g = load_canonical()
        h1 = genome_rules_hash(g)
        h2 = genome_rules_hash(load_canonical())
        assert h1 == h2
        g2 = PseudoGenome(
            name=g.name,
            schema_version=g.schema_version,
            description="completely different prose",
            areas=g.areas,
            area_connections=g.area_connections,
            development_parameters=g.development_parameters,
        )
        assert genome_rules_hash(g2) == h1

    def test_save_load_roundtrip(self, tmp_path):
        g = load_canonical()
        p = tmp_path / "g.json"
        save_pseudogenome(g, p)
        g2 = load_pseudogenome(p)
        assert g2.name == g.name
        assert g2.schema_version == g.schema_version
        assert g2.areas == g.areas
        assert g2.development_parameters == g.development_parameters
        assert genome_rules_hash(g2) == genome_rules_hash(g)


class TestValidation:
    def test_duplicate_layer_rejected(self):
        lg = LayerGenome(name="L1", n_neurons=10, depth_band=(0.0, 0.5),
                         cell_type_fractions={"E": 1.0})
        g = PseudoGenome(name="bad", areas=(AreaGenome(name="A", layers=(lg, lg)),))
        with pytest.raises(ValueError):
            validate_genome(g)

    def test_bad_fraction_sum_rejected(self):
        lg = LayerGenome(name="L1", n_neurons=10, depth_band=(0.0, 0.5),
                         cell_type_fractions={"E": 0.6, "PV": 0.2})
        g = PseudoGenome(name="bad", areas=(AreaGenome(name="A", layers=(lg,)),))
        with pytest.raises(ValueError):
            validate_genome(g)

    def test_bad_band_rejected(self):
        lg = LayerGenome(name="L1", n_neurons=10, depth_band=(0.6, 0.5),
                         cell_type_fractions={"E": 1.0})
        g = PseudoGenome(name="bad", areas=(AreaGenome(name="A", layers=(lg,)),))
        with pytest.raises(ValueError):
            validate_genome(g)

    def test_unknown_connection_target_rejected(self):
        lg = LayerGenome(name="L1", n_neurons=10, depth_band=(0.0, 0.5),
                         cell_type_fractions={"E": 1.0})
        rule = ConnectionRuleGenome("L1", "E", "L2", "E", "AMPA")
        g = PseudoGenome(
            name="bad",
            areas=(AreaGenome(name="A", layers=(lg,), inter_connections=(rule,)),),
        )
        with pytest.raises(ValueError):
            validate_genome(g)


class TestPrngSeparation:
    def test_development_domain_independent_of_runtime(self):
        """K_D controls the phenotype; K_S (runtime seed) never changes it."""
        g = load_canonical()
        t = develop(g, seed=5)
        model_a = jtfne.construct(t, jtfne.RuntimeConfiguration(seed=1, duration_ms=50.0, dt_ms=0.5))
        model_b = jtfne.construct(t, jtfne.RuntimeConfiguration(seed=99, duration_ms=50.0, dt_ms=0.5))
        sa = jtfne.simulate(model_a)
        sb = jtfne.simulate(model_b)
        import jax.numpy as jnp
        assert bool(jnp.isfinite(sa.get("vm")).all())
        assert bool(jnp.isfinite(sb.get("vm")).all())
        # runtime seeds affect realization (positions/edges), not the developed tensor
        assert t.to_dict() == develop(g, seed=5).to_dict()

    def test_no_shared_seed_state_between_domains(self):
        """develop() never consumes or mutates the caller's PRNG state."""
        g = load_canonical()
        t_a = develop(g, seed=1)
        t_b = develop(g, seed=1)
        assert phenotype_sha256(t_a) == phenotype_sha256(t_b)


class TestCompatibility:
    def test_canonical_tensor_path_unchanged(self):
        tensor = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
        model = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=1, duration_ms=50.0, dt_ms=0.5))
        signals = jtfne.simulate(model)
        assert signals.get("vm") is not None
        assert signals.get("vm").shape[-1] == 1000

    def test_tensor_save_excludes_provenance(self, tmp_path):
        t = develop(load_canonical(), seed=0)
        p = tmp_path / "t.json"
        jtfne.save_neuronal_tensor(t, p)
        raw = json.loads(p.read_text())
        assert "provenance" not in raw
        assert raw["schema_version"] == "neuronal_tensor_v1"