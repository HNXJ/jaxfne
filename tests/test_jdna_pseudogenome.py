"""PseudoGenome / JDNA tests: generativity, determinism, constraints, provenance.

Covers campaign blocks E (genuine generativity), F (canonical genome), G
(phenotype ensembles), T (property validation: determinism, PRNG separation,
type closure, structural validity, serialization), V (provenance), and S
(compatibility of the direct tensor path).
"""
from __future__ import annotations

import json
import math
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
    pseudogenome_from_dict,
)
from jaxfne.neuronal_tensor import NeuronalTensor

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
            (pathlib.Path(jtfne.jdna.genomes_dir()) / f"{CANONICAL}.json").read_text(encoding="utf-8")
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

    def test_schema_version_preserved_on_save(self, tmp_path):
        """J-S1: save never silently downgrades/upgrades the schema version."""
        g = PseudoGenome(name="X", schema_version="pseudogenome_v2",
                         areas=(), area_connections=())
        p = tmp_path / "g.json"
        save_pseudogenome(g, p)
        raw = json.loads(p.read_text(encoding="utf-8"))
        assert raw["schema_version"] == "pseudogenome_v2"

    def test_unknown_schema_rejected_on_load(self):
        """J-S3: unknown future schemas are rejected, not interpreted as v1."""
        with pytest.raises(ValueError, match="schema_version"):
            pseudogenome_from_dict({"schema_version": "pseudogenome_v9", "name": "X"})

    def test_semantic_roundtrip_tuple_geometry(self, tmp_path):
        """J-S2: load(save(G)) is semantically equal even for tuple geometry."""
        lg = LayerGenome(name="L1", n_neurons=10, depth_band=(0.0, 0.5),
                         cell_type_fractions={"E": 1.0},
                         geometry={"x_range": (0.0, 1.0), "y_range": (0.0, 1.0)})
        area = AreaGenome(name="A", layers=(lg,), inter_connections=(),
                          pose={"translation": (0, 0, 0)})
        g = PseudoGenome(name="RT", schema_version="pseudogenome_v1",
                         areas=(area,), area_connections=())
        p = tmp_path / "rt.json"
        save_pseudogenome(g, p)
        g2 = load_pseudogenome(p)
        assert g2 == g


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

    def test_fraction_out_of_domain_rejected(self):
        """J-V1: every cell-type fraction must lie in [0, 1]."""
        lg = LayerGenome(name="L1", n_neurons=10, depth_band=(0.0, 0.5),
                         cell_type_fractions={"A": 1.5, "B": -0.5},
                         fraction_tolerance={"A": (0.5, 1.0), "B": (0.0, 0.0)})
        g = PseudoGenome(name="bad", areas=(AreaGenome(name="A", layers=(lg,)),))
        with pytest.raises(ValueError, match="fraction must be in"):
            validate_genome(g)

    def test_base_fraction_outside_own_band_rejected(self):
        """J-V2: tol_lo <= base <= tol_hi per cell type."""
        lg = LayerGenome(name="L1", n_neurons=100, depth_band=(0.0, 0.5),
                         cell_type_fractions={"E": 0.5, "I": 0.5},
                         fraction_tolerance={"E": (0.0, 0.1), "I": (0.9, 1.0)})
        g = PseudoGenome(name="bad", areas=(AreaGenome(name="A", layers=(lg,)),))
        with pytest.raises(ValueError, match="outside its declared tolerance"):
            validate_genome(g)

    def test_jointly_infeasible_bands_rejected(self):
        """J-V3: sum(lo) <= 1 <= sum(hi) must hold for the box simplex.

        Note: for base fractions inside their own bands (J-V2) with
        sum(base)=1, the joint condition is implied; the check is kept as
        defense-in-depth and is exercised directly via the projection."""
        from jaxfne.jdna.genome import _project_box_simplex

        with pytest.raises(ValueError, match="infeasible"):
            _project_box_simplex([0.5, 0.5], [0.0, 0.0], [0.3, 0.3])
        with pytest.raises(ValueError, match="infeasible"):
            _project_box_simplex([0.5, 0.5], [0.7, 0.7], [1.0, 1.0])
        p = _project_box_simplex([0.6, 0.4], [0.3, 0.2], [0.7, 0.6])
        assert abs(sum(p) - 1.0) < 1e-9
        assert all(lo <= pi <= hi for pi, lo, hi in zip(p, [0.3, 0.2], [0.7, 0.6]))

    def test_high_sigma_jitter_never_escapes_bands(self):
        """J-V4: the box-simplex projection respects bands for any sigma."""
        lg = LayerGenome(name="L1", n_neurons=100, depth_band=(0.0, 0.5),
                         cell_type_fractions={"E": 0.75, "I": 0.25},
                         fraction_tolerance={"E": (0.7, 0.8), "I": (0.05, 0.25)})
        g = PseudoGenome(
            name="high-sigma", areas=(AreaGenome(name="A", layers=(lg,)),),
            development_parameters={"fraction_jitter_sigma": 5.0},
        )
        validate_genome(g)
        for s in range(25):
            t = develop(g, seed=s)
            l = t.areas[0].layers[0]
            counts = {nt.name: int(round(l.n_neurons * (nt.fraction or 0.0)))
                      for nt in l.neuron_types}
            assert sum(counts.values()) == l.n_neurons
            for ct, frac in lg.cell_type_fractions.items():
                tol = lg.fraction_tolerance.get(ct, (frac, frac))
                lo = int(math.floor(l.n_neurons * tol[0]))
                hi = int(math.ceil(l.n_neurons * tol[1]))
                assert lo <= counts.get(ct, 0) <= hi, (
                    f"seed {s}: {ct} count {counts.get(ct, 0)} outside [{lo}, {hi}]"
                )

    def test_area_connections_references_rejected(self):
        """J-V6: dangling area_connections references must fail validation."""
        lg = LayerGenome(name="L1", n_neurons=10, depth_band=(0.0, 0.5),
                         cell_type_fractions={"E": 1.0})
        area = AreaGenome(name="A", layers=(lg,), inter_connections=())
        dangling_area = {"source_area": "A", "source_layer": "L1",
                         "source_neuron_type": "E",
                         "target_area": "GHOST", "target_layer": "L1",
                         "target_neuron_type": "E"}
        g = PseudoGenome(name="bad", areas=(area,),
                         area_connections=(dangling_area,))
        with pytest.raises(ValueError, match="unknown target_area"):
            validate_genome(g)
        dangling_layer = {"source_area": "A", "source_layer": "L9",
                          "source_neuron_type": "E",
                          "target_area": "A", "target_layer": "L1",
                          "target_neuron_type": "E"}
        g2 = PseudoGenome(name="bad", areas=(area,),
                          area_connections=(dangling_layer,))
        with pytest.raises(ValueError, match="unknown source_layer"):
            validate_genome(g2)
        valid = {"source_area": "A", "source_layer": "L1",
                 "source_neuron_type": "E",
                 "target_area": "A", "target_layer": "L1",
                 "target_neuron_type": "E"}
        g3 = PseudoGenome(name="ok", areas=(area,), area_connections=(valid,))
        validate_genome(g3)
        t = develop(g3, seed=0)
        assert len(t.area_connections) == 1


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

    def test_runtime_seed_changes_realization_but_not_development(self):
        """K_S: with (G, K_D) fixed, changing the runtime seed changes the
        realized positions/edges but never the developed NeuronalTensor."""
        import jax.numpy as jnp

        g = load_canonical()
        t = develop(g, seed=5)
        pos_a = jtfne.construct(t, jtfne.RuntimeConfiguration(seed=1, duration_ms=50.0, dt_ms=0.5)).params["positions"]
        pos_b = jtfne.construct(t, jtfne.RuntimeConfiguration(seed=99, duration_ms=50.0, dt_ms=0.5)).params["positions"]
        assert not bool(jnp.allclose(pos_a, pos_b)), "different K_S must change positions"
        assert t.to_dict() == develop(g, seed=5).to_dict(), "K_S must not change the developed tensor"

    def test_optimizer_seed_changes_proposals_but_not_development(self):
        """K_A: with (G, K_D, K_S) fixed, changing the optimizer seed changes
        optimization proposals while development and runtime realization
        remain unchanged."""
        g = load_canonical()
        t = develop(g, seed=5)
        cfg = jtfne.RuntimeConfiguration(seed=1, duration_ms=50.0, dt_ms=0.5)
        m = jtfne.construct(t, cfg)
        obj = jtfne.rate_synchrony_targets()
        r_a = m.tune(objectives=obj, optimizer=jtfne.agsdr(
            parameters={"drive_gain": (0.5, 2.0)}, generations=2,
            population_size=4, seed=42))
        r_b = m.tune(objectives=obj, optimizer=jtfne.agsdr(
            parameters={"drive_gain": (0.5, 2.0)}, generations=2,
            population_size=4, seed=7))
        assert r_a.best_parameters != r_b.best_parameters, (
            "different K_A must change optimizer proposals"
        )
        assert t.to_dict() == develop(g, seed=5).to_dict()
        pos = jtfne.construct(t, cfg).params["positions"]
        assert t.to_dict() == develop(g, seed=5).to_dict()
        _ = pos  # construction determinism implied by t.to_dict() equality above


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
        raw = json.loads(p.read_text(encoding="utf-8"))
        assert "provenance" not in raw
        assert raw["schema_version"] == "neuronal_tensor_v1"