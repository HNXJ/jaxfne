"""JDNA — JAX Developmental Neural Architecture (pseudo-genomic generation).

JDNA is the theory/framework governing pseudo-genomic generation. The primary
object is a :class:`PseudoGenome`: a finite, model-defined generative
specification whose coordinates and rules determine the development of a
neuronal phenotype. The genome analogy is computational — its components need
not correspond to literal genes, DNA sequences, chromosomes, or molecular
genomic mechanisms.

Canonical grammar:

    PseudoGenome --develop--> NeuronalTensor --construct--> Model --simulate--> Signals

A PseudoGenome describes *how* a phenotype is generated; it does not store the
terminal phenotype. Development is a function

    develop(G, K_D) -> NeuronalTensor

with independent PRNG domain K_D (development seed). Construction and
simulation then use the ordinary jaxfne pipeline with their own PRNG domains
(runtime seed K_S, optimizer seed K_A). JDNA is optional: the existing
Configuration / NeuronalTensor direct paths remain first-class.
"""
from __future__ import annotations

from .genome import (
    PSEUDOGENOME_SCHEMA_VERSION,
    PseudoGenome,
    AreaGenome,
    LayerGenome,
    ConnectionRuleGenome,
    develop,
    genome_rules_hash,
    phenotype_sha256,
    declared_constraints,
    validate_genome,
    load_pseudogenome,
    load_canonical_pseudogenome,
    list_canonical_pseudogenomes,
    save_pseudogenome,
    pseudogenome_from_dict,
    genomes_dir,
)

__all__ = [
    "PSEUDOGENOME_SCHEMA_VERSION",
    "PseudoGenome",
    "AreaGenome",
    "LayerGenome",
    "ConnectionRuleGenome",
    "develop",
    "genome_rules_hash",
    "phenotype_sha256",
    "declared_constraints",
    "validate_genome",
    "load_pseudogenome",
    "load_canonical_pseudogenome",
    "list_canonical_pseudogenomes",
    "save_pseudogenome",
    "pseudogenome_from_dict",
    "genomes_dir",
]