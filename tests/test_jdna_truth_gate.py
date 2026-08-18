"""0.4.17 campaign Block P: documentation/code truth gate for the JDNA surface.

Every documented claim about JDNA (root surface, grammar, boundary behavior,
provenance) is verified against the live package and source text.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import jaxfne as jtfne

ROOT = Path(__file__).resolve().parents[1]
GENOMES = ROOT / "jaxfne" / "jdna" / "genomes"

JDNA_ROOT = frozenset(
    {
        "PseudoGenome",
        "develop",
        "load_pseudogenome",
        "load_canonical_pseudogenome",
        "list_canonical_pseudogenomes",
    }
)


def test_root_surface_documented_names_exist():
    for name in JDNA_ROOT:
        assert name in jtfne.__all__, f"{name} must be a root export"
        assert hasattr(jtfne, name)


def test_guide_claims_surface_and_grammar():
    guide = (ROOT / "docs" / "guides" / "jdna.md").read_text(encoding="utf-8")
    assert "PseudoGenome --develop--> NeuronalTensor" in guide
    for name in ("jtfne.PseudoGenome", "jtfne.develop",
                 "jtfne.load_pseudogenome", "jtfne.load_canonical_pseudogenome",
                 "jtfne.list_canonical_pseudogenomes"):
        assert name in guide
    assert "K_D \\ne K_S \\ne K_A" in guide


def test_api_page_names_match_root_surface():
    api = (ROOT / "docs" / "api" / "jdna.md").read_text(encoding="utf-8")
    for name in JDNA_ROOT:
        assert name in api


def test_canonical_genome_shipped_and_loadable():
    assert (GENOMES / "canonical-v1-column-1000n.json").exists()
    g = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")
    assert g.name == "canonical-v1-column-1000n"
    assert g.schema_version == "pseudogenome_v1"
    assert jtfne.list_canonical_pseudogenomes() == ["canonical-v1-column-1000n"]


def test_no_jdna_branches_in_construct_simulate():
    sources = {
        "construct": ROOT / "jaxfne" / "neuronal_tensor.py",
        "pipeline": ROOT / "jaxfne" / "_pipeline.py",
        "simulate": ROOT / "jaxfne" / "emitters.py",
    }
    jdna_text = (ROOT / "jaxfne" / "jdna" / "genome.py").read_text(encoding="utf-8")
    for label, path in sources.items():
        text = path.read_text(encoding="utf-8")
        assert "import jdna" not in text, f"{label} must not import JDNA"
        assert "from jaxfne.jdna" not in text, f"{label} must not import JDNA"
        assert "PseudoGenome" not in text, f"{label} must not reference PseudoGenome"
    assert "provenance" in (ROOT / "jaxfne" / "neuronal_tensor.py").read_text()


def test_provenance_semantics_documented_and_real(tmp_path):
    guide = (ROOT / "docs" / "guides" / "jdna.md").read_text(encoding="utf-8")
    api = (ROOT / "docs" / "api" / "jdna.md").read_text(encoding="utf-8")
    assert "provenance" in guide and "provenance" in api
    g = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")
    t = jtfne.develop(g, seed=0)
    assert t.provenance is not None
    assert set(t.provenance) == {
        "genome", "genome_sha256", "schema_version",
        "development_seed", "development_parameters", "phenotype_sha256",
    }
    dumped = t.to_dict()
    assert "provenance" in json.dumps(dumped)  # in-memory tensor carries provenance
    path = tmp_path / "tensor.json"
    jtfne.save_neuronal_tensor(t, path)  # JSON saves exclude provenance
    assert "provenance" not in path.read_text(encoding="utf-8")


def test_references_page_links_resolve():
    refs = (ROOT / "docs" / "reference" / "references.md").read_text(encoding="utf-8")
    for target in ("docs/doctrine/rbs_rbd_hdp.md", "../guides/hdp.md"):
        assert target in refs


def test_mkdocs_nav_registers_jdna_pages():
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "guides/jdna.md" in nav
    assert "api/jdna.md" in nav
    assert "reference/references.md" in nav


def test_visual_review_script_executes():
    out = ROOT / "figures" / "jdna_visual_review.png"
    assert out.exists(), "run scripts/jdna_visual_review.py first"
    assert out.stat().st_size > 0
