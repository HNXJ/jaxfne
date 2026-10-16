"""JDNA compact grammar etude package (private, not shipped)."""
from .parser import parse_text, parse_file, desugar_file, _apply_tweaks_to_genome, _merge_genomes, develop_with_provenance, merge_tensors_after_develop

__all__ = ["parse_text", "parse_file", "desugar_file", "_apply_tweaks_to_genome", "_merge_genomes", "develop_with_provenance", "merge_tensors_after_develop"]
