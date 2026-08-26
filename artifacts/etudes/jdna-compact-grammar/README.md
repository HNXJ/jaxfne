# JDNA Compact Grammar — Etude (F2)

Private etude implementing the compact grammar from `jdna-compact-grammar-preregistration.md`.

- **Location:** `artifacts/etudes/jdna-compact-grammar/` (not `jaxfne/jdna/`)
- **No public-surface change:** `jaxfne/jdna/*.py` untouched.
- **Desugar:** `define`/`inherit`/`use` → `PseudoGenome` → `validate_genome` → `develop(G, K_D)` → `NeuronalTensor`
- **Provenance:** `genome_rules_hash` (existing, no new hash)
- **K_D determinism:** `develop(G, K_D)` PRNG split per area/layer (existing implementation)
- **Denominator semantics:** `A-80` absolute, `A*0.08` fractional with `N_enclosing = N_area(G0)` preserving mode (§5.3), largest-remainder integer rounding
- **Composition:** `use ... compose with ...` → `merge_neuronal_tensors` at tensor level (§7)
- **H11:** tests generate outputs only in `tmp_path`; no reliance on gitignored artifacts

Parser: `parser.py` (Lexer + Parser + desugar)
Tests: `tests/test_jdna_compact_grammar.py` (isolated `tmp_path`)

Run:
```
pytest tests/test_jdna_compact_grammar.py -v
```
