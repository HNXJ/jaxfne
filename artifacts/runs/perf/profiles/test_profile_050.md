# Test/gate runtime profile + defect→gate matrix (0.5.0 item 4, draft)

Source: `artifacts/attestations/rc-pytest-broad.xml` (2026-09-20) +
measured gate runs on the release workstation (Windows, py3.14).
Local-environment receipt only.

## Gate wall times (measured)

| Gate | Wall time | Notes |
|---|---|---|
| dev (`run_test_gate.py dev`) | ~153 s | curated targets |
| broad (`-m "not slow"`) | ~2216 s (test-time 2252 s / 4160 tests) | dominates |
| slow | ~91 s | 7 tests |
| notebook | ~1384 s | 30 tests |
| audits (language/vocab/semantic/orphans/integrity) | ~10–20 s total | integrity gate ~8 s |
| mkdocs strict | ~15 s | |

## Slowest test files (broad)

| File | Time | Tests |
|---|---|---|
| test_equivalence_gate_v20260815 | 115 s / 7 | reruns generators per test |
| test_neuronal_tensor | 56 s / 33 | canonical JSON construct (32 s single test) |
| test_rep01_edge_class_storage | 52 s / 33 | 5k-threshold equivalence |
| test_connectivity_scaling | 50 s / 15 | scaling sweeps |
| test_closure_hp_reconciliation | 48 s / 21 | |
| test_protocol_w_w3a_stability | 45 s / 7 | |
| test_fig06_hwd_evidence | 44 s / 3 | generator rerun |
| test_jaxley_emitter_bridge_e2e | 41 s / 9 | controller stabilization |
| test_atlas_suite | 37 s / 29 | full atlas builds |

## Defect → cheapest detecting gate (draft)

| Historical defect | Cheapest gate | Cost |
|---|---|---|
| Version drift (P-005) | `test_docs_version_alignment.py` alone | seconds |
| Signature drift (P-006) | `audit_doc_code_integrity.py --check` | ~8 s |
| Tune silent drops (P-007) | `test_tune_mixed_args_warn.py` | ~5 s |
| TFNE refusal regressions (PARAM class) | dev gate (TFNE targets included) | ~153 s |
| Numerical drift | equivalence-gate track probes | 57 s each (see below) |
| Docs prose/vocab/orphans | audit battery | seconds |
| Release identity | gate0 + authority tests | seconds |

## Recommendations (proposals, not actions)

1. Narrow the equivalence gate to representative probes: the two 57 s
   tests rerun full generators; a cheapest-detector already exists per
   defect class — keep one full track, move the rest to release-only.
2. Keep adversarial TFNE/JDNA semantic tests in dev (cheap, high-value).
3. Cache immutable generated evidence where scientifically valid instead
   of regenerating per test (fig06 pattern: 44 s).
4. Parallelize independent test families (xdist already a dev dep —
   measure before/after).
5. Every historical defect keeps its cheap local detector (table above);
   broad stays the backstop, not the first line.
