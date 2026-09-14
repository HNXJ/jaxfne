# 24-VERIFY-01 receipt — technical seal (SEAL_GO)

**Candidate:** source `7f89effccc7e4b2b6105d99033acce2cb12b2fbe`
(tree `4269c15816aaefb9f3e0887f39706ed7067a5626`).
Version identity 0.4.24 (candidate; published PyPI stays 0.4.23).

## Failure loop exercised once
First RC run: 3874 + 1 failure (`test_canonical_manifest_provenance` —
atlas stamp 0.4.23 vs runtime 0.4.24). Repaired via owning generator
(`scripts/generate_readme_atlas.py`; stamp + timestamps only,
config_hash `4b0d96456d56bc1a` pinned) → new candidate → full RC re-run.

## Gate matrix on exact SHA (single-process ledger)
- env parity, compileall, ruff, vocab, docs orphans/notebook grammar: PASS
- broad: 3875 passed, 75 skipped, 37 deselected, 4 xfailed
- slow: 7 passed; notebooks: 30 passed; mkdocs strict: built
- release examples: all ran green
- JUnit parity vs release CI run 34804574833 @ exact SHA: rc=3991 ci=3991,
  0 unjustified, 0 failures (py3.11 + py3.14)
- CI (Fast) on candidate: run 34801205249 success; release CI dispatched
  read-only (run 34804574833 success)
- build (hatchling 1.29.0, isolated) + twine: PASSED
- isolated venv smoke (site-packages, 0.4.24): PASS
- attestation: `artifacts/attestations/rc_gate_attestation.json`
  (untracked by design), status PASS, 16/16 families, subsumes CI
- Local RC hashes (informational; authoritative bytes come from CI
  release-dist at publish): wheel 605792
  `b0cbb134e8dc8ea098812c66df4ba10ebae3bfdc1a2dab97c0734090f7c67a56`,
  sdist 38051167
  `52a51c347194e133f2fcfe91dedf028bc6ce7fcef5b3785a87eedc42e5950eb2`;
  wheel 111 py, zero research leaks

## Independent audit (AUDIT-01 external half)
Two read-only critic sessions (PRO-01) + one declared-grammar HDP-GEN
challenge, all findings triaged in receipts with file:line evidence.
Same-model-family provenance recorded in PRO-01 receipt; a genuinely
external party remains the human's call at seal.

## Unresolved warnings (explained, none material)
- `working_tree_clean: False` in attestation = one pre-existing foreign
  untracked file (`artifacts/_temp_opencode_handoff.md`), never touched.
- float64→float32 UserWarnings (x64 off by design), TuneResult/legacy
  DeprecationWarnings (declared compat), nbformat cell-id warnings
  (upstream), recompilation-guard warnings (intentional cache tests).

## SEAL_GO (technical only)
All required gates PASS; zero unexplained failures; candidate exact;
tree synced. STOP before tag / main / GitHub / PyPI / RTD (not authorized).
Readiness: 100/100 technical.
