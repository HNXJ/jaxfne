# Seal E2 PRGS — Harness + Memory (2026-08-25) — Deep consolidation H1–H5

HEAD: f9be6ae (parents 0b8c22e→555ad78→f9be6ae, E1 07b1c04, frozen 28/28 aeca6d0)
Authority packet: ping b89a09c466186330a58eb70c632d597a7989803f6e418a2d9d778385a498af1f ssa 6ca10a960fc408819618f2f0b4032be900d00c7420620409766066955a1781b0 sha 0E9D602B/D4DEBFD8 code_head f9be6ae parent 07b1c04
DESIGN_READY: V1=(1,1,1,1) V2=(1,1,1,1) E2a=E2b=0

## Harness lessons promoted (durable) — H1–H5 synthesis

- H1 External review is hypothesis generation, not authority — preserved via lineage, not mutation solely from finding.
- H2 Hard-gate claims require receipts — READY/PASS/100% require exact declared gate completed on state being sealed; feature works ≠ release sealed.
- H3 Reconcile arithmetic before Seal — test counts, file counts, hashes mechanically reconciled (28/28, 7/7, 189 exports).
- H4 Serialization is epistemic boundary — IN_MEMORY_ONLY/PRESERVED/PRESERVED_ELSEWHERE/PARTIALLY_LOST/LOST tested separately; persistence not inferred.
- H5 Adversarial validation must include counterexamples — validators test invalid/boundary, not only happy paths.

Synthesis: Parallel pattern `P -> {W_i}|| -> {R_j}|| -> G_integrate -> R_final -> S` with bounded packets; reviewers comparable only under identical cryptographic authority packets (ping/ssa spec_hash + sha256 + code_head + parent_e1); stale packet reviews excluded. Every finding gets lineage REAL_BLOCKER/RESOLVED/STALE_REVIEW/HISTORICAL_REJECTED/UNRESOLVED; stronger reproducible evidence wins. Executable values have one owner (JSON owns constants/classifiers, prose owns rationale). Ranking artifact > reproduced observation > inference > assertion; correlated reviewers not independent. One writer default. Registries go stale silently — prefer discovery over recorded observation; manifest refreshed via `scripts/harness/sync_skills.py --update --manifest` and verified via `check_harness_integrity`.

## Repository minimization PRGS (this seal) — f9be6ae

- C_root: experiments/jdna_v0_drive_experiment.py -> artifacts/etudes/jdna_v0_drive_experiment.py (0 consumers, provenance_relocation_jdna.json sha e762c1f5 verified, delta 0), empty skills dirs pruned (13), experiments dir removed.
- C_harness: .opencode/skills + .cursor/skills mirrors regenerated (14 files), .opencode/HARNESS_MANIFEST.json project_config hash fixed ad5ce6a (was db710cc stale), integrity PASS; stale 36_V1_PING_REAUDIT.* archived to artifacts/developer/stale_reviews/ (private, gitignored).
- C_prose/C_duplicate/C_terminology: 420 lines duplication across 11 blocks single-owner map (doctrine/*.md, operator_doctrine.md, source_field_equations.md, protocol_h_rbd_memory.md, etc.); final RG cycle link-out applied docs/guides/hdp.md (27→3 lines) + docs/computation_basis.md (22→1 line) with qualifier preservation (coordinates H_k=z_k/z*, RBD enumeration, LFP proxy retained), remaining 9 blocks low-value deferred per stopping rule duplicate=0 material, not word count. No high-value root move blocked.
- E2a blinding: artifacts/e2/preregistration/E2A_BLINDING_SPEC.json v1 — 17 forbidden (G_spec, SI, ΔPLV, S0-S4, D-N1) vs 14 allowed (G_adequate, mean_rate), technical grep 0 hits required, f_select blinded.
- Delta science/API/E1/E2 semantics =0 (verified: no jaxfne/*, 28/28 frozen intact, E2 spec hashes b89/6ca invariant post-minimization with fs2000 and mask disambiguation).
- S = S_repo + S_harness + S_memory: repo minimized, harness refreshed, memory consolidated, blinding enforced.

## Memory consolidation (aggressive) — H1–H5 synthesis

Durable (project memory): scientific grammar E->S->F->P->O->A->M, execution grammar CircuitSpec->construct->Model->simulate->Signals, RBS container doctrine, relative quantity grammar B/R/E/T, source/field/probe tiers, containment vs composition, proxy truth gate, delay vs wave, attenuation vs adaptation, bounded vs stable, H1–H5 discipline + harness lessons above.

Ephemeral (handoff only, not project memory): current HEAD f9be6ae, parent 35291f8→0b8c22e→555ad78, E1 07b1c04, spec hashes b89a09c/6ca10a96, sha 0E9D602B/D4DEBFD8, counts 189 public exports, resolved E2 blockers (PSD, delay grid, PV, mean_degree, epsilon, mask h_rule/h_bytes/h_impl, sampling fs2000), 13 empty skill dirs, 2 stale reviews archived, etc. These remain in handoff receipts, not in doctrine.

## Replay validation (final minimized state f9be6ae) — synthesis

Historical suite rerun on f9be6ae with JAX_PLATFORMS=cpu, dev==origin/dev clean (2 stale files archived to private, not in active surface):
- F02/F03/NO_WAVE: tests/test_protocol_c_c1_synthetic.py + c3_c4 42 passed 1 skipped
- C_head: tests/test_public_api_snapshot_v034.py 2 passed (189 exports) + tests/test_equivalence_gate_v20260815.py 3 passed 1 skipped 7/7 decoded_pixel_equal
- NOT_FOUND: tests/test_closure_hp_reconciliation.py / test_config_json_roundtrip PASS (via hygiene)
- JSON/prose: tests/test_agent_context_hygiene.py 20 passed, docs/publication manuscript terminology PASS, check_agent_refs governed PASS, check_harness_integrity integrity PASS (project+global, frozen 28/28)
- E2 deterministic preflight: h_rule 51105b03 h_bytes 83af8971 h_impl 027bdd97 fs2000 spec_hash b89/6ca invariant, provenance_relocation_jdna.json sha e762c1f5 verified, path relocation not silent redefinition

All six must-reproduce decisions correct on final minimized state.

## Next gate

Only after minimized repo sealed, synchronized (dev==origin/dev), replay PASS, and E2 preflight PASS on new HEAD:

E2a --adequacy-only--> theta* --write-once freeze--> E2b

JDNA deferred: `define/inherit, A-80, A*0.08` compiler concept survives as future design item, not on E2 path. Current E2 audit packet and DESIGN_READY receipts relocated with provenance, not rewritten.
