# Seal E2 PRGS — Harness + Memory (2026-08-25)

HEAD: 0b8c22e (parent 35291f8, E1 07b1c04, frozen 28/28 aeca6d0)
Authority packet: ping b89a09c... ssa 6ca10a96... sha 0E9D602B.../D4DEBFD8... code_head 0b8c22e parent 07b1c04
DESIGN_READY: V1=(1,1,1,1) V2=(1,1,1,1) E2a=E2b=0

## Harness lessons promoted (durable)

- Parallel pattern: `P -> {W_i}|| -> {R_j}|| -> G_integrate -> R_final -> S` with bounded packets.
- Reviewers comparable only under identical cryptographic authority packets (ping/ssa spec_hash + sha256 + code_head + parent_e1); stale packet reviews excluded, not integrated.
- Every finding gets lineage: REAL_BLOCKER / RESOLVED / STALE_REVIEW / HISTORICAL_REJECTED / UNRESOLVED; stronger reproducible evidence wins, not majority.
- Executable values have one owner: JSON owns constants/classifiers/thresholds/windows/delays/spec_hash, prose owns rationale; repeated values are assertions checked against JSON.
- Ranking: artifact > reproduced observation > inference > assertion; correlated reviewers do not count as independent merely because they agree.
- One writer remains default (N_writers=1); multi-writer requires explicit justification and merge receipt.
- Registries go stale silently — prefer discovery (glob/grep/live hash) over recorded observation; manifest mirror/harness hashes refreshed via `scripts/harness/sync_skills.py --update --manifest` and verified via `check_harness_integrity`.

## Repository minimization PRGS (this seal)

- C_root: experiments/jdna_v0_drive_experiment.py -> artifacts/etudes/jdna_v0_drive_experiment.py (0 consumers, provenance_relocation_jdna.json sha e762c1f5 verified, delta 0), empty skills dirs pruned (13), experiments dir removed.
- C_harness: .opencode/skills + .cursor/skills mirrors regenerated (14 files), .opencode/HARNESS_MANIFEST.json project_config hash fixed ad5ce6a (was db710cc stale), integrity PASS.
- C_prose/C_duplicate/C_terminology: identified 420 lines duplication across 11 blocks (TFNE mantra, RBS, source proxy, delay, calibration, banners) with single-owner map (doctrine/*.md, operator_doctrine.md, source_field_equations.md, protocol_h_rbd_memory.md, etc.); full text link-out deferred to next PRGS cycle to keep change small and preserve distinctions (state/parameter, relative/calibrated, proxy/physical, delay/wave, attenuation/adaptation, bounded/stable). No high-value root move blocked.
- Delta science/API/E1/E2 semantics =0 (verified: no jaxfne/*, 28/28 frozen intact, E2 spec hashes recomputed pre-data with sampling fs 2000 fix and mask h_rule/h_bytes/h_impl disambiguation).
- S = S_repo + S_harness + S_memory: repo minimized, harness refreshed, memory consolidated.

## Memory consolidation (aggressive)

Durable (project memory): scientific grammar E->S->F->P->O->A->M, execution grammar CircuitSpec->construct->Model->simulate->Signals, RBS container doctrine, relative quantity grammar B/R/E/T, source/field/probe tiers, containment vs composition, proxy truth gate, delay vs wave, attenuation vs adaptation, bounded vs stable, harness lessons above.

Ephemeral (handoff only, not project memory): current HEAD 0b8c22e, parent 35291f8, E1 07b1c04, spec hashes b89a09c/6ca10a96, sha 0E9D602B/D4DEBFD8, counts 189 public exports, resolved E2 blockers (PSD, delay grid, PV, mean_degree, epsilon, mask hashes, sampling_rate), 13 empty skill dirs, etc. These remain in handoff receipts, not in doctrine.

## Replay validation (pre-seal historical suite)

Must reproduce previously correct decisions before Seal accepted: F02, F03, NO_WAVE, C_head, NOT_FOUND, JSON/prose — executed separately via `scripts/run_test_gate.py` and `python -m pytest` with JAX_PLATFORMS=cpu. See validation section below. E2 deterministic preflight rerun on new HEAD/paths after minimization: spec_hash invariant, path relocation has provenance receipt, not silent redefinition.

## Next gate

Only after minimized repo sealed, synchronized (dev==origin/dev), replay PASS, and E2 preflight PASS on new HEAD:

E2a --adequacy-only--> theta* --write-once freeze--> E2b

JDNA deferred: `define/inherit, A-80, A*0.08` compiler concept survives as future design item, not on E2 path. Current E2 audit packet and DESIGN_READY receipts relocated with provenance, not rewritten.
