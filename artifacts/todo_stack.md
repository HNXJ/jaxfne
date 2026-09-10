# jaxfne finite programme stack (v0.4.22 → v0.4.24)

**Authority.** Planning and execution-order state for the private reduction programme.
**Not evidence.** Tests, RC/CI attestations, release receipts, tags, and PyPI state
remain evidence authority. Mark `DONE` only with matching current or immutable evidence.

**Baseline (2026-09-10).** `main = dev = 2338b00`. `v0.4.21` immutable @ `8823520`.
Pre-bump RC + CI complete on `2338b00`; package surfaces still declare `0.4.21`.

**Work loop.** `P [R G]^N S` — inspect stack (P), review/select one item (R), execute
and verify acceptance (G), seal release and reconcile stack (S). Do not execute past a
release seal line.

**Roadmap.** `artifacts/roadmap/ROADMAP_0422_0424.md` is scope reference; this file
is the executable queue. Where they differ, this file wins for *what runs next* after
reconciliation; roadmap wins for *scientific scope boundaries*.

---

# v0.4.22

Theme: truth + measurement floor. No representation change (roadmap §3.4).

## DONE (retained for context)

| ID | Task | Evidence |
|----|------|----------|
| 22-ENT-01 | Programme entry: v0.4.21 published; `main==dev` clean | `v0.4.21`→`8823520`; `docs/install.md`; `origin/main==origin/dev` @ `2338b00` |
| 22-INT-01 | Integration baseline validated @ `2338b00` | RC attestation PASS `2338b00`; `release_ci` 34491802335; main-push CI 34504921847/34504921892 |
| 22-W1 | Null-distribution diagnostics | Roadmap DONE; `2eef9bb`; `tests/test_objective_null_accounting_w1.py` |
| 22-W2 | Atlas dt provenance classification | `0c7d079`; `tests/test_atlas_suite.py` dt gates |
| 22-W3 | Broad-handler classification + scientific defects | `w3_broad_handler_tally.json` (132 classified, 0 UNKNOWN); `w3_scientific_handler_review.json` (23 reviewed, 2 fixed, 0 UNKNOWN); `e047177`; `tests/test_w3_scientific_handlers.py` |
| 22-W6 | Dual-representation UNKNOWN → AVOIDABLE (`emitter.W`) | W10 measurement; roadmap §1.1 |
| 22-W7 | Dense N×N site inventory | `w7_dense_nxn_inventory.json` (16 in-scope, 0 UNKNOWN); `2abf952`; `tests/test_w7_dense_nxn_inventory.py` |
| 22-W8 | Benchmark harness + baseline | `w8_baseline.json`; `03177b5`; `tests/test_w8_baseline.py` |
| 22-W10 | Allocation / usage map (U_k) | Roadmap CLOSED; `scripts/perf/w10_allocation_map.py` |
| 22-W11 | Bounded-degree scaling atlas | Roadmap MEASURED; `scripts/perf/w11_scaling_atlas.py`; `tests/test_backend_request_realization_w11.py` |
| 22-W12 | Environment-dependent figure gate | `4d696ad`; matplotlib pin `<3.11`; `tests/test_equivalence_gate_v20260815.py` |
| 22-W13 | Connectivity truth D1–D4 (realize or refuse) | `a825ed2`, `cd6161a`, `e6218c5`; `tests/test_connectivity_request_realization.py` |
| 22-W4W5-CLS | Public unfinished-content classification | Roadmap CLASSIFIED 2026-09-09 (88 nav pages) |

## TODO

| ID | Task | State | Why 0.4.22 | Acceptance | Evidence / authority | Depends |
|----|------|-------|------------|------------|----------------------|---------|
| **22-AUTH-01** | **Establish v0.4.22 acceptance authority + receipt chain** | **TODO** | RELEASE/SEAL gates require version-specific receipt and goal paths; none exist for 0.4.22 | `artifacts/release/v0_4_22_release_receipt.json` (or successor) authored; acceptance goal authority designated; Gate 0 RELEASE path defined | `current_release_authorities.json` schema; `gate0_git_reality.py`; `jaxfne-release`/`jaxfne-seal` skills | — |
| 22-AUTH-02 | Align `current_release_authorities.json` to 0.4.22 | TODO | Stale `0.4.17` blocks RELEASE mode | `release_target_version==0.4.22`; receipt/acceptance paths match AUTH-01 artifacts | `artifacts/release/current_release_authorities.json`; `validate_release_authorities` | 22-AUTH-01 |
| 22-VER-01 | Coherent 0.4.22 version identity transition | TODO | Tree still declares `0.4.21` while carrying post-0.4.21 work | `pyproject.toml`, `jaxfne/_model.py` `_JAXFNE_VERSION`, `sync_release_metadata.py --check` PASS; `scripts/sync_docs_version.py` run | `tests/test_package_version_alignment.py`, `test_docs_version_alignment.py` (tree fields) | 22-AUTH-01 (target confirmed) |
| 22-GEN-01 | Regenerate stale `gallery_manifest.json` | TODO | Manifest at `0.4.20` vs runtime `0.4.21` | `docs/_static/gallery/gallery_manifest.json` version matches `jaxfne.__version__` after bump | `scripts/generate_release_gallery.py` | 22-VER-01 |
| 22-GEN-02 | Regenerate canonical atlas manifest if version-gated | TODO | `docs/_static/atlas/manifest.json` may drift after bump | `test_canonical_manifest_provenance` PASS | `scripts/generate_readme_atlas.py` | 22-VER-01 |
| 22-DOCS-01 | Candidate-vs-published public docs semantics | TODO | PyPI stays `0.4.21` until publish; tree becomes `0.4.22` candidate | `docs/install.md` / `colab.md` distinguish published `0.4.21` @ `8823520` from unreleased candidate; `test_published_version_matches_pyproject_when_not_mid_candidate` PASS | `tests/test_docs_version_alignment.py` | 22-VER-01 |
| 22-DOCS-02 | Fix `sync_docs_version.py` install.md pattern drift | TODO | Sync script misses current install wording | Script updates `docs/install.md` published lines OR docs updated manually with script fix | `scripts/sync_docs_version.py` | 22-VER-01 |
| 22-W45-01 | W4/W5 STALE public doc repairs (3 lines) | TODO | Roadmap §3.2 STALE class | `guides/jdna.md` stamp; tutorial placeholder figures resolved or reclassified | Roadmap W4 table STALE rows | — |
| 22-W45-02 | Tutorial runner stored≠consumed flags | TODO | `--smoke` / `--out-root` documented inert; 6 CI doc lines wrong | Implement flags or remove from CLI and docs; no public doc instructs dead flags | `scripts/run_all_tutorials.py`; roadmap W4/W5 defect note | — |
| 22-W45-03 | Public stub contract consistency (`__init__` vs `__all__`) | TODO | Five names reachable but excluded from contract | Each disposition: experimental doc, remove import, or add to contract; no silent `NotImplementedError` on undocumented public names | Roadmap W5 table; `public_surface.py` | — |
| 22-W9-01 | `LegacyMultiAreaSpectrolaminarObjective` deprecation horizon | TODO | Roadmap W9; warning noise in canonical examples | Declared removal version or permanent compatibility; examples updated | Roadmap §3.2 W9 | — |
| 22-W15-01 | BMTK/SONATA architecture study (planning only) | TODO | Roadmap item 8; no interoperability in 0.4.22 | Written study artifact; no SONATA import/export code | Roadmap W15; §3.4 non-goals | — |
| 22-W16-01 | Scientific/developmental programme definition (planning) | TODO | Roadmap item 9 | W16.1–2 definitions recorded; no implementation claims in public docs | Roadmap §8 W16 | — |
| 22-W17-01 | Downstream mechanism truth semantics (Jomission) | TODO | Roadmap item 10; 0.4.22 semantics only | W17.1–4 surfaces specified; prototype only if low-risk | `jomission_downstream_lessons.md`; roadmap §9 | — |
| 22-W14-01 | AI harness refine (not expand) | TODO | Roadmap W14; checkpoint census done | Minimal workflow preserved; no duplicate skills; continuation-equivalence retained | `gate0_git_reality.py`; skills under `artifacts/skills/` | — |
| 22-VERIFY-01 | Identity-sensitive verification after version bump | TODO | Successor SHA must be checked without full narrative re-run | `sync_release_metadata.py --check`; targeted version/alignment tests PASS | `tests/test_docs_version_alignment.py`, `test_package_version_alignment.py` | 22-VER-01, 22-DOCS-01 |
| 22-RC-01 | Full same-SHA RC on final 0.4.22 candidate | TODO | Publication requires observed RC attestation @ candidate SHA | `rc_gate_attestation.json` `commit_sha==candidate`, `status==PASS`, 16/16 families | `scripts/run_test_gate.py rc` | 22-VERIFY-01 |
| 22-CI-01 | `release_ci` evidence for final candidate SHA | TODO | `junit_parity` requires successful main CI @ exact SHA | `reconcile_release_target.py` PASS for `0.4.22` @ candidate | `.github/workflows/release_ci.yml` | 22-RC-01 (parallel after push) |
| 22-SEAL-01 | Independent release seal | TODO | Programme rule; 0.4.21 gap acknowledged in roadmap §5.1 | SEAL_GO with scorecard; no self-repair by executor | `jaxfne-seal` skill; AUTH-01 acceptance set | 22-RC-01, 22-CI-01 |
| 22-PUB-01 | Tag `v0.4.22` @ sealed candidate (new SHA) | TODO | Immutable release identity | Tag peels to candidate; does not move `v0.4.21` | `reconcile_release_target.py` | 22-SEAL-01 + explicit authorization |
| 22-PUB-02 | GitHub release + retained CI artifacts | TODO | Provenance chain | Release binds manifest SHA256 to CI bytes | `docs/ci_policy.md`; `release_ci` build job | 22-PUB-01 |
| 22-PUB-03 | PyPI publish from retained bytes | TODO | Distribution identity | `reconcile_release_target.py` `safe_to_upload`; clean install smoke | `twine_check`; isolated wheel smoke | 22-PUB-02 |
| 22-PUB-04 | RTD / deployed docs identity verification | TODO | Docs must match release commit | RTD build SHA == tag SHA; public install docs show `0.4.22` published | `docs/ci_policy.md` RTD section | 22-PUB-03 |
| 22-POST-01 | Post-publication clean-install + identity checks | TODO | Seal completion | Fresh venv `jaxfne==0.4.22`; version/tag/PyPI reconcile | `reconcile_release_target.py`; `test_install_md_latest_pypi_version` constants updated | 22-PUB-04 |
| 22-EXT-01 | External/durable project-source current-target wording | TODO | Stale `0.4.20` engineering-target claims outside repo evidence | Owned external sources name `0.4.22` target after formal establishment | User policy; not gated by in-repo tests alone | 22-PUB-04 |
| 22-STACK-01 | **Review and update this stack from sealed v0.4.22 before v0.4.23** | TODO | Release boundary | v0.4.22 sealed; stale entries reconciled; discoveries routed to 0.4.23 queue | Fresh attestation + tag + PyPI evidence | 22-POST-01 |

## DEFERRED (out of 0.4.22 scope — do not execute here)

| ID | Task | State | Note |
|----|------|-------|------|
| 22-DEF-01 | Remove `emitter.W` / dual representation | DEFERRED | Roadmap §3.4 non-goal; v0.4.23 §4.2 |
| 22-DEF-02 | Performance optimisation merges | DEFERRED | 0.4.22 measures only |
| 22-DEF-03 | SONATA/BMTK interoperability code | DEFERRED | W15 study-only in 0.4.22 |

## REJECTED

| ID | Task | State | Note |
|----|------|-------|------|
| 22-REJ-01 | Retag or republish `v0.4.21` @ `2338b00` | REJECTED | `v0.4.21`→`8823520` immutable; `reconcile_release_target` blocks |

**Next executable item:** `22-AUTH-01` (establish v0.4.22 acceptance authority + receipt chain).

---

# v0.4.23

Theme: exact representation + development substrate. Every item needs measured
before/after (roadmap §4). **Entry:** v0.4.22 shipped; zero UNKNOWN in W7/W3
classifications; W8 baseline recorded.

| ID | Task | State | Why 0.4.23 | Acceptance | Evidence / authority | Depends |
|----|------|-------|------------|------------|----------------------|---------|
| 23-ENT-01 | Confirm 0.4.23 entry criteria | TODO | Gate | 22-STACK-01 DONE; W7/W3 zero UNKNOWN; W8 baseline current | Artifacts §4.1 | 22-STACK-01 |
| 23-REP-01 | Eliminate avoidable `emitter.W` persistent storage | TODO | W6/W10/W11; headline representation item | Measured `M_before/M_after`, `t_before/t_after` on W8 matrix; bit-exact observables | Roadmap §4.2 dual representation | 23-ENT-01 |
| 23-REP-02 | Realized topology authoritative; backend layouts execution-specific | TODO | W15 principle; Jomission W17 | Inspect/simulate use realized edges only; mechanism reports where feasible | Roadmap §4.2 connectivity | 23-ENT-01 |
| 23-REP-03 | Sparsity routing: honour `p_connect` or refuse; lower `_SPARSE_DIRECT_N` | TODO | Roadmap §1.1c silent inert levers | Sparse request at N=1000 allocates sparse; no silent no-op | `tests` migrated not deleted | 23-ENT-01 |
| 23-PARAM-01 | Parameter sharing (class-shared edge/neuron fields) | TODO | §1.1b ~57% class-shared at N=1000 | Bit-exact equivalence; sharing taxonomy applied | Roadmap §4.2 parameter sharing | 23-REP-01 |
| 23-H-01 | H/RBS/RBD representation reduction (observable-specific) | TODO | Roadmap §4.2 | Per-coordinate written argument + tests; no biology deleted on intuition | Roadmap H section | 23-ENT-01 |
| 23-HDP-01 | HDP: null plasticity bit-exact; no dense mutable W when disabled | TODO | Roadmap §4.2 | Mechanical guards; `W(t)=W(0)` when HDP off | Existing HDP tests + new guards | 23-ENT-01 |
| 23-DELAY-01 | Delay/history compaction (delay classes, ring buffers) | TODO | W10 `delay_steps` class-sharing prize | Bit-exact continuation where promised | `409ea7b` checkpoint delays; roadmap §4.2 | 23-ENT-01 |
| 23-STOCH-01 | Continuation-owned stochastic inputs (where required) | TODO | Jomission rejection of incompatible Poisson path | Per-path verification before merge | Roadmap §4.2 | 23-ENT-01 |
| 23-REC-01 | Streaming/decimated recording + memory preflight (W17.2) | TODO | Primary 0.4.23 performance target | `dynamic state != requested observation`; preflight API | Roadmap §4.2 recording | 23-ENT-01 |
| 23-FIN-01 | Long-run transactional finalization (W17.4) | TODO | If justified by evidence | Progress/liveness; atomic persist; manifest terminal state | Roadmap §4.2 | 23-W17-01 semantics from 0.4.22 |
| 23-SYSID-01 | Operating-point system ID utility (W17.3) | TODO | If evidenced | Generic `S_ij` utility; not Jomission controller | Roadmap §4.2 | 23-W17-01 |
| 23-JDNA-01 | Minimal `evolve()` runtime substrate (PROVISIONAL) | TODO | Only if semantics complete | Tests + manifest before public doc claims | Roadmap W16.3–4; W16 JDNA rules | 23-H-01, 23-HDP-01 |
| 23-EDGE-01 | Edge selectors + immutable transforms pipeline | TODO | W17 + W16 structural G | Provenance diff; not parallel mutation system | Roadmap §4.2 | 23-REP-02 |
| 23-LAW-01 | Custom HDP/RBD law interface | TODO | Jomission downstream need | Structured law object; preserves H≠RBD≠HDP≠G | Roadmap §4.2 | 23-HDP-01 |
| 23-VERIFY-01 | Full 0.4.23 RC + CI + seal + publish chain | TODO | Same provenance model as 0.4.22 | Attestation @ candidate; tag `v0.4.23`; reconcile PASS | `run_test_gate.py`; `reconcile_release_target.py` | 23-* programme items |
| 23-STACK-01 | **Review and update this stack from sealed v0.4.23 before v0.4.24** | TODO | Release boundary | 0.4.23 sealed; queue reconciled | Tag + attestation | 23-VERIFY-01 |

## DEFERRED (needs scope decision — not in roadmap programme text)

| ID | Task | State | Note |
|----|------|-------|------|
| 23-DEF-MCC3 | MCC3 programme item | DEFERRED | Historical handoff mention; no entry in `ROADMAP_0422_0424.md` — promote only with new authority |
| 23-DEF-E2 | E2 dedupe / preregistration cleanup | DEFERRED | E2 artifacts classified NOT_MODEL_CHECKPOINT (W14); dedupe not a roadmap release requirement |

**Next executable item (after 0.4.22 seal):** `23-ENT-01`.

---

# v0.4.24

Theme: bounded approximation + final programme audit (roadmap §5). **Entry:**
v0.4.23 shipped; harness shows cumulative gains from `8823520` baseline.

| ID | Task | State | Why 0.4.24 | Acceptance | Evidence / authority | Depends |
|----|------|-------|------------|------------|----------------------|---------|
| 24-ENT-01 | Confirm 0.4.24 entry criteria | TODO | Programme close | 23-STACK-01 DONE; W8 cumulative receipts | Roadmap §5 | 23-STACK-01 |
| 24-W16-6 | Neurobiophysical Geometry / pseudo-neurogeneration (W16.6) | TODO | Only release allowing approximation | Each result: observable, metric `d`, ε, cost before/after | Roadmap §5.1 | 24-ENT-01 |
| 24-FIELD-01 | Field/LFP proxy computational audit | TODO | Pairwise work reduction with declared error | No weakening Q≠I_syn; proxy≠calibrated | Roadmap §5.1 fields | 24-ENT-01 |
| 24-JAX-01 | JAX execution profile programme | TODO | Measure before JIT changes | Profile receipts; no guess-based optimisation | Roadmap §5.1 JAX | 24-ENT-01 |
| 24-EQUIV-01 | Per-observable equivalence acceptance | TODO | Approximation discipline | Exact: bit-exact; approx: `d≤ε` named pre-merge | Roadmap §5.1 scientific equivalence | 24-W16-6 |
| 24-AUDIT-01 | Fresh independent adversarial core audit | TODO | Close 0.4.21 seal gap (roadmap §5.1) | Independent critic; implementer ≠ sole auditor | Roadmap §5.1 | 24-ENT-01 |
| 24-EXIT-01 | Programme exit criteria verification | TODO | Formal programme close | All §5.2 bullets evidenced | Roadmap §5.2 | 24-* items |
| 24-VERIFY-01 | Full 0.4.24 RC + CI + seal + publish chain | TODO | Final public release | Attestation; tag `v0.4.24` | Same as 22-VERIFY pattern | 24-EXIT-01 |
| 24-STACK-01 | **Seal v0.4.24 and reconcile this stack against final programme state** | TODO | Programme terminus | Stack archived or marked complete; no open P0/P1 | Final tag + audit receipt | 24-VERIFY-01 |

**Next executable item (after 0.4.23 seal):** `24-ENT-01`.

---

## Conflicts / reconciliations (inspection 2026-09-10)

| Source | Stack treatment |
|--------|-----------------|
| `ROADMAP_0422_0424.md` baseline SHA `8823520` | Superseded for *integration* by `2338b00`; roadmap baseline line should update at 22-VER-01, not before |
| `current_release_authorities.json` @ `0.4.17` | Stale; fixed in 22-AUTH-02 after 22-AUTH-01 |
| Package version `0.4.21` @ `2338b00` | Intentional pre-candidate state; 22-VER-01 creates successor SHA |
| Pre-bump RC @ `2338b00` | Valid for integration evidence; **not** sufficient for 0.4.22 publish (22-RC-01 required on final SHA) |
| MCC3 / E2 dedupe (historical handoff) | DEFERRED pending roadmap authority — not promoted |
| W4/W5 classification vs repair | Classification DONE; repairs remain TODO (22-W45-*) |
