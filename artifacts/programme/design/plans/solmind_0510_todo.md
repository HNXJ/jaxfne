# 0.5.10 Solmind execution todo (planning branch)

Status: PROPOSED; no implementation authorized by this file. Baseline `jaxfne@dev@10c417adced394e799b922e78315a0be1f8a82dd` (2026-11-20). Owner: Hamm. Read `artifacts/etudes/meta/plans/release_plan_0.5.9-0.6.0.md`, `artifacts/programme/design/plans/arxiv_paper_plan.md`, canonical stacks and issue log before execution. This is a scoped planning overlay, not a replacement for those homes.

**Score contract:** Frozen 0.5.10 checklist: 70 package points + 30 release-gate points, each scored only on its named check; last recorded 43/100 at 2026-11-19, NOT a current verified score. Gate 5 (6 points) requires publication; 100/100 under current authority means published, not merely ready. Owner's desired prepublication readiness=100 is a distinct proposed metric, NOT adopted; do not rewrite historical scoring without ruling.

**Dependencies:** R01-R11 first; R12-R27 ephaptic/theory can proceed as bounded lane beside R28-R41 release packages; R42-R50 docs and code hygiene only after API inventory; R51-R55 sequential release closure. Max three lanes including reviewers. No unapproved changes to frozen figures, preregs, scores or release targets. Negative replication results remain evidence, not release failure unless their registered gate says so. Preserve original 15-stage manuscript authority and new figure remapping.

**Task protocol:** For each row, record base SHA, read/write scope, expected minutes, falsifier, positive/negative controls, exact closing command and non-writer review before execution. The closing phrases below are acceptance designs, not observed PASS. Smallest sufficient change, no automatic scope expansion. Owner decisions remain OPEN.

| ID | Priority | Action | Authority | Closing evidence |
|---|---|---|---|---|
| R01 | P0 | Reconcile release score against exact dev SHA | 10A-G and gates | Checklist recomputed with evidence per scored item |
| R02 | P0 | Audit source authority and 15-stage figure remapping | Release/paper plans | Owner rulings mapped; no frozen figure silently changed |
| R03 | P0 | Reconcile current open/todo/issue records | Stacks | Every inherited open defect has one owner and closing check |
| R04 | P0 | Run matching-host broad baseline at integration SHA | Gate 2 | Linux/Windows receipt and CI same SHA |
| R05 | P0 | Resolve O-FIG05 frozen equivalence panel | Gate 2 | Owner ruling plus macOS gate pass; no unauthorized refreeze |
| R06 | P0 | Resolve O-NUM platform bitwise assertions | Gate 2 | Owner-approved numerics criteria, negative mutant, Mac/CI pass |
| R07 | P0 | Resolve O-73 positional time-unit hazard | 10F | Old usage fails demonstrably; safe behavior tested |
| R08 | P0 | Resolve O-74 noninteger bin-width API transition | 10F | Compatibility/deprecation contract and regression test |
| R09 | P0 | Resolve O-75 Fleiss-kappa semantic change | 10F | 0.5.8 comparison and documented version behavior |
| R10 | P0 | Turn six non-strict xfails into meaningful gates | Gate 2 | Each failure detectable; intentional skips justified |
| R11 | P0 | Audit silent data-dependent skips | Gate 2 | Clean data-free clone lists all skips explicitly |
| R12 | P1 | Inventory ephaptic operator routes and units | TASK-EPHAPTIC | LIF/Izh voltage, timestep, alpha dimension table |
| R13 | P1 | Separate lateral voltage coupling from HDP/RBD taxonomy | TASK-EPHAPTIC | Single canonical operator description, no mislabeled dynamics |
| R14 | P1 | Derive ephaptic discrete update and alpha dimensional contract | TASK-EPHAPTIC | Unit derivation and dt-refinement falsifier |
| R15 | P1 | Pin zero-coupling identity on LIF and Izhikevich | TASK-EPHAPTIC | Bitwise/declared tolerance identity and mutant |
| R16 | P1 | Verify spatial kernel normalization and symmetry | TASK-EPHAPTIC | Row normalization, edge cases, permutation test |
| R17 | P1 | Resolve depth-fraction vs mm geometry P-066 | TASK-UNITS-CSD | Owner-approved coordinate units and affected-rule inventory |
| R18 | P1 | Test ephaptic lag across supported execution routes | TASK-EPHAPTIC | Same causal lag in scalar and batched routes |
| R19 | P1 | Compare exogenous-field drive to endogenous operator | T-94 | Matched kernel and imposed field; differential effect receipt |
| R20 | P1 | Run zero/shuffle/spatial-perturbation controls | Paper stage 7 | Prespecified effect and null estimands; report negatives |
| R21 | P1 | Freeze mechanistic claims before ephaptic figure | Paper stage 7 | Association vs causal attribution clearly bounded |
| R22 | P1 | Determine whether ephaptic core addition is necessary | Release scope | Keep experimental if not required by frozen 0.5.10 acceptance |
| R23 | P1 | Verify source domain (state,input) and proxy scope | TASK-SOURCE-SCOPE | LFP/CSD/EEG/MEG claims separated from physical solutions |
| R24 | P1 | Check kernel LFP factorization and analytic oracles | T-90 | Tone, gain, shuffle and zero controls remain non-vacuous |
| R25 | P1 | Resolve P-056 inputs or retain owner blocker | T-90 | Population/weights/drive/threshold recorded before run |
| R26 | P1 | Test discrete-map consistency and dt refinement | TASK-DMAP | Declared order, bounded error, planted break |
| R27 | P1 | Test H-state invariance vs clamp vs stability | TASK-HINV | Three distinct claims and matching checks |
| R28 | P1 | Build AT-10-N20 receipt and F2/new Fig 3 | 10A A1 | Figure regenerates from pinned atlas receipt |
| R29 | P1 | Implement NeuronalTensor flattening identity/schema figure | 10A A2 | Identity/topology/sign/position preserved in tests |
| R30 | P1 | Recheck geometry mutants and rotation invariance | 10A A3-A4 | All required planted breaks fail |
| R31 | P1 | Score CPU scaling on preregistered sizes | 10B B2 | Hash-pinned throughput/compile/run receipts |
| R32 | P1 | Score GPU scaling with fixed accuracy and seeds | 10B B2 | GPU receipt matched to CPU and prereg |
| R33 | P1 | Score Brian2/NEST/Jaxley baselines | 10B B3 | Matched model classes or documented incompatibility |
| R34 | P1 | Verify LIF edge GPU kernel parity | 10B B5 | Declared CPU/GPU tolerance met with receipt |
| R35 | P1 | Render S6 and main scaling figure from receipts | 10B | Axes, units, uncertainty and provenance audited |
| R36 | P1 | Execute controlled ensemble augmentation study | 10C C1 | Frozen seeds and metrics; independent test |
| R37 | P1 | Render S7 determinism/provenance panel | 10C C1 | Pinned receipts and replayable figure |
| R38 | P1 | Score Mejias 2016 G6 after prereg signature | 10D D3 | Owner-signed estimands and honest verdict |
| R39 | P1 | Complete AdEx/LIF/Model.tune batching tests | 10F F5 | Seed parity, shapes, gradients and negative tests |
| R40 | P1 | Finish dashboard P1/P2 only after text recovered | 10F F6 | Owner ruling recovered, reproducible outputs |
| R41 | P1 | Run replication conformance audit | 10G G2 | 0 unlisted violations; exceptions owner-authorized |
| R42 | P2 | Inventory active docs, archived tutorials and navigation | Docs | Each public page classified canonical/archive/stale |
| R43 | P2 | Inspect jomission subproject tutorial candidates | Docs | Source SHA, license, API compatibility and scientific claim status |
| R44 | P2 | Select minimal tutorial migration candidates | Docs | Only executable transferable examples; no stale copying |
| R45 | P2 | Build 2-3 clean-room tutorials using current public API | Docs | Fresh install runs, deterministic figures, no jomission results promoted |
| R46 | P2 | Merge duplicate docs by canonical subject home | Docs | Link map, redirects where needed, no lost unique semantics |
| R47 | P2 | Compact README, quickstart and API prose | Docs | No unsupported claims, reduced repetition, original meaning preserved |
| R48 | P2 | Consolidate redundant code files only after dependency map | Maintainability | One bounded merger, API parity and tests; no broad rewrite |
| R49 | P2 | Audit figure captions, terminology and abbreviations | Docs | Independent interpretability at final print size |
| R50 | P2 | Regenerate docs/nav/notebooks and strict build | Docs | mkdocs strict, link checks, notebook smoke and dashboard fresh |
| R51 | P0 | Close package items with exact evidence or owner deferral | Gate 1 | Every 10A-10G item accepted or dated owner ruling |
| R52 | P0 | Prepare candidate version, changelog and release body | Gate 3 | Version identity and release_body --check pass |
| R53 | P0 | Run full RC gates, clean install and adversarial faults | Gate 2 | Same candidate SHA all gates, wheel/sdist hashes and mutants |
| R54 | P0 | Obtain independent release and scientific review | Gate 4 | Non-writer ACCEPT, findings dispositioned |
| R55 | P0 | Owner release authorization and publication | Gate 5 | Only after owner go: tag/main/Release/PyPI and receipt |

**Release invariant:** Do not call 0.5.10 READY until all prepublication requirements pass on the exact candidate SHA, issue-log Open section is empty under the release skill, full package/CI/documentation checks pass and owner has reviewed the remaining publication action. Never tag/publish without exact owner go.
