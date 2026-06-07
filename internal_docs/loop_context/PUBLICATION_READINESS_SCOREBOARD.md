<!--
Updated jaxfne project-source bundle.
Generated from attached repo zip: jaxfne-pub-ed08-tutorial-atlas-coverage.zip
Zip SHA256: ebcc98621b0542a9fca1de1ad5790d6508258fe66c63365a95cefe3bbbde6761
Repo checklist SHA: 9a8c7db58f588bde9f5e8c31b664d56c4982958e
Repo checklist branch: pub/ed08-tutorial-atlas-coverage
jaxfne version: 0.3.29
Generated UTC: 2026-06-07T22:34:39Z
-->
# Publication Readiness Scoreboard

Scores are operational scores out of 100 for a strong methods-journal package paper. They are not claims of biological validity.

## Snapshot

| Item | Value |
|---|---|
| Version | `0.3.29` |
| Checklist SHA | `9a8c7db58f588bde9f5e8c31b664d56c4982958e` |
| Checklist branch | `pub/ed08-tutorial-atlas-coverage` |
| Main figures | `8/8` |
| Extended Data | `8/10` |
| Remaining ED | `ED9 failure/null controls; ED10 release archive receipt` |
| Output manifests in attached zip | `missing; rerun scripts on live checkout` |

## Factor table

| # | Factor | Score now | Target | Evidence now | Next action | Stop condition |
|---:|---|---:|---:|---|---|---|
| 1 | Branch/release hygiene | 88 | 100 | Checklist SHA present; zip lacks `.git` metadata | Re-freeze live `cur`, clean tree, immutable SHA | Dirty tree or unknown branch before mutation |
| 2 | Main figure stack | 94 | 100 | 8/8 PNGs and scripts present | Final rerun with manifests and hashes | Hand-edited figure without script |
| 3 | Extended Data stack | 80 | 100 | ED1-ED8 implemented; ED9/ED10 missing | Implement ED9, then ED10 after approval | ED panel without script/manifest/receipt |
| 4 | Manifest/hash closure | 50 | 100 | Checklist paths exist, but `outputs/publication/*` absent from zip | Rerun publication scripts and inventory | Claiming artifact integrity without manifests |
| 5 | Notebook execution evidence | 60 | 95 | ED3/ED8 present; full output receipts absent from zip | Rerun smoke/full notebooks on clean install | Claiming execution without receipts |
| 6 | JSON/schema validation | 78 | 95 | Checklist and generated inventory are strict JSON | Validate all regenerated manifests/receipts | NaN/Inf or ndarray in JSON |
| 7 | API stability | 72 | 95 | Root import works; 149 exports observed; wrappers exist | Refresh API snapshot after ED10 | Public API removal without wrapper |
| 8 | Optional dependency laziness | 60 | 95 | ED4 exists; local targeted test contaminated by preloaded pandas | Repeat in clean venv; patch if real | Root import requires optional heavy dependency |
| 9 | JAX numerical discipline | 74 | 95 | Runtime rules and tests exist | Add scan/vmap/jit receipts where benchmarked | I/O/plotting/JSON inside JIT |
| 10 | Source bookkeeping | 76 | 95 | Source modes and double-count guard documented/tested | Add source accounting panel or ED9 row | Synaptic/current double count |
| 11 | Probe/readout contracts | 84 | 95 | ED7 and probe contract tests present | Final readout manifest crosswalk | Physical labels on proxy arrays |
| 12 | Electromagnetic admissibility | 48 | 90 | P0-P5 ladder and Poisson admissibility docs/tests | Keep physical solver experimental only | PDE/field claim without residual |
| 13 | Physical amplitude discipline | 92 | 100 | Gate false in checklist | Preserve across all manifests | Calibrated claim without calibration |
| 14 | Mechanism discipline | 72 | 95 | Objective/null infrastructure exists | ED9 null/failure controls | Objective success called mechanism proof |
| 15 | Benchmark evidence | 72 | 85 | ED6 local CPU receipt implemented | Add hardware/timing-phase receipt bundle | Speedup claim from local smoke |
| 16 | Adjacent-tool positioning | 82 | 95 | Fig8 implemented | Add citations and neutral capability table | Superiority claim |
| 17 | Tutorial-to-package discipline | 74 | 90 | Etude thinness tests and package vis helpers present | Dedupe remaining tutorial glue | Notebook-local scientific engine |
| 18 | Release archive readiness | 22 | 100 | Release requirements declared; no ED10 | Approval-gated tag/archive/wheel hash | Release or DOI without approval |
| 19 | Manuscript alignment | 72 | 95 | Roadmap/checklist exists | Replace placeholders with receipts | Numeric claim without manifest |
| 20 | Empirical validation readiness | 15 | 70 | Synthetic/proxy package-paper posture | Keep empirical validation future scope | Empirical claim in scaffold paper |
| 21 | Config-first backbone | 72 | 95 | Config object present; README object grammar updated | Verify Net/Model naming consistency | Breaking old tutorials |
| 22 | Identity/selectors | 84 | 95 | v0.3.29 selector tests present | Preserve stable area/layer/type IDs | Nondeterministic selector results |
| 23 | Connectivity rules | 76 | 95 | Connection-rule tests present | Harden sparse deterministic compiler | Silent empty rule selection |
| 24 | Weld/reconstruct/flatten | 35 | 90 | Experimental HPC contracts present | Plan before public promotion | Lost identity/tracking map |
| 25 | Solver readiness | 46 | 90 | Admissibility docs/tests exist; no solver claim | Stable schemas before solver | Solver work before schema stability |
| 26 | Jaxley bridge | 60 | 90 | Optional bridge modules/tests present | Guarded bridge receipts | Top-level Jaxley dependency |
| 27 | PyNWB bridge | 25 | 85 | Doctrine only; no confirmed writer in inspected zip | Design lazy writer + round trip later | NWB export called empirical validation |

## Interpretation

The repo is close to a package-methods evidence bundle, not to a physical-field or empirical-mechanism paper. The main work left is ED9/ED10 plus full manifest/output closure from a live checkout.
