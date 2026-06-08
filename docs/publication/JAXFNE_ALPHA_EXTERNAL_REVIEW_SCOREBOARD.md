# jaxfne alpha external review scoreboard

Recalibrated after alpha bundle execution on live `cur`. Operational scores (0–100) for a **software/methods** paper — not biological validity.

## Freeze receipt

| Field | Value |
|---|---|
| branch | `cur` |
| SHA | `408cb7960e374636ef3c72fbe545e0e1b5dd4f10` (alpha bundle closure) |
| version | `0.3.29` |
| inventory | `8/8 main + 10/10 ED` |
| output manifests | regenerated locally in `outputs/publication/` (gitignored) |
| release/tag/publish | `pending_approval` / not executed |

## Verdict split

| Dimension | Verdict | Score | Reason |
|---|---|---:|---|
| package methods paper readiness | **follow-up** | 82 | Artifact stack complete; manifest bundle + manuscript TBDs remain |
| biophysical solver readiness | **reject** (for solver claims) | 28 | laminar proxy / no PDE; admissibility docs only |
| empirical mechanism readiness | **reject** (for empirical claims) | 12 | out of scope for alpha scaffold paper |

## 27-factor table

| # | Factor | Score now | Target | Gap | Evidence paths | Classification | Stop condition |
|---:|---|---:|---:|---|---|---|---|
| 1 | Branch/release hygiene | 92 | 100 | tag at HEAD pending | `cur`, `docs/publication/JAXFNE_ALPHA_HANDOUT.md`, ED10 receipt | follow-up | dirty tree before release |
| 2 | Main figure stack | 90 | 100 | final PNG commit at regen SHA | `figures/publication/fig01–08`, `scripts/publication/fig*.py` | accept | hand-edited PNG |
| 3 | Extended Data stack | 98 | 100 | none structural | ED1–ED10 PNGs + scripts | accept | ED without script |
| 4 | Manifest/hash closure | 85 | 100 | outputs gitignored in remote | `ed05_manifest_hashes.py`, `outputs/publication/*` | follow-up | claim integrity without manifests |
| 5 | Notebook execution evidence | 48 | 95 | structural receipts only | ED3, ED8 | follow-up | full execution claim |
| 6 | JSON/schema validation | 78 | 95 | broaden schema fixtures | ED2, `test_artifact_json_safety_v0330.py` | accept | NaN/Inf in JSON |
| 7 | API stability | 72 | 95 | refresh ED1 at release | ED1, `test_api_smoke.py` | follow-up | public API removal |
| 8 | Optional dependency laziness | 82 | 95 | formal clean-venv receipt | ED4, Task 06 receipt | accept | root import loads pandas |
| 9 | JAX numerical discipline | 68 | 95 | more jit/scan receipts | runtime rules, benchmark tests | follow-up | I/O inside jit |
| 10 | Source bookkeeping | 58 | 95 | accounting panel | source mode docs/tests | follow-up | double-counting |
| 11 | Probe/readout contracts | 84 | 95 | maintain ED7 | ED7, `docs/probe_operators.md` | accept | physical labels on proxies |
| 12 | Electromagnetic admissibility | 38 | 90 | no solver residuals | `docs/poisson_admissibility.md` | follow-up | PDE claim without residual |
| 13 | Physical amplitude discipline | 92 | 100 | preserve gates | truth gates all manifests | accept | calibrated claim |
| 14 | Mechanism-claim discipline | 86 | 95 | maintain ED9 | ED9, null tests | accept | objective = proof |
| 15 | Benchmark evidence | 70 | 85 | local receipt only | ED6 | accept | speedup superiority |
| 16 | Adjacent-tool positioning | 76 | 95 | citation pass | fig08 | follow-up | superiority claim |
| 17 | Tutorial-to-package discipline | 50 | 90 | dedupe notebook glue | ED8 atlas | follow-up | notebook-local engine |
| 18 | Release archive readiness | 75 | 100 | approval-gated execution | ED10 receipt | follow-up | publish without approval |
| 19 | Manuscript alignment | 55 | 95 | TBD replacement plan | `MANUSCRIPT_TBD_REPLACEMENT_PLAN.md` | patch | numeric claim w/o manifest |
| 20 | Empirical validation readiness | 12 | 70 | future scope | scope docs | accept | empirical claim in scaffold |
| 21 | Config-first backbone | 38 | 95 | 0.3.28+ ladder | `Configuration`, glossary | follow-up | breaking tutorials |
| 22 | Identity/selectors | 35 | 95 | stable IDs | v0.3.29 selector tests | follow-up | nondeterministic selectors |
| 23 | Connectivity rules | 28 | 95 | typed rules | connection tests | follow-up | silent empty rules |
| 24 | Weld/reconstruct/flatten | 18 | 90 | experimental | HPC contracts | follow-up | lost identity map |
| 25 | Solver-readiness | 22 | 90 | schemas first | admissibility ladder | follow-up | solver before schema |
| 26 | Jaxley bridge | 62 | 90 | lazy guarded bridge | optional bridge modules | follow-up | mandatory Jaxley |
| 27 | PyNWB bridge | 28 | 85 | design only | doctrine | follow-up | NWB = validation |

## Top 10 strengths

1. Complete publication artifact ladder (8/8 + 10/10 ED) with scripts and PNGs.
2. Explicit truth gates preserved across manifests (`physical_amplitude_claim_allowed: false`).
3. ED9 null/failure controls reduce objective overinterpretation risk.
4. ED7 probe contract matrix separates proxy readout families.
5. ED5 manifest/hash integrity panel with synthesized manifest closure.
6. ED10 records release/archive as pending without executing release actions.
7. Public docs hygiene test suite (`test_public_docs_hygiene.py`) with broad coverage.
8. Optional dependency laziness passes clean venv check (Task 06).
9. Strict JSON outputs from publication generators (Task 02 finite gate pass).
10. Conservative root cleanup without breaking package/docs/test paths (Task 03).

## Top 10 blockers

1. `outputs/publication/*` not in git — review zip must include regenerated bundle.
2. Manuscript TBD slots not yet replaced with exact SHA/inventory receipts.
3. Notebook execution evidence is structural, not universal PASS execution.
4. Version tag at HEAD not created (approval-gated).
5. Wheel/sdist/PyPI/GitHub release not executed (approval-gated).
6. Archive/DOI not assigned (approval-gated).
7. Biophysical solver path remains proxy-only — cannot support field-solver claims.
8. Empirical validation datasets absent by design for this paper scope.
9. Stale `PUBLICATION_READINESS_SCOREBOARD.md` in loop_context (superseded by this file).
10. Some publication PNGs drift on regeneration — must commit at regen SHA for hash-stable review.

## Required patches before submission

| Priority | Patch | Owner task |
|---|---|---|
| P0 | Bundle `outputs/publication/` into external review zip | sender |
| P0 | Manuscript TBD → receipt table | Task 05 plan |
| P1 | Commit regenerated publication PNGs at regen SHA | post Task 02 |
| P1 | Release approval workflow (separate from ED10) | Task 08 |
| P2 | Refresh ED1 API snapshot at release tag | release branch |
| P2 | Notebook smoke receipts expansion | post-alpha |
