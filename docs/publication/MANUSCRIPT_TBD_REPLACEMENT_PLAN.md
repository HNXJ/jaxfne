# Manuscript TBD replacement plan

Manuscript PDF is external to this repo. This table maps placeholder slots to alpha receipts. **Do not invent** tags, DOI, wheel hashes, or empirical datasets.

| Manuscript section | Current placeholder | Replacement source | Blocking receipt |
|---|---|---|---|
| 2.1 demonstrated behavior | release tested version/SHA/tutorial notebooks | `docs/publication/JAXFNE_ALPHA_HANDOUT.md`, `outputs/publication/inventory.json` | Tasks 01/02 |
| 2.4 validation/testing | compileall/pytest/mkdocs/execution counts | `outputs/publication/qa_logs/`, Task 07 | Task 07 |
| 2.5 software availability | repo URL/tag/DOI/install | ED10 receipt + approval decision | Task 08 |
| A.6 checklist | install/deps/runtime/tests/docs | `publication_checklist.json`, output manifests | Tasks 02/07 |
| Field/proxy scope | laminar proxy / no PDE | truth gates, `docs/scope_and_limitations.md` | glossary §1 |
| Physical amplitude | requires calibration + solver evidence | `physical_amplitude_claim_allowed: false` | ED2/ED9/ED10 |

## Wording posture for manuscript edits

Use: computational scaffold, proxy readout, local receipt, release pending approval.

Avoid unless evidenced: real EEG/MEG, calibrated amplitude, mechanism proof, Poisson/Maxwell solver, empirical validation.
