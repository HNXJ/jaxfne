# Release plan 0.5.9–0.6.0

Hamm 2026-10-26: 0.5.9 packages follow the arXiv plan
(`artifacts/programme/design/plans/arxiv_paper_plan.md`); 0.6.0 is the post-peer-review acceptance
release, hard and complete. Replaces §4 of `artifacts/etudes/meta/release_plan_0.5.7-0.5.10.md`; its
§5 (execution) and §6 (exit gate) stay. Tags, main and PyPI need Hamm's word per release.

| Release | Theme | Scope |
|---|---|---|
| 0.5.9 | Language, H-state, replications | F1, F3, F5–F7; S1, S2, S5, S8–S14; C3, C5, C8 |
| 0.5.10 | Geometry, compute, extras | F2, F4; S3, S4, S6, S7, S16; C4, C6, C7 |
| 0.5.11 | Submission | F8, S15; manuscript, refreeze, review, arXiv; C1, C2, C9–C12 at 100 |
| 0.6.0 | Acceptance | peer-review revisions; the exit gate below |

Re-homed: Potjans 2014 to 0.5.9; AT-10-R6, the LIF GPU kernel and Mejias 2016 to 0.5.10. Lee,
Pennartz and Mejias 2025, Mazzoni 2015 and EEG validity name no figure and wait for an amendment.

## 0.5.9 packages

| Id | Package | Work | Done when |
|---|---|---|---|
| 9A | Language (F1, S1, S2, C3) | 8 open rows of `definition_map.md` (D17, D21, D26, D28, D31, D32, D34, D35); F1, S2 generators | each row names a public symbol with a test; F1, S1, S2 regenerate in one command |
| 9B | H-state (F3, S5, C5) | typed coupling check on D34; S5 per rule; F3 | a coupling-bypass mutant fails; F3, S5 regenerate |
| 9C | Emitters (F5, S8, S9, C8) | Brunel S1 (`fixed_indegree`); Izhikevich E1: per-edge uniform weights, per-cell Gaussian noise on `construct`, then the library-path prereg | E1 finding closed; scored under signed preregs |
| 9D | Circuits (F6, S10, S11, C8) | Potjans P3; CAMINOS scored sweeps rerun on the core route | caminos S2 findings closed; Potjans scored |
| 9E | Plasticity, fit (F7, S12–S14, C8) | Vogels rerun and P3; PCL R2–R5 (R4 prereg signed first); GSDR pilot with MCDP, prereg, P3 | F7 shows passes; each failure in its S figure |
| 9F | Mechanics (C2, C9) | JUnit parity classifier; figure lock file; clean-checkout figure job | rc gate end to end on the candidate |
| 9G | Carried | jnwb dashboard schema and receipt hashes (C8); dark twins, page test, theme, docs leftovers | nothing blocks 0.5.9 |

Agent work starts now (9F, E1 core, generators, PCL design study, pilots, CAMINOS rerun); scored
runs wait for signatures. At most 3 lanes.

| Hamm | Decision | Recommendation |
|---|---|---|
| H1 | F1–F8 against frozen fig01–fig07 | Ruled 2026-10-26: F1 fig01, F3 fig06, S15 fig05; fig02–04 and fig07 stay frozen evidence outside the paper |
| H2 | 9A names | Ruled 2026-10-26: D21 maps to the existing public `resolve_edge_weight`; D34 becomes public `rbd_recurrent_gain` |
| H3 | Signatures | Open: the Potjans and Vogels preregs are templates; estimands and pilots first, then signature; GSDR, Destexhe after pilots; Izhikevich E1; PCL-R4 |
| H4 | C1 numbering CL-22 to CL-26 | Ruled 2026-10-26: the ledger keeps CL-1 to CL-26; E2 claims are CL-27 to CL-34 |
| H5 | LICENSE name, data statement, data licences | Ruled 2026-10-26: LICENSE names Hamed Nejat; third-party inputs are cited by DOI and licence and never redistributed |

## 0.5.9 checklist and score

Weights sum to 100: packages 70, exit gate (`release_plan_0.5.7-0.5.10.md` §6) 30. A row earns its
weight only when its "Done when" check holds; partial credit is the share of its named checks that hold.
Score on 2026-11-04 after publication: 100/100 (Gate 5 0 to 6: v0.5.9 tagged on main e0160dfc, Release, PyPI). 94.0 after the rc gate on bb3680d5 (9F 6 to 8, Gate 1 4 to 6, Gate 2 6 to 8); 88.0 after the candidate and its second review: Gate 3 0 to 4, Gate 4 0 to 6; 78.0 after R5-exact, S13 and F7: 9D 6 to 12, 9E 9.3 to 14, Gate 1 0 to 4; 69.0 before F7; 63.3 at ba788393, 63.3 at fd5a9c27, 58.7 at 702c1d61, 52.7 at dceb49b0, 48 at ff12072c, 40 at 8da66a42, 28 at 4ce58b4c).

| Id | Weight | Evidence 2026-10-31 | Score |
|---|---|---|---|
| 9A Language | 12 | `definition_map.md` rows all name a symbol and a test; `supplement.py` regenerates S1 (35 of 35 mapped), S2 (2 calibration rows, 0 failed) and S5 in one command; F1 re-renders RGBA-identical in `run_test_gate.py publication`. The specializer `PV[model=IZH]` (T-45) is outside the "Done when" check | 12 |
| 9B H-state | 10 | `tests/test_d34_typed_coupling.py`: the neutral reduction and the composition check fail under a bypass and a map-dropping mutant (hand-applied); S5 regenerates for 7 rules; F3 (fig06) re-renders RGBA-identical | 10 |
| 9C Emitters | 10 | `fixed_indegree` in `jaxfne/connectivity.py`; Izhikevich P6 v2 signed (Hamm 2026-10-31) and scored PASS, Q1-Q5 on seeds 2-6 (`RESULT_P6.md`, `core/p6_v2_score.json`); E1 closed in `check_replications.py` (the core runner calls `construct` and `.simulate`) | 10 |
| 9D Circuits | 12 | Potjans P3 v2 signed (Hamm 2026-11-02) and scored off on seeds 4-8 (`potjans2014/RESULT_P3.md`, 42e7f618): 1 of 2 checks. CAMINOS: the Cell Reports core port open (T-30; lanes a-d landed); equivalence E1 (TOST ±5%) at the confirmatory n 12 not equivalent, 20 of 24 cells (`caminos/RESULT_FIGURES.md`, ba788393); Hamm kept the oracle 2026-11-03 and granted permanent exceptions for the `cellrep_figures.py` and `attribute_o83.py` S2 entries, so no CAMINOS S2 finding is outside Hamm's exceptions (`check_replications.py --check`: 0 unlisted) | 12 |
| 9E Plasticity, fit | 14 | GSDR P3 fit signed (Hamm 2026-11-01) and scored FAIL, 0 of 5 seeds (`RESULT_P3_fit.md`), shown in S14 (`supplement.py write_s14`). Vogels P3 v4 signed (Hamm 2026-11-02) and scored off on R-E, 4.13 Hz against 3 Hz (`vogels2011/RESULT_P3.md`), shown in S12 (`write_s12`, fd5a9c27). PCL: R4b signed (Hamm 2026-11-03); R5-exact scored FAIL, 4 of 16 files (`pcl/prereg/P3_fig3_r5_exact_result.md`), shown in S13 with the signed v4 and v5 failures (`write_s13`); R5-approximate moved to 0.5.10 (Hamm 2026-11-03). Checks: each failure in its S figure holds (3 of 3 papers); F7 shows the passes (`fig07_validation.py`: PCL T1, T3, T4 under v5, Vogels CONV and O-DROP, 11 of 11 criteria recomputed from the score JSONs; `tests/test_fig07_validation.py`) | 14 |
| 9F Mechanics | 8 | `check_junit_parity.py` in the gate; `.github/figure_env_lock.txt`; `publication` job in `release_ci.yml`; rc gate PASS end to end at dev 5934aac1 (2026-11-02, Release CI run 41488687157 JUnit reused; attestation `F:/warehouse/jaxfne/rc_5934aac1/rc_gate_attestation.json`, sha256 715095b8…); on the 0.5.9 candidate bb3680d5: RC GATE PASS 2026-11-04 (Release CI run 41720575544 JUnit reused, tree d162c31d; wheel smoke 0.5.9; attestation `F:/warehouse/jaxfne/rc_bb3680d5/rc_gate_attestation.json`, sha256 f52bdb34…, commit bb3680d5, working tree clean) | 8 |
| 9G Carried | 4 | dashboard schema `jaxfne-replications/2` with receipt hashes, dark twins, the replications page check and the theme are in place (agy sweep, observed); changelog: each of the 55 `jaxfne/` commits since v0.5.8 maps to a fragment or changes no behavior (Muse sweep 2026-10-31; its two unmatched commits read by hand: 9c9cd3af docstring only, ac29227a gained `evidence_dirs_under_temp.changed.md`); assembled section 26 of 30 ghost hits | 4 |
| Gate 1 packages closed or moved | 6 | 9A-9G closed: 9D and 9E closed 2026-11-03 (PCL R5-approximate and P-052 moved to 0.5.10 by Hamm 2026-11-03), 9F closed 2026-11-04 (rc gate PASS on bb3680d5) | 6 |
| Gate 2 broad, CI, replications, dashboard | 8 | On the release SHA bb3680d5: Release CI run 41720575544 success (broad, slow and notebook pytest on Python 3.11 and 3.14), CI (Fast) success; `check_replications.py --check` 28 known, 0 unlisted, 0 stale; `build_dashboard_data.py --check` fresh. The local broad gate at 190432a4 failed 7 tests on 120 s example timeouts under the R5 load (passed alone) and one stale doc exemption (fixed in bb3680d5) | 8 |
| Gate 3 `release_body.py --check` | 4 | 0.5.9 candidate (2026-11-03): version 0.5.9 in `pyproject.toml` and the version files; `assemble_changelog.py --version 0.5.9 --date 2026-11-03` assembled 83 fragments; `release_body.py --tag v0.5.9 --check` PASS | 4 |
| Gate 4 second review, receipt | 6 | opencode Muse review of the 0.5.9 candidate 190432a4 (2026-11-03): NON-BLOCKING, 0 blocking, 1 should-fix (an `in_degree` refusal message, moved to 0.5.10, todo 30), 2 notes; every Changed and Fixed bullet and the first 10 Added read against `jaxfne/`, none wrong. Receipt `F:/warehouse/jaxfne/rc_190432a4/second_review_muse.md` (sha256 4d07f396…) with its packet (b6586f4d…) | 6 |
| Gate 5 tag, main, PyPI | 6 | Hamm's go 2026-11-04. Main CI failed on two missing test-gate imports (PyYAML, Markdown), each fixed with a failing-first test (PRs #121, #122); main CI green on e0160dfc (CI (Fast) 41922217570, Release CI 41922217824); tag v0.5.9 (e621eff7) on e0160dfc; GitHub Release; PyPI by publish run 41935856859, sha256 equal to the release-dist bytes (wheel 6e962c1e…, sdist 2aef7eac…); fresh-venv install imports 0.5.9. Receipt `artifacts/release/v0_5_9_release_receipt.json` | 6 |

## 0.5.10 packages and checklist (draft 2026-11-05, for Hamm's review)

Written from the 0.5.10 scope below, the arXiv plan rows F2, F4, S3, S4, S6, S7, S16, C4, C6, C7, and the todo rows
moved to 0.5.10. Weights sum to 100 as in 0.5.9: packages 70, exit gate 30 (Gates 1-5 with the 0.5.9 weights).
Scoring: a package's weight splits equally over its listed items; an item scores when its done-when check is cited.

Score 2026-11-05: 8.5/100 (packages 8.5 of 70, gates 0 of 30).

| Id | Weight | Package | Done when | State 2026-11-05 | Score |
|---|---|---|---|---|---|
| 10A Geometry (F2, S3, S4, C4) | 14 | Atlas AT-10-N20, NeuronalTensor schema and flattening, geometry property tests, the C4 turn-invariance wording | S3 tests fail on a broken invariant (round trip, permutation, rigid transform); F2 renders from AT-10-N20 | C4 radius test landed (de86aa67, `tests/test_geometry_radius_rotation.py`); atlas, schema item and S3 panel open | 3.5 |
| 10B Compute (F4, S6, C6) | 14 | benchmark prereg; CPU and GPU receipts; Brian2, NEST and Jaxley baselines; the `build_config` repair; the LIF GPU kernel | receipts hash-pinned; baselines run where the model class applies | C6 inventory only; no baseline run | 0 |
| 10C Augmentation (S7, C7) | 8 | ensemble study with provenance | two seeds change output; the same seed reproduces | not started | 0 |
| 10D Extras (S16) | 14 | Mejias 2016 G6; Destexhe 2009; kernel LFP (T-47: `6eb3c6b8`, `44cd0111`, `ff45364b` from `_temp_kernel-lfp`, then the iSTDP out-degree check and `coherence_radius_um`); Jansen-Rit neural-mass emitter and its P1 (T-48) | each on dev with its tests; S16 regenerates | kernel-lfp commits unpicked; Jansen-Rit not started | 0 |
| 10E PCL Fig 3 | 10 | R5-approximate under signed R4b, scored in `prereg/`; P-055 attributed | result file and score JSON pinned | R5 approx_core running: 920 of 2562 files (`F:/warehouse/pcl/r5/approx_core.log`, 2026-11-05) | 0 |
| 10F Carried | 10 | AGSDR log-scale bounds (T-31); T-32 open rows (GSDR on `simulate_hh_batch`, seed and candidate batching, core homes for lane analysis); T-R28 pieces (jitter reader, S3 panel); the `in_degree` refusal message (Gate 4 should-fix); P-BALANCED (`BackgroundNoise`, `baseline_drive`, the balanced-1000n preset) | each with its test or receipt cited in the todo row | `in_degree` message done (9e7c2044); P-BALANCED ported (e179a2d0..9dd1c111, `tests/test_balanced_preset_acceptance.py` 6 passed; Hamm 2026-11-05: retune under the unchanged bounds; seeds 0-4 CV 0.742-0.755, shift 0.236-0.257); T-31 log-scale bounds landed (733b3960), the GSDR lane's log-scale parameters not named (half); T-32 and T-R28 open | 5 |
| Gates 1-5 | 30 | as in 0.5.9 | as in 0.5.9 | not started | 0 |
| Owner | 0 | P-052 surrogate fitness; T-62 Fig 1D raw-data request; a named figure for Lee 2025 and Mazzoni 2015; approval of these packages and weights | Hamm rules on each | open | - |

## Later releases

- 0.5.10: F2 with AT-10-N20 and the C4 turn-invariance wording; F4 with the benchmark prereg, Brian2,
  NEST and Jaxley baselines, the `build_config` repair, the LIF GPU kernel; the augmentation study;
  S16 (Mejias, Destexhe, kernel LFP, Jansen-Rit). Packages written at the 0.5.9 seal.
- 0.5.11: F8 and S15 against the dashboard data; manuscript with at most 80 DOI references and
  audited figures; refreeze and fresh-clone figure build; second reviewer, then an external reader;
  arXiv build. Submission on Hamm's word.
- 0.6.0 exit gate (proposed, Hamm amends): every review point answered; C1–C12 rechecked on the
  accepted text; 0 open issues; no replication finding outside Hamm's exceptions; every skip and
  xfail with a reason; rc gate end to end; figures, docs and dashboard rebuilt from a clean tag.

One Haiku 5.5 subagent per lane in its own clone; the dispatcher keeps science, verdicts and pushes.
Heavy runs on Colab. Main figures carry no negative results. Missed work moves on with its reason.
