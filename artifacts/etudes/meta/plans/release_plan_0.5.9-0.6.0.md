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
Score on 2026-11-03 at dev ba788393: 63.3/100, unchanged (63.3 at fd5a9c27, 58.7 at 702c1d61, 52.7 at dceb49b0, 48 at ff12072c, 40 at 8da66a42, 28 at 4ce58b4c).

| Id | Weight | Evidence 2026-10-31 | Score |
|---|---|---|---|
| 9A Language | 12 | `definition_map.md` rows all name a symbol and a test; `supplement.py` regenerates S1 (35 of 35 mapped), S2 (2 calibration rows, 0 failed) and S5 in one command; F1 re-renders RGBA-identical in `run_test_gate.py publication`. The specializer `PV[model=IZH]` (T-45) is outside the "Done when" check | 12 |
| 9B H-state | 10 | `tests/test_d34_typed_coupling.py`: the neutral reduction and the composition check fail under a bypass and a map-dropping mutant (hand-applied); S5 regenerates for 7 rules; F3 (fig06) re-renders RGBA-identical | 10 |
| 9C Emitters | 10 | `fixed_indegree` in `jaxfne/connectivity.py`; Izhikevich P6 v2 signed (Hamm 2026-10-31) and scored PASS, Q1-Q5 on seeds 2-6 (`RESULT_P6.md`, `core/p6_v2_score.json`); E1 closed in `check_replications.py` (the core runner calls `construct` and `.simulate`) | 10 |
| 9D Circuits | 12 | Potjans P3 v2 signed (Hamm 2026-11-02) and scored off on seeds 4-8 (`potjans2014/RESULT_P3.md`, 42e7f618): 1 of 2 checks. CAMINOS: the Cell Reports core port open (T-30; lanes a-d landed); equivalence E1 (TOST ±5%) at the confirmatory n 12 not equivalent, 20 of 24 cells (`caminos/RESULT_FIGURES.md`, ba788393); Hamm kept the oracle 2026-11-03, so the S2 entry stays (exception or move is Hamm's) | 6 |
| 9E Plasticity, fit | 14 | GSDR P3 fit signed (Hamm 2026-11-01) and scored FAIL, 0 of 5 seeds (`RESULT_P3_fit.md`), shown in S14 (`supplement.py write_s14`). Vogels P3 v4 signed (Hamm 2026-11-02) and scored off on R-E, 4.13 Hz against 3 Hz (`vogels2011/RESULT_P3.md`), shown in S12 (`write_s12`, fd5a9c27): 2 of 3 papers. PCL: R2 oracle equals the C++ reference on weights (CALIBRATION R2); R3 core arm equals the C++ after file 0 (CALIBRATION R3, b829cef0); R4 prereg open | 9.3 |
| 9F Mechanics | 8 | `check_junit_parity.py` in the gate; `.github/figure_env_lock.txt`; `publication` job in `release_ci.yml`; rc gate PASS end to end at dev 5934aac1 (2026-11-02, Release CI run 41488687157 JUnit reused; attestation `F:/warehouse/jaxfne/rc_5934aac1/rc_gate_attestation.json`, sha256 715095b8…), repeat on the candidate | 6 |
| 9G Carried | 4 | dashboard schema `jaxfne-replications/2` with receipt hashes, dark twins, the replications page check and the theme are in place (agy sweep, observed); changelog: each of the 55 `jaxfne/` commits since v0.5.8 maps to a fragment or changes no behavior (Muse sweep 2026-10-31; its two unmatched commits read by hand: 9c9cd3af docstring only, ac29227a gained `evidence_dirs_under_temp.changed.md`); assembled section 26 of 30 ghost hits | 4 |
| Gate 1 packages closed or moved | 6 | 9D, 9E and 9F open | 0 |
| Gate 2 broad, CI, replications, dashboard | 8 | broad gate exit 0 and CI (Fast) success on 4ce58b4c; `check_replications.py --check` 0 unlisted; dashboard `--check` fresh; repeat on the release SHA | 6 |
| Gate 3 `release_body.py --check` | 4 | needs the release-day steps: version 0.5.9 in `pyproject.toml`, then `assemble_changelog.py --version 0.5.9 --date <day>` | 0 |
| Gate 4 second review, receipt | 6 | not started | 0 |
| Gate 5 tag, main, PyPI | 6 | Hamm's word | 0 |

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
