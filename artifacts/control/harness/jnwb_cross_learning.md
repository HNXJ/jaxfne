# jaxfne and jnwb: what each can take from the other (2026-10-12)

Sources: an opencode read-only sweep of jnwb at 9fd2e7e0 (each item checked against its test
file), greps of both trees, and jaxfne at 2b446fab. jnwb analyses recorded data; jaxfne
simulates it. They already meet at one interface.

## Shared usage

| What | Where | State |
|---|---|---|
| Simulation output to jnwb arrays | `jaxfne.jnwb_view.to_jnwb` | `tests/test_jnwb_view.py` imports jnwb when installed; 25 passed with jnwb 0.2.10 |
| Simulation output to NWB and back | `jaxfne.pynwb_compat.write_nwb`, `read_nwb` (through `jnwb.read_nwb`) | proxies keep `RELATIVE_PROXY` as unit; never written as `ElectricalSeries` |
| Ground truth for jnwb estimators | CAMINOS p = 0.07 seed 1 excerpt (G-3, O-2) | waiting for jnwb to name the path |
| Reference format | `docs/reference/references.md` in jnwb's form since 2b446fab | both pages machine-checked against docstrings |

Gaps at the interface: jnwb's docs and code never mention jaxfne, and the venv carries jnwb
0.2.10 while jnwb is at 0.2.11. The venv's editable `jaxfne` points at `E:\repos\jaxfne-theory`,
so a script run outside pytest imports that tree; pytest imports the tree under test.

## jaxfne takes from jnwb (ranked by value per cost)

| # | Practice | jnwb evidence | Why jaxfne needs it |
|---|---|---|---|
| 1 | Changelog fragments assembled at release | `tests/test_changelog_fragments.py`, `scripts/assemble_changelog.py` | Lanes are cherry-picked into one branch, and every lane edits `docs/reference/changelog.md` |
| 2 | One newline policy, tested on generated files | `tests/test_generated_files_are_byte_stable.py` | `docs/reference/changelog.md` holds 653 CRLF and 5 LF lines; docs mix CRLF and LF files |
| 3 | Errors page with a test that every exported error is on it | `docs/errors.md`, `tests/test_errors_documented.py` | jaxfne refuses many inputs by name and documents none of the refusals in one place |
| 4 | Claim-wording gate | `tests/test_claim_wording.py` | jaxfne's equivalent risk is calling a proxy a measured or calibrated quantity |
| 5 | Frozen function bodies by AST hash | `tests/test_frozen_validated.py`, `artifacts/frozen_validated.json` | Kernels validated bit for bit (AdEx, Izhikevich) can change without notice; jaxfne freezes files, not functions |
| 6 | Findings ledger with a closed set of dispositions | `tests/test_findings_ledger.py` | The rule "fix or record every review finding" has no mechanical check |
| 7 | Gates tested on trees they must reject; every gate runs after one fails | `tests/test_gates_reject_the_trees_they_passed.py`, `tests/test_every_gate_runs.py` | The lesson "test a gate on a known positive" is applied by hand |
| 8 | CI conclusion gate: cancelled, skipped or absent legs fail | `tests/test_ci_conclusion_gate.py` | jaxfne reads CI by hand through `ci_runs.py` |

Not worth porting now: the 22-gate harness runner and its meta-tests (weight), and the pixel
tolerance plus font hash for figures (brittle across OS).

## jnwb takes from jaxfne

| # | Practice | jaxfne evidence | Why jnwb could use it |
|---|---|---|---|
| 1 | Preregistered replications scored from receipts, shown with four classes | `dashboard/sources.json`, `scripts/build_dashboard_data.py`, `artifacts/etudes/*/PREREG_*.md` | jnwb validates estimators against closed forms and oracles; a preregistered replication of a published analysis (e.g. the Bastos 2018 crossover) tests them against data |
| 2 | Allowlist of consumed fields; every other field refused by name, and each consumed field shown to change the output end to end | `jaxfne/_construct_adex.py` (`ADEX_CONSUMED`), `tests/test_adex_route.py` | Extends jnwb's silent-no-op test from bad values to accepted options that do nothing |
| 3 | Figures checked by a hash of their chart data | `scripts/build_dashboard_figures.py --check` | Replaces pixel tolerance and the font hash for figures whose data is the claim (jnwb weakness W4) |
| 4 | Tracked-tree size budget with an allowlist | `scripts/check_tree_size.py`, `scripts/tree_size_allowlist.json` | jnwb has no size gate |
| 5 | Bit-for-bit transcription tests against an author's reference code | `tests/test_emitters_adex.py` | For ports of published pipelines, exact equality settles what tolerance cannot |
| 6 | A page for simulated input | `jaxfne.jnwb_view`, `jaxfne.pynwb_compat` | jnwb's docs do not say how to analyse jaxfne output or what a proxy unit means |

## Next steps

- jaxfne: items 1 and 2 first (they remove churn in every lane), then 3 and 5; add each to the
  todo stack with its check.
- jnwb: offered through jchat; jnwb's keeper decides.
- Shared: upgrade the venv's jnwb when jnwb tags 0.2.11, and land the G-3 excerpt.
