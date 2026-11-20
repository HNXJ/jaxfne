# Goal-match scorecard: JaxFNE and OCMA (2026-11-12)

**Status:** provisional, evidence-limited planning estimate; not an acceptance verdict, a CI gate, or a claim that the scored experiments were rerun. Baselines: jaxfne `dev@342279db791a005e556cb52c6894970b0ed9c940`; ocma `dev@ee020fa382a00524bd3999fed77e6f4618d5a8ca`. GitHub files read directly; local F:/ and E:/ receipts, private unpushed changes and execution environments not inspected. Scores are analyst assessments, not owner-approved release criteria.

## Scoring contract

Question: how much of each project's **stated final objective** has evidence-backed coverage today? Each dimension is rated 0–100: **0** = no demonstrated required behavior; **100** = every declared mandatory check demonstrated at matching source revision with no known material defect in checked scope. Intermediate ratings estimate demonstrated coverage, not probability. UNKNOWN checks receive zero *completion* credit within the dimension; a missing test is not a failure of the scientific hypothesis. Scores are not comparable across different project goals. The weights below are **proposed** and must be frozen before using this instrument as a longitudinal metric. No hidden CI or external-data inference.

`S = sum(weight_i * rating_i) / 100`, with integer percentage weights summing to 100. Scores rounded to nearest integer. Any hard release/acceptance gate still fails independently of S. Uncertainty: qualitative and substantial, because no new gate or full artifact audit was run. Avoid interpreting differences under ~10 points as measured progress.

### JaxFNE: 52/100 provisional

| Goal component | Weight | Rating | Weighted points | Evidence and missing check |
|---|---:|---:|---:|---|
| Active 0.5.10 release | 20 | 24 | 4.8 | Release plan records 24/100 (2026-11-08), not independently rescored; current-SHA broad/RC unknown |
| G-0 manuscript and figure acceptance | 25 | 45 | 11.25 | F1–F8/S1–S16 plan and generated figures exist; C1–C12 all-green, fresh-clone regeneration, external review and arXiv submission not demonstrated |
| Scientific replication/qualification | 20 | 55 | 11.0 | Jansen-Rit P3 5/5 and CAMINOS P6 bounded PASS; PCL R5-exact 4/16 FAIL, GSDR fit FAIL, Destexhe FAIL, CAMINOS E1 20/24 not equivalent; field calibration bounded |
| Executable reliability and semantics | 20 | 80 | 16.0 | Recorded broad 8263 PASS at 9020f50e; latest Fast CI PASS at current HEAD; RNG permutation, surrogate fitness, sparse STDP and matched backend gates open |
| G-1–G-4 release evidence, jnwb, dashboard | 15 | 60 | 9.0 | G-2 bounded P6 PASS; release scripts/dashboard exist; jnwb fixture test and current deployment freshness not independently inspected |
| **Total** | **100** | | **52.05 → 52** | **Not a release score** |

Hard gates open: 0.5.10 package/RC closure, exact candidate broad suite, independent review, human release go; G-0 C1–C12 and submission. The goal-stack reference cap conflicts with the 2026-10-28 paper-plan amendment removing the cap; preserve the owner goal, follow the later paper-plan amendment for paper execution, seek owner reconciliation if changing G-0.

### OCMA: 55/100 provisional

| Goal component | Weight | Rating | Weighted points | Evidence and missing check |
|---|---:|---:|---:|---|
| G-1 reference audit coverage and agreement | 30 | 78 | 23.4 | 117 inventoried, 88 in scope; independent alpha H1/H2/H3 0.844/0.880/0.815 above shuffled p95 0.283/0.106/0.284; some citations null; full current-SHA reproduction not run |
| G-2 manuscript submission readiness | 25 | 20 | 5.0 | `paper/main.md` draft and claim table; `paper/main.tex` build + fresh critic ACCEPT not demonstrated; claims 2–4 hypotheses |
| Cross-family and causal validity | 20 | 65 | 13.0 | Round-2 25-row sensitivity and GLM family arm recorded; round-4 88-row swap not run; agreement is not truth, human ground-truth accuracy and defect detection unknown |
| W8 work-mode verified utility | 15 | 65 | 9.75 | Round 3: 4 tasks, 12 agents; one write task red/green + keeper merge, read tasks checked; held-out benefit/cost versus simple baseline unknown |
| Operational reproducibility and control | 10 | 35 | 3.5 | `state.md` records 14 deleted tracked local files, hold on push/clone work; Cursor CLI blocks round 4; remote clean does not establish local recovery |
| **Total** | **100** | | **54.65 → 55** | **Not a paper score** |

Hard gates open: local incident recovery and explicit hold clearance; exact-input rerun; round-4 independent arm or a recorded decision to stop; G-2 manuscript build and fresh critic ACCEPT. `open_stack.md` contains R4-Q and HOLD despite `state.md` saying R4-1 ruled: reconcile the duplicate decision status without overwriting the recorded ruling.

## Unified next actions and closing checks

| Priority | Owner/repo | Action | Evidence required / stop |
|---|---|---|---|
| 0 | OCMA owner/local keeper | Resolve byod HOLD, inventory deleted files and recover without destroying surviving work; verify Defender status and tool integrity | Owner clears HOLD, restore/test receipts and exact source SHA; **no OCMA remote write while hold remains** |
| 0 | JaxFNE keeper | Finish PCL R5-approximate output and score, attribute P-055 at first divergent complex update; preserve signed R5-exact FAIL | Pinned R5 result/score JSON and minimal differential trace, or explicit UNKNOWN with blocker |
| 0 | JaxFNE B1 lane | Reconcile HH OU sigma units, Jaxley channel parameters, E_L and dt/duration; pilot matched model before scoring | Independent parameter/unit equivalence, smoke and signed benchmark receipts |
| 1 | OCMA methods | Resume R4-1 only after hold and CLI unblock; compare mixed A/B/G vs original A/B/C on 88 rows with 1000 shuffled draws | Per-group delta alpha, uncertainty/limits, validation receipts; no post hoc model choice |
| 1 | OCMA paper | Resolve claim table: 1 from reproducible full-scope evidence; 2–4 demonstrated or removed | Source-to-claim ledger, `main.tex` build, independent critic |
| 1 | JaxFNE paper | Complete 0.5.10 then 0.5.11 F8/S15/C1–C12, respecting negative evidence | Existing signed criteria, fresh-clone figures, exact SHA and human submission |
| 2 | Both | Controlled W8/OCMA-vs-baseline defect experiment: known red bugs, planted mutants, clean cases, blinded allocation, held-out tests, equal budgets | Freeze sensitivity, precision, verified repair, regressions, cost and stopping criteria **before** execution |
| 2 | JaxFNE | 0.5.12–0.5.15 proposed audit, numerical semantics, thinning and integration | Existing roadmap's explicit gates; no scope expansion or automatic 0.6.0 delay |

## Decision and execution rules

- The JaxFNE roadmap through 0.5.15 is a **proposal**, not a change to 0.5.10/0.5.11/0.6.0 acceptance authority. No release gate is waived by an aggregate score.
- OCMA `open_stack.md` has an active local hold; do not push, clone, or run affected OCMA work until the owner-authorized closing check. A GitHub-only planning proposal can be reviewed in JaxFNE without changing OCMA.
- Agent ratings are observations, not independent authority. For a code change: base red → smallest causal fix → green + adversarial control → non-writer review → exact SHA gate → permitted push.
- Maintain <=3 active lanes, one writer per file set; route mutable tasks to each repo's own todo stack only when authorized. Preserve negative/null results, frozen protocols, and all owner goal/fact stacks.
- Rescore only after collecting the same rubric's required receipts; publish per-component deltas, uncertainty, and changed SHA, never only the headline.
