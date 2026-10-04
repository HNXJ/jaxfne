# MEMORY.md (jaxfne) - verified reusable lessons only

Scope: this project only. Never store current state, secrets, or unverified claims.
Format per lesson: {trigger,cause,repair,evidence,scope}.

## Lessons

- trigger: release-gate script passes locally but fails on CI fresh clone
  cause: skill doc references a gitignored runtime-generated path (scratch/CURRENT_TASK.md); existence check is environment-dependent
  repair: declared `generated_skill_refs` allowlist in doc_code_integrity_allowlist.json; gate skips listed paths; proved via file-absent simulation + CI green
  evidence: 0.5.0 closure (CI Fast red on 4 dev commits → green on e1527ec)
  scope: jaxfne harness

- trigger: freezing a numeric equivalence gate with conjunctive abs+rel legs
  cause: abs leg unscaled to magnitude demands sub-ulp cross-implementation agreement, unsatisfiable by any port; rel leg alone had 41x margin
  repair: allclose form at freeze (abs governs near-zero only) encoded in tests/_numeric_gates.py; original FAIL kept immutable; human adjudication recorded
  evidence: 0.5.0 item 10 (re-evaluation 19 floats, 0 failures, worst rel 1.8e-07)
  scope: jaxfne harness

- trigger: shared numeric helper upcasts float32 sim data to float64
  cause: np.asarray(x, dtype=float) in a unification helper; changed committed figure payloads (f4->f8) beyond the intended contract
  repair: preserve input dtype in shared contracts; verify via normalized figure diff (UUID/timestamp/version-normalized, expect stamp-only bytes)
  evidence: 0.5.0 atlas regen (oscillatory +1942B caught, corrected to -1B stamp-only)
  scope: jaxfne vis

- trigger: edit applied to large file, diff far bigger than intended
  cause: tool_requires exact oldString; parallel same-file edits and formatter churn (P-004) slip through success messages
  repair: re-read every edited region; check `git diff --stat` scale matches intent; unattributed reflow gets a P-ID until proven harmless
  evidence: P-004 (450-line reflow; ruff+regen proved harmless); recurrence in test files via edit tool (P-008), caught by diff-stat review
  scope: jaxfne

- trigger: PowerShell command fails with ParserError or silent wrong results
  cause: unix aliases absent; `$env:X='v';` inline, `?` in URLs, and embedded quotes break parsing
  repair: use Select-String/Select-Object/Get-ChildItem; single-purpose calls; complex quoting goes in a temp .py file under Temp\opencode
  evidence: repeated ParserError episodes 2026-09-20, all resolved by splitting/file-ifying
  scope: jaxfne (windows host)

- trigger: dt_ms=0.1 long-run atlas builds fail INVALID_TIME_GRID
  cause: float32 time-grid drift over thousands of steps trips the uniformity gate
  repair: use binary-exact dt_ms=0.5 for doc atlases; label actual dt in run labels
  evidence: 3 regen failures → 0 after switch; gate unchanged
  scope: jaxfne docs/vis

- trigger: regenerated Plotly HTML differs byte-wise from committed copy
  cause: plotly embeds random per-render div UUIDs; manifest sha covers metadata (incl. byte counts), not content bytes
  repair: treat UUID-only diffs as identical; verify via manifest sha + linkcheck, not byte diff
  evidence: schema.html regen diffs (single_neuron etc.), ei_pop sha stable c9aa81
  scope: jaxfne docs/vis

- trigger: frozen etude bundle trajectories don't reproduce under tip tree
  cause: V_m/Q float hashes drift (JAX/XLA version unrecorded at bundle time); spikes/positions exact
  repair: two-tier check — hash tier, else claims tier (spike-exact ==, geometry-exact ==, floats within protocol 1e-3); future bundles must record JAX/lib versions (P-003)
  evidence: multiscale/heterogeneous/mcc3 reruns 2026-09-20
  scope: jaxfne etudes

- trigger: release commit red on CI after green local gates
  cause: local env (py3.14/windows, no-viz-dev split) differs from CI (py3.11/ubuntu, viz); POSIX-only tests skip locally; no log auth from here
  repair: monitor check-runs via public API; ask owner for failing test names from Actions UI; fix forward; never create the GitHub Release from a red commit
  evidence: 0.4.25 push-run 35496154417 (broad-gate step, 24 min in)
  scope: jaxfne release

- trigger: docs build green but docs lie about the API (13 signature drifts)
  cause: no mechanical check between doc blocks and live code; rendering ≠ truth
  repair: `audit_doc_code_integrity.py` as permanent `doc_code_integrity_audit`
  family in broad/release/rc gates + explicit ci.yml step; allowlist stays explicit
  evidence: P-006 round (143 signatures checked), gate green in 8s
  scope: jaxfne harness

- trigger: subagent (bounded, no-commit) work
  cause: overlapping file sets and interrupted runs leave partial/unknown tree state
  repair: disjoint file sets per worker; exact stop-report format; on interruption, `git status` + gate battery before trusting anything; workers never commit/push
  evidence: 7 worker rounds 2026-09-20 (vocab/verbosity/nav/code-audit), 2 interrupted, all integrated cleanly
  scope: jaxfne

- trigger: python text-mode file write on Windows rewrites every line
  cause: open()/write_text translate \n to os.linesep (CRLF) on write; whole-file diff with zero semantic change
  repair: binary-safe patch scripts (read_bytes/write_bytes) for existing .py files; verify via git diff --stat before commit
  evidence: 0.5.3 item 2 (w1-53): 7718/7533-line diff reduced to +194/-9, tests re-green
  scope: jaxfne (windows host)

- trigger: pre-existing stash entry in `git stash list`
  cause: bare `git stash pop` takes the top entry regardless of ownership; unknown-ownership stashes predate the session
  repair: never bare-pop; treat pre-existing stashes read-only; if one is popped by mistake, `git reset --hard HEAD` recovers only when the tree was sealed+pushed (verify HEAD==origin/dev and status 0 first)
  evidence: P-008 (conflicted jaxfne/vis/__init__.py; clean reset, stash preserved, import smoke OK)
  scope: jaxfne (git)

- trigger: consecutive per-step pushes land red on CI while local gates pass
  cause: per-step verification ran targeted test subsets only; public-surface additions (snapshot/docs counts) and lint break CI's full gate, which nothing per-step executed
  repair: after any push touching jaxfne/ exports or surface artifacts, read that push's CI result before the next step; repo is public so GitHub Actions API answers without a gh token — check Fast on the exact pushed SHA, not just local broad
  evidence: 0.5.4 (9 pushes 6792bc7..0ad95e0 red on CI, green locally; e13a511 fixed lint, 26c4299 fixed surface; Fast green confirmed on 86e5224 via public API)
  scope: jaxfne (git/CI)

- trigger: scripted patch over 100+ files reports success per file
  cause: success messages prove the script ran, not that each file is correct (hit: whole-file reformat, use-before-bind from reordered imports, double application, deleted sibling imports)
  repair: every generated edit carries machine-checkable postconditions in the applier itself (re-parse + behavior-relevant asserts such as name-binding order), idempotency via the gate's own verdict (import the gate test's functions; never a text marker), and a rerun that must report zero changes; verify with git diff --stat scale plus an independent whole-tree check, never the applier's log alone
  evidence: P-025 (119 files; 4 applier bugs caught only by diff-stat scale, gate re-runs, and binding-order asserts)
  scope: jaxfne (scripted edits)

- trigger: bit-exact float hash pinned across CI legs
  cause: float32 reductions/matmul already carry ~2e-6 relative rounding locally, so any BLAS/XLA reorder on another runner flips the hash while values agree; unpinned deps make it a hardware lottery
  repair: keep bit-pins only for exactly-representable passthroughs; for reduction/matmul leaves use twin-equality (same op and inputs: identical bytes on any backend) plus numpy-float64 agreement at the repo gate epsilons with measured margin stated; leave unproven leaves pinned and name them as watch items
  evidence: PR#97 release 3.11/3.14 flipped eeg_t then proj_source on different runners (oracle file, Oct 2026)
  scope: jaxfne (tests/CI)

- trigger: ruff E402 on imports following a sys.path guard block
  cause: assignments/conditionals in the guard count as code, but ruff exempts imports after a bare sys.path.insert expression (probe-proven); PowerShell jq quoting breaks inline filters
  repair: canonical guard is bare `sys.path.insert(0, str(Path(__file__).resolve().parents[N]))` with no assignment wrapper; parse gh JSON output with ConvertFrom-Json instead of jq in pwsh; always pass `-R HNXJ/jaxfne` to gh when the cwd is outside that repo (gh infers repo from git remote and silently queries jynx)
  evidence: P-025 (125 files, 0 new ruff findings vs HEAD under repo select)
  scope: jaxfne (windows host)

- trigger: jchat claims board still shows a claim after posting a release
  cause: the release text used a different item id than the claim text, so the board never matched them (claim `untested-tail` vs release `untested-pair-gate`)
  repair: a release reuses the claim's exact item id verbatim; verify with `claims` after posting
  evidence: untested-tail ghost claim Oct 2026, fixed by re-releasing with the matching id
  scope: jchat (claims board)

- trigger: a refusal test passes while the guarded entry point stays callable
  cause: testing the check helper directly leaves the call-site mutant alive (deleting the call inside the public function survives the test file)
  repair: every refusal gate ships an end-to-end test through the public entry point with registry-valid names for the targeted axis; kill both the check body and the call site before trusting green, restore byte-identical after
  evidence: pair-gate review (agy finding #1); call-site mutant killed by the new test, check-body mutant by the direct tests
  scope: jaxfne (tests)

- trigger: encoding a measured compatibility table from a docstring
  cause: re-reading rows as one axis and columns as the other transposes the measured set (first 7-pair set misread conductance rows/cols; also an invalid registry name in a new test hit a different refusal first)
  repair: re-read the table's axis labels explicitly, cross-check each claimed cell against the test files that pin it, and use registry-valid names per axis in tests; let the suite (not the table) be the arbiter — two existing tests caught the transposition here
  evidence: pair-gate set corrected 7 to 8 pairs via stage1/stage2 failures
  scope: jaxfne (tests)

- trigger: PR merge BLOCKED with all checks green and no review required
  cause: the protect-main ruleset sets required_review_thread_resolution, so one unresolved bot review thread blocks the merge; the bot objected to specified fail-closed behavior (UNTESTED-exact refusal) as if it were a defect
  repair: read the thread body, rebut with the stack-item + invariant citation via a review-comment reply when the code does what the authorized task specified, then resolve; verify resolution with a GraphQL reviewThreads query (comment node ids are not thread ids) and re-check mergeStateStatus before merging
  evidence: PR#98 macroscope thread on the pair-gate (rebutted, resolved, merged d77a937a)
  scope: jaxfne (git/CI)
