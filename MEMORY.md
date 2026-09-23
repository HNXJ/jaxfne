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

- trigger: pre-existing stash entry in `git stash list`
  cause: bare `git stash pop` takes the top entry regardless of ownership; unknown-ownership stashes predate the session
  repair: never bare-pop; treat pre-existing stashes read-only; if one is popped by mistake, `git reset --hard HEAD` recovers only when the tree was sealed+pushed (verify HEAD==origin/dev and status 0 first)
  evidence: P-008 (conflicted jaxfne/vis/__init__.py; clean reset, stash preserved, import smoke OK)
  scope: jaxfne (git)
