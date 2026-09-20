# MEMORY.md (jaxfne) - verified reusable lessons only

Scope: this project only. Never store current state, secrets, or unverified claims.
Format per lesson: {trigger,cause,repair,evidence,scope}.

## Lessons

- trigger: edit applied to large file, diff far bigger than intended
  cause: tool_requires exact oldString; parallel same-file edits and formatter churn (P-004) slip through success messages
  repair: re-read every edited region; check `git diff --stat` scale matches intent; unattributed reflow gets a P-ID until proven harmless
  evidence: P-004 (generate_doc_page_atlases.py 450-line reflow; ruff+regen proved harmless)
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
