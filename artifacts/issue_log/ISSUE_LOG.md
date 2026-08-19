# jaxfne issue log — 40-day frozen-use period

Active issue log for the v0.4.17 frozen period (approximately 2026-08-19 to
2026-09-28). During this period the released core is frozen:

```
observe -> reproduce -> log
```

not

```
observe -> patch core.
```

Every entry records at least:

```text
date
type
area
observation
severity
minimal reproduction
expected behavior
actual behavior
evidence
possible future change
```

Types: `BUG` `FRICTION` `DOC` `PERF` `SCIENCE` `IDEA`.

Core bugs are accumulated here and are NOT repaired during the 40-day period.
At the end of the period, this log becomes the evidence for the next
development cycle. A severe issue may be proposed as an emergency patch but
requires separate authorization.

---

## Open issues

### I-001
- **date:** 2026-08-19
- **type:** DOC
- **area:** public surface / release evidence
- **observation:** `jaxfne/public_surface.py:105` contains a stale internal
  comment reporting "259 symbols at 0.4.13 baseline"; the frozen 0.4.13 set
  was actually 260.
- **severity:** MINOR (cosmetic comment; the live 191 public-export claim in
  docs + contract + `__all__` is correct)
- **minimal reproduction:** `grep -n "259" jaxfne/public_surface.py`
- **expected behavior:** comment matches frozen 0.4.13 count or is removed
- **actual behavior:** comment says 259
- **evidence:** final-hostile-review 5bc0400, check A/B; verified at commit
  0678557 that the frozen set was 260
- **possible future change:** correct comment at next code-touch (not during
  frozen period)

### I-002
- **date:** 2026-08-19
- **type:** DOC
- **area:** release receipt / provenance
- **observation:** `artifacts/release/v0_4_17_release_receipt.json` records
  `branches.dev/main = cd5738a` (seal-time truth). `dev` now points at
  5bc0400 (intentional docs-only delta); `final_candidate=cd5738a` unchanged.
- **severity:** MINOR (receipt is write-once by design; not a core
  contradiction)
- **minimal reproduction:** compare `git rev-parse dev` with
  `branches.dev` in the receipt
- **expected behavior:** receipt is immutable; branch snapshots are
  seal-time truth
- **actual behavior:** branch snapshot differs from current dev HEAD after a
  docs-only commit
- **evidence:** final-hostile-review check F (receipt branches snapshot note)
- **possible future change:** final release receipt should snapshot the exact
  release commit/tag after S2 completes

### I-003
- **date:** 2026-08-19
- **type:** DOC
- **area:** build artifacts / packaging
- **observation:** untracked `dist/` holds stale 0.3.42 wheels. The 0.4.17
  wheel/sdist referenced by the release receipt are build-gate outputs, not
  committed. A rebuilt 0.4.17 wheel from HEAD is clean and leak-free.
- **severity:** MINOR (no release blocker)
- **minimal reproduction:** `ls dist/`
- **expected behavior:** 0.4.17 wheel/sdist published from the release build
- **actual behavior:** stale 0.3.42 wheels in `dist/`; 0.4.17 built per gate
- **evidence:** final-hostile-review check C; clean-room build log
- **possible future change:** publish the 0.4.17 wheel/sdist in S2; consider
  cleaning or ignoring `dist/`

### I-004
- **date:** 2026-08-19
- **type:** DOC
- **area:** provenance / observed values
- **observation:** value `1.2165` is an observed `m_EI` mean-weight-trace from
  a 0.4.8-era checkpoint artifact
  (`artifacts/mcc3_10s_checkpoint/mcc3_10s_metrics.json`), not a documented
  `drive_gain` result. No public doc ties it to drive_gain.
- **severity:** MINOR (no contradiction, but flagged for precision)
- **minimal reproduction:** `jq '.weights.B.m_EI.mean_W_t' <  ...metrics.json`
- **expected behavior:** observed values remain traceable to their exact
  source run
- **actual behavior:** value exists only in the old checkpoint artifact
- **evidence:** final-hostile-review check 23 note
- **possible future change:** when quoting observed values in docs, cite the
  exact checkpoint artifact and version