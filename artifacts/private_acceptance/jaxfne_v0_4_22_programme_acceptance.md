# JaxFNE v0.4.22 programme acceptance

Status: **PRE_RELEASE** — acceptance authority for the v0.4.22 truth and
measurement floor. Derived from `artifacts/roadmap/ROADMAP_0422_0424.md` §3.3.

Nothing in this file is a completed claim until sealed with receipts on the final
release candidate SHA.

## Programme scope

v0.4.22 establishes classification, instrumentation, and public-surface truth
without representation reduction or performance claims. Scientific numerics must
remain bit-exact against the published v0.4.21 baseline unless a gate below
explicitly authorizes a documented exception (none are declared here).

## Integration baseline

| Field | Value |
| --- | --- |
| SHA | `2338b00af5c7c822aa35af60d68be4d62f18469d` |
| Evidence | RC PASS + CI PASS (pre-identity bump) |
| Published predecessor | `v0.4.21` @ `88235208950425ce76723115f1cb61fe3a828ab2` |

## Acceptance gates (all required)

Each gate must pass with named receipts on the final `0.4.22` candidate SHA
before seal authorization.

1. **Scientific numeric invariance.** `V_m`, `spikes`, `W`, `H`, and field
   outputs are bit-exact against the v0.4.21 baseline across the benchmark
   matrix.
2. **New failure-path coverage.** Null-objective and atlas regression tests
   prove the repaired failure paths fire (no silent swallow).
3. **Classification closure.** Zero `UNKNOWN` in dense-site and handler
   classifications (`artifacts/audit/` tallies current at seal time).
4. **Public unfinished-work sweep.** Rendered public docs contain no
   unclassified unfinished-work statement (W4/W5 repair complete).
5. **Release verification chain.** Full RC gate, CI, notebook gate, artifact
   provenance, clean-room install, and jomission checks PASS on the candidate.

## Explicit non-goals (v0.4.22)

No representation change. No performance claim. No approximation programme.

## Gate 0 RELEASE path

`artifacts/release/current_release_authorities.json` still points at v0.4.17
authorities until task `22-AUTH-02` updates it to:

- `release_target_version`: `0.4.22`
- `release_receipt`: `artifacts/release/v0_4_22_release_receipt.json`
- `acceptance_goals`: this file
- `acceptance_goal_count`: `5`

Until that update, RELEASE-mode Gate 0 correctly reports stale authority relative
to the package version after `22-VER-01`.
