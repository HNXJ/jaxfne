# Evidence and workflow rules (moved verbatim from artifacts/AGENTS.md, 2026-09-27)

Router: `artifacts/AGENTS.md`. These rules are unchanged; only their location moved.

## Work loop

Use:

```text
P (R G)^N S
```

**P — Prepare**

- inspect current state and evidence;
- update `todo_stack.md`;
- order remaining work.

**R — Review**

- review the last result;
- update `todo_stack.md`;
- choose the next item;
- if there are validated Git changes: commit, push to `dev`, verify sync.

**G — Progress**

- do the selected item;
- test it;
- return to R.

**N**

- keep doing R → G while useful work remains;
- stop only when an important human decision blocks all remaining work.

**S — Seal**

- verify the version/release is complete;
- update `todo_stack.md`;
- remove completed items;
- commit, push to `dev`, and verify clean sync.

### Rules

- `todo_stack.md` = remaining work only. It is not history or evidence.
- Do not stop after one TODO.
- Do not make empty commits.
- Do not skip commit/push of validated changes.
- Do not pass a version boundary before sealing it.

## Review and evidence discipline (H-series, 0.4.17 reconciliation)

- **H1 External review is hypothesis generation, not authority.** Findings
  from another model, reviewer, benchmark, static analyzer, or prior session
  are hypotheses until independently reproduced against the current
  authoritative state. Preserve the finding and its provenance; do not
  mutate solely from the finding.
- **H2 Hard-gate claims require receipts.** Never infer `READY`, `PASS`,
  `100/100`, release readiness, or scientific validation from partial or
  focused tests. A hard-gate claim requires the exact declared gate to have
  completed successfully on the state being sealed.
  `feature works ≠ release is sealed`.
- **H3 Reconcile arithmetic before Seal.** Before Seal, mechanically
  reconcile test counts, score sums, file counts, hashes, and other
  arithmetic appearing in the report. Contradictory receipts invalidate the
  corresponding claim until resolved.
- **H4 Serialization is an epistemic boundary.** For state/provenance/
  identity claims, test in-memory behavior and serialization roundtrip
  separately. Do not infer persistence from in-memory presence. Use
  `IN_MEMORY_ONLY` / `PRESERVED` / `PRESERVED_ELSEWHERE` / `PARTIALLY_LOST`
  / `LOST` when useful.
- **H5 Adversarial validation must include counterexamples.** Validators
  and constrained generative operators must be tested with adversarial
  invalid and boundary inputs, not only canonical happy paths. When an
  operator promises a feasible constrained output, test feasibility,
  invalid-domain rejection, boundary cases, and roundtrip semantics.
- **H6 Algorithm names are mathematical specifications.** If multiple
  implementations share an algorithm name, compare their state, update
  equations, hyperparameter semantics, randomness, selection, bounds, and
  termination. Do not call materially non-equivalent engines canonical
  instances of one algorithm without an explicit relationship. A public
  hyperparameter with the same name must not silently mean different
  mathematics across canonical paths.
- **H7 Dead public parameters are defects until classified.** If a public
  parameter is accepted but has no effect on a reachable canonical path,
  classify it explicitly as dead, compatibility-only, ignored-by-design,
  or defective. Never document it as active without behavioral evidence.
- **H8 General theory versus specialization.** A valid specialization is
  not a contradiction of a general theory. For a finite-dimensional state
  `H ∈ R^{d_H}`, `d_H=1` is a valid finite-dimensional realization unless
  rule explicitly requires `d_H>1`. `RBS ≠ homeostasis` means RBS is
  not intrinsically defined as homeostasis; it does not prohibit a
  particular RBS realization from having homeostatic dynamics.
- **H9 Do not impose textbook semantics over project-defined mathematics.**
  For project-defined or published algorithms, verify semantics against
  the project's canonical mathematical definition and original publication
  before importing external textbook requirements.
- **H10 Scholarly references require primary verification.** Before adding
  or preserving a scholarly citation used to justify theory, verify title,
  authors, year, venue/identifier, and relevance against a primary or
  authoritative bibliographic source. Never infer a citation from a search
  snippet or model memory. A citation mismatch in public scientific
  documentation is release-blocking until corrected.
- **H11 Generated-artifact tests must work from a fresh clone.** Tests must
  not depend on untracked, gitignored, or locally generated artifacts
  unless the test itself generates them in an isolated temporary directory
  or the artifact is an explicitly tracked release input.
- **H12 Concurrent-worktree drift is a first-class event.** Capture
  worktree state before long tests/generators. If tracked state changes
  unexpectedly, do not attribute the change to the current task without
  evidence.
- **H13 Model/reviewer identity and provenance.** Every audit/report must
  record model identity if exposed, exact repository SHA, package version,
  date/time, tool profile, and protocol version. Use `UNKNOWN`; never guess
  model identity.
- **H14 Review should challenge both positive and negative conclusions.**
  Adversarial Review must attempt to falsify both "this works" and "this
  is broken." A criticism is not validated merely because a counterexample
  sounds plausible.

## Step completion rule (operational)

At the end of each successfully validated discrete development or protocol step:

```text
specify → implement → test → freeze evidence → commit → push dev → verify → next step
```

1. Commit the complete scoped delta for that step only.
2. Push to `origin/dev`.
3. Verify `dev == origin/dev` and a clean working tree before declaring the step complete or beginning the next step.
4. Do not bundle unfinished work from the next step into that commit.