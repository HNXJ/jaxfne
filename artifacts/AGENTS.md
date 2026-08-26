# jaxfne — minimal persistent context

## Harness (2026-08-15, v2.1)

- **Identity vs integrity.** Workspace identity = (root, git root, remote, workspace id),
  validated by `~/.config/opencode/gates/workspace-gate.sh` against
  `workspaces/jaxfne-analysis.json`. Editing rule files (AGENTS.md, skills) never invalidates
  identity. The gate protects filesystem mutation targeting only; effective injected context
  is proven by fresh-session launch from this clone.
- **Integrity.** `HARNESS_MANIFEST.json` records hashes of kernel, router, canonical skills,
  generated mirrors, gates, schemas. Change flow: edit canonical `artifacts/skills/` → run
  `scripts/harness/sync_skills.py --update --manifest` → verify. Mirrors (tool-local,
  outside the repository, e.g. `~/.config/opencode/skills/`) are generated, never manually edited.
- **Frozen evidence.** `artifacts/publication/frozen_manifest.json` explicitly enumerates the immutable
  publication artifacts (`artifacts/publication/**`, `artifacts/figures/publication/*.png`). New polish
  layer outputs (`artifacts/figures/publication/final/**`, `fig*_polish_{spec,audit,receipt}.json`) are
  writable under the authorized publication task; frozen files are not.
- **Checkpoint.** `scratch/CURRENT_TASK.md` YAML frontmatter is authoritative for
  workspace/mode/facets/freeze/authorities/next_checkpoint/last_verified_head.
- **Routing.** mode ∈ {READ, CODE, SCIENCE, RELEASE, PUBLICATION}; facets ⊆ {CODE, REPO,
  SCIENCE, EVIDENCE_AUDIT, RELEASE}. Skills load compositionally from both.
- Root freeze patch: .opencode/ and `scripts/harness/` approved 2026-08-15 as the client
  harness artifact (same reasoning that kept `skills/`).

## Purpose

jaxfne expresses neural biophysics as modular Tensor-Field Neural Equations (TFNE): a **containment and composition framework** for neural models of different resolution — not a single prescribed biophysical equation. Nested biological semantics and geometry are preserved while numerical tensors may be computationally flattened.

Scientific grammar:

```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer -> Manifest
```

Execution grammar:

```text
CircuitSpec -> construct -> Model -> simulate -> Signals
```

`CircuitSpec` includes supported Configuration and NeuronalTensor forms.

Paradigm, Objective, optimization/training utilities, visualization, and export are optional downstream workflow components where they exist — not stages of either invariant grammar.

## Mathematical invariants

- Internal quantities may remain relative. Absolute units arise through explicit calibration transformations at semantic boundaries.
- **RBS (Relative Biophysical State):** `H` is a finite-dimensional dependency-state container — not intrinsically homeostasis and not one scalar controlling all operators. Coordinates may be ions, traces, modulators, or reduced \(\mathcal R(\mathbf z)\); influence on \(E,S,F,P\) requires **typed coupling maps**. **RBD** is \(\dot H=F_H(\ldots)\); **HDP** is \(\dot W=F_W(H,\ldots)\). Authority: `docs/doctrine/tfne_containment_architecture.md`, `docs/doctrine/rbs_rbd_hdp.md`, `artifacts/project_sources/4_tfne_theory_and_neural_tensor.md`.
- General adaptive dynamics are conceptually `dX/dt = F_X`, `dH/dt = F_H`, `dTheta/dt = F_Theta`. RBD with fixed `W` is valid; plasticity rules are realizations of this grammar, not separate architectural subsystems by default.
- Preserve biological identity, topology, signs, receptor/mechanism identity, geometry, locality, and declared parameter ownership through compilation and optimization.
- Source, field, probe, objective, and calibration semantics remain explicit. A projection, proxy, PDE solve, calibration, and validation status are distinct concepts.

## Authority

For current mathematical specification, use the repository's authoritative project-source set when present. For implemented truth, inspect live `jaxfne/` code and tests. For public explanation, inspect README/docs. For current repository state, use generated state/audit scripts when available.

Do not store SHAs, versions, benchmark timings, test counts, bug lists, implementation line numbers, or temporary release state in persistent doctrine.

## Evidence

Keep these distinct:

- SPECIFIED — required by authoritative specification.
- IMPLEMENTED — present in the checkout.
- TESTED — covered by an executable test/verification receipt.
- OBSERVED — measured in a named run/environment.

Scientific experiments preserve failed prospective receipts. Do not tune a frozen protocol/controller after observing its validation outcome unless a new protocol is explicitly declared.

## Repository behavior

- Read the smallest relevant skill under `artifacts/skills/` for procedure.
- Verify unfamiliar public symbols against live code before using them.
- Prefer package-native scientific operators over notebook/script-local duplicate engines.
- Keep reusable plotting in the visualization layer.
- Use targeted tests during development; broader/release gates are separate evidence tiers.
- Do not commit, push, tag, release, or mutate remote state without explicit authorization. Under an explicitly authorized task and the standing completion rule below, routine non-force `git push origin dev` is part of step completion; tagging, main merge, release publication, force push, and other exceptional remote operations always require separate authorization.
- Public README/docs should be compact mathematical descriptions using positive definitions; engineering history and agent governance stay outside public scientific documentation.

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
- **H6 Algorithm names are mathematical contracts.** If multiple
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
  doctrine explicitly requires `d_H>1`. `RBS ≠ homeostasis` means RBS is
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