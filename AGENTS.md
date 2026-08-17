# jaxfne — minimal persistent context

## Harness (2026-08-15, v2.1)

- **Identity vs integrity.** Workspace identity = (root, git root, remote, workspace id),
  validated by `~/.config/opencode/gates/workspace-gate.sh` against
  `workspaces/jaxfne-analysis.json`. Editing rule files (AGENTS.md, skills) never invalidates
  identity. The gate protects filesystem mutation targeting only; effective injected context
  is proven by fresh-session launch from this clone.
- **Integrity.** `HARNESS_MANIFEST.json` records hashes of kernel, router, canonical skills,
  generated mirrors, gates, schemas. Change flow: edit canonical `skills/` → run
  `scripts/harness/sync_skills.py --update --manifest` → verify. Mirrors (`.opencode/skills/`,
  `.cursor/skills/`) are generated, gitignored, never manually edited.
- **Frozen evidence.** `.opencode/frozen_paths.json` explicitly enumerates the immutable
  publication artifacts (`artifacts/publication/**`, `figures/publication/*.png`). New polish
  layer outputs (`figures/publication/final/**`, `fig*_polish_{spec,audit,receipt}.json`) are
  writable under the authorized publication task; frozen files are not.
- **Checkpoint.** `scratch/CURRENT_TASK.md` YAML frontmatter is authoritative for
  workspace/mode/facets/freeze/authorities/next_checkpoint/last_verified_head.
- **Routing.** mode ∈ {READ, CODE, SCIENCE, RELEASE, PUBLICATION}; facets ⊆ {CODE, REPO,
  SCIENCE, EVIDENCE_AUDIT, RELEASE}. Skills load compositionally from both.
- Root freeze patch: `.opencode/` and `scripts/harness/` approved 2026-08-15 as the client
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

- Read the smallest relevant skill under `skills/` for procedure.
- Verify unfamiliar public symbols against live code before using them.
- Prefer package-native scientific operators over notebook/script-local duplicate engines.
- Keep reusable plotting in the visualization layer.
- Use targeted tests during development; broader/release gates are separate evidence tiers.
- Do not commit, push, tag, release, or mutate remote state without explicit authorization. Under an explicitly authorized task and the standing completion rule below, routine non-force `git push origin dev` is part of step completion; tagging, main merge, release publication, force push, and other exceptional remote operations always require separate authorization.
- Public README/docs should be compact mathematical descriptions using positive definitions; engineering history and agent governance stay outside public scientific documentation.

## Step completion rule (operational)

At the end of each successfully validated discrete development or protocol step:

```text
specify → implement → test → freeze evidence → commit → push dev → verify → next step
```

1. Commit the complete scoped delta for that step only.
2. Push to `origin/dev`.
3. Verify `dev == origin/dev` and a clean working tree before declaring the step complete or beginning the next step.
4. Do not bundle unfinished work from the next step into that commit.