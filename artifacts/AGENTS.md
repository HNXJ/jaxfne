# jaxfne — minimal persistent context

## Harness

- **Integrity.** `scripts/harness/HARNESS_MANIFEST.json` hashes kernel, this router, canonical skills, mirrors, gates and schemas. Change flow: edit canonical `artifacts/skills/` or this file → `python scripts/harness/sync_skills.py --update --manifest` → verify. Tool-local mirrors are generated, never hand-edited.
- **Frozen evidence.** `artifacts/publication/frozen_manifest.json` enumerates immutable publication artifacts. Release authorities: `artifacts/release/current_release_authorities.json`. `scratch/CURRENT_TASK.md` supplies the active `mode:` for Gate 0.
- **Routing.** mode ∈ {READ, CODE, SCIENCE, RELEASE, PUBLICATION}; facets ⊆ {CODE, REPO, SCIENCE, EVIDENCE_AUDIT, RELEASE}. Skills load compositionally from both.

## Purpose

jaxfne expresses neural biophysics as modular Tensor-Field Neural Equations (TFNE): a **containment and composition model** for neural models of different resolution — not a single prescribed biophysical equation. Nested biological semantics and geometry are preserved while numerical tensors may be computationally flattened.

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
- General adaptive dynamics are conceptually `dX/dt = F_X`, `dH/dt = F_H`, `dTheta/dt = F_Theta`. RBD with fixed `W` is valid; plasticity rules are realizations of this grammar, not separate subsystems by default.
- Preserve biological identity, topology, signs, receptor/mechanism identity, geometry, locality, and declared parameter ownership through compilation and optimization.
- Source, field, probe, objective, and calibration semantics remain explicit. A projection, proxy, PDE solve, calibration, and validation status are distinct concepts.
- A public parameter either changes the realized object or is refused; stored-but-unconsumed parameters are defects (P-014, P-015).

## Authority and evidence

- Mathematical specification: the project-source set (`artifacts/project_sources/`). Implemented behavior: live `jaxfne/` code and tests. Public explanation: README/docs. Repository state: generated state/audit scripts.
- Keep distinct: SPECIFIED · IMPLEMENTED · TESTED · OBSERVED. Preserve failed prospective receipts; never tune a frozen protocol after seeing its outcome unless a new protocol is declared.
- Do not store SHAs, versions, timings, test counts, bug lists or line numbers in persistent rules.

## Repository behavior

- Read `artifacts/memory.md` (project brief) and the smallest relevant skill under `artifacts/skills/` (router: `artifacts/context.md`).
- Verify unfamiliar public symbols against live code. Prefer package-native operators; reusable plotting lives in the visualization layer.
- Targeted tests while iterating; `python scripts/run_test_gate.py broad` before pushing code.
- Routine non-force `git push origin dev` is part of step completion under an authorized task; tags, main, releases, force push need separate authorization.
- Public docs are compact positive mathematical descriptions; agent governance stays out of them.

## Project control

| File | Role |
|------|------|
| `artifacts/fact_stack.md` | Stable, human-authorized facts. Read, use, test, challenge — **do not edit without explicit human authorization.** Not evidence. |
| `artifacts/todo_stack.md` | Remaining work only; done items are deleted (git and receipts keep history). Sealed stacks are archived byte-for-byte under `artifacts/archive/`. |
| `artifacts/issue_log/ISSUE_LOG.md` | Open issues (P-NNN). |

Work loop `P (R G)^N S`: prepare from the todo stack → review the last result → do the next item → test → commit + push `dev` → repeat; stop only for a human decision that blocks all remaining work. After each feature, run an adversarial review with a separate agent (opencode muse, `review` tier; packet template `artifacts/harness/review_packet_template.md`) and fix or record every finding.

Full rules (work loop, H1–H14 review and evidence discipline, step completion): `artifacts/harness/evidence_and_workflow_rules.md`.
