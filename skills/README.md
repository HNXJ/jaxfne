# jaxfne skills

**Documentation written for AI coding agents** — same package source-of-truth as
the human docs under `docs/`, not a parallel spec. Human-oriented overview:
[docs/for_ai_agents.md](../docs/for_ai_agents.md).

Versioned skills live in this folder (`skills/`). Global copies may also exist in
each contributor's `~/.claude/skills/` and `~/.agents/skills/` — **the repo copy
wins** when they diverge:

```bash
bash skills/SYNC_GLOBAL.sh
```

## Glossary (agent shorthand)

| Term | Meaning |
|------|---------|
| **PRP** | Progress–Review–Plan backlog in `artifacts/developer/{plans,progress,review}.json` (local-only, gitignored — not in a fresh clone) |
| **TBI / TBD** | To-be-investigated / to-be-done items on a `progress.json` file row |
| **P0 / P1** | Priority labels in audit notes — P0 = fix before release claim |
| **Truth gates** | Conservative defaults (`claim_level`, `*_proxy`, etc.) — see [Scope & status](../docs/scope_and_status.md) |
| **HDP** | Homeostasis-Dependent Plasticity — the H-factor stabilization module (`RuntimeConfig(enable_hdp=True, ...)`, `DEFAULT_HDP`); see `jaxfne-neural-tensor` |
| **AGSDR / GSDR** | jaxfne's own optimizer families for `Model.tune` (Adaptive-Gradient/Gradient Stochastic Descent with Restarts) — not third-party optimizers; see `jaxfne-neural-network` |

**Friction ledger:** `FRICTIONS_STACK.md` — contradictions between skills, docs,
and code. Resolve before escalating claims.

## First-class skills (`skill-name/SKILL.md`)

Consolidated 2026-06-30 (17 → 13 folders). Merged-away names are listed in `PATCH.md`.

| Skill | Role |
|-------|------|
| `catalog-glossary-jaxfne` | Public API catalog — check before writing helpers |
| `jaxfne-objective-grammar` | Top-level chain: Config → Model → Signals → tune → manifest |
| `jaxfne-config` | `Configuration` fluent API + canonical laminar column |
| `jaxfne-neural-tensor` | `NeuronalTensor` path + HDP |
| `jaxfne-neural-network` | `construct` → `simulate` → probe/tune/manifest |
| `jaxfne-vis-modules` | `jaxfne.vis` plotting — package-level, proxy-safe |
| `jaxfne-modeling-optimization-schema` | Connectivity, selectors, objective conventions |
| `jaxfne-paradigm-design` | Sequential paradigms / oddball builders |
| `jaxfne-notebook-release-gate` | Notebook/tutorial validation before "done" claims |
| `jaxfne-release-mutation-guard` | Remote mutation gates (push, tag, PyPI) |
| `jaxfne-sha256-artifact-integrity` | Content hashes for configs/figures/wheels |
| `jaxfne-worker-context-router` | Route tasks to the right skill/lane |
| `jaxfne-spectrolaminar-suite` | Spectrolaminar études and LFP-proxy caveats |

These are guidance files, not importable package code (`jaxfne/` is the package).

## Repo-hardening rules (flat `0N_*.md`)

Numbered enforcement checklists (`00_INDEX.md` … `11_*.md`) at this folder root.
Start at `00_INDEX.md`. They overlap the folder skills at a shorter rule altitude.

Keep this flat set at the repo root of `skills/` — do not nest duplicate folders.
