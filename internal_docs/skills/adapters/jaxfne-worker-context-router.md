# jaxfne-worker-context-router

**Triggers:** jaxfne, repo, file, module, branch, SHA, worker, handoff, tests, tutorial, release.

**Purpose:** Route work to the correct layer before editing. Prevents changes in the wrong module.

**Canonical pipeline:**

```text
Config -> Net -> Paradigm -> Objective -> Trainer -> Signals -> Vis/Export
```

Legacy aliases: `Configuration`→`Config`, `Model`→`Net`, `FlatModel`→`FlatNet`.

**Preflight (run first):**

```bash
git status --short
git branch --show-current
git rev-parse HEAD
```

**Full skill:** user-installed `jaxfne-worker-context-router` (see `internal_docs/skills/README.md`). Fallback context: `internal_docs/loop_context/00_MANIFEST.md`, `internal_docs/loop_context/02_PUBLIC_API_CONTRACT.md`.
