# jaxfne-worker-handoff

**Triggers:** session start, session finish, handoff, locks, work report.

**Purpose:** Auditable worker discipline: branch ownership, active locks, identity header/footer, one next safe action.

**On start:**

```bash
git fetch origin
git status --short --branch
git log --oneline -3
```

Read active locks before editing. On finish: run relevant verification skills; update work log only when user requests commit.

**Full skill:** user-installed `jaxfne-worker-handoff`.
