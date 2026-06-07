# jaxfne-release-mutation-guard

**Triggers:** push, merge, main, tag, PyPI, TestPyPI, GitHub Release, publish, release freeze.

**Purpose:** Prevent accidental remote mutations without explicit user authorization.

**Mutation classes:**

| Class | Examples | Authorization |
|---|---|---|
| Read-only | `git status`, `git fetch`, `gh run view` | Allowed |
| Local-only | build, test, compile | Allowed |
| Remote branch | `git push` | Explicit scope required |
| Tag / distribution / GitHub Release | tag push, twine upload, `gh release` | Explicit authorization required |

**Reconcile before release mutation:**

```bash
git fetch origin --prune --tags
git status --short
```

**Full skill:** user-installed `jaxfne-release-mutation-guard`. Related: `mutation-intent-gate`, `release-target-reconciler`, `tag-auditor`.
