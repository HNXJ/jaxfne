---
name: jaxfne-release-mutation-guard
description: >-
  Guard every remote mutation: push to main/dev, tag create/delete, GitHub
  Release, TestPyPI/PyPI upload, release freeze, SHA reconciliation, and
  version identity. Use before any task that mentions push, merge, main,
  dev, tag, retag, delete tag, version bump, release, GitHub Release,
  PyPI, TestPyPI, twine upload, publish, CI, headSha, origin/main, release
  freeze, or mutation.
---

# jaxfne Release Mutation Guard

## Purpose

Prevent accidental branch, tag, GitHub Release, and PyPI mutations. Use before any remote mutation.

## Mutation classes

```text
read-only: git status, git log, git fetch, gh run view
local-only: build, test, compile, inspect
remote branch mutation: git push origin main/dev
tag mutation: create/delete/recreate/push tag
distribution mutation: TestPyPI/PyPI upload
GitHub Release mutation: create/edit/publish release
```

| Class | Authorization |
|---|---|
| Read-only (status, ls-remote, gh run view) | Allowed |
| Local-only (build, test, inspect) | Allowed unless risky |
| Remote branch push | Explicit task scope |
| Tag create/delete/recreate/push | Explicit authorization |
| TestPyPI/PyPI upload | Explicit authorization (+2FA if available) |
| GitHub Release create/edit/publish | Explicit authorization |

## Required reconciliation before release mutation

```bash
git fetch origin --prune --tags
git status --short
git branch --show-current
git rev-parse HEAD
git ls-remote origin main dev
git ls-remote origin refs/tags/<version>
git ls-remote origin "refs/tags/<version>^{}"
gh run list --branch main --limit 10
```

Report:

```text
local HEAD SHA
origin/main SHA
origin/dev SHA
CI headSha/status/conclusion
tag object SHA
tag peeled commit SHA
working tree clean/dirty
package version
```

## One `intended_release_sha` per release (invariant)

Before tag repair / Release edit / TestPyPI / PyPI, verify all four gates:

```bash
git fetch origin --prune --tags
origin_main_sha=$(git ls-remote origin main | awk '{print $1}')
ci_head_sha=$(gh run view <run-id> --json headSha -q .headSha)
# GATES: origin/main == ci_head_sha == intended_release_sha · CI conclusion == success · tree clean
```

origin/main ≠ CI headSha → stop, report reconciliation options, do not repair/upload.

**Annotated tags:** use the peeled commit SHA as release identity, never the tag-object SHA —
`git ls-remote origin "refs/tags/vX.Y.Z^{}"` (peeled) vs `refs/tags/vX.Y.Z` (object).

## Release freeze rules

Freeze is active when release CI for a candidate SHA is running or passed and the release has not yet been published or cancelled.

During freeze, block unrelated commits:

```text
docs cleanup
formatting-only changes
new features
branch hygiene
receipt cleanup
```

Allowed during freeze:

```text
read-only checks
explicit release-only tag repair
explicit TestPyPI/PyPI upload
explicit GitHub Release publication
```

## Correct PyPI release rule

Before upload:

```bash
python -m build
python -m twine check dist/*
ls dist | grep '<version>'
```

Post-install smoke:

```bash
python -m venv /tmp/jaxfne-release-smoke
source /tmp/jaxfne-release-smoke/bin/activate
python -m pip install -U pip
python -m pip install "jaxfne==<version>"
python - <<'PY'
import jaxfne as jtfne
print(jtfne.__version__)
PY
```

## Before any remote mutation: report intent, get explicit authorization

Do not combine mutation classes in one authorization (tag · TestPyPI · PyPI · GitHub Release
are separate gates). Intent template:

```text
type · operation · scope · target ref · current SHA · intended SHA · reason ·
risk · rollback · permanent · Authorization: PENDING
```

## Long CI job → ONE receipt at terminal state only

No heartbeats — wait for success/failure/cancelled/timed_out, then report once: run URL ·
status+conclusion · headSha · job-matrix · origin/main SHA · tree status ·
no-unauthorized-mutations confirmation · next safe action.

## Fix root clutter BEFORE release-candidate CI, not during freeze

`git ls-files . | wc -l`; `find . -maxdepth 1 -type f`; `git status --short` clean — do this as a
default before kicking off release CI, not as freeze-window cleanup (freeze blocks it, see above).

## Stop conditions

```text
pyproject version does not match intended release
jaxfne.__version__ does not match intended release
tag target is not origin/main release commit
main CI is missing/failing/running for wrong SHA
working tree is dirty before tag/build/upload
attempt to force-push main
attempt to delete/recreate a public tag without authorization
attempt to upload to PyPI without explicit authorization
```
