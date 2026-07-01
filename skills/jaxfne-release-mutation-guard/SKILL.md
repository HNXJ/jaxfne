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

Remote branch, tag, distribution, and GitHub Release mutations need explicit user authorization.

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
