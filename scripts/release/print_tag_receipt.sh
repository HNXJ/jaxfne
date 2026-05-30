#!/bin/bash
VERSION=$(grep -E '^version =' pyproject.toml | head -n1 | cut -d'"' -f2)
if [ -z "$VERSION" ]; then
  VERSION=$(grep -E "^version =" pyproject.toml | head -n1 | cut -d"'" -f2)
fi
echo "Reconciliation tag receipt for v$VERSION:"
git ls-remote origin refs/tags/v$VERSION 2>/dev/null || echo "No tag on origin/remote refs/tags/v$VERSION"
git ls-remote origin "refs/tags/v$VERSION^{}" 2>/dev/null || echo "No peeled tag on origin/remote refs/tags/v$VERSION^{}"
