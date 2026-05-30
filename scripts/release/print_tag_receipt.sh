#!/bin/bash
# Print the annotated tag object SHA and peeled commit SHA for a release tag.
#
# Usage:
#   bash scripts/release/print_tag_receipt.sh v0.3.14
#   bash scripts/release/print_tag_receipt.sh v0.3.14 --allow-missing
#
# Exit codes:
#   0  — tag found on origin; object SHA and peeled commit SHA printed
#   1  — tag missing on origin (unless --allow-missing)

set -euo pipefail

VERSION="${1:-}"
ALLOW_MISSING=false

if [[ "${2:-}" == "--allow-missing" ]]; then
    ALLOW_MISSING=true
fi

if [[ -z "$VERSION" ]]; then
    # Fall back to pyproject.toml version
    VERSION="v$(grep -E '^version =' pyproject.toml | head -n1 | sed 's/.*= *["'"'"']//' | sed 's/["'"'"'].*//')"
    if [[ "$VERSION" == "v" ]]; then
        echo "ERROR: No version argument and could not parse pyproject.toml" >&2
        exit 1
    fi
fi

# Ensure the version starts with 'v'
[[ "$VERSION" == v* ]] || VERSION="v$VERSION"

echo "Tag receipt for $VERSION:"

git fetch origin --prune --tags 2>/dev/null || true

OBJECT_SHA=$(git ls-remote origin "refs/tags/$VERSION" 2>/dev/null | awk '{print $1}')
PEELED_SHA=$(git ls-remote origin "refs/tags/$VERSION^{}" 2>/dev/null | awk '{print $1}')

if [[ -z "$OBJECT_SHA" ]]; then
    if [[ "$ALLOW_MISSING" == true ]]; then
        echo "Tag $VERSION not found on origin. --allow-missing set. Skipping."
        exit 0
    else
        echo "ERROR: Tag $VERSION not found on origin (refs/tags/$VERSION)" >&2
        exit 1
    fi
fi

echo "  tag object SHA : $OBJECT_SHA"
echo "  peeled commit  : ${PEELED_SHA:-$OBJECT_SHA  (lightweight tag — same as object)}"

if [[ -n "$PEELED_SHA" ]]; then
    echo ""
    echo "Release identity uses peeled commit SHA: $PEELED_SHA"
else
    echo ""
    echo "Release identity uses tag SHA (lightweight): $OBJECT_SHA"
fi
