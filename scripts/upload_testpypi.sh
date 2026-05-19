#!/usr/bin/env bash
# scripts/upload_testpypi.sh
# Upload jaxfne to TestPyPI. Requires ~/.pypirc with [testpypi] section.
# Run from repo root: ./scripts/upload_testpypi.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXPECTED_VERSION="0.1.0"
WHEEL="dist/jaxfne-${EXPECTED_VERSION}-py3-none-any.whl"
SDIST="dist/jaxfne-${EXPECTED_VERSION}.tar.gz"

echo "=== jaxfne TestPyPI upload — v${EXPECTED_VERSION} ==="

# Guard: credentials
if [ ! -f "$HOME/.pypirc" ]; then
  echo "BLOCKED: ~/.pypirc not found. Create it with [testpypi] credentials first."
  echo "See docs/RELEASE_CHECKLIST.md for the template."
  exit 1
fi

# Guard: artifacts
[ -f "$WHEEL" ] || { echo "FAIL: wheel not found at $WHEEL. Run scripts/release_rehearsal.sh first."; exit 1; }
[ -f "$SDIST" ] || { echo "FAIL: sdist not found at $SDIST. Run scripts/release_rehearsal.sh first."; exit 1; }

echo "target:   https://test.pypi.org"
echo "wheel:    $WHEEL"
echo "sdist:    $SDIST"
echo ""

python3 -m twine upload --repository testpypi dist/jaxfne-"${EXPECTED_VERSION}"*
echo ""
echo "=== TestPyPI upload complete ==="
echo "Install test: pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ jaxfne==${EXPECTED_VERSION}"
