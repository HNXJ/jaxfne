#!/usr/bin/env bash
# scripts/upload_pypi.sh
# Upload jaxfne to real PyPI. Requires JAXFNE_CONFIRM_REAL_PYPI=1.
# Run from repo root: JAXFNE_CONFIRM_REAL_PYPI=1 ./scripts/upload_pypi.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXPECTED_VERSION="0.1.0"
WHEEL="dist/jaxfne-${EXPECTED_VERSION}-py3-none-any.whl"
SDIST="dist/jaxfne-${EXPECTED_VERSION}.tar.gz"

echo "=== jaxfne real PyPI upload — v${EXPECTED_VERSION} ==="

# Guard: explicit confirmation required
if [ "${JAXFNE_CONFIRM_REAL_PYPI:-}" != "1" ]; then
  echo "BLOCKED: Set JAXFNE_CONFIRM_REAL_PYPI=1 to confirm upload to real PyPI."
  echo "This is irreversible. Run scripts/upload_testpypi.sh and Colab smoke first."
  exit 1
fi

# Guard: credentials
if [ ! -f "$HOME/.pypirc" ]; then
  echo "BLOCKED: ~/.pypirc not found. Create it with [pypi] credentials first."
  exit 1
fi

# Guard: artifacts
[ -f "$WHEEL" ] || { echo "FAIL: wheel not found at $WHEEL"; exit 1; }
[ -f "$SDIST" ] || { echo "FAIL: sdist not found at $SDIST"; exit 1; }

echo "TARGET:   https://pypi.org  (REAL PyPI — PERMANENT)"
echo "wheel:    $WHEEL"
echo "sdist:    $SDIST"
echo ""

python3 -m twine upload dist/jaxfne-"${EXPECTED_VERSION}"*
echo ""
echo "=== Real PyPI upload complete ==="
echo "Verify: pip install jaxfne==${EXPECTED_VERSION}"
