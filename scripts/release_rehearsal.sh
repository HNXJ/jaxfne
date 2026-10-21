#!/usr/bin/env bash
# scripts/release_rehearsal.sh
# Full release gate rehearsal — no credentials required, no upload, no tag.
# Run from repo root: ./scripts/release_rehearsal.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Read expected version dynamically from pyproject.toml
EXPECTED_VERSION=$(python3 -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(d['project']['version'])" 2>/dev/null || \
    grep '^version = ' pyproject.toml | sed 's/version = "//;s/"//')
WHEEL="dist/jaxfne-${EXPECTED_VERSION}-py3-none-any.whl"
SDIST="dist/jaxfne-${EXPECTED_VERSION}.tar.gz"
VENV_WHEEL="/tmp/jaxfne_rehearsal_wheel"
VENV_SDIST="/tmp/jaxfne_rehearsal_sdist"

echo "=== jaxfne release rehearsal — v${EXPECTED_VERSION} ==="
echo "repo: $REPO_ROOT"

# 1. Version check
echo ""
echo "--- 1. Version check ---"
PYPROJECT_VERSION="$EXPECTED_VERSION"
JAXFNE_VERSION=$(python3 -c "import jaxfne; print(jaxfne.__version__)" 2>/dev/null)
if [ "$JAXFNE_VERSION" != "$PYPROJECT_VERSION" ]; then
  echo "FAIL: jaxfne.__version__=$JAXFNE_VERSION, pyproject.toml=$PYPROJECT_VERSION"
  exit 1
fi
echo "version: $PYPROJECT_VERSION OK (jaxfne.__version__ matches)"

# 2. Clean build
echo ""
echo "--- 2. Clean build ---"
rm -rf dist build jaxfne.egg-info
python3 -m build
echo "build: OK"

# 3. Artifact check + twine
echo ""
echo "--- 3. Artifact + twine check ---"
[ -f "$WHEEL" ] || { echo "FAIL: wheel not found"; exit 1; }
[ -f "$SDIST" ] || { echo "FAIL: sdist not found"; exit 1; }
python3 -m twine check dist/*
echo "twine: OK"

# 4. Wheel smoke (body owned by scripts/run_test_gate.py: wheel_smoke_code;
#    this rehearsal only builds the venv and calls the `wheel-smoke` gate)
echo ""
echo "--- 4. Wheel install smoke ---"
rm -rf "$VENV_WHEEL"
python3 -m venv "$VENV_WHEEL"
"$VENV_WHEEL/bin/python" -m pip install --upgrade pip --quiet
"$VENV_WHEEL/bin/python" -m pip install "$WHEEL" --quiet
PYTHONPATH=. python3 scripts/run_test_gate.py wheel-smoke \
  --venv-python "$VENV_WHEEL/bin/python" --expected-version "$EXPECTED_VERSION"

# 5. Sdist smoke
echo ""
echo "--- 5. Sdist install smoke ---"
rm -rf "$VENV_SDIST"
python3 -m venv "$VENV_SDIST"
"$VENV_SDIST/bin/python" -m pip install --upgrade pip --quiet
"$VENV_SDIST/bin/python" -m pip install "$SDIST" --quiet
cd /tmp
EXPECTED_VERSION_PY="$EXPECTED_VERSION"
"$VENV_SDIST/bin/python" - <<PY
import json, jaxfne as jtfne
assert "site-packages" in jtfne.__file__
assert jtfne.__version__ == "$EXPECTED_VERSION_PY", f"version={jtfne.__version__}, expected $EXPECTED_VERSION_PY"
cfg = jtfne.configuration().network(n=8).emitter().field().probe(n_contacts=4)
model = jtfne.construct(cfg)
signals = model.simulate(jtfne.simulation(duration_ms=5.0, dt_ms=0.1, seed=0))
manifest = model.manifest(signals)
json.dumps(manifest, allow_nan=False)
print("sdist smoke: OK", jtfne.__version__)
PY
cd "$REPO_ROOT"

# 6. Test suite
echo ""
echo "--- 6. pytest ---"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest -q --tb=short

# 7. Examples (list owned by scripts/run_test_gate.py: RELEASE_EXAMPLES;
#    no subset reason applies here, so this rehearsal calls the `examples`
#    gate instead of carrying its own list)
echo ""
echo "--- 7. Examples ---"
PYTHONPATH=. python3 scripts/run_test_gate.py examples

echo ""
echo "=== release rehearsal PASSED — v${EXPECTED_VERSION} ready for TestPyPI ==="
echo "next: create ~/.pypirc then run scripts/upload_testpypi.sh"
