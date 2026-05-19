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

# 4. Wheel smoke
echo ""
echo "--- 4. Wheel install smoke ---"
rm -rf "$VENV_WHEEL"
python3 -m venv "$VENV_WHEEL"
"$VENV_WHEEL/bin/python" -m pip install --upgrade pip --quiet
"$VENV_WHEEL/bin/python" -m pip install "$WHEEL" --quiet
cd /tmp
EXPECTED_VERSION_PY="$EXPECTED_VERSION"
"$VENV_WHEEL/bin/python" - <<PY
import json, jaxfne as jtfne
assert "site-packages" in jtfne.__file__, f"CWD shadowing! file={jtfne.__file__}"
expected_version = "$EXPECTED_VERSION_PY"
assert jtfne.__version__ == expected_version, f"version={jtfne.__version__}, expected {expected_version}"
cfg = (jtfne.configuration()
    .network(n=8).emitter(family="izhikevich", preset="cortical_eig")
    .field(domain="laminar_column", conductivity="proxy",
           boundary="mean_zero_neumann", gauge="mean_zero")
    .probe(name="lp", n_contacts=8))
model = jtfne.construct(cfg)
signals = model.simulate(jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0))
readouts = model.compute_readout(signals, [jtfne.readout_spec("r", "spike_rate_hz")])
manifest = model.manifest(signals, readouts)
json.dumps(manifest, allow_nan=False)
assert manifest["truth_mode"] == "truth_safe_unverified"
assert manifest["physical_amplitude_claim_allowed"] is False
print("wheel smoke: OK", jtfne.__version__)
PY
cd "$REPO_ROOT"

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

# 7. Examples
echo ""
echo "--- 7. Examples ---"
for ex in examples/00_minimal_column.py examples/01_source_field_manifest.py \
          examples/02_omission_scaffold.py examples/03_objective_and_tune_smoke.py \
          examples/04_blackbox_tuning_loop.py examples/05_dataset_bridge_manifest.py \
          examples/06_edge_list_recurrent_backend.py; do
  PYTHONPATH=. python3 "$ex" > /dev/null 2>&1 && echo "PASS: $ex" || { echo "FAIL: $ex"; exit 1; }
done

echo ""
echo "=== release rehearsal PASSED — v${EXPECTED_VERSION} ready for TestPyPI ==="
echo "next: create ~/.pypirc then run scripts/upload_testpypi.sh"
