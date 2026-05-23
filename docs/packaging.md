# Packaging and Distribution (v0.2.27)

**How to build, test, and distribute jaxfne.**

**Version:** v0.2.27  
**Last updated:** 2026-05-22  
**truth_mode:** truth_safe_unverified

---

## Overview

jaxfne is distributed as:
- **Wheel** (`jaxfne-0.2.27-py3-none-any.whl`) — binary distribution (recommended for most users)
- **Source tarball** (`jaxfne-0.2.27.tar.gz`) — source distribution (for custom builds)
- **PyPI package** — published at https://pypi.org/project/jaxfne/

**Requirements:**
- Python >= 3.10
- pip, build, twine (for packaging/upload)

---

## Local Installation Modes

### 1. Development Install (Editable)

Install from source with development extras:

```bash
cd /Users/hamednejat/workspace/main/jaxfne
pip install -e '.[dev]'
```

**Installs:**
- Core package (`jaxfne`) in editable mode
- Development tools: pytest, pytest-cov, ruff, black
- JAX dependencies: jax, jaxlib, optax (optional)
- Visualization: matplotlib (optional)

**Result:** Changes to source code are immediately reflected in imports.

### 2. Core Install (Minimal)

Install only core dependencies (no JAX, no dev tools):

```bash
pip install -e .
```

**Installs:**
- Core package (`jaxfne`)
- Core dependencies: numpy, scipy, pandas, pyyaml

**Omits:**
- JAX (jax, jaxlib, optax) — optional extras
- Visualization (matplotlib) — optional extras
- Development tools — optional extras

### 3. Full Install (All Extras)

Install with all optional extras:

```bash
pip install -e '.[all]'
```

**Installs:**
- Core package
- Core dependencies
- JAX extras: jax, jaxlib, optax
- Visualization: matplotlib
- Development tools: pytest, ruff, black

---

## Building Distribution Packages

### 1. Prerequisites

Ensure you have build and twine installed:

```bash
pip install build twine
```

### 2. Clean Prior Artifacts

```bash
cd /Users/hamednejat/workspace/main/jaxfne
rm -rf build/ dist/ *.egg-info
```

### 3. Build Wheel and Source Distribution

```bash
python -m build
```

**Output:**

```bash
# Check build artifacts
ls -lh dist/
# Expected:
# -rw-r--r--  jaxfne-0.2.27-py3-none-any.whl  (typical: ~50KB)
# -rw-r--r--  jaxfne-0.2.27.tar.gz             (typical: ~100KB)
```

### 4. Verify Wheel Contents

```bash
python -m zipfile -l dist/jaxfne-0.2.27-py3-none-any.whl | head -30
```

**Expected files in wheel:**
- `jaxfne/__init__.py`
- `jaxfne/core.py`
- `jaxfne/fields.py`
- `jaxfne/emitters/izhikevich.py`
- `jaxfne/emitters/hodgkin_huxley.py`
- `jaxfne/probes.py`
- `jaxfne/objectives.py`
- `jaxfne/validation.py`
- `jaxfne-0.2.27.dist-info/METADATA`
- `jaxfne-0.2.27.dist-info/WHEEL`
- `jaxfne-0.2.27.dist-info/RECORD`

### 5. Verify Source Distribution

```bash
tar -tzf dist/jaxfne-0.2.27.tar.gz | head -20
```

**Expected files in tarball:**
- `jaxfne-0.2.27/pyproject.toml`
- `jaxfne-0.2.27/README.md`
- `jaxfne-0.2.27/jaxfne/__init__.py`
- `jaxfne-0.2.27/tests/` (all test files)
- `jaxfne-0.2.27/examples/` (all example scripts)
- `jaxfne-0.2.27/docs/` (all documentation)

---

## Testing Built Packages

### 1. Test Wheel in Fresh Virtual Environment

```bash
# Create temporary venv
python -m venv /tmp/test_wheel

# Activate and install wheel
source /tmp/test_wheel/bin/activate
pip install dist/jaxfne-0.2.27-py3-none-any.whl

# Test import and version
python -c "
import jaxfne
print('Package version:', jaxfne.__version__)
print('Core modules available:')
print('  - jaxfne.core:', hasattr(jaxfne, 'configuration'))
print('  - jaxfne.emitters:', hasattr(jaxfne, 'IzhikevichEmitter'))
print('  - jaxfne.fields:', hasattr(jaxfne, 'compute_conservation_proxy_diagnostics'))
"

# Run example
python -c "
from jaxfne.core import configuration, construct, simulate
from jaxfne.emitters import IzhikevichEmitter

config = configuration()
model = construct(config, emitters=[IzhikevichEmitter()])
signals = simulate(model, duration_ms=10, dt_ms=0.1)
print('Example run successful')
"

# Cleanup
deactivate
rm -rf /tmp/test_wheel
```

### 2. Test Source Distribution in Fresh Virtual Environment

```bash
# Create temporary venv
python -m venv /tmp/test_sdist

# Activate and install from source tarball
source /tmp/test_sdist/bin/activate
pip install dist/jaxfne-0.2.27.tar.gz

# Test import
python -c "import jaxfne; print('Version:', jaxfne.__version__)"

# Cleanup
deactivate
rm -rf /tmp/test_sdist
```

### 3. Verify No Secrets in Distribution

```bash
# Unpack tarball and scan for secrets
tar -xzf dist/jaxfne-0.2.27.tar.gz
cd jaxfne-0.2.27

grep -RInE 'api[_-]?key|secret|token|password|private[_-]?key|bearer' \
  jaxfne/ tests/ examples/ docs/ 2>/dev/null || echo "Safe: no secrets detected"

cd ..
rm -rf jaxfne-0.2.27
```

---

## PyPI Upload

### 1. Test PyPI (Optional Pre-Flight)

Before uploading to production PyPI, you can test on TestPyPI:

```bash
# Setup credentials
# Create account at https://test.pypi.org/
# Generate token at https://test.pypi.org/manage/account/token/
# Add to ~/.pypirc:
# [testpypi]
# repository = https://test.pypi.org/legacy/
# username = __token__
# password = pypi-AgEIcHlwaS5vcmc...

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/jaxfne-0.2.27*
```

**Verify on TestPyPI:**
```bash
# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ jaxfne==0.2.27

# Should work (with JAX deps from main PyPI if not in TestPyPI)
python -c "import jaxfne; print(jaxfne.__version__)"
```

### 2. Production PyPI Upload

Setup production PyPI credentials (if not already configured):

```bash
# Create account at https://pypi.org/
# Generate token at https://pypi.org/manage/account/token/
# Add to ~/.pypirc:
# [pypi]
# username = __token__
# password = pypi-AgEIcHlwaS5vcmc...

# Upload to production PyPI
python -m twine upload dist/jaxfne-0.2.27*
```

**Expected output:**
```
Uploading jaxfne-0.2.27-py3-none-any.whl
100% |████████████████████████████| 50KB/0.5s
Uploading jaxfne-0.2.27.tar.gz
100% |████████████████████████████| 100KB/0.3s
```

### 3. Verify on PyPI

Open https://pypi.org/project/jaxfne/0.2.27/ and verify:
- ✓ Version 0.2.27 is listed
- ✓ Upload timestamp is current
- ✓ Wheel and source distributions are available for download
- ✓ README renders correctly (no broken markup)
- ✓ Dependencies are listed (numpy, scipy, pandas, pyyaml, optional jax/optax)

### 4. Install from Production PyPI

```bash
# Fresh install from PyPI
pip install jaxfne==0.2.27

# Verify
python -c "import jaxfne; print(jaxfne.__version__)"
# Expected: 0.2.27
```

---

## Using Environment Variables for Credentials

For CI/CD or automated uploads, use environment variables instead of ~/.pypirc:

```bash
# Option 1: TWINE environment variables
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-AgEIcHlwaS5vcmc...
python -m twine upload dist/jaxfne-0.2.27*

# Option 2: Repository URL (TestPyPI)
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-AgEIcHlwaS5vcmc...
export TWINE_REPOSITORY_URL=https://test.pypi.org/legacy/
python -m twine upload dist/jaxfne-0.2.27*
```

**Security note:** Never commit credentials to git. Use GitHub Secrets or environment variables for CI/CD.

---

## Troubleshooting

### Build Fails: "ModuleNotFoundError: No module named 'build'"

```bash
pip install build
python -m build
```

### Upload Fails: "Invalid distribution"

Verify the build completed successfully:
```bash
ls -lh dist/
python -m zipfile -l dist/jaxfne-0.2.27-py3-none-any.whl | head -5
```

### Upload Fails: "401 Unauthorized"

Check PyPI credentials:
- Verify token is valid (not expired)
- Verify token is in ~/.pypirc or environment variables
- Verify correct repository URL (testpypi vs. pypi)

### Upload Fails: "Filename already exists"

Version already uploaded. Check:
```bash
# List releases on PyPI
pip search jaxfne  # (deprecated, use PyPI web)
# Or: https://pypi.org/project/jaxfne/#history
```

If reupload is needed (e.g., to fix a build artifact), increment patch version:
- Current: 0.2.27
- New: 0.2.28

---

## Version Bumping Workflow

When creating a new release:

1. **Update version in pyproject.toml:**
   ```bash
   # Before: version = "0.2.27"
   # After: version = "0.2.28"
   ```

2. **Update version in jaxfne/core.py:**
   ```bash
   # Before: _JAXFNE_VERSION = "0.2.27"
   # After: _JAXFNE_VERSION = "0.2.28"
   ```

3. **Rebuild and test:**
   ```bash
   rm -rf build/ dist/ *.egg-info
   python -m build
   # Test as above
   ```

4. **Commit and tag:**
   ```bash
   git add pyproject.toml jaxfne/core.py
   git commit -m "chore: bump version to 0.2.28"
   git tag -a v0.2.28 -m "Release v0.2.28"
   git push origin main v0.2.28
   ```

---

## See Also

- [Release Checklist](release_checklist.md) — Step-by-step release process
- [README.md](../README.md) — Installation instructions for end users
- [pyproject.toml](../pyproject.toml) — Build configuration and dependencies
