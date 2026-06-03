# Release Checklist (v0.3.4+) — CURRENT DOCUMENT

> This is the **current release checklist** for jaxfne v0.3.4 and beyond.  
> For legacy v0.1.0 release process, see [RELEASE_CHECKLIST_LEGACY.md](RELEASE_CHECKLIST_LEGACY.md).

**Version:** v0.3.4  
**Last updated:** 2026-05-26  
**truth_mode:** truth_safe_unverified

> **Note:** Code examples in this checklist use `v0.3.4` as a placeholder. Substitute the
> actual target version (e.g., `0.3.4`, `0.3.5`) when following these steps.
> The current released version is **v0.3.4** (PyPI: https://pypi.org/project/jaxfne/0.3.4/).

---

## Pre-Release Validation

### 1. Verify Branch State

```bash
cd /Users/hamednejat/workspace/main/jaxfne
git status --short --branch
# Expected: ## main...origin/main (clean, no uncommitted changes)

git log --oneline -1
# Expected: HEAD matches the final commit for the release
```

### 2. Confirm Python Environment

```bash
python --version
# Expected: Python 3.10 or higher

source .venv/bin/activate
pip list | grep -E 'jax|numpy|scipy'
# Expected: jax>=0.4.25, jaxlib>=0.4.25, numpy>=1.24, scipy>=1.10
```

### 3. Run Full Test Suite

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/ -q --tb=line
# Expected: all tests pass (current baseline: 903 passed, 5 skipped)
```

### 4. Syntax Check All Python Files

```bash
python -m py_compile jaxfne/*.py jaxfne/**/*.py
# Expected: exit code 0
```

### 5. Validate JSON-Safe Outputs

```bash
python -c "
import json
from jaxfne.core import configuration, construct, simulate
from jaxfne.emitters import IzhikevichEmitter

# Run minimal example
config = configuration()
model = construct(config, emitters=[IzhikevichEmitter()])
signals = simulate(model, duration_ms=10, dt_ms=0.1)
manifest = model.manifest(signals)

# Validate all outputs are JSON-safe
json.dumps(manifest, allow_nan=False)
print('JSON-safe output validated')
"
# Expected: JSON-safe output validated (no NaN/Inf)
```

### 6. No-Secrets Scan

```bash
grep -RInE 'api[_-]?key|secret|token|password|private[_-]?key|bearer|BEGIN .*PRIVATE KEY' jaxfne/ docs/ tests/ 2>/dev/null || echo "Safe: no secrets detected"
# Expected: Safe: no secrets detected
```

### 7. Verify All Claim Gates are Immutable

```bash
python -c "
from jaxfne.core import default_basis_spec, compute_conservation_proxy_diagnostics

basis = default_basis_spec()
diag = compute_conservation_proxy_diagnostics()

assert basis.physical_amplitude_claim_allowed == False, 'Basis gate violated'
assert basis.biological_metabolism_claim_allowed == False, 'Basis gate violated'
assert diag['physical_amplitude_claim_allowed'] == False, 'Diagnostics gate violated'
assert diag['biological_metabolism_claim_allowed'] == False, 'Diagnostics gate violated'
assert diag['poisson_solver_status'] == 'not_implemented', 'Solver gate violated'
assert diag['maxwell_solver_status'] == 'not_implemented', 'Solver gate violated'
print('All claim gates frozen and immutable')
"
# Expected: All claim gates frozen and immutable
```

---

## Version Bump

### 1. Update pyproject.toml

```bash
# Edit pyproject.toml: version = "0.2.27"
# Verify:
grep '^version = ' pyproject.toml
# Expected: version = "0.2.27"
```

### 2. Update _JAXFNE_VERSION in jaxfne/core.py

```bash
# Edit jaxfne/core.py: _JAXFNE_VERSION = "0.2.27"
# Verify:
grep '_JAXFNE_VERSION = ' jaxfne/core.py
# Expected: _JAXFNE_VERSION = "0.2.27"
```

### 3. Update Version Assertions in Test Files

**The following 11 test files reference version strings and must be updated:**

- tests/test_runtime_module_v0211.py
- tests/test_semantic_correctness_v020.py
- tests/test_single_neuron_colab_v028.py
- tests/test_single_neuron_multimodal_v023.py
- tests/test_spectrolaminar_readiness_v011.py
- tests/test_tutorial_smoke_runner_v0217.py
- tests/test_two_neuron_ei_colab_v029.py
- tests/test_docs_equations_plotly_v0214.py
- tests/test_field_solution_metadata_v0213.py
- tests/test_network_100_ei_colab_v0210.py
- tests/test_probe_report_contract_v0212.py

**For each file:**

```bash
# Open file, find line with version assertion
# Example: assert jaxfne.__version__ == "0.2.26", ...
# Update to current version: "0.2.27"
# Verify:
grep '__version__' tests/test_<filename>.py
```

### 4. Update CHANGELOG.md

Add a new entry at the top (v0.2.27 most recent):

```markdown
## v0.2.27 — Conservation-Inspired Proxy Diagnostics

**Date:** 2026-05-22  
**Status:** Released  
**Baseline:** 903 tests passed, 5 skipped

### Features
- Added `compute_conservation_proxy_diagnostics()` function for safe scalar diagnostics over existing source/field proxy arrays
- Supports source norms (L1, L2), field-gradient proxy, CSD/LFP norms, source conservation residual
- Explicit immutable claim gates: physical_amplitude_claim_allowed=False, biological_metabolism_claim_allowed=False
- Hardcoded not-implemented status for Poisson, Maxwell, stress-energy, J·E, and Poynting solvers

### Documentation
- New: `docs/conservation_proxy_diagnostics.md` — full mathematical basis and API
- Updated: `docs/computation_basis.md` — v0.2.27 regime documented, Poisson solver gated future
- Updated: `docs/poisson_admissibility.md` — stale references removed, spec-only status clarified
- Updated: `docs/index.md` — added conservation-proxy-diagnostics reference

### Validation
- 42 new tests in `tests/test_conservation_proxy_diagnostics_v027.py`
- All 903 tests pass, 5 skipped (cumulative baseline)
- JSON-safe outputs validated
- No NaN/Inf in diagnostic scalars

### BETA Audit
- Scope drift patch: removed implication of Poisson solver in v0.2.27
- Confirmed: no Poisson solver, no Maxwell solver, no stress-energy tensor computation
- Confirmed: j_dot_e_proxy and poynting_flux_proxy hardcoded None (future doctrine)

### Truth Status
- Proxy diagnostics only (no physical amplitude claim)
- Computational scaffold (subject to future redesign)
- No biological metabolism calibration
- All solver gates: not_implemented
```

---

## Build and Package

### 1. Build Distribution

```bash
cd /Users/hamednejat/workspace/main/jaxfne

# Clean prior build artifacts
rm -rf build/ dist/ *.egg-info

# Build wheel and source distribution
python -m build
# Expected: Successfully built jaxfne-0.2.27-py3-none-any.whl and jaxfne-0.2.27.tar.gz
```

### 2. Verify Wheel Contents

```bash
python -m zipfile -l dist/jaxfne-0.2.27-py3-none-any.whl | head -20
# Expected: jaxfne/__init__.py, jaxfne/core.py, jaxfne/fields.py, ... present
```

### 3. Test Wheel Installation in Fresh Environment

```bash
# Optional: Create temporary venv for testing
python -m venv /tmp/test_jaxfne_wheel
source /tmp/test_jaxfne_wheel/bin/activate

pip install dist/jaxfne-0.2.27-py3-none-any.whl
python -c "import jaxfne; print(jaxfne.__version__)"
# Expected: 0.2.27

deactivate
rm -rf /tmp/test_jaxfne_wheel
```

---

## Tag and Push

### 1. Create Annotated Tag

```bash
git tag -a v0.2.27 -m "Release v0.2.27: Conservation-Inspired Proxy Diagnostics"
# Expected: tag created locally
```

### 2. Verify Tag

```bash
git tag -v v0.2.27
# Expected: tag signature (or verify signed)
```

### 3. Push Tag and Main Branch

```bash
git fetch origin
git pull --rebase origin main
# Run full tests again after rebase
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/ -q --tb=line

git push origin main
git push origin v0.2.27
# Expected: both pushed successfully
```

---

## PyPI Upload

### 1. Verify PyPI Credentials

```bash
# Ensure ~/.pypirc is configured with token (not plain password)
# Example:
# [testpypi]
# repository = https://test.pypi.org/legacy/
# username = __token__
# password = pypi-AgEIcHlwaS5vcmc... (token value)

# OR use environment variable:
export TWINE_USERNAME=__token__
export TWINE_PASSWORD="pypi-AgEIcHlwaS5vcmc..."
```

### 2. Upload to PyPI (Production)

```bash
python -m twine upload dist/jaxfne-0.2.27*
# Expected: Uploading jaxfne-0.2.27-py3-none-any.whl ... 100%
#           Uploading jaxfne-0.2.27.tar.gz ... 100%
```

### 3. Verify on PyPI

Open https://pypi.org/project/jaxfne/0.2.27/ and confirm:
- Version 0.2.27 listed
- Upload date = today
- Wheel and source distributions available
- README renders correctly

---

## Documentation Site (ReadTheDocs)

### 1. Trigger ReadTheDocs Build

Push main and tag to GitHub:
- ReadTheDocs should auto-detect and build
- Monitor: https://readthedocs.org/projects/jaxfne/

### 2. Verify Documentation Build

```bash
# After RTD build completes:
curl -s https://jaxfne.readthedocs.io/en/latest/ | grep -c "Conservation Proxy Diagnostics"
# Expected: count >= 1 (documentation includes new content)
```

### 3. Verify Version Selector

On https://jaxfne.readthedocs.io:
- Check version dropdown
- Confirm v0.2.30 is listed and set as "latest"

---

## Post-Release Validation

### 1. Install from PyPI

```bash
# In a fresh environment
python -m venv /tmp/test_pypi_install
source /tmp/test_pypi_install/bin/activate

pip install jaxfne==0.2.27
python -c "
import jaxfne
print('jaxfne version:', jaxfne.__version__)
from jaxfne import compute_conservation_proxy_diagnostics
print('Conservation proxy diagnostics available')
"
# Expected: version 0.2.27 confirmed, function available

deactivate
rm -rf /tmp/test_pypi_install
```

### 2. Test Example Scripts

```bash
cd /Users/hamednejat/workspace/main/jaxfne
python examples/03_single_neuron_multimodal_probe.py
# Expected: completion without errors, JSON output generated
```

### 3. Close Related Issues

If there are GitHub issues resolved by this release:
- Link commit/tag in the issue comment
- Close with comment: "Fixed in v0.2.27. See commit https://github.com/HNXJ/jaxfne/commit/<sha>"

---

## Sign-Off

**Release is complete when:**
- ✓ All pre-release validations pass
- ✓ Version bumped in pyproject.toml and core.py
- ✓ All 11 test files updated with new version assertions
- ✓ CHANGELOG.md entry added
- ✓ Tag created and pushed
- ✓ PyPI upload succeeds
- ✓ ReadTheDocs build succeeds
- ✓ Post-release validation confirms installation

**Report template:**
```
[model-llm-name][/Users/hamednejat/workspace/main/jaxfne][yyyymmdd-hhmm]

### v0.2.27 Release Complete

- Tag: v0.2.27, commit: [sha]
- PyPI upload: ✓ https://pypi.org/project/jaxfne/0.2.27/
- ReadTheDocs: ✓ https://jaxfne.readthedocs.io/en/latest/
- Baseline: 903 tests passed, 5 skipped
- No secrets, all claim gates frozen

[model-llm-name][/Users/hamednejat/workspace/main/jaxfne][yyyymmdd-hhmm]
```

---

## See Also

- [docs/packaging.md](packaging.md) — Build and distribution details
- [docs/ci_policy.md](ci_policy.md) — CI/CD infrastructure
- [CHANGELOG.md](../CHANGELOG.md) — Release history
- [README.md](../README.md) — Quick start and installation
