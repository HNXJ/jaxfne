# Release Checklist

This document tracks verification gates for jaxfne releases.

## Release-Control State Machine (v0.3.15+)

To prevent accidental release drift or tag/commit mutation, the following machine checks are enforced prior to tagging and PyPI upload:
1. **Intended Release SHA Lock**: The file `CLAUDE.md` or `AGENTS.md` must declare `intended_release_sha: "<commit_sha>"` and `release_freeze: true`.
2. **Release Freeze Enforcement**: Running `python scripts/release/assert_release_freeze.py` must verify that `HEAD` matches `intended_release_sha`.
3. **Peeled vs Object Tag Audit**: Prior to PyPI upload, running `bash scripts/release/print_tag_receipt.sh` must verify that the peeled ref matches `intended_release_sha`.
4. **Reconciled Target Report**: Running `python scripts/release/reconcile_release_target.py` must produce a validated target report JSON with `"release_target_reconciled": true`.
5. **CI headSha == Release SHA Gate**: The head SHA compiled in CI must exactly match the `intended_release_sha`.
6. **Noiseless CI Monitor**: No warnings or errors can exist in the CI pipeline run.
7. **Root Hygiene Preflight**: The repository working tree must be completely clean without untracked files or clutter.

---

## v0.3.21 Release Checklist

### Pre-release Validation (Local)
- [ ] `compileall`: `python -m compileall jaxfne/` passes without syntax errors
- [ ] Full pytest: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/` — all pass with expected counts
- [ ] Import smoke: `python -c "import jaxfne; print(jaxfne.__version__)"` — confirms 0.3.21
- [ ] Runtime receipt: `python -c "import jax; print(jaxfne.__version__, jax.__version__, jax.devices(), jax.config.jax_enable_x64, jnp.array([1.0]).dtype)"` — confirms cpu, x64=False, float32
- [ ] MkDocs strict build: `mkdocs build --strict` — no errors or warnings
- [ ] Package build: `python -m build` — generates `dist/jaxfne-0.3.21.tar.gz` and `dist/jaxfne-0.3.21-py3-none-any.whl`
- [ ] Twine check: `python -m twine check dist/*` — no errors
- [ ] CI run URL: GitHub Actions run captured for commit SHA
- [ ] Notebook audit: `python scripts/audit_notebooks_and_assets.py` — produces v0_3_21 report, all notebooks pass

### Branch & Main Verification (Ownership-gated)
- [ ] Verify `main` branch state: `git status --short`, `git diff --stat`
- [ ] Verify `main` branch is clean: `git fetch --dry-run`
- [ ] Verify remote `main` at release SHA: `git ls-remote origin main`
- [ ] All test suites pass: `pytest tests/ --tb=short`

### Distribution & TestPyPI (Local + Remote)
- [ ] TestPyPI upload: `python -m twine upload --repository testpypi dist/*` (if authorized)
- [ ] TestPyPI install smoke: `pip install --index-url https://test.pypi.org/simple/ jaxfne==0.3.21 && python -c "import jaxfne; print(jaxfne.__version__)"`
- [ ] PyPI upload: `python -m twine upload dist/*` (if authorized)
- [ ] Post-release install smoke: `pip install --upgrade jaxfne && python -c "import jaxfne; print(jaxfne.__version__)"`

### Tag & Release (After all gates pass)
- [ ] Tag provenance verified: `git cat-file -t v0.3.21`, `git branch --contains v0.3.21`, `git ls-remote --tags origin v0.3.21`
- [ ] Create annotated release tag on `main`: `git tag -a v0.3.21 -m "Release v0.3.21: Etude No. 1, template standardization, hygiene rule clarifications"` (pending authorization)
- [ ] Push tag to remote: `git push origin v0.3.21` (pending authorization)
- [ ] GitHub release created from tag: Visit https://github.com/HNXJ/jaxfne/releases/new?tag=v0.3.21

### Release Metadata
- [ ] CHANGELOG.md updated with v0.3.21 entry ✅
- [ ] docs/changelog.md updated with v0.3.21 entry ✅
- [ ] pyproject.toml version: 0.3.21 ✅
- [ ] docs/RELEASE_CHECKLIST.md created ✅
- [ ] Version strings synchronized across:
  - [ ] `pyproject.toml` → `version = "0.3.21"` ✅
  - [ ] `jaxfne/__init__.py` → version constant
  - [ ] `mkdocs.yml` → version reference (if present)
  - [ ] `docs/_generated/version.md` (if auto-generated, verify post-build)

### Truth & Scope Validation
- [ ] Truth posture confirmed: `truth_safe_unverified`, `computational_scaffold`, no `physical_amplitude_claim_allowed`
- [ ] Claim language audit: CHANGELOG and release notes avoid "validated", "proved", "physical" amplitude
- [ ] Field solver status: all data classes carry `field_solver_status=laminar_proxy_no_pde` or equivalent
- [ ] No hardcoded EEG/MEG/LFP/CSD amplitude claims in tutorials or notebooks

### Final Sign-off
- [ ] All gates pass locally or on CI
- [ ] No blockers remain
- [ ] Release checklist signed by owner(s)
- [ ] Ready for `main` merge and PyPI upload

---

## How to Use This Checklist

1. **Local gates** (starting point): Run all local validation gates before requesting main merge.
2. **Ownership gates** (decision required): Request authorization for merge/tag/upload actions.
3. **CI gates** (automated): Watch GitHub Actions run to confirm remote build and test results.
4. **Distribution gates** (final): After all other gates pass, proceed with TestPyPI and PyPI uploads.

## Release Decision Flow

```
Local validation complete
    ↓
Request: Merge dev → main
    ↓
Main branch verified
    ↓
Request: Create/push tag
    ↓
Tag verified on remote
    ↓
Request: Upload to TestPyPI
    ↓
TestPyPI smoke pass
    ↓
Request: Upload to PyPI
    ↓
Post-release install smoke
    ↓
Release complete ✅
```
