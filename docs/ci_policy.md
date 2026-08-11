# CI and Validation Policy

## Gate vocabulary

jaxfne uses four named validation gates. **The executable definitions live in
`scripts/run_test_gate.py`** (Makefile targets mirror them). Documentation
refers to gate names only — not duplicated file lists.

| Gate | Command | Scope | Typical wall time |
|------|---------|-------|-------------------|
| **dev** | `make test-dev` | Curated architectural tests + MCCs + compile + docs-language audit | ~1 min |
| **broad** | `make test-broad` | Repository-wide `pytest -m "not slow"` | ~8 min |
| **release** | `make test-release` | Broad + slow tests + strict MkDocs + example scripts | CI / pre-tag |
| **publication** | `make test-publication` | Frozen scientific experiments + artifact validation hooks | Manual |

```bash
make test-dev        # daily development gate
make test-broad      # broad non-slow suite (do not call this the dev gate)
make test-release    # pre-release validation
make test-publication
```

## What each gate runs

See `scripts/run_test_gate.py` for the authoritative step list. Summary:

### dev gate

- `compileall` on `jaxfne`, `tests`, `scripts`
- Curated pytest targets (API smoke, continuation, neuronal tensor, MCC, …) with
  `-m "not slow and not release"`
- `scripts/audit_public_docs_language.py --check`

### broad gate

- `compileall`
- Full `tests/` collection with `-m "not slow"` and standard ignores for the
  two long spectrolaminar projector tests

### release gate

- Everything in **broad**
- Additional `-m slow` tests
- `mkdocs build --strict`
- Release example scripts (`examples/00`–`08` core set)

### publication gate

- Repository state snapshot and hooks for frozen experiment manifests
- Extend when publication-facing receipts have a single entrypoint

## CI workflows

GitHub Actions should converge on the same gate names over time:

| Workflow | Branch | Current mapping |
|----------|--------|-----------------|
| `.github/workflows/ci.yml` | `dev` | Fast PR checks; moving toward **dev** + docs build |
| `.github/workflows/release_ci.yml` | `main` | **broad** / **release** on main and nightly |

`.github/workflows/notebook_execution.yml` handles `notebook`-marked tests separately
(long-running notebook execution).

## Truth gates on outputs

All manifests and evaluation reports carry conservative defaults:

| Field | Default |
|-------|---------|
| `physical_amplitude_calibrated` | `False` |
| `claim_level` | `computational_scaffold` |
| `field_claim_level` | `proxy_readout` |
| `field_solver_status` | `linear_solver` |

These cannot be escalated without explicit calibration evidence.

## Extended validation (manual)

Large tutorial sweeps and figure regeneration remain manual or nightly:

```bash
python scripts/run_all_tutorials.py --smoke --write-figures --out-root outputs/
python scripts/validate_tutorial_outputs.py outputs/
```

## Before tagging a release

1. `make test-release` (or equivalent CI green on intended SHA)
2. `python scripts/repo_state_snapshot.py`
3. Fresh venv wheel smoke (CI release job)
4. Working tree clean; `origin/main` matches release candidate SHA

## Performance benchmarks

Deterministic benchmark scripts exist for documentation; they are **not**
automatic CI gates. See `docs/performance_baseline.md`.
