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
- `-m "slow and not notebook"` tests
- `-m notebook` tests
- `mkdocs build --strict`
- Release example scripts (`examples/00`–`08` core set)

The three pytest sweeps use the named selectors `BROAD_MARKER_EXPR`,
`SLOW_MARKER_EXPR` and `NOTEBOOK_MARKER_EXPR` in `scripts/run_test_gate.py`, and
together they must be **exhaustive** over the `(slow, notebook)` marker algebra.

This matters because release CI on `main` runs `pytest tests` with no marker
filter. Matching check-family *names* does not prove matching test *sets*: the
gate previously ran only `not slow` and `slow and not notebook`, so the 30 node
ids carrying the `notebook` marker were executed by release CI and never by the
RC gate — every family name lined up while the invariant was broken underneath.
`tests/test_release_gate_hierarchy.py` now proves exhaustiveness mechanically,
and fails if release CI ever gains a marker filter or an `--ignore`, since the
coverage argument depends on release CI selecting everything.

### publication gate

- Repository state snapshot and hooks for frozen experiment manifests
- Extend when publication-facing receipts have a single entrypoint

## CI workflows

GitHub Actions should converge on the same gate names over time:

| Workflow | Branch | Current mapping |
|----------|--------|-----------------|
| `.github/workflows/ci.yml` | `dev` | Fast PR checks; moving toward **dev** + docs build |
| `.github/workflows/release_ci.yml` | `main` | **broad** / **release** on main and nightly; installs `.[dev,viz,jaxley]` for parity with the RC gate (reportlab required) and uploads per-node JUnit (`-rs --junitxml`) via `scripts/compare_pytest_junit.py` |

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
5. `python scripts/run_test_gate.py rc` **on the candidate SHA itself**, which
   writes the release candidate attestation

## Release candidate attestation

Publication authorization is mechanical and reads only observed execution.

`run_test_gate.py rc` records one ledger entry per check family as it runs --
`family`, `command`, `started`, `completed`, `exit_code`, `status`, `evidence` --
and writes them to `artifacts/attestations/rc_gate_attestation.json`
(override with `JAXFNE_RC_ATTESTATION`).

The attestation is **untracked by design**. A receipt committed into the tree
changes the SHA it records, so it can never certify the released commit itself;
that was the defect in the previous `jaxfne.rc_gate_receipt.v1` design, whose
`check_families` list was copied from a constant rather than observed. Keep the
attestation local, or publish it as a CI artifact.

`scripts/release/reconcile_release_target.py` authorizes a candidate only when
all of the following hold:

- `attestation.commit_sha == intended_sha`, and the tree was clean at gate time;
- every family in `RELEASE_CI_GATE_FAMILIES` has a record naming a command that
  actually ran, with both timestamps and a zero exit;
- `observed_pass_families >= RELEASE_CI_GATE_FAMILIES`.

`observed_pass_families` and `pre_release_subsumes_ci` are **re-derived** from
the ledger during verification and cross-checked against the stored values, so
editing either field by hand cannot authorize a release. See
`tests/test_release_gate_hierarchy.py` for the adversarial cases.

## Environment parity

The subsumption invariant is about *executed tests*, not about check family
names, so it depends on the gate environment being able to run everything CI
runs. When an optional dependency is absent, a module guarded by
`pytest.importorskip` is skipped at **collection**: its tests produce no node
IDs, nothing fails, and the family still reports PASS while the gate quietly
covers less than CI.

That happened. With the `jaxley` extra missing locally, three modules skipped at
collection and 16 tests release CI runs were never executed by the gate -- local
collection 3746 against CI's 3762 -- while every check family name matched and
`pre_release_subsumes_ci` read `true`. Matching family names had made the gap
invisible; only comparing test populations exposed it.

`scripts/check_environment_parity.py` now runs first in the `rc` gate, as the
`environment_parity` family, and fails when either:

- a distribution required by an extra the release workflow installs is absent
  (the extras are read from the workflow, so the check cannot drift from CI); or
- any test module is skipped at collection, which catches the same class of
  failure for causes no dependency list would predict.

It is deliberately **not** a member of `RELEASE_CI_GATE_FAMILIES`: CI does not
run it, and claiming otherwise would be the same kind of unobserved assertion
the attestation design exists to prevent.

## Interpreter coverage

JaxFNE supports Python 3.11-3.14. Blocking CI validates the supported range
at its 3.11 and 3.14 endpoints rather than every minor version: `3.11` is the
oldest line the pinned extras resolve on, `3.14` the newest and the
interpreter the local release-candidate gate runs on (without it the gate
would certify releases on an interpreter CI never exercises). `3.12` and
`3.13` are supported but not independently exercised by the full CI matrix;
do not describe them as independently CI-tested.

`3.10` is not supported: the `dev`/`viz` extras pin `scipy==1.17.1`, which
publishes no cp310 wheels, and the `jax`/`jaxlib` line the package resolves
against already floors at `>=3.11`. Dropping nominal 3.10 support is truth
alignment, not removal of a validated configuration.

Wheel availability for both endpoints was checked against the package index
rather than assumed -- `scipy 1.17.1`, `matplotlib 3.10.9` and `jaxlib 0.10.2`
all publish cp311 and cp314 manylinux wheels (and the pinned 3.14
release-candidate environment installs cleanly, which further evidences
cp314 availability).

A regression test (`test_python_support_policy`) pins `requires-python`,
the `Programming Language :: Python` classifiers, the CI matrices, the
publish interpreter, and this policy sentence together, so metadata,
classifiers, CI endpoints, publish interpreter, and documentation cannot
silently diverge.

## Artifact provenance

The bytes uploaded to PyPI are the bytes the gates validated. Nothing rebuilds
at publish time.

`release_ci.yml` runs `python -m build` exactly once per run. That single build
feeds the twine check, the isolated wheel smoke, and
`scripts/build_release_manifest.py`, which writes `RELEASE_MANIFEST.json` beside
the artifacts recording source commit, tree hash, version, declared and observed
build backend, and the filename, size and full SHA256 of each distribution. The
wheel, the sdist and the manifest are retained together as the `release-dist`
CI artifact.

`publish.yml` downloads that artifact for the exact commit being published,
re-derives both SHA256 values, and requires equality with the manifest before
any upload step runs. If no successful `release_ci` run exists for the commit,
or the manifest names a different commit, or a hash differs, publishing fails
rather than falling back to a build.

The manifest is generated, never committed. Committing it would change the tree,
and therefore the commit SHA it claims to certify — the same self-reference the
release-candidate attestation avoids. Retained artifacts live in the gitignored
`artifacts/release_candidate/` locally and as CI artifacts remotely.

`build-system.requires` pins the backend exactly (`hatchling==1.29.0`) rather
than ranging it. A wheel embeds `Generator: hatchling <version>` in its `WHEEL`
metadata, so a different resolution inside a range changes the artifact SHA256
while every gate still passes.

## JUnit parity

The `junit_parity` family compares the release-candidate sweeps node-by-node
against the blocking CI evidence for the same commit. `scripts/run_test_gate.py`
executes `scripts/check_junit_parity.py` in the `rc` gate, which downloads the
`pytest-results-*` artifacts for HEAD and runs `scripts/compare_pytest_junit.py`
against the three RC sweeps.

The comparison fails on a node present on one side only, on any failure or
error, and on any outcome difference not classified
`INTENTIONAL_PLATFORM_DIFFERENCE`. `ENVIRONMENT_DEFECT` and `UNKNOWN` both fail:
an unclassified difference is not a justified one. The justified differences are
the POSIX-only shell gates and executable-bit checks, which skip on the Windows
RC host and pass on Linux CI; they are listed by node ID in the report rather
than filtered out.

This family belongs to the `rc` gate only. Release CI produces one side of the
comparison and cannot compare against itself.

## Branch protection

`main` requires status checks. Job names are unique across workflows
(`fast-test` / `fast-build` in `ci.yml`, `release-test` / `release-build` in
`release_ci.yml`) because required checks are keyed by job name: while both
workflows exposed `test (3.12)`, `test (3.13)` and `build`, a required context
could be satisfied by whichever run reported last, making enforcement weaker
than it looked. A regression test rejects reintroduced collisions.

`enforce_admins` is deliberately **off**. Turning it on would also enforce the
existing `required_pull_request_reviews` against repository admins, which is a
workflow decision separate from requiring CI, and is left to explicit approval.
Admin pushes therefore still bypass the required checks.

## Documentation surfaces

[ReadTheDocs](https://jaxfne.readthedocs.io/) is the **authoritative** published
documentation: it builds `mkdocs.yml` from the repository on every push, and the
build's recorded commit is what the docs correspond to.

The `gh-pages` branch is **historical and inactive**. It holds an older
deployment (release candidate docs for v0.4.20) and is intentionally not
synchronized with `main`; nothing publishes to it and no gate reads it. It is
retained only so existing links do not 404. Do not treat its contents as
current, and do not reconcile it against a release SHA.

## Performance benchmarks

Deterministic benchmark scripts exist for documentation; they are **not**
automatic CI gates. See the repository-local performance baseline notes (not part of the built site).
