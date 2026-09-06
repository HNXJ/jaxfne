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
