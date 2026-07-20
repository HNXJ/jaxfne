---
name: jaxfne-sha256-artifact-integrity
description: >-
  Use SHA256 as content identity for jaxfne artifacts, configs, notebooks,
  trained models, and release files. Use when a task mentions sha256,
  SHA256SUMS, checksum, hash, artifact_hashes, stale figure, stale output,
  reproducibility, deterministic run, cache, candidate hash, manifest.json,
  validation_report.json, metrics.json, trained_model_agsdr.json, notebook
  hash, source hash, wheel hash, PyPI hash, release artifact, or verifying
  whether an output changed after a code/config change.
---

# jaxfne SHA256 Artifact Integrity Skill

## Purpose

Use SHA256 as a content identity tool for jaxfne artifacts, configs, notebooks, trained models, optimization candidates, weights, figures, and release files.

SHA256 proves content identity:

```text
same bytes  -> same hash
changed byte -> different hash
```

It does not prove biological truth, solver validity, calibration, or scientific correctness. Always pair hashes with finite-output checks, shape checks, JSON strictness, rate gates, proxy/status gates, and notebook execution receipts.

## When to invoke

Invoke this skill when the task involves any of these:

```text
sha256
checksum
hash
artifact_hashes
SHA256SUMS
stale figure
stale output
cache key
candidate hash
reproducibility
deterministic run
manifest integrity
notebook source hash
weight artifact
PyPI wheel hash
release artifact hash
```

## jaxfne status gates to preserve

Never use hashes to upgrade scientific status.

```yaml
claim_level: computational_scaffold
field_solver_status: linear_solver
field_claim_level: proxy_readout
physical_amplitude_calibrated: false
```

Use wording like:

```text
content identity verified
artifact hash matched
source hash changed
output hash changed
release file hash recorded
```

Avoid wording like:

```text
biologically verified
field solved
calibrated amplitude confirmed
mechanism proven
```

## Core helper functions

Implemented in `scripts/hash_utils.py` (extracted 2026-07-14 so this skill is the contract, not
the monolith — per `~/.claude/CLAUDE.md` § INTELLIGENCE-PER-TOKEN DOCTRINE). Import them rather
than reimplementing:

```python
from scripts.hash_utils import (
    sha256_file,            # SHA256 hex digest of a file's exact bytes
    stable_json_bytes,      # sorted/allow_nan=False JSON bytes, for deterministic hashing
    sha256_json,            # SHA256 of an object's stable_json_bytes
    notebook_source_sha256, # hash notebook cell type+source only, ignoring outputs/exec counts
    make_asset_hashes,      # {relpath: sha256} for *.json/*.png/*.html/*.npz under a directory
    write_asset_hashes,     # make_asset_hashes(...) + write asset_hashes.json
    diff_hashes,            # {path: {old, new}} for every changed/added/removed key
    candidate_sha256,       # stable hash of an AGSDR/training candidate params dict
    load_weight_artifact,   # load an artifact_ref's array, raising on hash mismatch
)
```

All 9 are plain-Python, no jaxfne import required — safe to use from any script/notebook/release
check. Verify the module still exports exactly these names before relying on it:
`python3 -c "import scripts.hash_utils as h; print([n for n in dir(h) if not n.startswith('_')])"`.

## Method 1: freeze and compare repo/source files

Use before a worker changes code:

```bash
find jaxfne tests tutorials docs scripts -type f \
  \( -name "*.py" -o -name "*.ipynb" -o -name "*.md" -o -name "*.yml" -o -name "*.yaml" \) \
  -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS.before.txt
```

After work:

```bash
find jaxfne tests tutorials docs scripts -type f \
  \( -name "*.py" -o -name "*.ipynb" -o -name "*.md" -o -name "*.yml" -o -name "*.yaml" \) \
  -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS.after.txt

diff -u SHA256SUMS.before.txt SHA256SUMS.after.txt || true
```

Interpretation:

```text
expected changed files changed -> good
unexpected files changed       -> inspect before commit
expected output unchanged      -> change may not reach runtime path
```

## Method 2: hash generated artifacts

After notebook or tutorial execution:

```python
from pathlib import Path

out_dir = Path("outputs/jaxfne_etude_no_1")
asset_hashes = write_asset_hashes(out_dir)
print(json.dumps(asset_hashes, indent=2, sort_keys=True))
```

Manifest should include:

```json
{
  "asset_hashes": {
    "manifest.json": "sha256:...",
    "validation_report.json": "sha256:...",
    "metrics.json": "sha256:...",
    "activity_suite.png": "sha256:...",
    "spectrolaminar_initial_V1.png": "sha256:..."
  }
}
```

Preferred value style:

```text
sha256:<hex>
```

## Method 3: normalized notebook hashing

Full `.ipynb` hashes change when outputs, execution counts, or metadata change. For logic changes, use source-only hashes.

```python
notebook = "tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb"
print("source_sha256:", notebook_source_sha256(notebook))
print("full_file_sha256:", sha256_file(notebook))
```

Use both:

```text
source hash: logic/text changed
full hash: exact packaged notebook changed
```

## Method 4: candidate hashes for AGSDR/training

Use `candidate_sha256` (from `scripts/hash_utils.py`) to deduplicate and cache candidate evaluations.

```python
params = {"cell.E.drive": 4.5, "cell.PV.drive": 3.5, "conn.feedforward_gain": 1.2}
key = candidate_sha256(params)
print(key)
```

Training history row:

```json
{
  "candidate_sha256": "sha256:...",
  "parameters": {
    "cell.E.drive": 4.5,
    "cell.PV.drive": 3.5
  },
  "score": 0.72,
  "rate_mean_hz": 8.1,
  "rate_max_hz": 31.4
}
```

Cache pattern:

```python
CACHE = {}


def evaluate_candidate(params):
    key = candidate_sha256(params)
    if key in CACHE:
        return CACHE[key]
    result = run_simulation(params)
    CACHE[key] = result
    return result
```

Persistent cache path:

```python
cache_path = Path("outputs/cache") / f"{candidate_sha256(params)}.json"
```

## Method 5: artifact refs for weight matrices

Do not store large raw arrays in config JSON. Use artifact refs.

```json
{
  "weight": {
    "mode": "artifact_ref",
    "path": "weights/V1_L1_E_to_V1_L2_PV.npz",
    "array_name": "W",
    "sha256": "sha256:7b1d..."
  }
}
```

Load with verification via `load_weight_artifact` (from `scripts/hash_utils.py` — raises
`ValueError` naming expected vs actual hash on mismatch):

```python
array = load_weight_artifact(ref, root=".")
```

## Method 6: JAX compile/performance keys

Hash static model properties to compare performance on the same shape.

```python
static_compile_key = sha256_json({
    "n_neurons": int(model.n_neurons),
    "n_edges": int(model.n_edges),
    "dt_ms": float(cfg.dt_ms),
    "dtype": str(cfg.dtype),
    "backend": "jax",
})
```

Benchmark row:

```json
{
  "static_compile_key": "sha256:...",
  "compile_s": 2.41,
  "run_s": 0.18,
  "n_neurons": 360,
  "n_edges": 18420
}
```

Use this to separate:

```text
shape/config changed -> compile key changes
kernel got faster/slower with same shape -> compile key same, timing changed
```

## Method 7: release file hashes

Before PyPI/GitHub release:

```bash
rm -rf dist build *.egg-info
python -m build
python -m twine check dist/*
shasum -a 256 dist/*
```

Record wheel/sdist hashes in the release receipt.

Post-upload, verify PyPI file hashes match the built artifacts.

## Debug decision table

| Observation | Likely meaning | Next action |
|---|---|---|
| Source hash changed, output hash unchanged | New code did not reach output path, or output is cached | Trace runtime path and clear outputs/cache |
| Output hash changed, source hash unchanged | Environment, seed, dependency, or nondeterminism changed | Check seed/backend/version/device |
| Manifest hash changed, metrics hash unchanged | Metadata/path/timestamp changed | Normalize metadata or accept as provenance |
| PNG hash changed, numeric metrics unchanged | Plot styling/layout changed | Check figure code and metadata |
| Weight artifact hash mismatch | Wrong/stale weight file | Stop and regenerate/verify artifact |
| Wheel hash differs after rebuild | Build is not reproducible or source changed | Inspect dist, version, build metadata |

## Common jaxfne hash targets

Hash these regularly:

```text
manifest.json
validation_report.json
metrics.json
asset_hashes.json
trained_model_agsdr.json
notebook source cells
figures/*.png
plotly/*.html
weights/*.npz
built wheels and sdists
```

Avoid relying only on exact full-file hashes for:

```text
.ipynb files with outputs
HTML with timestamps
figures with nondeterministic metadata
```

Use normalized/source hashes where practical.

## Acceptance checks for worker reports

A worker report using this skill should include:

```text
repo / branch / SHA
changed files
commands run
exact validation results
hashes frozen before work, when relevant
new artifact hashes, when relevant
hash diffs, when relevant
interpretation of changed/unchanged hashes
truth/status unchanged
blockers
next safe action
```

## Stop conditions

Stop and report when:

```text
expected artifact hash did not change after an intended output-path change
unexpected source file hash changed
weight artifact hash mismatch appears
manifest/metrics JSON cannot be serialized with allow_nan=False
release dist filename/version does not match intended tag
PyPI/GitHub release artifact hash does not match local checked artifact
hash evidence is being used to imply biological truth or calibrated amplitude
```
