---
name: jaxfne-sha256-artifact-integrity
summary: Use when a jaxfne task mentions sha256, hashes, checksums, artifact hashes, stale outputs, reproducibility, cache keys, manifest integrity, notebook source hash, release wheel hash, or debugging whether generated figures/JSON changed.
trigger: Use whenever the task mentions sha256, SHA256SUMS, checksum, hash, artifact_hashes, stale figure, stale output, reproducibility, deterministic run, cache, candidate hash, manifest.json, validation_report.json, metrics.json, trained_model_agsdr.json, notebook hash, source hash, wheel hash, PyPI hash, release artifact, or verifying that an output changed after a code/config change.
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
truth_mode: truth_safe_unverified
claim_level: computational_scaffold
field_solver_status: laminar_proxy_no_pde
physical_amplitude_claim_allowed: false
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

Use these helpers in scripts, notebooks, release checks, and debugging cells.

```python
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    """Return SHA256 hex digest for a file's exact bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def stable_json_bytes(obj: Any) -> bytes:
    """Return stable strict-JSON bytes for deterministic hashing."""
    return json.dumps(
        obj,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(obj: Any) -> str:
    """Return SHA256 for stable strict JSON representation."""
    return hashlib.sha256(stable_json_bytes(obj)).hexdigest()


def notebook_source_sha256(path: str | Path) -> str:
    """Hash notebook cell type + source only, ignoring outputs/execution counts."""
    nb = json.loads(Path(path).read_text())
    source_only = [
        {
            "cell_type": cell.get("cell_type"),
            "source": cell.get("source", []),
        }
        for cell in nb.get("cells", [])
    ]
    return sha256_json(source_only)


def make_asset_hashes(root: str | Path, patterns=("*.json", "*.png", "*.html", "*.npz")) -> dict[str, str]:
    """Hash generated artifacts under a directory."""
    root = Path(root)
    paths = []
    for pattern in patterns:
        paths.extend(root.rglob(pattern))
    out = {}
    for path in sorted(set(paths)):
        if path.is_file():
            out[str(path.relative_to(root))] = sha256_file(path)
    return out


def write_asset_hashes(root: str | Path, output_name: str = "asset_hashes.json") -> dict[str, str]:
    """Write asset_hashes.json with strict sorted JSON."""
    root = Path(root)
    hashes = make_asset_hashes(root)
    (root / output_name).write_text(json.dumps(hashes, indent=2, sort_keys=True, allow_nan=False))
    return hashes


def diff_hashes(old: dict[str, str], new: dict[str, str]) -> dict[str, dict[str, str | None]]:
    """Return changed/missing/new file hashes."""
    keys = sorted(set(old) | set(new))
    return {
        k: {"old": old.get(k), "new": new.get(k)}
        for k in keys
        if old.get(k) != new.get(k)
    }
```

For Python 3.9 support, replace `str | Path` with `Union[str, Path]` if needed.

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

Use stable hashes to deduplicate and cache candidate evaluations.

```python
def candidate_sha256(params: dict) -> str:
    return sha256_json(params)

params = {
    "cell.E.drive": 4.5,
    "cell.PV.drive": 3.5,
    "conn.feedforward_gain": 1.2,
}

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

Load with verification:

```python
import numpy as np
from pathlib import Path


def load_weight_artifact(ref: dict, root: str | Path = "."):
    path = Path(root) / ref["path"]
    expected = str(ref["sha256"]).removeprefix("sha256:")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"Weight artifact hash mismatch: {path}\n"
            f"expected={expected}\n"
            f"actual={actual}"
        )
    data = np.load(path)
    return data[ref["array_name"]]
```

For Python 3.8/3.9 compatibility, replace `.removeprefix("sha256:")` with:

```python
expected = expected[7:] if expected.startswith("sha256:") else expected
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
