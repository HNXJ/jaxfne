---
name: jaxfne-sha256-artifact-integrity
description: >-
  Use SHA256 for content identity of jaxfne configs, notebooks, models,
  figures, manifests, and release files. Use when checking reproducibility or
  whether an artifact changed.
---

# jaxfne artifact integrity procedure

SHA256 establishes content identity only:

```text
same bytes -> same hash
changed bytes -> different hash
```

It does not establish biological truth, solver validity, calibration, or
scientific correctness. Pair hashes with finite-output, shape, status, and
execution receipts.

## Canonical helpers

Use `scripts/hash_utils.py` rather than reimplementing hashing:

```python
from scripts.hash_utils import (
    sha256_file,
    stable_json_bytes,
    sha256_json,
    notebook_source_sha256,
    make_asset_hashes,
    write_asset_hashes,
    diff_hashes,
    candidate_sha256,
    load_weight_artifact,
)
```

Verify the current export surface before relying on a helper.

## Artifact procedure

1. Identify the source/configuration and output scope.
2. Hash exact source files or stable JSON.
3. Record hashes in the manifest or sidecar.
4. Check finite values and expected shapes separately.
5. Compare before/after hashes and inspect unexpected changes.
6. Preserve write-once receipt behavior where required.

Use `allow_nan=False` for JSON. Never use a content hash to upgrade a
proxy/scaffold status or amplitude claim.

## Validation

Run the relevant hashing helper, strict JSON check, and targeted artifact test.
Report the exact command and hash result. Do not embed current hashes,
versions, or SHAs into persistent skill prose.
