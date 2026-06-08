# Release and archive approval gate

ED10 is an **evidence receipt**, not a release event. Alpha closure preserves this gate.

## Allowed alpha state

| Field | Status |
|---|---|
| release tag | `pending_approval` or `not_created` |
| wheel/sdist | `not_built_without_approval` |
| PyPI/TestPyPI | `not_published_without_approval` |
| GitHub release | `not_created_without_approval` |
| archive/DOI | `pending_approval` / `not_applicable_without_release_approval` |
| clean install smoke | isolated import recorded; full pip install pending approval |

## Forbidden without explicit approval

```bash
git tag
python -m build
twine upload
gh release create
# zenodo / archive DOI actions
```

## ED10 verification (Task 08)

- `scripts/publication/ed10_release_archive_receipt.py` records status only; no tag/build/publish calls.
- Receipt fields: `release_action_taken: false`, `tag_action_taken: false`, `archive_action_taken: false`.
- Unsafe executable actions in ED10 script: **none found** (instruction/status fields only).

## Separate release task (future)

When approved, run from pinned `cur` SHA with:

1. tag at HEAD
2. wheel/sdist build + SHA256
3. clean-room install smoke
4. optional PyPI/GitHub release
5. optional archive/DOI
6. ED10 receipt refresh or supplemental release manifest
