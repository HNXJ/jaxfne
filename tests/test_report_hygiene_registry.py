"""The hash registry names pinned legacy paths; staging it must not fail the hygiene gate.

``scripts/record_hashes.json`` is staged whenever a prereg is pinned, and its keys include
``artifacts/legacy/...`` paths, which the gate excludes as files. The registry is excluded too.
"""

from scripts.report_hygiene_check import REPO_ROOT, is_excluded, scan_file


def test_record_hash_registry_excluded():
    registry = REPO_ROOT / "scripts" / "record_hashes.json"
    assert scan_file(registry), "fixture must hold a pattern the gate forbids"
    assert is_excluded(registry)
