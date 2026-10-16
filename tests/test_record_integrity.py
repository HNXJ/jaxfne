"""Record-integrity gate: past records may move but their bytes may not change.

Fail-first tests for the record-integrity lane: one fixture file per record
class (receipts, archive, legacy, dated, prereg, benchmark, frozen). Per-path
pins (``{rel: sha256}`` plus per-class counts): a one-byte edit fails as
``changed-record`` naming the file, a class-preserving move passes (after a
clean check before the move), a move outside every pinned class fails as
``declassified-move`` (as do launders onto an excluded script or another
pinned path), deleting one of two identical records fails, a within-class
swap fails twice, a deletion fails naming its class, and per-class live counts
must equal the pins (a dropped frozen key around an archived record fails
naming frozen). ``--add`` runs the check first and refuses a ``mv``+edit
launder through the writing function with the hashes file left byte-identical;
a move is repinned only onto a discovered path, and a duplicate-bytes add is
accepted only inside a class with counts consistent. The allowlist holds
exactly the three live prereg scripts, and no script is pinned under dated or
prereg.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_record_integrity import (
    CLASSES,
    NON_RECORD_PATHS,
    _analyze,
    _frozen_paths,
    add_paths,
    check_tree,
    discover_records_by_class,
    find_violations,
    pinned_counts_for,
    plan_add,
    sha256_bytes,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_record_integrity.py"

CLASS_FILES = {
    "receipts": "artifacts/release/v9_9_9_release_receipt.json",
    "archive": "artifacts/archive/note.md",
    "legacy": "artifacts/legacy/note.md",
    "dated": "artifacts/audit/note_2026-10-04.md",
    "prereg": "artifacts/prereg_fixture/note.md",
    "benchmark": "artifacts/runs/benchmark/agent_tasks_999.json",
    "frozen": "artifacts/frozen_data/record.json",
}

SECOND_FILES = {
    "receipts": "artifacts/release/v9_9_8_release_receipt.json",
    "archive": "artifacts/archive/second.md",
    "legacy": "artifacts/legacy/second.md",
    "dated": "artifacts/audit/second_2026-10-05.md",
    "prereg": "artifacts/prereg_fixture/second.md",
    "benchmark": "artifacts/other/benchmark/agent_tasks_998.json",
    "frozen": "artifacts/frozen_data/second.json",
}

CLASS_MOVES = {
    "receipts": "artifacts/release/renamed/v9_9_9_release_receipt.json",
    "archive": "artifacts/archive/renamed/note.md",
    "legacy": "artifacts/legacy/renamed/note.md",
    "dated": "artifacts/other/note_2026-10-04.md",
    "prereg": "artifacts/relocated_prereg/note.md",
    "benchmark": "artifacts/other/benchmark/agent_tasks_999.json",
    "frozen": "artifacts/frozen_data/moved.json",
}

EXPECTED_ALLOWLIST = frozenset(
    [
        "artifacts/e2/preregistration/E2a_search/e2a_blinded_executor.py",
        "artifacts/e2/preregistration/E2a_search/freeze_audit.py",
        "artifacts/e2/preregistration/E2b_confirmatory/e2b_v1_executor.py",
    ]
)


def _record_bytes(rel: str) -> bytes:
    if rel.endswith("_release_receipt.json"):
        return (
            json.dumps(
                {
                    "schema": "jaxfne.release_receipt.v1",
                    "status": "published",
                    "file": rel,
                }
            )
            + "\n"
        ).encode("utf-8")
    if rel.startswith("artifacts/frozen_data/"):
        return json.dumps({"frozen": rel}).encode("utf-8")
    return f"record-bytes-{rel}\n".encode("utf-8")


def _build_tree(root: Path, skip: str = "", extra: bool = False) -> dict[str, bytes]:
    """Write one record per class (except ``skip``); return rel -> bytes."""
    written: dict[str, bytes] = {}
    rels = [CLASS_FILES[cls] for cls in CLASSES if cls != skip]
    if extra:
        rels += [SECOND_FILES[cls] for cls in CLASSES if cls != skip]
    for rel in rels:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _record_bytes(rel)
        path.write_bytes(data)
        written[rel] = data
    manifest = root / "artifacts" / "publication" / "frozen_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    keys = [r for r in rels if r.startswith("artifacts/frozen_data/")]
    manifest.write_text(json.dumps({"files": {key: {} for key in keys}}), encoding="utf-8")
    return written


def _pin_all(root: Path) -> dict[str, dict]:
    """Per-path pins plus per-class counts for every discovered record."""
    paths: dict[str, str] = {}
    for ps in discover_records_by_class(root).values():
        for path in ps:
            rel = path.relative_to(root).as_posix()
            paths[rel] = sha256_bytes(path.read_bytes())
    return {"paths": paths, "counts": pinned_counts_for(paths, _frozen_paths(root))}


def test_live_tree_passes_check():
    """The unmodified working tree passes ``--check``."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, (
        f"--check failed on the live tree:\n{result.stdout}\n{result.stderr}"
    )


@pytest.mark.parametrize("cls", CLASSES)
def test_edited_byte_fails_and_names_file(tmp_path, cls):
    """A one-byte edit of a record fails as changed-record with the file named."""
    written = _build_tree(tmp_path)
    pinned = _pin_all(tmp_path)
    assert find_violations(tmp_path, pinned) == []

    target = tmp_path / CLASS_FILES[cls]
    original = written[CLASS_FILES[cls]]
    edited = bytearray(original)
    if cls == "receipts":
        # Flip a byte inside the schema value so the JSON still parses with
        # its final status (flipping byte 0 would unlist it as a record).
        anchor = original.find(b"receipt")
        assert anchor > 0
        edited[anchor] = (edited[anchor] + 1) % 256
    else:
        edited[0] = (edited[0] + 1) % 256
    target.write_bytes(bytes(edited))

    violations = check_tree(tmp_path, pinned)
    assert any(
        target.name in violation and "changed-record" in violation for violation in violations
    ), f"edit of {cls} names no file: {violations}"


@pytest.mark.parametrize("cls", CLASSES)
def test_moved_record_passes(tmp_path, cls):
    """The same bytes under a class-preserving path still pass."""
    _build_tree(tmp_path)
    pinned = _pin_all(tmp_path)
    assert check_tree(tmp_path, pinned) == []

    src = tmp_path / CLASS_FILES[cls]
    dst = tmp_path / CLASS_MOVES[cls]
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    if cls == "frozen":
        # The manifest key is the frozen record's locator: moving the file
        # means moving its key; the bytes (and their pinned hash) are unchanged.
        manifest = tmp_path / "artifacts" / "publication" / "frozen_manifest.json"
        manifest.write_text(json.dumps({"files": {CLASS_MOVES["frozen"]: {}}}), encoding="utf-8")

    assert check_tree(tmp_path, pinned) == []


def test_move_to_unmatched_path_fails(tmp_path):
    """A move to a path no class matches fails as declassified-move."""
    _build_tree(tmp_path, extra=True)
    pinned = _pin_all(tmp_path)
    assert check_tree(tmp_path, pinned) == []

    src = tmp_path / CLASS_FILES["archive"]
    dst = tmp_path / "artifacts" / "misc" / "note.md"
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)

    violations, moves = _analyze(tmp_path, pinned)
    assert moves == [], f"classless move must not be a move: {moves}"
    assert any(
        "declassified-move" in violation and CLASS_FILES["archive"] in violation
        for violation in violations
    ), f"classless move not reported: {violations}"


def test_move_onto_excluded_script_fails(tmp_path):
    """Bytes laundered onto an excluded script fail as declassified-move."""
    _build_tree(tmp_path, extra=True)
    pinned = _pin_all(tmp_path)
    assert check_tree(tmp_path, pinned) == []

    script = "artifacts/e2/preregistration/E2a_search/e2a_blinded_executor.py"
    src = tmp_path / CLASS_FILES["archive"]
    dst = tmp_path / script
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    src.unlink()

    violations, moves = _analyze(tmp_path, pinned)
    assert moves == [], f"script overwrite must not be a move: {moves}"
    assert any(
        "declassified-move" in violation and script in violation for violation in violations
    ), f"script overwrite not reported: {violations}"


def test_identical_pair_deletion_is_not_a_move(tmp_path):
    """Deleting one of two identical records fails; the twin is no rescue."""
    _build_tree(tmp_path, extra=True)
    first = tmp_path / CLASS_FILES["archive"]
    second = tmp_path / SECOND_FILES["archive"]
    second.write_bytes(first.read_bytes())
    pinned = _pin_all(tmp_path)
    assert check_tree(tmp_path, pinned) == []
    # Sanity: the pair really shares bytes, so only the overwrite distinguishes.
    assert pinned["paths"][CLASS_FILES["archive"]] == (pinned["paths"][SECOND_FILES["archive"]])

    first.unlink()

    violations, moves = _analyze(tmp_path, pinned)
    assert moves == [], f"pinned-path launder must not be a move: {moves}"
    assert any(
        "deleted-record" in violation and CLASS_FILES["archive"] in violation
        for violation in violations
    ), f"pinned-path launder not reported: {violations}"


def test_overwrite_pinned_path_with_other_bytes_fails(tmp_path):
    """Bytes moved onto a live pinned path fail naming both paths."""
    _build_tree(tmp_path, extra=True)
    pinned = _pin_all(tmp_path)
    assert check_tree(tmp_path, pinned) == []

    first = tmp_path / CLASS_FILES["archive"]
    second = tmp_path / SECOND_FILES["archive"]
    second.write_bytes(first.read_bytes())
    first.unlink()

    violations = check_tree(tmp_path, pinned)
    assert any(
        "declassified-move" in violation and SECOND_FILES["archive"] in violation
        for violation in violations
    ), f"pinned-path overwrite not reported: {violations}"


def test_frozen_key_removal_around_archived_record_fails(tmp_path):
    """Dropping a manifest key demotes the record: the frozen count fails."""
    repo = tmp_path / "repo"
    rels = [
        "artifacts/archive/frozen_dual_2026-10-04.md",
        "artifacts/archive/frozen_dual_2026-10-05.md",
    ]
    for rel in rels:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_bytes(f"dual-class record {rel}\n".encode())
    manifest = repo / "artifacts" / "publication" / "frozen_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"files": {rel: {} for rel in rels}}))
    pinned = _pin_all(repo)
    assert set(pinned["paths"]) == set(rels)
    assert check_tree(repo, pinned) == []

    manifest.write_text(json.dumps({"files": {rels[1]: {}}}), encoding="utf-8")

    violations = check_tree(repo, pinned)
    assert any(
        "class-count-mismatch" in violation and "class=frozen" in violation
        for violation in violations
    ), f"frozen demotion not reported: {violations}"


def test_add_after_frozen_key_removal_is_refused_and_writes_nothing(tmp_path):
    """A dropped manifest key is not laundered by ``--add`` of a fresh file."""
    repo = tmp_path / "repo"
    rels = [
        "artifacts/archive/frozen_dual_2026-10-04.md",
        "artifacts/archive/frozen_dual_2026-10-05.md",
    ]
    for rel in rels:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_bytes(f"dual-class record {rel}\n".encode())
    manifest = repo / "artifacts" / "publication" / "frozen_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"files": {rel: {} for rel in rels}}))
    pinned = _pin_all(repo)
    hashes_path = _write_hashes(tmp_path, pinned)
    assert check_tree(repo, pinned) == []
    before = hashes_path.read_bytes()

    manifest.write_text(json.dumps({"files": {rels[1]: {}}}), encoding="utf-8")
    fresh = "artifacts/audit/fresh_2026-10-06.md"
    (repo / fresh).parent.mkdir(parents=True, exist_ok=True)
    (repo / fresh).write_bytes(b"fresh dated record\n")

    assert add_paths([fresh], repo_root=repo, hashes_path=hashes_path) == 1
    assert hashes_path.read_bytes() == before, "refused --add must write nothing"


def test_moved_and_edited_record_fails(tmp_path):
    """A move whose bytes changed fails (missing hash, nowhere to be found)."""
    _build_tree(tmp_path, extra=True)
    pinned = _pin_all(tmp_path)
    assert check_tree(tmp_path, pinned) == []

    src = tmp_path / CLASS_FILES["archive"]
    dst = tmp_path / "artifacts" / "misc" / "note.md"
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    dst.write_bytes(b"edited after move\n")

    violations = check_tree(tmp_path, pinned)
    assert any(
        "deleted-record" in violation and "class=archive" in violation for violation in violations
    ), f"moved-and-edited record not reported: {violations}"


@pytest.mark.parametrize("cls", CLASSES)
def test_deleted_record_fails_and_names_class(tmp_path, cls):
    """Deleting a record fails naming its class."""
    _build_tree(tmp_path)
    pinned = _pin_all(tmp_path)

    (tmp_path / CLASS_FILES[cls]).unlink()

    violations = check_tree(tmp_path, pinned)
    assert any(f"class={cls}" in violation for violation in violations), (
        f"deletion in {cls} names no class: {violations}"
    )


def test_delete_one_of_identical_pair_fails(tmp_path):
    """Two pinned paths sharing bytes: deleting one still fails."""
    _build_tree(tmp_path, extra=True)
    first = tmp_path / CLASS_FILES["archive"]
    second = tmp_path / SECOND_FILES["archive"]
    second.write_bytes(first.read_bytes())
    pinned = _pin_all(tmp_path)
    assert check_tree(tmp_path, pinned) == []

    first.unlink()

    violations = check_tree(tmp_path, pinned)
    assert any(
        "deleted-record" in violation and CLASS_FILES["archive"] in violation
        for violation in violations
    ), f"identical-pair deletion not reported: {violations}"


@pytest.mark.parametrize("cls", CLASSES)
def test_swap_within_class_fails(tmp_path, cls):
    """Swapping two records' bytes fails once per path (multisets go blind)."""
    written = _build_tree(tmp_path, extra=True)
    assert written[CLASS_FILES[cls]] != written[SECOND_FILES[cls]]
    pinned = _pin_all(tmp_path)
    assert check_tree(tmp_path, pinned) == []

    first = tmp_path / CLASS_FILES[cls]
    second = tmp_path / SECOND_FILES[cls]
    first.write_bytes(written[SECOND_FILES[cls]])
    second.write_bytes(written[CLASS_FILES[cls]])

    violations = check_tree(tmp_path, pinned)
    changed = [v for v in violations if "changed-record" in v]
    assert len(changed) == 2, f"swap in {cls} must fail per path: {violations}"
    assert any(first.name in v for v in changed)
    assert any(second.name in v for v in changed)


@pytest.mark.parametrize("cls", CLASSES)
def test_empty_class_fails_only_when_pinned(tmp_path, cls):
    """No live records with a positive pin fails naming the class."""
    _build_tree(tmp_path)
    pinned = _pin_all(tmp_path)
    assert pinned["counts"][cls] > 0, f"fixture must pin {cls}"

    (tmp_path / CLASS_FILES[cls]).unlink()

    violations = check_tree(tmp_path, pinned)
    assert any(
        f"class={cls}" in violation and violation.startswith("class-count-mismatch")
        for violation in violations
    ), f"empty {cls} not reported: {violations}"


def test_empty_unpinned_class_passes(tmp_path):
    """Pinned 0 with live 0 is green, even beside pinned sibling classes."""
    _build_tree(tmp_path, skip="archive")
    pinned = _pin_all(tmp_path)
    assert pinned["counts"]["archive"] == 0
    assert check_tree(tmp_path, pinned) == []


def test_nonfinal_receipt_is_not_a_record(tmp_path):
    """A receipt whose JSON status is not final is excluded from records."""
    src = tmp_path / "artifacts" / "release" / "v9_9_9_release_receipt.json"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(
        (
            json.dumps({"schema": "jaxfne.release_receipt.v1", "status": "gates_pass_not_sealed"})
            + "\n"
        ).encode("utf-8")
    )
    assert discover_records_by_class(tmp_path)["receipts"] == []


def test_allowlist_is_exactly_the_three_live_scripts():
    """Every allowlisted path exists, is a live script, and matches prereg."""
    assert NON_RECORD_PATHS == EXPECTED_ALLOWLIST, (
        f"allowlist drifted: {sorted(set(NON_RECORD_PATHS) ^ EXPECTED_ALLOWLIST)}"
    )
    for rel in NON_RECORD_PATHS:
        assert (ROOT / rel).is_file(), f"stale allowlist entry: {rel}"
        assert rel.endswith(".py"), f"non-script allowlisted: {rel}"
        assert "prereg" in rel.lower(), f"allowlist entry outside prereg: {rel}"
    prereg_rels = {
        p.relative_to(ROOT).as_posix() for p in discover_records_by_class(ROOT)["prereg"]
    }
    leaked = [rel for rel in NON_RECORD_PATHS if rel in prereg_rels]
    assert leaked == [], f"allowlisted paths still pinned: {leaked}"


@pytest.mark.parametrize("cls", ["dated", "prereg"])
def test_no_scripts_in_record_class(cls):
    """No live ``.py`` file is pinned under the dated or prereg classes."""
    scripts = [
        p.relative_to(ROOT).as_posix()
        for p in discover_records_by_class(ROOT)[cls]
        if p.suffix == ".py"
    ]
    assert scripts == [], f"scripts pinned in {cls}: {scripts}"


def _write_hashes(tmp_path: Path, pinned: dict) -> Path:
    hashes_path = tmp_path / "hashes.json"
    hashes_path.write_text(json.dumps(pinned, indent=2, sort_keys=True) + "\n")
    return hashes_path


def test_add_pins_new_record_and_prints_it(tmp_path, capsys):
    """``--add`` of a fresh path pins path->hash and prints path and digest."""
    repo = tmp_path / "repo"
    rel = "artifacts/archive/note.md"
    (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    data = b"fresh record\n"
    (repo / rel).write_bytes(data)
    hashes_path = tmp_path / "hashes.json"

    assert add_paths([rel], repo_root=repo, hashes_path=hashes_path) == 0
    out = capsys.readouterr().out
    digest = sha256_bytes(data)
    assert rel in out and digest in out, f"--add printed no path+digest: {out}"
    saved = json.loads(hashes_path.read_text(encoding="utf-8"))
    assert saved["paths"][rel] == digest
    assert saved["counts"]["archive"] == 1


def test_add_move_then_edit_is_refused_and_writes_nothing(tmp_path):
    """``mv a b``, edit b, ``--add b`` is refused; the hashes file is untouched."""
    repo = tmp_path / "repo"
    rel = "artifacts/archive/note.md"
    (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / rel).write_bytes(b"original record\n")
    hashes_path = tmp_path / "hashes.json"
    assert add_paths([rel], repo_root=repo, hashes_path=hashes_path) == 0
    before = hashes_path.read_bytes()

    moved = "artifacts/archive/renamed.md"
    (repo / moved).parent.mkdir(parents=True, exist_ok=True)
    (repo / rel).rename(repo / moved)
    (repo / moved).write_bytes(b"edited after mv\n")

    assert add_paths([moved], repo_root=repo, hashes_path=hashes_path) == 1
    assert hashes_path.read_bytes() == before, "refused --add must write nothing"


def test_add_repins_a_clean_move(tmp_path):
    """``mv a b`` with unchanged bytes is repinned to the new path."""
    repo = tmp_path / "repo"
    rel = "artifacts/archive/note.md"
    (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / rel).write_bytes(b"original record\n")
    hashes_path = tmp_path / "hashes.json"
    assert add_paths([rel], repo_root=repo, hashes_path=hashes_path) == 0

    moved = "artifacts/archive/renamed.md"
    (repo / rel).rename(repo / moved)
    assert add_paths([moved], repo_root=repo, hashes_path=hashes_path) == 0

    saved = json.loads(hashes_path.read_text(encoding="utf-8"))
    assert moved in saved["paths"] and rel not in saved["paths"]
    assert check_tree(repo, saved) == []


def test_add_refused_by_unrelated_violation(tmp_path):
    """A dirty tree blocks ``--add`` even when the listed path itself is clean."""
    repo = tmp_path / "repo"
    _build_tree(repo)
    pinned = _pin_all(repo)
    hashes_path = _write_hashes(tmp_path, pinned)

    (repo / CLASS_FILES["archive"]).write_bytes(b"tampered\n")
    fresh = "artifacts/legacy/fresh.md"
    (repo / fresh).write_bytes(b"fresh legacy record\n")
    before = hashes_path.read_bytes()

    assert add_paths([fresh], repo_root=repo, hashes_path=hashes_path) == 1
    assert hashes_path.read_bytes() == before, "refused --add must write nothing"


def test_plan_add_reports_move_for_repin(tmp_path):
    """The planner surfaces a clean move as a repin, not a violation."""
    repo = tmp_path / "repo"
    _build_tree(repo)
    pinned = _pin_all(repo)

    moved = "artifacts/archive/renamed.md"
    (repo / CLASS_FILES["archive"]).rename(repo / moved)

    ok, merged, refusals = plan_add(repo, [moved], pinned)
    assert refusals == [], f"clean move must not be refused: {refusals}"
    assert any("repinned-move" in line and moved in line for line in ok), ok
    assert merged["paths"][moved] == pinned["paths"][CLASS_FILES["archive"]]
    assert CLASS_FILES["archive"] not in merged["paths"]


def test_add_move_to_undiscovered_path_is_refused_and_writes_nothing(tmp_path):
    """A move outside every class is no repin: ``--add`` refuses, writing nothing."""
    repo = tmp_path / "repo"
    _build_tree(repo, extra=True)
    pinned = _pin_all(repo)
    hashes_path = _write_hashes(tmp_path, pinned)
    before = hashes_path.read_bytes()

    moved = "artifacts/misc/note.md"
    (repo / moved).parent.mkdir(parents=True, exist_ok=True)
    (repo / CLASS_FILES["archive"]).rename(repo / moved)

    assert add_paths([moved], repo_root=repo, hashes_path=hashes_path) == 1
    assert hashes_path.read_bytes() == before, "refused --add must write nothing"


def test_add_duplicate_bytes_in_class_is_accepted(tmp_path):
    """A new in-class path holding a pinned record's bytes pins cleanly."""
    repo = tmp_path / "repo"
    _build_tree(repo, extra=True)
    pinned = _pin_all(repo)
    hashes_path = _write_hashes(tmp_path, pinned)

    dup = "artifacts/archive/duplicate.md"
    (repo / dup).write_bytes((repo / CLASS_FILES["archive"]).read_bytes())

    assert add_paths([dup], repo_root=repo, hashes_path=hashes_path) == 0
    saved = json.loads(hashes_path.read_text(encoding="utf-8"))
    assert saved["paths"][dup] == pinned["paths"][CLASS_FILES["archive"]]
    assert saved["counts"] == pinned_counts_for(saved["paths"], _frozen_paths(repo))
    assert check_tree(repo, saved) == []


def test_add_duplicate_bytes_outside_class_is_refused(tmp_path):
    """Duplicate bytes at a classless path are refused; nothing is written."""
    repo = tmp_path / "repo"
    _build_tree(repo, extra=True)
    pinned = _pin_all(repo)
    hashes_path = _write_hashes(tmp_path, pinned)
    before = hashes_path.read_bytes()

    dup = "artifacts/misc/duplicate.md"
    (repo / dup).parent.mkdir(parents=True, exist_ok=True)
    (repo / dup).write_bytes((repo / CLASS_FILES["archive"]).read_bytes())

    assert add_paths([dup], repo_root=repo, hashes_path=hashes_path) == 1
    assert hashes_path.read_bytes() == before, "refused --add must write nothing"
