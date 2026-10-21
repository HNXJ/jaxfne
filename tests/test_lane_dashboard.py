"""Lane dashboard generator (scripts/lane_dashboard/jaxfne_dashboard.py): one card per replication lane,
every field read from the lane folder, the registry, git and the todo and open stacks."""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "scripts" / "lane_dashboard" / "jaxfne_dashboard.py"
BLOB_A = "/_blob/" + "a" * 32
BLOB_B = "/_blob/" + "b" * 32


def _load():
    spec = importlib.util.spec_from_file_location("jaxfne_dashboard_t", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _git(root: Path, *args: str, env: dict | None = None) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


def _png(path: Path) -> None:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (40, 30), (200, 10, 10)).save(buf, "PNG")
    path.write_bytes(buf.getvalue())


def _commit(root: Path, msg: str, when: str, *paths: str) -> None:
    env = {**os.environ, "GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when}
    _git(root, "add", *paths)
    _git(root, "commit", "-q", "-m", msg, env=env)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Two lanes: demo2001 (registry entry, README, render, two receipts, a data file, named by a todo row and
    an open item) and orphan (no registry entry, no README, nothing else)."""
    D = _load()
    root = tmp_path / "repo"
    lane = root / "artifacts" / "etudes" / "papers" / "demo2001"
    lane.mkdir(parents=True)
    (root / "artifacts" / "etudes" / "papers" / "orphan").mkdir()
    (root / "artifacts" / "etudes" / "papers" / "orphan" / "notes.py").write_text("x = 1\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    (lane / "README.md").write_text("# Demo and Doe 2001, Fig. 2\n\nA demo lane. It has two sentences.\n\n"
                                    "## More\n\ntext\n", encoding="utf-8")
    _png(lane / "fig2.png")
    (lane / "a_receipt.json").write_text("{}", encoding="utf-8")
    (lane / "scores").mkdir()
    (lane / "scores" / "p3.json").write_text("{}", encoding="utf-8")
    src = root / "artifacts" / "etudes" / "papers" / "src2009"
    src.mkdir()
    (src / "SOURCE_P1.md").write_text("# Src (2009): P1 source record\n\nStatus: draft.\n\nTarget: A. Author, Fig. 2 paper. More.\n",
                                      encoding="utf-8")
    _commit(root, "first", "2026-01-02T03:04:05+00:00", ".")
    (lane / "b_receipt.json").write_text("{}", encoding="utf-8")
    (lane / "rates.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    _commit(root, "second", "2026-02-03T04:05:06+00:00", ".")
    (root / "dashboard" / "data").mkdir(parents=True)
    (root / "dashboard" / "sources.json").write_text(json.dumps({"papers": [
        {"id": "demo-fig2", "stage": "P3", "stage_artifacts": {
            "P1": [{"label": "readme", "path": "artifacts/etudes/demo2001/README.md"}]}}]}),
        encoding="utf-8")
    (root / "dashboard" / "data" / "replications.json").write_text(json.dumps({
        "generated": {"utc": "2026-02-04T00:00:00Z", "dev_sha": "f" * 40},
        "papers": [{"id": "demo-fig2", "stage": "P3", "class": "reproduced", "release": "0.5.7"}]}),
        encoding="utf-8")
    (root / "artifacts" / "todo_stack.md").write_text(
        "# Todo\n\n| Order | Item about demo2001 | Size |\n|---|---|---|\n| 7 | Demo 2001 scored run | 1 h |\n"
        "| 8 | unrelated | 1 h |\n\n- T-40 [A] Lanes: brunel and\n  demo2001 pilots.\n- T-41 other\n",
        encoding="utf-8")
    (root / "artifacts" / "open_stack.md").write_text(
        "# open\n\n- O-9 demo2001 receipt rerun | keeper | since 2026-01-01 | closes when x\n"
        "- O-10 other | keeper | since 2026-01-01 | closes when y\n", encoding="utf-8")
    monkeypatch.setattr(D, "ROOT", root)
    monkeypatch.setattr(D, "PAPERS", root / "artifacts" / "etudes" / "papers")
    monkeypatch.setattr(D, "ARTIFACTS", root / "artifacts")
    return D, root


def _row(D, lane):
    return next(r for r in D.lane_rows() if r["id"] == lane)


def test_lane_card_reads_every_field_from_the_files(repo):
    D, root = repo
    r = _row(D, "demo2001")
    assert r["title"] == "Demo and Doe 2001, Fig. 2"
    assert r["caption"] == "A demo lane."
    assert (r["registry_ids"], r["stage"], r["class"], r["release"]) == ("demo-fig2", "P3", "reproduced", "0.5.7")
    assert r["standing"] == "P3, reproduced"
    assert r["rendered_utc"] == "2026-01-02T03:04:05Z" and r["render_source"] == "git"
    assert r["caption_source"] == "README.md"
    assert (r["n_receipts"], r["newest_receipt"], r["newest_receipt_utc"]) == (3, "b_receipt.json",
                                                                               "2026-02-03T04:05:06Z")
    assert (r["n_data_files"], r["data_files"]) == (1, "rates.csv")
    assert r["open_items"] == "todo 7, T-40, O-9" and r["n_open_items"] == 3
    head = base64.b64decode(r["thumb"].split(",", 1)[1])[:3]
    assert r["thumb"].startswith("data:image/jpeg;base64,") and head == b"\xff\xd8\xff", "thumbnail is a scaled JPEG"


def test_lane_without_registry_or_readme_says_so(repo):
    D, _ = repo
    r = _row(D, "orphan")
    assert (r["title"], r["caption"], r["standing"], r["thumb"], r["rendered_utc"]) == (
        "orphan", "", "not in the registry", "", "")
    assert (r["n_receipts"], r["n_data_files"], r["n_open_items"]) == (0, 0, 0)


def test_lane_without_readme_reads_its_source_record(repo):
    D, _ = repo
    r = _row(D, "src2009")
    assert (r["title"], r["caption"], r["caption_source"]) == ("Src (2009)", "Target: A. Author, Fig. 2 paper.", "SOURCE_P1.md")


def test_parsing_edge_cases(repo):
    """Review findings 2026-10-20: header rows, code-fence captions, abbreviations, hyphenated names, and a
    registry id with no replications.json row."""
    D, root = repo
    labels = [a for a, _ in D.stack_items()]
    assert "todo Order" not in labels and "todo 7" in labels
    lane = root / "artifacts" / "etudes" / "papers" / "fence"
    lane.mkdir()
    (lane / "README.md").write_text("# Fence\n\n```python\nx = 1\n```\n\n* a list\n\nSee e.g. Fig. 3 vs. Eq. 2 here. "
                                    "Next.\n", encoding="utf-8")
    assert D.readme_fields(lane)[:2] == ("Fence", "See e.g. Fig. 3 vs. Eq. 2 here.")
    assert D.named_by(["pcl"], [("T-1", "the pcl-port lane"), ("T-2", "pcl runs")]) == ["T-2"]
    (root / "dashboard" / "data" / "replications.json").write_text(json.dumps({"papers": []}), encoding="utf-8")
    assert _row(D, "demo2001")["standing"] == "demo-fig2 registered, no row in replications.json"


def test_untracked_render_uses_file_time_and_says_so(repo):
    D, root = repo
    _png(root / "artifacts" / "etudes" / "papers" / "orphan" / "local.png")
    r = _row(D, "orphan")
    assert r["render_source"] == "file time (untracked)" and r["rendered_utc"]


def test_full_run_stamps_commit_and_read_time_into_every_dataset(repo, tmp_path, monkeypatch):
    D, root = repo
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["jaxfne_dashboard.py", "--out", str(out)])
    assert D.main() == 0
    head = _git(root, "rev-parse", "--short", "HEAD")
    for name in ("lanes.json", "analyses.json"):
        d = json.loads((out / name).read_text(encoding="utf-8"))
        assert d["commit"] == head and d["read_utc"].endswith("Z") and d["rows"], name
    for name in ("doc_ds.json", "doc_ds_analyses.json", "doc_index.json", "doc_app.json"):
        assert (out / name).stat().st_size <= 256 * 1024, name


def test_out_inside_the_repository_is_refused(repo, monkeypatch):
    D, root = repo
    monkeypatch.setattr(sys, "argv", ["jaxfne_dashboard.py", "--out", str(root / "dash")])
    with pytest.raises(SystemExit, match="inside the repository"):
        D.main()


def test_asset_refresh_rewrites_only_the_dataset_documents(repo, tmp_path, monkeypatch):
    D, _ = repo
    out = tmp_path / "out"
    out.mkdir()
    stamp = {"rows": [], "commit": "abc1234", "read_utc": "2026-10-01T00:00:00Z"}
    for n in ("lanes.json", "analyses.json"):
        (out / n).write_text(json.dumps(stamp), encoding="utf-8")
    before = {n: hashlib.sha256((out / n).read_bytes()).hexdigest() for n in ("lanes.json", "analyses.json")}
    monkeypatch.setattr(sys, "argv", ["jaxfne_dashboard.py", "--out", str(out), "--asset-url", BLOB_A,
                                      "--analyses-asset-url", BLOB_B])
    assert D.main() == 0
    assert {n: hashlib.sha256((out / n).read_bytes()).hexdigest() for n in before} == before
    a = json.loads((out / "doc_ds.json").read_text(encoding="utf-8"))
    b = json.loads((out / "doc_ds_analyses.json").read_text(encoding="utf-8"))
    assert a["updated"]["at"] == b["updated"]["at"] == "2026-10-01T00:00:00Z"
    assert "abc1234" in a["description"] and (a["source"]["url"], b["source"]["url"]) == (BLOB_A, BLOB_B)


def test_bad_asset_url_is_refused(repo, tmp_path, monkeypatch):
    D, _ = repo
    monkeypatch.setattr(sys, "argv", ["jaxfne_dashboard.py", "--out", str(tmp_path / "o"), "--asset-url",
                                      "C:/Program Files/Git/_blob/" + "a" * 32])
    with pytest.raises(SystemExit, match="_blob"):
        D.main()
