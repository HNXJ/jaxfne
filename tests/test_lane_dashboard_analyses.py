"""Lane dashboard, analyses dataset: one row per saved analysis output under artifacts/, rows counted from
the files, write time from git where tracked."""
from __future__ import annotations

import gzip
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "scripts" / "lane_dashboard" / "jaxfne_dashboard.py"


def _load():
    spec = importlib.util.spec_from_file_location("jaxfne_dashboard_t", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_analysis_rows_kinds_counts_and_skipped_folders(tmp_path, monkeypatch):
    D = _load()
    root = tmp_path / "repo"
    lane = root / "artifacts" / "etudes" / "papers" / "demo2001"
    lane.mkdir(parents=True)
    (lane / "rates.csv").write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
    with gzip.open(lane / "spikes.csv.gz", "wb") as fh:
        fh.write(b"t\n1\n2\n")
    (lane / "run_receipt.json").write_text("{}", encoding="utf-8")
    (lane / "log.jsonl").write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    (lane / "w.npz").write_bytes(b"npz")
    (lane / "README.md").write_text("# x\n", encoding="utf-8")
    (lane / "_temp_x.csv").write_text("a\n1\n", encoding="utf-8")
    runs = root / "artifacts" / "runs" / "perf"
    runs.mkdir(parents=True)
    (runs / "timing.json").write_text("{}", encoding="utf-8")
    skip = root / "artifacts" / "control" / "harness"
    skip.mkdir(parents=True)
    (skip / "manifest.json").write_text("{}", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for k, v in (("user.email", "t@example.com"), ("user.name", "t")):
        subprocess.run(["git", "-C", str(root), "config", k, v], check=True)
    subprocess.run(["git", "-C", str(root), "add", "artifacts/etudes/papers/demo2001/rates.csv"], check=True)
    env = {**os.environ, "GIT_AUTHOR_DATE": "2026-03-04T05:06:07+00:00",
           "GIT_COMMITTER_DATE": "2026-03-04T05:06:07+00:00"}
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "m"], check=True, env=env)
    monkeypatch.setattr(D, "ROOT", root)
    monkeypatch.setattr(D, "ARTIFACTS", root / "artifacts")
    monkeypatch.setattr(D, "PAPERS", root / "artifacts" / "etudes" / "papers")
    rows = {r["path"].split("/")[-1]: r for r in D.analysis_rows()}
    assert set(rows) == {"rates.csv", "spikes.csv.gz", "run_receipt.json", "log.jsonl", "w.npz", "timing.json"}
    assert rows["rates.csv"]["n_rows"] == 3 and rows["spikes.csv.gz"]["n_rows"] == 2 and rows["log.jsonl"]["n_rows"] == 2
    assert rows["w.npz"]["n_rows"] == "" and rows["w.npz"]["kind"] == "array"
    assert rows["run_receipt.json"]["kind"] == "receipt" and rows["timing.json"]["kind"] == "json"
    assert rows["rates.csv"]["area"] == "papers/demo2001" and rows["timing.json"]["area"] == "runs/perf"
    assert (rows["rates.csv"]["tracked"], rows["rates.csv"]["written_utc"], rows["rates.csv"]["written_source"]) == (
        True, "2026-03-04T05:06:07Z", "git")
    assert rows["log.jsonl"]["tracked"] is False and rows["log.jsonl"]["written_source"] == "file time (untracked)"
