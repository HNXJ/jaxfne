"""Tests for scripts/build_dashboard_data.py.

Each test commits fake scorer JSONs to a throwaway git repo under pytest's tmp_path, copies the
generator into that repo and runs it as a subprocess. No network. DASH_GEN_BUILD_SCRIPT may name
another script to test; it is used to show that these tests fail against a stub.
"""

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path(os.environ.get("DASH_GEN_BUILD_SCRIPT", REPO / "scripts" / "build_dashboard_data.py"))
REPO_URL = "https://github.com/hnxj/jaxfne"
REL_OUT = Path("dashboard") / "data" / "replications.json"
NO_VALUE = "—"


def git(root, *args):
    proc = subprocess.run(["git", *args], cwd=str(root), capture_output=True, stdin=subprocess.DEVNULL)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    return proc.stdout.decode("utf-8").strip()


def exists_at(root, sha, path):
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}:{path}"], cwd=str(root), capture_output=True, stdin=subprocess.DEVNULL
    )
    return probe.returncode == 0


def blob_bytes(root, rev, path):
    """The bytes git holds at rev:path (git show; honours replace refs, as the generator does)."""
    proc = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=str(root), capture_output=True,
                          stdin=subprocess.DEVNULL)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    return proc.stdout


def make_repo(root, files):
    """Commit `files` ({path: JSON-able object}), then copy the generator and the references page (the
    study-list source) in, uncommitted. Returns the SHA."""
    root.mkdir(parents=True)
    git(root, "init", "-q")
    git(root, "config", "user.name", "test")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "commit.gpgsign", "false")
    for rel, obj in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(json.dumps(obj), encoding="utf-8")
    git(root, "add", "--", *files)
    git(root, "commit", "-q", "-m", "fixture scores")
    (root / "scripts").mkdir()
    shutil.copy(SCRIPT, root / "scripts" / "build_dashboard_data.py")
    (root / "docs" / "reference").mkdir(parents=True)
    shutil.copy(REPO / "docs" / "reference" / "references.md", root / "docs" / "reference" / "references.md")
    (root / "dashboard").mkdir(exist_ok=True)
    shutil.copy(REPO / "dashboard" / "replications.schema.json", root / "dashboard" / "replications.schema.json")
    return git(root, "rev-parse", "HEAD")


def write_sources(root, papers, **extra):
    (root / "dashboard").mkdir(exist_ok=True)
    doc = {"schema": "jaxfne-dashboard-sources/2", "repo_url": REPO_URL, "papers": papers, **extra}
    (root / "dashboard" / "sources.json").write_text(json.dumps(doc), encoding="utf-8")


def paper(sha, **over):
    """A minimal valid paper entry; keyword arguments replace or add fields."""
    base = {
        "id": "fixture", "title": "Fixture", "citation": "test", "release": "0.0.0", "tier": "T-test",
        "class": "in progress",
        "stage": "P0", "lane": {"branch": "test", "sha": sha}, "summary": "fixture",
        "preregs": [], "files": {}, "criteria": [], "sensitivity": [], "charts": [], "open": [],
    }
    base.update(over)
    return base


def run(root, *args):
    return subprocess.run(
        [sys.executable, str(root / "scripts" / "build_dashboard_data.py"), *args],
        cwd=str(root), capture_output=True, text=True, encoding="utf-8", errors="replace",
        stdin=subprocess.DEVNULL,
    )


def load_out(root):
    return json.loads((root / REL_OUT).read_text(encoding="utf-8"))


def load_module(path):
    spec = importlib.util.spec_from_file_location("build_dashboard_data_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def criterion(**over):
    base = {
        "id": "R1", "group": "", "claim": "c", "estimand": "e", "criterion": "k", "unit": "",
        "value_label": "v", "verdict": {"file": "main", "key": "ok"}, "prereg": "v1",
    }
    base.update(over)
    return base


def test_per_run_where_and_seed_split(tmp_path):
    runs = []
    for seed in (3, 1, 2):
        for p in (0.07, 0.02, 0.05):
            runs.append({"seed": seed, "p": p, "E": seed + p})
    runs.append({"seed": 9, "p": 0.07, "E": 9.07})
    tolerance_item = next(r for r in runs if r["seed"] == 2 and r["p"] == 0.07)
    tolerance_item["p"] = 0.07 + 1e-12  # stored a hair off 0.07; the where match must still take it
    # fixture preconditions
    assert tolerance_item["p"] != 0.07 and abs(tolerance_item["p"] - 0.07) < 1e-9
    assert sum(abs(r["p"] - 0.07) <= 1e-9 for r in runs) == 4
    assert all(r["seed"] != 5 for r in runs), "seed 5 must be absent from the data"

    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": {"per_run": runs}})
    chart = {
        "id": "c", "kind": "lines", "title": "t", "x": {"label": "p"}, "y": {"label": "E"},
        "series": [
            {"group": "g", "file": "main", "from": "per_run", "where": {}, "x": "p", "y": "E",
             "split": "seed", "seeds": [1, 2, 3, 5]},
            {"group": "ref", "file": "main", "from": "per_run", "where": {"p": 0.07}, "x": "seed", "y": "E",
             "markers_only": True},
        ],
        "refs": [], "caption": "c",
    }
    write_sources(root, [paper(sha, files={"main": "scores/main.json"}, charts=[chart])])

    proc = run(root)
    assert proc.returncode == 0, proc.stderr
    traces = {t["name"]: t for t in load_out(root)["papers"][0]["charts"][0]["traces"]}
    assert sorted(traces) == ["g seed 1", "g seed 2", "g seed 3", "ref"], "seed 5 is omitted, the rest kept"
    assert traces["g seed 1"]["x"] == pytest.approx([0.02, 0.05, 0.07])
    assert traces["g seed 1"]["y"] == pytest.approx([1.02, 1.05, 1.07])
    assert traces["g seed 2"]["x"] == pytest.approx([0.02, 0.05, 0.07]), "the 1e-12 point must be kept"
    assert traces["g seed 2"]["x"] == sorted(traces["g seed 2"]["x"]), "split traces are sorted by x"
    assert sorted(traces["ref"]["x"]) == [1, 2, 3, 9]
    assert traces["ref"]["markers_only"] is True


def test_allow_missing_file_gives_pending_and_drops_its_series(tmp_path):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": {"per_run": [{"p": 0.07, "seed": 1, "E": 2.0}],
                                                "val": 1.5, "ok": True}})
    assert not exists_at(root, sha, "scores/v7.json"), "fixture must lack the optional file"
    r9 = criterion(id="R9", value={"file": "v7", "key": "r4.smallest"}, unit="",
                   extra=[{"label": "n", "file": "v7", "key": "r4.n"}],
                   verdict={"file": "v7", "key": "r4.pass"}, prereg="v7")
    r1 = criterion(id="R1", value={"file": "main", "key": "val"}, unit="Hz")
    chart = {
        "id": "c", "kind": "lines", "title": "t", "x": {"label": "p"}, "y": {"label": "E"},
        "series": [
            {"group": "kept", "file": "main", "from": "per_run", "where": {}, "x": "p", "y": "E"},
            {"group": "dropped", "file": "v7", "from": "per_run", "where": {}, "x": "p", "y": "E"},
        ],
        "refs": [], "caption": "",
    }
    write_sources(root, [paper(sha, files={"main": "scores/main.json",
                                           "v7": {"path": "scores/v7.json", "allow_missing": True}},
                               criteria=[r1, r9], charts=[chart])])

    proc = run(root)
    assert proc.returncode == 0, proc.stderr
    out = load_out(root)["papers"][0]
    got_r1, got_r9 = out["criteria"]
    assert got_r1["verdict"] == "pass" and got_r1["value"] == 1.5 and got_r1["value_text"] == "1.5"
    assert got_r9["value"] is None and got_r9["value_text"] == NO_VALUE
    assert got_r9["verdict"] == "pending"
    assert got_r9["evidence"] == {"path": "scores/v7.json", "url": None}
    assert got_r9["extra"] == [{"label": "n", "value": None, "value_text": NO_VALUE}]
    assert [t["name"] for t in out["charts"][0]["traces"]] == ["kept"], "the absent file's series is dropped"


def test_dev_lane_sha_off_head_history_is_refused(tmp_path):
    # A rebase leaves the old lane SHA in a local clone, but CI's clone of dev lacks it.
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": {"val": 1.5, "ok": True}})
    stray = git(root, "commit-tree", f"{sha}^{{tree}}", "-m", "rebased away")
    probe = subprocess.run(["git", "merge-base", "--is-ancestor", stray, "HEAD"], cwd=str(root),
                           capture_output=True, stdin=subprocess.DEVNULL)
    assert probe.returncode == 1, "fixture must build a commit off HEAD's history"
    write_sources(root, [paper(stray, lane={"branch": "dev", "sha": stray}, files={"main": "scores/main.json"})])

    proc = run(root)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "fixture" in proc.stderr and stray in proc.stderr and "dev" in proc.stderr, proc.stderr

    write_sources(root, [paper(sha, lane={"branch": "dev", "sha": sha}, files={"main": "scores/main.json"})])
    assert run(root).returncode == 0


@pytest.mark.parametrize("kind", ["missing_key", "missing_file"])
def test_missing_required_key_or_file_exits_1_and_names_it(tmp_path, kind):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": {"val": 1.5, "ok": True}})
    if kind == "missing_key":
        assert "nope" not in json.loads((root / "scores/main.json").read_text(encoding="utf-8"))
        value, files, needle, name = ({"file": "main", "key": "nope.value"}, {"main": "scores/main.json"},
                                      "nope.value", "main")
    else:
        assert not exists_at(root, sha, "scores/gone.json"), "fixture must lack the required file"
        value, files, needle, name = ({"file": "gone", "key": "val"},
                                      {"main": "scores/main.json", "gone": "scores/gone.json"},
                                      "scores/gone.json", "gone")
    write_sources(root, [paper(sha, files=files, criteria=[criterion(value=value)])])

    proc = run(root)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "fixture" in proc.stderr and name in proc.stderr and needle in proc.stderr, proc.stderr
    assert not (root / REL_OUT).exists(), "no data file may be written on error"


def test_pass_fail_verdict_form(tmp_path):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": {"v": 1.0, "a": "pass", "b": "fail", "c": "incomplete",
                                                "d": "maybe"}})
    crits = [criterion(id=k, value={"file": "main", "key": "v"},
                       verdict={"file": "main", "key": k, "form": "pass_fail"}) for k in "abc"]
    write_sources(root, [paper(sha, files={"main": "scores/main.json"}, criteria=crits)])
    proc = run(root)
    assert proc.returncode == 0, proc.stderr
    got = [c["verdict"] for c in load_out(root)["papers"][0]["criteria"]]
    assert got == ["pass", "fail", "fail"]
    bad = [criterion(id="d", value={"file": "main", "key": "v"},
                     verdict={"file": "main", "key": "d", "form": "pass_fail"})]
    write_sources(root, [paper(sha, files={"main": "scores/main.json"}, criteria=bad)])
    proc = run(root)
    assert proc.returncode == 1 and "is not pass, fail or incomplete" in proc.stderr, proc.stderr


@pytest.mark.parametrize("energies, want_verdict, want_reason", [
    ([0.5, 0.3, 0.2], "pass", "source reason"),
    ([0.5, 0.05, 0.2], "fail", "E-rate floor not met"),
])
def test_floor_keeps_or_fails_a_true_verdict(tmp_path, energies, want_verdict, want_reason):
    runs = [{"p": 0.02, "seed": s, "rates_Hz": {"E": e}} for s, e in zip((1, 2, 3), energies)]
    runs.append({"p": 0.07, "seed": 1, "rates_Hz": {"E": 9.0}})  # outside the floor's where
    selected = [r for r in runs if abs(r["p"] - 0.02) <= 1e-9]
    assert len(selected) == 3, "fixture must select three runs at p 0.02"
    if want_verdict == "fail":
        assert min(r["rates_Hz"]["E"] for r in selected) < 0.1, "fixture must break the floor"

    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/v7.json": {"per_run": runs,
                                              "r4": {"pass": True, "smallest_oscillating_p": 0.02}}})
    floor = {"file": "v7", "from": "per_run", "where": {"p": 0.02}, "key": "rates_Hz.E", "min": 0.1}
    c = criterion(id="R4v7", value={"file": "v7", "key": "r4.smallest_oscillating_p"},
                  verdict={"file": "v7", "key": "r4.pass"}, floor=floor, reason="source reason",
                  prereg="v7")
    write_sources(root, [paper(sha, files={"v7": "scores/v7.json"}, criteria=[c])])

    proc = run(root)
    assert proc.returncode == 0, proc.stderr
    got = load_out(root)["papers"][0]["criteria"][0]
    assert got["value"] == 0.02
    assert got["verdict"] == want_verdict
    assert got["reason"] == want_reason


def test_bars_keys_and_pairs(tmp_path):
    main = {"a1": 1.5, "a2": 2.5, "b1": 3.5, "b2": 4.5}
    s1 = {"a2": 8.0}
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": main, "scores/S1.json": s1})
    chart = {
        "id": "bars", "kind": "bars", "title": "t", "categories": ["c1", "c2"], "y": {"label": "v"},
        "series": [
            {"group": "keys", "file": "main", "keys": ["a1", "a2"]},
            {"group": "pairs", "pairs": [["main", "b1"], ["S1", "a2"]]},
        ],
        "refs": [{"kind": "hline", "y": 2, "label": "l"}], "caption": "cap",
    }
    assert len(chart["series"][0]["keys"]) == len(chart["categories"]) == len(chart["series"][1]["pairs"])
    write_sources(root, [paper(sha, files={"main": "scores/main.json", "S1": "scores/S1.json"}, charts=[chart])])

    proc = run(root)
    assert proc.returncode == 0, proc.stderr
    got = load_out(root)["papers"][0]["charts"][0]
    assert got["categories"] == ["c1", "c2"]
    assert got["traces"] == [
        {"name": "keys", "group": "keys", "y": [1.5, 2.5]},
        {"name": "pairs", "group": "pairs", "y": [3.5, 8.0]},
    ]
    assert got["evidence"] == {"path": "scores/main.json",
                               "url": f"{REPO_URL}/blob/{sha}/scores/main.json",
                               "sha256": hashlib.sha256(blob_bytes(root, sha, "scores/main.json")).hexdigest()}


def test_check_flags_a_changed_value_and_ignores_generated(tmp_path):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": {"val": 1.5, "ok": True}})
    write_sources(root, [paper(sha, files={"main": "scores/main.json"},
                               criteria=[criterion(value={"file": "main", "key": "val"})])])
    assert run(root).returncode == 0
    fresh = run(root, "--check")
    assert fresh.returncode == 0 and "fresh" in fresh.stdout, fresh.stderr

    out = root / REL_OUT
    doc = load_out(root)
    assert doc["papers"][0]["criteria"][0]["value"] == 1.5, "fixture value must be 1.5 before the edit"
    doc["papers"][0]["criteria"][0]["value"] = 2.5
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    stale = run(root, "--check")
    assert stale.returncode == 1, stale.stdout
    assert "papers[0].criteria[0].value" in stale.stderr, stale.stderr

    doc["papers"][0]["criteria"][0]["value"] = 1.5
    doc["generated"]["utc"] = "1999-01-01T00:00:00Z"
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    assert run(root, "--check").returncode == 0, "only generated differs, so the check must pass"


def test_sensitivity_caminos_form_uses_prefix_reference_and_range(tmp_path):
    main = {"per_run": [
        {"p": 0.07, "seed": 1, "rates_Hz": {"E": 1.0}, "f": 33.0},
        {"p": 0.07, "seed": 2, "rates_Hz": {"E": 2.0}, "f": 34.0},
        {"p": 0.07, "seed": 3, "rates_Hz": {"E": 3.0}, "f": 35.0},
        {"p": 0.02, "seed": 1, "rates_Hz": {"E": 50.0}, "f": 99.0},
    ]}
    sens = {"S1": {"rates_Hz": {"E": 1.5}, "f": 33.5}, "S4": {"rates_Hz": {"E": None}, "f": None}}
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": main, "scores/sens.json": sens})
    assert len([r for r in main["per_run"] if r["p"] == 0.07 and r["seed"] == 1]) == 1
    write_sources(root, [paper(
        sha, files={"main": "scores/main.json", "sens": "scores/sens.json"},
        sensitivity=[{"id": "S1", "label": "l1", "status": "agrees", "note": "n1"},
                     {"id": "S4", "label": "l4", "status": "invalid", "note": "n4"}],
        sensitivity_metrics=[{"metric": "f_pk", "unit": "Hz", "key": "f"},
                             {"metric": "E rate", "unit": "Hz", "key": "rates_Hz.E"}],
        sensitivity_source={"file": "sens", "key_prefix": "<id>."},
        sensitivity_ref={"file": "main", "from": "per_run", "where": {"p": 0.07, "seed": 1}},
        sensitivity_range={"file": "main", "from": "per_run", "where": {"p": 0.07}},
    )])

    proc = run(root)
    assert proc.returncode == 0, proc.stderr
    s1, s4 = load_out(root)["papers"][0]["sensitivity"]
    f, e = s1["rows"]
    assert (f["value"], f["ref"], f["ref_range"]) == (33.5, 33.0, [33.0, 35.0])
    assert (e["value"], e["ref"], e["ref_range"]) == (1.5, 1.0, [1.0, 3.0])
    assert s4["rows"][0]["value"] is None and s4["rows"][1]["value"] is None
    assert s1["status"] == "agrees" and s1["note"] == "n1"


def test_sensitivity_pcl_form_applies_prefix_and_skip(tmp_path):
    files_json = {
        "scores/main.json": {"t1": 3.0, "t2": 1.2, "mx": 0.5},
        "scores/S1.json": {"t1": 2.9, "t2": 1.1, "mx": 0.4},
        "scores/S2.json": {"s2_t1": 3.1, "s2_t2": 1.3},
    }
    assert "s2_mx" not in files_json["scores/S2.json"], "the skip is what keeps S2 valid"
    root = tmp_path / "repo"
    sha = make_repo(root, files_json)
    write_sources(root, [paper(
        sha, files={"main": "scores/main.json", "S1": "scores/S1.json", "S2": "scores/S2.json"},
        sensitivity=[{"id": "S1", "label": "l1", "status": "differs", "file": "S1", "note": "n1"},
                     {"id": "S2", "label": "l2", "status": "agrees", "file": "S2", "keys_prefix": "s2_",
                      "note": "n2"}],
        sensitivity_metrics=[{"metric": "R1", "key": "t1"}, {"metric": "R2", "key": "t2"},
                             {"metric": "MX", "key": "mx", "skip_with_prefix": True}],
        sensitivity_ref={"file": "main"},
    )])

    proc = run(root)
    assert proc.returncode == 0, proc.stderr
    s1, s2 = load_out(root)["papers"][0]["sensitivity"]
    assert [r["metric"] for r in s1["rows"]] == ["R1", "R2", "MX"]
    assert [r["metric"] for r in s2["rows"]] == ["R1", "R2"]
    assert (s2["rows"][0]["value"], s2["rows"][0]["ref"], s2["rows"][0]["ref_range"]) == (3.1, 3.0, None)
    assert (s1["rows"][2]["value"], s1["rows"][2]["ref"]) == (0.4, 0.5)


def test_value_text_three_significant_figures():
    module = load_module(SCRIPT)
    cases = [
        (33.8, "33.8"), (7.040000000000001, "7.04"), (5.2966, "5.3"), (0.02, "0.02"),
        (9.996, "10"), (-0.12171, "-0.122"), (1234.7, "1235"), (560, "560"), (34.0, "34"),
        (None, NO_VALUE), (1.5e-05, "1.5e-05"),
    ]
    assert len(cases) >= 8
    for value, want in cases:
        assert module.value_text(value) == want, f"value_text({value!r}) != {want!r}"


@pytest.mark.parametrize("over, needle", [
    ({"class": "replicated"}, "class 'replicated'"),
    ({"class": "not reproduced", "preregs": [{"id": "v1", "signed": "2026-01-01", "path": "prereg.md"}]},
     "class 'not reproduced' is not one of"),
    ({"class": "verified", "preregs": [{"id": "v1", "signed": "2026-01-01", "path": "prereg.md"}]},
     "class 'verified' is not one of"),
    ({"class": "demonstration"}, "class 'demonstration' is not one of"),
    ({"class": "reproduced", "preregs": []}, "needs a signed preregistration"),
    ({"class": "failed", "preregs": []}, "needs a signed preregistration"),
    ({"class": "not started", "verdict": "pass"}, "a not-started paper carries no verdict and no criteria"),
    ({"class": "reproduced (conditional)", "verdict": "pass",
      "preregs": [{"id": "v1", "signed": "2026-01-01", "path": "prereg.md"}]},
     "a conditional reproduction states its condition in verdict_note"),
    ({"class": "mixed", "verdict": "mixed",
      "preregs": [{"id": "v1", "signed": "2026-01-01", "path": "prereg.md"}]},
     "a mixed verdict names what reproduced and what missed in verdict_note"),
    ({"class": "mixed", "verdict_note": "f_pk reproduced; lag missed", "preregs": []},
     "needs a signed preregistration"),
])
def test_class_rules_refuse_and_name_the_paper(tmp_path, over, needle):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": {"val": 1.5}})
    write_sources(root, [paper(sha, **over)])

    proc = run(root)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "fixture" in proc.stderr and needle in proc.stderr, proc.stderr
    assert not (root / REL_OUT).exists(), "no data file may be written on error"


def test_in_progress_builds_with_a_stage_and_no_verdict(tmp_path):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": {"val": 1.5}})
    spec = paper(sha, **{"class": "in progress", "stage": "P3"})
    assert "verdict" not in spec and spec["preregs"] == [], "fixture must carry no verdict and no prereg"
    write_sources(root, [spec])

    proc = run(root)
    assert proc.returncode == 0, proc.stderr
    got = load_out(root)["papers"][0]
    assert (got["class"], got["stage"], got["verdict"], got["preregs"]) == ("in progress", "P3", None, [])


def test_in_progress_with_a_verdict_is_refused(tmp_path):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": {"val": 1.5}})
    write_sources(root, [paper(sha, **{"class": "in progress", "stage": "P3", "verdict": "pass"})])

    proc = run(root)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "fixture" in proc.stderr and "an in-progress paper carries no verdict" in proc.stderr, proc.stderr
    assert not (root / REL_OUT).exists(), "no data file may be written on error"


def test_in_progress_with_an_empty_stage_is_refused(tmp_path):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": {"val": 1.5}})
    write_sources(root, [paper(sha, **{"class": "in progress", "stage": ""})])

    proc = run(root)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "fixture" in proc.stderr and "an in-progress paper needs a non-empty stage" in proc.stderr, proc.stderr
    assert not (root / REL_OUT).exists(), "no data file may be written on error"


def replica(*statuses, **over):
    """A replica block with one panel per status, ids F1..Fn."""
    block = {"scope": "fixture paper", "panels": [{"id": f"F{i + 1}", "status": s} for i, s in enumerate(statuses)]}
    block.update(over)
    return block


def test_replica_score_counts_each_result_panel(tmp_path):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": {"val": 1.5}})
    block = replica("pass", "pass", "partial", "fail", "not run",
                    excluded=[{"id": "F0", "reason": "schematic"}])
    write_sources(root, [paper(sha, replica=block)])

    proc = run(root)
    assert proc.returncode == 0, proc.stderr
    got = load_out(root)["papers"][0]["replica"]
    # (1 + 1 + 0.5 + 0 + 0) / 5 = 0.5; the excluded schematic is outside the denominator.
    assert got["score"] == 50, got
    assert got["counts"] == {"pass": 2, "partial": 1, "fail": 1, "not run": 1}, got
    assert (got["panels"], got["excluded"], got["same_as"]) == (5, 1, None), got


def test_replica_absent_is_unscored(tmp_path):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": {"val": 1.5}})
    spec = paper(sha)
    assert "replica" not in spec, "fixture must carry no replica block"
    write_sources(root, [spec])

    proc = run(root)
    assert proc.returncode == 0, proc.stderr
    got = load_out(root)["papers"][0]["replica"]
    assert (got["score"], got["panels"]) == (None, 0), got


def test_replica_same_as_takes_the_named_paper_score(tmp_path):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": {"val": 1.5}})
    first = paper(sha, id="paper-a", replica=replica("pass", "fail", "fail", "fail"))
    second = paper(sha, id="paper-b", replica={"same_as": "paper-a"})
    write_sources(root, [first, second])

    proc = run(root)
    assert proc.returncode == 0, proc.stderr
    got = {p["id"]: p["replica"] for p in load_out(root)["papers"]}
    assert got["paper-a"]["score"] == got["paper-b"]["score"] == 25, got
    assert (got["paper-a"]["same_as"], got["paper-b"]["same_as"]) == (None, "paper-a"), got


@pytest.mark.parametrize("block, needle", [
    (replica("pass", "done"), "replica panel F2 status 'done'"),
    ({"scope": "s", "panels": []}, "replica needs at least one panel"),
    (replica("pass", "pass", panels=[{"id": "F1", "status": "pass"}, {"id": "F1", "status": "fail"}]),
     "replica panel id 'F1' repeats"),
    ({"same_as": "nowhere"}, "replica same_as 'nowhere' names no paper with panels"),
    (replica("pass", panels=[{"id": "F1", "status": "pass", "evidence": "scores/absent.json"}]),
     "replica panel F1 evidence 'scores/absent.json' does not exist"),
    (replica("pass", excluded=[{"id": "F1", "reason": "schematic"}]), "replica id 'F1' is both scored and excluded"),
])
def test_replica_refuses_and_names_the_paper(tmp_path, block, needle):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": {"val": 1.5}})
    write_sources(root, [paper(sha, replica=block)])

    proc = run(root)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "fixture" in proc.stderr and needle in proc.stderr, proc.stderr
    assert not (root / REL_OUT).exists(), "no data file may be written on error"


def test_stage_artifacts_link_lane_files_in_stage_order(tmp_path):
    root = tmp_path / "repo"
    sha = make_repo(root, {"spec.json": {}, "prereg.json": {}, "result.json": {}})
    artifacts = {"P5": [{"label": "result", "path": "result.json"}],
                 "P2": [{"label": "port spec", "path": "spec.json"}],
                 "P3": [{"label": "prereg", "path": "prereg.json"}]}
    write_sources(root, [paper(sha, stage="P5", stage_artifacts=artifacts)])

    proc = run(root)
    assert proc.returncode == 0, proc.stderr
    got = load_out(root)["papers"][0]["stage_artifacts"]
    assert [s["stage"] for s in got] == ["P2", "P3", "P5"], "stages must come out in pipeline order"
    assert got[0]["items"] == [{"label": "port spec", "path": "spec.json",
                                "url": f"{REPO_URL}/blob/{sha}/spec.json",
                                "sha256": hashlib.sha256(blob_bytes(root, sha, "spec.json")).hexdigest()}]


@pytest.mark.parametrize("artifacts, needle", [
    ({"P6": [{"label": "x", "path": "spec.json"}]}, "past the paper's stage P5"),
    ({"P9": [{"label": "x", "path": "spec.json"}]}, "key 'P9' is not one of"),
    ({"P2": [{"label": "x", "path": "absent.json"}]}, "P2 artifact absent.json is missing at lane sha"),
])
def test_stage_artifacts_refuse_and_name_the_paper(tmp_path, artifacts, needle):
    root = tmp_path / "repo"
    sha = make_repo(root, {"spec.json": {}})
    assert not exists_at(root, sha, "absent.json"), "fixture must lack absent.json"
    write_sources(root, [paper(sha, stage="P5", stage_artifacts=artifacts)])

    proc = run(root)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "fixture" in proc.stderr and needle in proc.stderr, proc.stderr
    assert not (root / REL_OUT).exists(), "no data file may be written on error"


def test_class_page_paper_value_and_roadmap_are_copied(tmp_path):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": {"val": 1.5, "ok": True}})
    (root / "dashboard" / "models").mkdir(parents=True)
    (root / "dashboard" / "models" / "fixture.md").write_text("## Question\n", encoding="utf-8")
    prereg = [{"id": "v1", "signed": "2026-01-01", "path": "prereg.md"}]
    r1 = criterion(value={"file": "main", "key": "val"}, paper_value="≈ 1.4")
    roadmap = [{"release": "9.9.9", "model": "m", "needs": "n"}]
    spec = {"class": "reproduced (conditional)", "verdict": "pass", "verdict_note": "under v2", "preregs": prereg,
            "page": "models/fixture.md", "files": {"main": "scores/main.json"}, "criteria": [r1]}
    write_sources(root, [paper(sha, **spec)], roadmap=roadmap)

    proc = run(root)
    assert proc.returncode == 0, proc.stderr
    doc = load_out(root)
    got = doc["papers"][0]
    assert (got["class"], got["verdict"], got["page"]) == ("reproduced (conditional)", "pass", "models/fixture.md")
    assert got["verdict_note"] == "under v2"
    assert got["criteria"][0]["paper_value"] == "≈ 1.4" and got["criteria"][0]["value"] == 1.5
    assert doc["roadmap"] == roadmap

    (root / "dashboard" / "models" / "fixture.md").unlink()
    proc = run(root)
    assert proc.returncode == 1 and "models/fixture.md does not exist" in proc.stderr, proc.stderr

    for bad in ("../README.md", "models/../x.md", "https://x/y.md", "models/a.html"):
        write_sources(root, [paper(sha, **{**spec, "page": bad})], roadmap=roadmap)
        proc = run(root)
        assert proc.returncode == 1 and "is not models/<name>.md" in proc.stderr, (bad, proc.stderr)


def core_criterion(check, **over):
    base = {
        "id": check, "entry": check, "claim": "c", "estimand": "e",
        "criterion": "error within tolerance",
        "value": {"file": "main", "key": f"checks.{check}.error"},
        "verdict": {"file": "main", "key": f"checks.{check}.pass"},
        "reference": {"file": "main", "key": f"checks.{check}.reference"},
    }
    base.update(over)
    return base


def core_score():
    return {"checks": {
        "meg_radial": {"error": 1e-9, "tolerance": 1e-6, "pass": True,
                       "reference": "primary-current field"},
        "eeg_mne": {"error": None, "tolerance": None, "pass": None,
                    "reference": "MNE single-shell", "skipped": "mne not installed"},
    }}


def core_paper(sha, criteria, grammar=("Field", "Probe")):
    spec = paper(sha, **{"class": "core method", "stage": None, "preregs": [],
                         "files": {"main": "scores/main.json"}, "criteria": criteria})
    if grammar is not None:
        spec["grammar"] = list(grammar)
    return spec


def test_core_method_builds_values_verdicts_and_references_from_the_scorer_file(tmp_path):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": core_score()})
    spec = core_paper(sha, [core_criterion("meg_radial"), core_criterion("eeg_mne")])
    assert "prereg" not in spec["criteria"][0] and spec.get("stage") is None, (
        "fixture must carry no prereg and no stage"
    )
    write_sources(root, [spec])

    proc = run(root)
    assert proc.returncode == 0, proc.stderr
    got = load_out(root)["papers"][0]
    assert (got["class"], got["stage"], got["verdict"], got["preregs"]) == ("core method", None, None, [])
    assert got["grammar"] == ["Field", "Probe"]
    live, skipped = got["criteria"]
    assert (live["value"], live["verdict"]) == (1e-9, "pass")
    assert live["reference"] == "primary-current field"
    assert live["reference_text"] == "primary-current field"
    assert (skipped["value"], skipped["verdict"]) == (None, "pending")

    flipped = core_score()
    flipped["checks"]["meg_radial"]["pass"] = False
    (root / "scores" / "main.json").write_text(json.dumps(flipped), encoding="utf-8")
    git(root, "add", "--", "scores/main.json")
    git(root, "commit", "-q", "-m", "flipped verdict")
    sha2 = git(root, "rev-parse", "HEAD")
    write_sources(root, [core_paper(sha2, [core_criterion("meg_radial")])])
    proc = run(root)
    assert proc.returncode == 0, proc.stderr
    assert load_out(root)["papers"][0]["criteria"][0]["verdict"] == "fail", (
        "the verdict comes from the lane scorer file, not the sources entry"
    )


@pytest.mark.parametrize("criteria, needle", [
    ([], "needs non-empty criteria"),
    ([{k: v for k, v in core_criterion("meg_radial").items() if k != "reference"}],
     "needs a non-empty reference"),
    ([{k: v for k, v in core_criterion("meg_radial").items() if k != "verdict"}],
     "takes its verdict from the scorer file"),
])
def test_core_method_refuses_and_names_the_paper(tmp_path, criteria, needle):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": core_score()})
    write_sources(root, [core_paper(sha, criteria)])

    proc = run(root)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "fixture" in proc.stderr and needle in proc.stderr, proc.stderr
    assert not (root / REL_OUT).exists(), "no data file may be written on error"


@pytest.mark.parametrize("grammar, needle", [
    (None, "names grammar stages from"),
    ([], "names grammar stages from"),
    (["Field", "Membrane"], "names grammar stages from"),
    (["Probe", "Field"], "unique and in grammar order"),
    (["Field", "Field"], "unique and in grammar order"),
])
def test_core_method_refuses_bad_grammar_stages(tmp_path, grammar, needle):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": core_score()})
    write_sources(root, [core_paper(sha, [core_criterion("meg_radial")], grammar=grammar)])

    proc = run(root)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "fixture" in proc.stderr and needle in proc.stderr, proc.stderr
    assert not (root / REL_OUT).exists(), "no data file may be written on error"


def test_grammar_stages_only_on_core_method_rows(tmp_path):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": core_score()})
    spec = paper(sha)
    spec["grammar"] = ["Emitter"]
    assert spec["class"] != "core method", "fixture must be a non-core paper"
    write_sources(root, [spec])

    proc = run(root)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "only a core-method entry names grammar stages" in proc.stderr, proc.stderr


def test_core_method_skippable_criterion_names_its_entry(tmp_path):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": core_score()})
    crit = core_criterion("eeg_mne", may_skip=True)
    assert "entry" in crit
    del crit["entry"]
    write_sources(root, [core_paper(sha, [crit])])

    proc = run(root)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "fixture" in proc.stderr and "names its scorer entry" in proc.stderr, proc.stderr
    assert not (root / REL_OUT).exists(), "no data file may be written on error"

    write_sources(root, [core_paper(sha, [core_criterion("eeg_mne", may_skip=True)])])
    proc = run(root)
    assert proc.returncode == 0, proc.stderr
    assert load_out(root)["papers"][0]["criteria"][0]["verdict"] == "pending"


# ---- DASH-P0: the three-verdict distinction, provenance and failure visibility ----

LIB_PASS = {"_comment": "mirrors scores/p6.json key p6.verdict at the lane SHA",
            "stage": "P6", "file": "p6", "key": "p6.verdict"}


def scored_paper(sha, verdict_value, **over):
    spec = {
        "class": "mixed", "verdict": "mixed",
        "verdict_note": "R1 reproduced; R4 missed",
        "preregs": [{"id": "v1", "signed": "2026-01-01", "path": "prereg.md"}],
        "stage": "P6",
        "files": {"main": "scores/main.json", "p6": "scores/p6.json"},
        "criteria": [criterion(value={"file": "main", "key": "val"})],
        "library_path": dict(LIB_PASS),
    }
    spec.update(over)
    return paper(sha, **spec)


def real_doc():
    return json.loads((REPO / "dashboard" / "data" / "replications.json").read_text(encoding="utf-8"))


def real_paper(doc, pid):
    return next(p for p in doc["papers"] if p["id"] == pid)


def test_verdict_summary_three_verdicts_for_a_scored_row(tmp_path):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": {"val": 1.5, "ok": True},
                           "scores/p6.json": {"p6": {"verdict": "pass"}}})
    write_sources(root, [scored_paper(sha, None)])

    proc = run(root)
    assert proc.returncode == 0, proc.stderr
    got = load_out(root)["papers"][0]["verdict_summary"]
    assert got["kind"] == "paper" and got["paper"] == "mixed"
    assert got["library"]["stage"] == "P6" and got["library"]["verdict"] == "pass"
    assert got["library"]["evidence"]["path"] == "scores/p6.json"
    assert got["methods"] is None
    assert got["failed_criteria"] == [] and got["pending_criteria"] == []


@pytest.mark.parametrize("lane_value, want", [
    ("pass", "pass"), ("fail", "fail"), ("incomplete", "fail"), (None, "pending"),
])
def test_library_path_maps_lane_verdicts(tmp_path, lane_value, want):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": {"val": 1.5, "ok": True},
                           "scores/p6.json": {"p6": {"verdict": lane_value}}})
    write_sources(root, [scored_paper(sha, None)])

    proc = run(root)
    assert proc.returncode == 0, proc.stderr
    assert load_out(root)["papers"][0]["verdict_summary"]["library"]["verdict"] == want


def test_removing_the_p6_receipt_flips_library_to_pending_and_keeps_paper(tmp_path):
    # Hamm's enforcement point: a missing library receipt renders the library
    # verdict pending (unknown), and the scientific verdict does not move.
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": {"val": 1.5, "ok": True}})
    spec = scored_paper(sha, None, library_path=dict(LIB_PASS),
                        files={"main": "scores/main.json",
                               "p6": {"path": "scores/p6.json", "allow_missing": True}})
    write_sources(root, [spec])

    proc = run(root)
    assert proc.returncode == 0, proc.stderr
    got = load_out(root)["papers"][0]["verdict_summary"]
    assert got["library"]["verdict"] == "pending", got
    assert got["library"]["stage"] == "P6"
    assert got["paper"] == "mixed", "the scientific verdict must not move"


def test_core_method_with_no_criteria_is_refused_not_pass(tmp_path):
    # An empty examined set is not a pass: a core-method row without criteria
    # is refused outright, so no empty pass can ever render.
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": core_score()})
    write_sources(root, [core_paper(sha, [])])

    proc = run(root)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "needs non-empty criteria" in proc.stderr, proc.stderr


def test_library_path_absent_without_a_block(tmp_path):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": {"val": 1.5, "ok": True}})
    spec = scored_paper(sha, None, files={"main": "scores/main.json"})
    del spec["library_path"]
    write_sources(root, [spec])

    proc = run(root)
    assert proc.returncode == 0, proc.stderr
    assert load_out(root)["papers"][0]["verdict_summary"]["library"] is None


def test_library_path_missing_key_is_refused(tmp_path):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": {"val": 1.5, "ok": True},
                           "scores/p6.json": {"p6": {}}})
    assert "verdict" not in json.loads((root / "scores" / "p6.json").read_text(encoding="utf-8"))["p6"]
    write_sources(root, [scored_paper(sha, None)])

    proc = run(root)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "fixture" in proc.stderr and "p6.verdict" in proc.stderr, proc.stderr
    assert not (root / REL_OUT).exists(), "no data file may be written on error"


def test_core_method_row_carries_null_paper_and_methods_verdict(tmp_path):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": core_score()})
    write_sources(root, [core_paper(sha, [core_criterion("meg_radial"), core_criterion("eeg_mne")])])
    assert "prereg" not in core_criterion("meg_radial"), "core criteria take no prereg"

    proc = run(root)
    assert proc.returncode == 0, proc.stderr
    got = load_out(root)["papers"][0]["verdict_summary"]
    assert got["kind"] == "core_method" and got["paper"] is None
    assert got["library"] is None
    assert got["methods"] == "pending", "one live pass and one pending check is pending, not passing"


def test_overview_counts_exclude_core_method_rows(tmp_path):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": dict(core_score(), **{"val": 1.5, "ok": True})})
    good = paper(sha, **{"class": "reproduced", "verdict": "reproduced",
                         "preregs": [{"id": "v1", "signed": "2026-01-01", "path": "prereg.md"}],
                         "stage": "P3", "files": {"main": "scores/main.json"},
                         "criteria": [criterion(value={"file": "main", "key": "val"})]})
    write_sources(root, [good, core_paper(sha, [core_criterion("meg_radial")])])

    proc = run(root)
    assert proc.returncode == 0, proc.stderr
    overview = load_out(root)["overview"]
    assert overview["papers"] == 1, "the core-method row must not enter the paper denominator"
    assert overview["by_paper"] == {"reproduced": 1}
    assert overview["criteria"] == {"pass": 1, "fail": 0, "pending": 0}
    assert overview["core_methods"]["rows"] == 1
    assert overview["core_methods"]["criteria"] == {"pass": 1, "fail": 0, "pending": 0}


def test_generated_carries_provenance_and_a_stable_revision(tmp_path):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": {"val": 1.5, "ok": True}})
    write_sources(root, [paper(sha, files={"main": "scores/main.json"},
                               criteria=[criterion(value={"file": "main", "key": "val"})])])

    proc = run(root)
    assert proc.returncode == 0, proc.stderr
    module = load_module(SCRIPT)
    first = load_out(root)
    gen = first["generated"]
    assert re.fullmatch(r"[0-9a-f]{40}", gen["dev_sha"]), "the generating commit must be recorded"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", gen["utc"]), "generation UTC must be recorded"
    assert re.fullmatch(r"[0-9a-f]{64}", gen["data_revision"]), "a deploy-comparable revision must be recorded"
    assert gen["data_revision"] == module.content_revision(first), "the revision must hash the content"

    again = run(root)
    assert again.returncode == 0, again.stderr
    assert load_out(root)["generated"]["data_revision"] == gen["data_revision"], (
        "the revision must be stable across regenerations (timestamps excluded)")


def test_short_sha_in_a_sha_field_fails_check(tmp_path):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": {"val": 1.5, "ok": True}})
    write_sources(root, [paper(sha[:12], files={"main": "scores/main.json"},
                               criteria=[criterion(value={"file": "main", "key": "val"})])])
    assert run(root).returncode == 0, "git resolves a short sha, so the fixture must build"
    assert len(load_out(root)["papers"][0]["lane"]["sha"]) == 12, "fixture must hold a 12-character lane sha"
    bad = run(root, "--check")
    assert bad.returncode == 1, bad.stdout
    assert "invalid" in bad.stderr and "lane" in bad.stderr and "40" in bad.stderr, bad.stderr


def test_receipt_digest_follows_the_lane_blob_not_the_working_tree(tmp_path):
    root = tmp_path / "repo"
    sha = make_repo(root, {"scores/main.json": {"val": 1.5, "ok": True}})
    write_sources(root, [paper(sha, files={"main": "scores/main.json"},
                               criteria=[criterion(value={"file": "main", "key": "val"})])])
    assert run(root).returncode == 0
    recorded = load_out(root)["papers"][0]["criteria"][0]["evidence"]
    assert recorded.get("sha256") == hashlib.sha256(blob_bytes(root, sha, "scores/main.json")).hexdigest(), (
        "the evidence item must carry the sha256 of the bytes at the lane sha")
    assert run(root, "--check").returncode == 0

    (root / "scores" / "main.json").write_bytes(b'{"val": 1.5, "ok": true, "note": "edited"}')  # not committed
    assert run(root, "--check").returncode == 0, "a working-tree edit alone must not change the digest"

    # Same lane sha, changed blob: a git replace ref makes git show return other bytes at rev:path.
    old_blob = git(root, "rev-parse", f"{sha}:scores/main.json")
    (root / "changed.json").write_bytes(b'{"val": 1.5, "ok": true, "note": "changed blob"}')
    new_blob = git(root, "hash-object", "-w", "changed.json")
    git(root, "replace", old_blob, new_blob)
    assert blob_bytes(root, sha, "scores/main.json") == (root / "changed.json").read_bytes(), (
        "fixture must make git show the changed blob at the lane sha")
    stale = run(root, "--check")
    assert stale.returncode == 1, stale.stdout
    assert "sha256" in stale.stderr, stale.stderr


def test_committed_data_file_passes_its_schema_and_unread_keywords_are_refused():
    module = load_module(SCRIPT)
    schema = json.loads((REPO / "dashboard" / "replications.schema.json").read_text(encoding="utf-8"))
    assert module.schema_errors(real_doc(), schema) == [], "the committed data file must conform to its schema"
    with pytest.raises(module.BuildError, match="patern"):
        module.schema_errors("x", {"type": "string", "patern": "^x$"})


def test_stale_generated_timestamp_flags_a_warning():
    module = load_module(SCRIPT)
    now = datetime(2026, 10, 23, tzinfo=timezone.utc)
    fresh = {"utc": "2026-10-22T12:00:00Z"}
    assert module.stale_warning(fresh, now) is None, "a day-old data file is fresh"
    stale = {"utc": "2026-01-01T00:00:00Z"}
    warning = module.stale_warning(stale, now)
    assert warning is not None and "tale" in warning and "2026-01-01" in warning, (
        "a deliberately stale fixture must warn, never pass silent")
    assert module.stale_warning({"utc": "not a timestamp"}, now) is not None, (
        "an unreadable timestamp must warn, never read green")
    assert "stale-banner" in (REPO / "dashboard" / "app.js").read_text(encoding="utf-8"), (
        "the renderer must wire the stale warning to a visible banner")
    assert 'id="stale-banner"' in (REPO / "dashboard" / "index.html").read_text(encoding="utf-8"), (
        "the page must hold the stale banner element")


def test_real_caminos_shows_all_three_verdicts_and_the_lag():
    got = real_paper(real_doc(), "caminos")["verdict_summary"]
    assert got["paper"] == "mixed" and got["kind"] == "paper"
    assert got["library"] == {"stage": "P6", "verdict": "pass",
                              "evidence": got["library"]["evidence"]}, "library must read P6 PASS"
    row = real_paper(real_doc(), "caminos")
    assert "6.47" in row["verdict_note"] and "8.5" in row["verdict_note"], (
        "the 6.47-vs-8.5 ms lag discrepancy must stay in the row")


def test_real_brunel_stays_conditional_with_its_disclosure():
    row = real_paper(real_doc(), "brunel2000")
    assert row["verdict_summary"]["paper"] == "reproduced (conditional)"
    assert row["stage"] == "P5"
    assert "classifier" in row["verdict_note"], "the classifier-change disclosure must stay visible"


def test_real_pcl_fig3_shows_its_signed_fail_and_fidelity_status():
    row = real_paper(real_doc(), "pcl-fig3")
    fails = {c["id"] + " " + c.get("group", ""): c["reason"] for c in row["verdict_summary"]["failed_criteria"]}
    assert "T2 v5 simple" in fails and "flat line" in fails["T2 v5 simple"], (
        "the signed T2 v5 simple FAIL must stay adjacent to the passes")
    assert "flat line" in row["summary"], "the fidelity-audit status must stay in the row"


def test_real_overview_excludes_core_method_rows():
    doc = real_doc()
    assert doc["overview"]["papers"] == sum(1 for p in doc["papers"]
                                           if p["verdict_summary"]["kind"] == "paper")
    assert all(p["verdict_summary"]["paper"] is None and p["verdict_summary"]["kind"] == "core_method"
               for p in doc["papers"] if p["class"] == "core method")
    assert sum(doc["overview"]["by_paper"].values()) == doc["overview"]["papers"]
