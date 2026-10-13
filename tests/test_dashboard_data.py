"""Tests for scripts/build_dashboard_data.py.

Each test commits fake scorer JSONs to a throwaway git repo under pytest's tmp_path, copies the
generator into that repo and runs it as a subprocess. No network. DASH_GEN_BUILD_SCRIPT may name
another script to test; it is used to show that these tests fail against a stub.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
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


def make_repo(root, files):
    """Commit `files` ({path: JSON-able object}), then copy the generator in (not committed). Returns the SHA."""
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
    return git(root, "rev-parse", "HEAD")


def write_sources(root, papers, **extra):
    (root / "dashboard").mkdir(exist_ok=True)
    doc = {"schema": "jaxfne-dashboard-sources/2", "repo_url": REPO_URL, "papers": papers, **extra}
    (root / "dashboard" / "sources.json").write_text(json.dumps(doc), encoding="utf-8")


def paper(sha, **over):
    """A minimal valid paper entry; keyword arguments replace or add fields."""
    base = {
        "id": "fixture", "title": "Fixture", "citation": "test", "release": "0.0.0", "tier": "T-test",
        "class": "demonstration",
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
                               "url": f"{REPO_URL}/blob/{sha}/scores/main.json"}


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
    ({"class": "demonstration", "verdict": "pass"}, "a demonstration carries no verdict"),
    ({"class": "partial", "preregs": []}, "needs a signed preregistration"),
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
                                "url": f"{REPO_URL}/blob/{sha}/spec.json"}]


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
    spec = {"class": "partial", "verdict": "pass", "verdict_note": "under v2", "preregs": prereg,
            "page": "models/fixture.md", "files": {"main": "scores/main.json"}, "criteria": [r1]}
    write_sources(root, [paper(sha, **spec)], roadmap=roadmap)

    proc = run(root)
    assert proc.returncode == 0, proc.stderr
    doc = load_out(root)
    got = doc["papers"][0]
    assert (got["class"], got["verdict"], got["page"]) == ("partial", "pass", "models/fixture.md")
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
