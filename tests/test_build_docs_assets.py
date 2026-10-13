"""Docs atlas registry: the --check comparison, the fresh-vs-registry compare and the state apply.

No simulation runs here. The tests read the tracked manifests and the tracked
registry, and write only under tmp_path.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "build_docs_assets", REPO / "scripts" / "build_docs_assets.py"
)
bda = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bda)


def _entry(slug):
    matches = [e for e in bda.load_registry()["atlases"] if e["slug"] == slug]
    assert len(matches) == 1, f"fixture must find exactly one registry entry for {slug}"
    return copy.deepcopy(matches[0])


def _manifest(entry):
    path = REPO / entry["dir"] / "manifest.json"
    assert path.exists(), f"fixture must have a tracked manifest at {entry['dir']}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_check_passes_on_tree(capsys):
    problems, examined = bda.check_registry(bda.load_registry(), bda.discover())
    assert problems == []
    assert examined == len(bda.discover()) > 0, "--check must examine every tracked atlas"
    assert bda.main(["--check"]) == 0
    assert f"{examined} atlases" in capsys.readouterr().out


def test_perturbed_config_hash_fails_and_names_field():
    entry = _entry("two_neuron_ei")
    manifest = _manifest(entry)
    assert bda.compare(entry, manifest) == [], "fixture must match before the perturbation"
    entry["config_hash"] = "0" * len(entry["config_hash"])
    problems = bda.compare(entry, manifest)
    assert len(problems) == 1 and problems[0].startswith("config_hash"), problems


def test_perturbed_registry_fails_check_and_names_field(tmp_path, monkeypatch, capsys):
    reg = bda.load_registry()
    target = [e for e in reg["atlases"] if e["slug"] == "two_neuron_ei"]
    assert len(target) == 1, "fixture must perturb one registry entry"
    target[0]["config_hash"] = "0" * len(target[0]["config_hash"])
    tmp_reg = tmp_path / "atlas_registry.json"
    tmp_reg.write_text(json.dumps(reg), encoding="utf-8")
    monkeypatch.setattr(bda, "REGISTRY", tmp_reg)
    assert bda.main(["--check"]) == 1
    out = capsys.readouterr().out
    assert "config_hash" in out and target[0]["dir"] in out


def test_perturbed_panel_status_fails():
    entry = _entry("two_neuron_ei")
    manifest = copy.deepcopy(_manifest(entry))
    panel = next(p for p in manifest["panels"] if p["file"] == "schema.html")
    assert panel["status"] != "ERROR", "fixture must start from a non-ERROR panel"
    panel["status"] = "ERROR"
    problems = bda.compare(entry, manifest)
    assert any("panels[schema.html].status" in p for p in problems), problems


def test_atlas_without_registry_entry_fails_check(tmp_path, monkeypatch, capsys):
    reg = bda.load_registry()
    dropped = [e for e in reg["atlases"] if e["slug"] == "two_neuron_ei"]
    assert len(dropped) == 1 and (REPO / dropped[0]["dir"] / "manifest.json").exists(), (
        "fixture must drop an entry whose manifest is on disk"
    )
    reg["atlases"] = [e for e in reg["atlases"] if e["slug"] != "two_neuron_ei"]
    tmp_reg = tmp_path / "atlas_registry.json"
    tmp_reg.write_text(json.dumps(reg), encoding="utf-8")
    monkeypatch.setattr(bda, "REGISTRY", tmp_reg)
    assert bda.main(["--check"]) == 1
    assert dropped[0]["dir"] in capsys.readouterr().out


def test_registry_state_applied_after_match(tmp_path):
    entry = _entry("two_neuron_ei")
    fresh = copy.deepcopy(_manifest(entry))
    # A freshly generated manifest: GENERATED state, no promotion history, same panels.
    fresh["figure_state"] = "GENERATED"
    fresh["promotion"] = []
    assert bda.compare(entry, fresh) == [], "the match is what licenses the apply"
    assert entry["figure_state"] != "GENERATED" and entry["promotion"], "fixture needs a promoted entry"

    applied = bda.apply_state(fresh, entry)
    assert applied["figure_state"] == entry["figure_state"]
    assert applied["promotion"] == entry["promotion"]

    out = tmp_path / "manifest.json"
    bda.write_manifest(out, applied)
    back = json.loads(out.read_text(encoding="utf-8"))
    assert back["figure_state"] == entry["figure_state"]
    assert back["promotion"] == entry["promotion"]
    assert back["panels"] == fresh["panels"]


def test_stale_registry_state_is_flagged():
    entry = _entry("two_neuron_ei")
    manifest = _manifest(entry)
    assert bda.compare_state(entry, manifest) == [], "fixture must match before the perturbation"
    entry["figure_state"] = "GENERATED"
    problems = bda.compare_state(entry, manifest)
    assert len(problems) == 1 and problems[0].startswith("figure_state"), problems


def _fake_run(calls, rc, stdout="", stderr=""):
    import subprocess

    def run(cmd):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, rc, stdout=stdout, stderr=stderr)
    return run


def test_build_attributes_a_crashed_generator_and_skips_the_gallery(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(bda, "_run", _fake_run(calls, 1, stderr="Traceback\nMemoryError: out of memory"))
    monkeypatch.setattr(bda, "_repo_guard", lambda: None)
    assert bda.build(["two_neuron_ei"]) == 1
    out = capsys.readouterr().out
    assert "ERROR" in out and "rc=1" in out and "MemoryError" in out
    assert "gallery: not rewritten" in out
    assert len(calls) == 1, "the gallery generator must not run after a failed atlas"


def test_build_fails_when_a_registry_html_file_is_not_written(tmp_path, monkeypatch, capsys):
    import subprocess

    entry = _entry("two_neuron_ei")
    atlas = tmp_path / entry["dir"]
    atlas.mkdir(parents=True)
    manifest_bytes = (REPO / entry["dir"] / "manifest.json").read_bytes()
    calls = []

    def run(cmd):
        # The generator rewrites the manifest and every panel, and skips index.html.
        calls.append(cmd)
        (atlas / "manifest.json").write_bytes(manifest_bytes)
        for name in entry["panels"]:
            (atlas / name).write_text("<html></html>", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    assert "index.html" not in entry["panels"], "fixture: index.html is outside the panel list"
    monkeypatch.setattr(bda, "ROOT", tmp_path)
    monkeypatch.setattr(bda, "_run", run)
    monkeypatch.setattr(bda, "_repo_guard", lambda: None)
    assert bda.build(["two_neuron_ei"]) == 1
    out = capsys.readouterr().out
    assert "HTML not written by this build: ['index.html']" in out, out
    assert len(calls) == 1, "the gallery generator must not run after a failed atlas"


def test_build_reads_a_failed_line_for_its_own_slug_only(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(bda, "_run", _fake_run(calls, 1, stdout="[hdp_1000] FAILED: ValueError('x')\n"))
    monkeypatch.setattr(bda, "_repo_guard", lambda: None)
    assert bda.build(["hdp_10", "hdp_1000"]) == 1
    lines = capsys.readouterr().out.splitlines()
    i10 = [i for i, ln in enumerate(lines) if ln.startswith("docs/_static/atlas/hdp_10 ")]
    i1000 = [i for i, ln in enumerate(lines) if ln.startswith("docs/_static/atlas/hdp_1000 ")]
    assert len(i10) == 1 and len(i1000) == 1, "fixture must report both atlases"
    assert "FAILED" not in lines[i10[0] + 1], "hdp_10 must not inherit hdp_1000's FAILED line"
    assert "[hdp_1000] FAILED" in lines[i1000[0] + 1]


def _link_problems(page_rel, text):
    problems, examined, _ = bda.check_links(bda.load_registry(), {REPO / page_rel: text})
    assert examined == 1, "fixture page must hold exactly one .html link"
    return problems


def test_links_on_tree_resolve():
    problems, examined, atlas_links = bda.check_links(bda.load_registry(), bda.docs_pages())
    assert problems == []
    assert atlas_links > 0 and examined > atlas_links, "the gate must examine atlas and other links"


def test_raw_iframe_resolves_from_the_page_url():
    # docs/guides/p.md is served at /guides/p/: a raw src needs two levels up to reach _static.
    frame = '<iframe src="{}_static/atlas/two_neuron_ei/index.html"></iframe>'
    shallow = _link_problems("docs/guides/p.md", frame.format("../"))
    assert len(shallow) == 1 and "raw HTML" in shallow[0], shallow
    assert _link_problems("docs/guides/p.md", frame.format("../../")) == []
    # An index page is served at its own directory: one level up is enough.
    assert _link_problems("docs/guides/index.md", frame.format("../")) == []


def test_markdown_link_resolves_from_the_file():
    assert _link_problems("docs/guides/p.md", "[i](../_static/atlas/two_neuron_ei/index.html)") == []
    deep = _link_problems("docs/guides/p.md", "[i](../../_static/atlas/two_neuron_ei/index.html)")
    assert len(deep) == 1 and "markdown" in deep[0], deep


def test_reference_style_and_uppercase_links_are_examined():
    ref = _link_problems("docs/guides/p.md", "[i]: ../_static/atlas/two_neuron_ei/missing.html\n")
    assert len(ref) == 1, ref
    angle = "[i](<../_static/atlas/two_neuron_ei/index.html>)"
    assert _link_problems("docs/guides/p.md", angle) == []
    upper = _link_problems("docs/guides/p.md", '<iframe SRC="../_static/atlas/two_neuron_ei/index.HTML">')
    assert len(upper) == 1 and "raw HTML" in upper[0], upper


def test_atlas_link_must_name_a_built_file():
    problems = _link_problems("docs/guides/p.md", "[x](../_static/atlas/two_neuron_ei/no_such_panel.html)")
    assert len(problems) == 1 and "is not an atlas file the build writes" in problems[0], problems


def test_other_html_link_must_name_a_file_on_disk():
    target = "../../_static/etudes/experiment_a/index.html"
    assert (REPO / "docs/_static/etudes/experiment_a/index.html").is_file(), "fixture needs a tracked page"
    assert _link_problems("docs/etudes/p.md", f'<iframe src="{target}"></iframe>') == []
    missing = _link_problems("docs/etudes/p.md", '<iframe src="../../_static/etudes/none.html"></iframe>')
    assert len(missing) == 1 and "names no file" in missing[0], missing


def test_registry_refuses_a_null_compared_field(tmp_path):
    import pytest

    manifest = _manifest(_entry("two_neuron_ei"))
    manifest["seed"] = None
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit, match="seed"):
        bda.entry_from_manifest(path)
