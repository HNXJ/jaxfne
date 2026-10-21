"""Docs atlas registry: the --check comparison, the fresh-vs-registry compare and the state apply.

No simulation runs here. The tests read the tracked manifests and the tracked
registry, and write only under tmp_path.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import re

import pytest

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


def _generated(path, data):
    """Write a fake generator's output, stamped just after the build start.

    The build compares each file's mtime with time.time() taken at its start; a write in the
    same clock tick as that start can read back earlier, so the stamp moves ahead.
    """
    import os
    import time

    path.write_bytes(data)
    later = time.time() + 2
    os.utime(path, (later, later))


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
        _generated(atlas / "manifest.json", manifest_bytes)
        for name in entry["panels"]:
            _generated(atlas / name, b"<html></html>")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    assert "index.html" not in entry["panels"], "fixture: index.html is outside the panel list"
    # The real registry's copies sources are not under tmp ROOT: check them out of this atlas case.
    reg = bda.load_registry()
    reg.pop("copies", None)
    tmp_reg = tmp_path / "atlas_registry.json"
    tmp_reg.write_text(json.dumps(reg), encoding="utf-8")
    monkeypatch.setattr(bda, "ROOT", tmp_path)
    monkeypatch.setattr(bda, "_run", run)
    monkeypatch.setattr(bda, "_repo_guard", lambda: None)
    monkeypatch.setattr(bda, "REGISTRY", tmp_reg)
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
    # After the docs-figures rewrite every figure link is an absolute
    # figure-site URL; no relative link may resolve into an atlas directory.
    assert atlas_links == 0
    assert examined > 0, "the gate must still examine non-figure .html links"


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
    # visuals/ is not a registry product: its tracked HTML is checked on disk.
    target = "../../_static/visuals/column_network.html"
    assert (REPO / "docs/_static/visuals/column_network.html").is_file(), "fixture needs a tracked page"
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


def _etude(slug="fixture_etude", html=("index.html", "fixture_panel.html")):
    """A registry etude entry. Its HTML is not on disk: the tests build their own case."""
    return {
        "slug": slug,
        "dir": f"docs/_static/etudes/{slug}",
        "generator": ["scripts/fixture_etude.py", "--etude", slug],
        "html": list(html),
    }


def _etude_link_problems(target):
    registry = {"atlases": [], "etudes": [_etude()]}
    text = f'<iframe src="{target}"></iframe>'
    problems, examined, _ = bda.check_links(registry, {REPO / "docs/etudes/p.md": text})
    assert examined == 1, "fixture page must hold exactly one .html link"
    return problems


def test_etude_link_resolves_against_its_registry_html():
    # fixture_panel.html is listed but not on disk: the registry says the build writes it.
    assert _etude_link_problems("../../_static/etudes/fixture_etude/fixture_panel.html") == []


def test_etude_link_outside_its_registry_html_fails():
    problems = _etude_link_problems("../../_static/etudes/fixture_etude/no_such_panel.html")
    assert len(problems) == 1 and "is not an etude file the build writes" in problems[0], problems


def _write_registry(tmp_path, monkeypatch, etudes):
    path = tmp_path / "atlas_registry.json"
    path.write_text(json.dumps({"schema": bda.SCHEMA, "atlases": [], "etudes": etudes}), encoding="utf-8")
    monkeypatch.setattr(bda, "REGISTRY", path)


def test_registry_refuses_an_etude_missing_a_required_key(tmp_path, monkeypatch):
    import pytest

    entry = _etude()
    del entry["html"]
    _write_registry(tmp_path, monkeypatch, [entry])
    with pytest.raises(SystemExit, match="lacks \\['html'\\]"):
        bda.load_registry()


def test_registry_refuses_duplicate_etude_slugs(tmp_path, monkeypatch):
    import pytest

    _write_registry(tmp_path, monkeypatch, [_etude(), _etude()])
    with pytest.raises(SystemExit, match="duplicate slug entries"):
        bda.load_registry()


def test_registry_write_keeps_the_etudes_list(tmp_path, monkeypatch):
    _write_registry(tmp_path, monkeypatch, [_etude()])
    assert bda.write_registry() == 0
    assert bda.read_json(bda.REGISTRY).get("etudes") == [_etude()]


TRACKED_ETUDE_MANIFEST = {"figure_state": "validated", "bytes": 1,
                          "promotion": [{"from": "generated", "to": "validated"}]}


def _build_with_etude(tmp_path, monkeypatch, entry, rc=0, written=("index.html",)):
    """build([]) over one etude whose fake generator writes `written` and exits with `rc`."""
    import subprocess

    registry = {"schema": bda.SCHEMA, "atlases": [], "etudes": [entry]}
    monkeypatch.setattr(bda, "load_registry", lambda: registry)
    monkeypatch.setattr(bda, "ROOT", tmp_path)
    monkeypatch.setattr(bda, "_repo_guard", lambda: None)
    out_dir = tmp_path / entry["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    bda.write_manifest(out_dir / "manifest.json", TRACKED_ETUDE_MANIFEST)

    def run(cmd):
        if str(cmd[1]).endswith("fixture_etude.py"):
            # The real generators rewrite the manifest with a reset figure state.
            bda.write_manifest(out_dir / "manifest.json", {"figure_state": "GENERATED", "bytes": 2})
            for name in written:
                _generated(out_dir / name, b"<html></html>")
            return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="Traceback\nboom" if rc else "")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(bda, "_run", run)
    return bda.build([])


def test_build_passes_an_etude_that_writes_every_listed_html(tmp_path, monkeypatch, capsys):
    both = ("index.html", "fixture_panel.html")
    assert _build_with_etude(tmp_path, monkeypatch, _etude(), written=both) == 0
    assert "docs/_static/etudes/fixture_etude  OK" in capsys.readouterr().out


def test_build_keeps_the_tracked_etude_figure_state(tmp_path, monkeypatch):
    both = ("index.html", "fixture_panel.html")
    assert _build_with_etude(tmp_path, monkeypatch, _etude(), written=both) == 0
    fresh = bda.read_json(tmp_path / _etude()["dir"] / "manifest.json")
    assert fresh["bytes"] == 2, "fixture: the generator's manifest must be the one kept"
    assert fresh["figure_state"] == "validated"
    assert fresh["promotion"] == TRACKED_ETUDE_MANIFEST["promotion"]


def test_build_fails_an_etude_with_a_listed_html_file_missing(tmp_path, monkeypatch, capsys):
    assert _build_with_etude(tmp_path, monkeypatch, _etude()) == 1
    assert "HTML not written by this build: ['fixture_panel.html']" in capsys.readouterr().out


def test_build_fails_an_etude_whose_generator_exits_nonzero(tmp_path, monkeypatch, capsys):
    assert _build_with_etude(tmp_path, monkeypatch, _etude(), rc=1) == 1
    out = capsys.readouterr().out
    assert "generator rc=1" in out and "boom" in out, out


def _copies_registry(copies):
    return {"schema": bda.SCHEMA, "atlases": [], "etudes": [], "copies": copies}


def test_build_copies_each_source_to_its_dest_byte_exact(tmp_path, monkeypatch, capsys):
    src = tmp_path / "artifacts/etudes/papers/pcl/confirm/c1_seed10_kernels.png"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"\x89PNG fixture-bytes")
    registry = _copies_registry([{
        "dest": "docs/assets/etudes/pcl/c1_kernels.png",
        "source": "artifacts/etudes/papers/pcl/confirm/c1_seed10_kernels.png",
    }])
    monkeypatch.setattr(bda, "load_registry", lambda: registry)
    monkeypatch.setattr(bda, "ROOT", tmp_path)
    monkeypatch.setattr(bda, "_repo_guard", lambda: None)
    monkeypatch.setattr(bda, "_run", _fake_run([], 0))
    assert bda.build([]) == 0
    assert (tmp_path / "docs/assets/etudes/pcl/c1_kernels.png").read_bytes() == b"\x89PNG fixture-bytes"
    assert "copies: 1 ok" in capsys.readouterr().out


def test_registry_refuses_a_malformed_copies_entry(tmp_path, monkeypatch):
    import pytest

    src = "artifacts/etudes/papers/pcl/confirm/c1_seed10_kernels.png"
    assert (REPO / src).is_file(), "fixture needs a tracked source figure"
    dest = "docs/assets/etudes/pcl/c1_kernels.png"
    bad = [
        [{"dest": dest}],  # no source
        [{"source": src}],  # no dest
        [{"dest": "assets/etudes/pcl/c1_kernels.png", "source": src}],  # dest outside docs/
        [{"dest": dest, "source": src}, {"dest": dest, "source": src}],  # duplicate dest
        [{"dest": dest, "source": "artifacts/etudes/no_such_figure.png"}],  # source missing
    ]
    for copies in bad:
        path = tmp_path / "atlas_registry.json"
        path.write_text(json.dumps(_copies_registry(copies)), encoding="utf-8")
        monkeypatch.setattr(bda, "REGISTRY", path)
        with pytest.raises(SystemExit):
            bda.load_registry()


def test_check_flags_a_copy_whose_dest_differs(tmp_path, monkeypatch, capsys):
    (tmp_path / "artifacts").mkdir(parents=True)
    (tmp_path / "artifacts/fig.png").write_bytes(b"source-bytes")
    dest_rel = "docs/assets/etudes/fig.png"
    (tmp_path / "docs/assets/etudes").mkdir(parents=True)
    (tmp_path / dest_rel).write_bytes(b"different-bytes")
    registry = _copies_registry([{"dest": dest_rel, "source": "artifacts/fig.png"}])
    monkeypatch.setattr(bda, "load_registry", lambda: registry)
    monkeypatch.setattr(bda, "ROOT", tmp_path)
    assert bda.check() == 1
    out = capsys.readouterr().out
    assert "ERROR" in out and dest_rel in out, out
    (tmp_path / dest_rel).write_bytes(b"source-bytes")
    assert bda.check_copies(registry) == []


# --- Standalone figure site (--site): one HTML + PNG still per panel ---

# Cheapest registry slug by realized size (n_neurons=1, n_steps=1000: the
# smallest N*steps in the registry), so the --site tests simulate the least.
CHEAPEST_SLUG = "single_neuron"


def _require_kaleido(tmp_path):
    pytest.importorskip("kaleido", reason="kaleido renders the --site PNG stills")
    import plotly.graph_objects as go

    probe = tmp_path / "kaleido_probe.png"
    try:
        go.Figure(go.Scatter(x=[0, 1], y=[0, 1])).write_image(str(probe))
    except Exception as exc:  # noqa: BLE001 -- kaleido fails without a browser
        pytest.skip(f"kaleido cannot start: {exc}")
    assert probe.stat().st_size > 0, "kaleido probe still must be non-empty"


@pytest.mark.slow
def test_site_writes_html_png_and_manifest_for_cheapest_slug(tmp_path):
    _require_kaleido(tmp_path)
    site = tmp_path / "figsite"
    assert bda.main(["--site", str(site), "--only", CHEAPEST_SLUG]) == 0
    manifest = json.loads((site / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == bda.SITE_SCHEMA
    assert len(manifest["build_sha"]) == 40, "build_sha must be the git HEAD"
    by_panel = {f["panel"]: f for f in manifest["figures"] if f["slug"] == CHEAPEST_SLUG}
    entry = _entry(CHEAPEST_SLUG)
    assert set(by_panel) == {name[:-len(".html")] for name in entry["panels"]}
    for name, status in entry["panels"].items():
        fig = by_panel[name[:-len(".html")]]
        html = site / fig["html"]
        assert html.is_file() and html.stat().st_size > 0, f"{name} html missing"
        assert isinstance(fig["docs_pages"], list)
        if status in ("OMITTED", "ERROR"):
            assert fig["status"] == "omitted" and "png" not in fig, name
        else:
            assert fig["status"] == "available", name
            png = site / fig["png"]
            assert png.is_file(), f"{name} png missing"
            assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and png.stat().st_size > 0, name


@pytest.mark.slow
def test_site_writes_nothing_under_docs(tmp_path):
    _require_kaleido(tmp_path)
    before = {p for p in (REPO / "docs").rglob("*")}
    assert bda.main(["--site", str(tmp_path / "figsite"), "--only", CHEAPEST_SLUG]) == 0
    assert {p for p in (REPO / "docs").rglob("*")} == before


def test_render_atlas_default_output_unchanged(tmp_path, monkeypatch):
    import datetime

    import jaxfne.vis.atlas_suite as suite

    real_datetime = datetime.datetime

    class _Frozen(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime(2026, 1, 1, tzinfo=tz)

    monkeypatch.setattr(datetime, "datetime", _Frozen)
    spec = importlib.util.spec_from_file_location(
        "generate_doc_page_atlases", REPO / "scripts" / "generate_doc_page_atlases.py")
    dpa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dpa)
    title, model, signals, kw = dpa.spec_two_neuron_ei()
    default = suite.render_atlas(model, signals, out_dir=str(tmp_path / "a"),
                                 title=title, **kw)
    assert isinstance(default, dict), "the default return stays the manifest dict"
    with_figures = suite.render_atlas(model, signals, out_dir=str(tmp_path / "b"),
                                      title=title, return_figures=True, **kw)
    assert isinstance(with_figures, tuple) and with_figures[0] == default
    manifest, figures = with_figures
    assert set(figures) == {p["file"] for p in manifest["panels"]
                             if p["status"] == "AVAILABLE"}
    # Plotly stamps a fresh random id on every to_html call (the div id,
    # repeated in the newPlot script, plus per-trace uids), so two runs are
    # never byte-identical; normalize those ids and require everything else
    # to match exactly.
    uuid = re.compile(rb'"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"')
    for name in sorted((tmp_path / "a").iterdir()):
        other = tmp_path / "b" / name.name
        assert other.is_file(), f"{name.name} missing from the keyword run"
        assert (uuid.sub(b'"X"', other.read_bytes())
                == uuid.sub(b'"X"', name.read_bytes())), f"{name.name} differs"


def test_site_builds_a_docs_visual_slug(tmp_path, monkeypatch):
    # The visual path keys its Figure by the panel file; a slug key failed
    # both visuals on the first CI build (Dashboard run 39679832206).
    import sys
    import types

    import plotly.graph_objects as go

    import jaxfne.vis.exporters as exporters

    slug = bda.VISUAL_SLUGS[0]
    fake = types.ModuleType("generate_docs_visuals")
    fake.FIGURES = {slug: lambda: go.Figure(go.Scatter(x=[0, 1], y=[0, 1]),
                                            layout={"title": {"text": "visual"}})}
    monkeypatch.setitem(sys.modules, "generate_docs_visuals", fake)
    monkeypatch.setattr(exporters, "_write_image",
                        lambda fig, path: pathlib.Path(path).write_bytes(b"\x89PNG\r\n\x1a\n"))
    site = tmp_path / "figsite"
    assert bda.main(["--site", str(site), "--only", slug]) == 0
    manifest = json.loads((site / "manifest.json").read_text(encoding="utf-8"))
    [entry] = manifest["figures"]
    assert entry["status"] == "available" and entry["title"] == "visual"
    assert (site / entry["html"]).is_file() and (site / entry["png"]).is_file()


def test_site_completeness_fails_when_png_missing(tmp_path):
    site = tmp_path / "site"
    (site / "s").mkdir(parents=True)
    (site / "s" / "raster.html").write_bytes(b"<html></html>")
    (site / "s" / "hdp.html").write_bytes(b"<html></html>")
    (site / "s" / "raster.png").write_bytes(b"\x89PNG" + b"\x00" * 16)
    manifest = {"schema": bda.SITE_SCHEMA, "build_sha": "0" * 40, "figures": [
        {"slug": "s", "panel": "raster", "html": "s/raster.html",
         "png": "s/raster.png", "title": "t", "status": "available",
         "docs_pages": []},
        {"slug": "s", "panel": "hdp", "html": "s/hdp.html",
         "title": "t", "status": "omitted", "docs_pages": []}]}
    (site / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert bda.check_site_figures(site) == []
    (site / "s" / "raster.png").unlink()
    problems = bda.check_site_figures(site)
    assert len(problems) == 1 and "png missing" in problems[0], problems


def test_site_covers_every_registry_etude():
    registry = bda.load_registry()
    assert [e["slug"] for e in registry["etudes"]] == [
        "experiment_a",
        "hdp_controllability_reachability",
        "multiscale_observation",
        "heterogeneous_emitters",
    ]
    known = bda.site_slugs(registry)
    for slug in ("experiment_a", "hdp_controllability_reachability",
                 "multiscale_observation", "heterogeneous_emitters"):
        assert slug in known, f"--site must cover the {slug} etude"
        assert slug in bda.ETUDE_SITE_MODES


def test_docs_figures_check_passes_on_tree(capsys):
    assert bda.docs_figures_check() == 0
    out = capsys.readouterr().out
    assert "docs-figures-check:" in out and "0 problems" in out


def test_docs_figures_check_fails_on_planted_iframe(tmp_path, monkeypatch, capsys):
    page = REPO / "docs" / "index.md"
    text = page.read_text(encoding="utf-8")
    planted = ('<iframe src="../_static/atlas/readme/raster.html"></iframe>\n' + text)
    monkeypatch.setattr(bda, "docs_pages", lambda: {page: planted})
    assert bda.docs_figures_check() == 1
    assert "not in allowlist" in capsys.readouterr().out


def test_docs_figures_check_fails_on_relative_figure_link(tmp_path, monkeypatch, capsys):
    page = REPO / "docs" / "index.md"
    text = "[raster](_static/atlas/calibration_100/raster.html)\n"
    monkeypatch.setattr(bda, "docs_pages", lambda: {page: text})
    assert bda.docs_figures_check() == 1
    assert "figure-site URL" in capsys.readouterr().out


def test_site_fails_when_docs_cite_an_omitted_figure():
    # The first etude build listed every rerun panel as omitted while the
    # docs showed their stills; the completeness check alone passed it.
    page = REPO / "docs" / "etudes" / "e.md"
    text = (f"![r]({bda.FIGURES_URL}e/raster.png)\n"
            f"[r]({bda.FIGURES_URL}e/raster.html) · [x]({bda.FIGURES_URL}other/raster.html)\n")
    omitted = [{"slug": "e", "panel": "raster", "html": "e/raster.html", "status": "omitted"}]
    problems = bda.check_cited_figures(omitted, ["e"], {page: text})
    assert len(problems) == 2 and all("e/raster." in p for p in problems), problems
    available = [dict(omitted[0], status="available", png="e/raster.png")]
    assert bda.check_cited_figures(available, ["e"], {page: text}) == []
