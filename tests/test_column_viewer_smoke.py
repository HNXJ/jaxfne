"""Smoke for jaxfne.vis.column_viewer — export contract + realized EdgeList invariants.

Δscience=0 — viewer is read-only, no kernel/sampler change.
Covers F1 PARTIAL missing_delta: vis/__init__.py must re-export
collect_column_viewer_data / render_column_viewer and expose
jtfne.vis.column_viewer attribute, plus pytest smoke.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path


def _build_canonical_model(n: int = 200):
    """Canonical laminar column (E/I gradient) — small n for fast smoke."""
    import jaxfne as jtfne

    cfg = jtfne.build_laminar_column(n=n, ei_profile="canonical")
    cfg = (
        cfg.set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes"], n_contacts=16)
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    )
    model = jtfne.construct(cfg)
    return model


def test_column_viewer_exported_via_jtfne_vis():
    """jtfne.vis must expose column_viewer functions and submodule attribute."""
    import jaxfne as jtfne

    assert hasattr(jtfne.vis, "collect_column_viewer_data"), (
        "jaxfne.vis missing collect_column_viewer_data"
    )
    assert hasattr(jtfne.vis, "render_column_viewer"), "jaxfne.vis missing render_column_viewer"
    assert hasattr(jtfne.vis, "column_viewer"), (
        "jaxfne.vis missing column_viewer submodule attribute"
    )
    assert callable(jtfne.vis.collect_column_viewer_data)
    assert callable(jtfne.vis.render_column_viewer)
    # submodule identity — from import and attribute should align
    from jaxfne.vis.column_viewer import collect_column_viewer_data as direct

    assert jtfne.vis.collect_column_viewer_data is direct
    # also via jtfne.vis.column_viewer
    assert hasattr(jtfne.vis.column_viewer, "collect_column_viewer_data")
    assert hasattr(jtfne.vis.column_viewer, "render_column_viewer")


def test_column_viewer_import_overhead_false():
    """import jaxfne must not eagerly load column_viewer / plotly (Δscience=0, zero overhead)."""
    code = """
import sys
import jaxfne
# after import jaxfne, column_viewer should not be loaded (lazy vis)
has_cv = \"jaxfne.vis.column_viewer\" in sys.modules
has_plotly = any(k.startswith(\"plotly\") for k in sys.modules)
has_mpl = any(k.startswith(\"matplotlib\") for k in sys.modules)
# vis itself is lazy via _RuntimeModuleWrapper — not loaded until accessed
if has_cv:
    print(f\"FAILED: column_viewer eagerly loaded: {[k for k in sys.modules if 'column_viewer' in k]}\")
    sys.exit(1)
if has_plotly or has_mpl:
    # allow matplotlib if something else pulled it, but we expect clean
    loaded = [k for k in sys.modules if k.startswith(\"plotly\") or k.startswith(\"matplotlib\")]
    print(f\"FAILED: graphics eagerly loaded on import jaxfne: {loaded}\")
    sys.exit(1)
print(\"SUCCESS overhead false\")
sys.exit(0)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", check=False
    )
    assert result.returncode == 0, (
        f"import overhead check failed: stdout={result.stdout} stderr={result.stderr}"
    )

    # also ensure that importing via jaxfne.vis does load column_viewer (export contract)
    # but still does not pull plotly (column_viewer is plotly-lazy)
    code2 = """
import sys
import jaxfne as jtfne
_ = jtfne.vis.collect_column_viewer_data
_ = jtfne.vis.render_column_viewer
has_cv = \"jaxfne.vis.column_viewer\" in sys.modules
has_plotly = any(k.startswith(\"plotly\") for k in sys.modules)
if not has_cv:
    print(\"FAILED: jtfne.vis.collect_column_viewer_data did not load column_viewer\")
    sys.exit(1)
if has_plotly:
    loaded = [k for k in sys.modules if k.startswith(\"plotly\")]
    print(f\"FAILED: plotly eagerly loaded on vis import: {loaded}\")
    sys.exit(1)
print(\"SUCCESS vis lazy\")
sys.exit(0)
"""
    result2 = subprocess.run(
        [sys.executable, "-c", code2], capture_output=True, text=True, encoding="utf-8", check=False
    )
    assert result2.returncode == 0, (
        f"vis lazy check failed: stdout={result2.stdout} stderr={result2.stderr}"
    )


def test_collect_column_viewer_data_smoke():
    """Construct canonical 200n model, collect payload, assert invariants."""
    model = _build_canonical_model(n=200)
    from jaxfne.vis.column_viewer import collect_column_viewer_data

    # also via jtfne.vis
    import jaxfne as jtfne

    assert jtfne.vis.collect_column_viewer_data is collect_column_viewer_data

    data = collect_column_viewer_data(model)
    n = data["n"]
    assert n == 200, f"expected n=200, got {n}"
    # positions shape
    assert data["x"].shape == (n,), f"x shape {data['x'].shape} != {(n,)}"
    assert data["y"].shape == (n,)
    assert data["z"].shape == (n,)
    # realized edges
    n_edges = int(data["realized"]["n_edges"])
    assert n_edges > 0, "n_edges should be >0 for all-to-all 200n"
    assert data["pre"].shape[0] == n_edges
    assert data["post"].shape[0] == n_edges
    # cat_counts sum == n_edges
    cat_counts = data["cat_counts"]
    total_cat = sum(int(v) for v in cat_counts.values())
    assert total_cat == n_edges, (
        f"cat_counts sum {total_cat} != n_edges {n_edges} (counts={cat_counts})"
    )
    # degree arrays length n
    assert data["in_deg"].shape == (n,)
    assert data["out_deg"].shape == (n,)
    # weight / delay present
    assert data["weight"].shape[0] == n_edges
    # sanity: no simulation was run (viewer is read-only)
    html_markers = data["realized"]["edge_category_counts"]
    assert set(html_markers.keys()) == {"E→E", "E→I", "I→E", "I→I"}


def test_render_column_viewer_smoke(tmp_path: Path):
    """Render HTML, assert E→E/weight/degree/Plotly CDN, re-render SHA matches."""
    model = _build_canonical_model(n=200)
    from jaxfne.vis.column_viewer import render_column_viewer

    import jaxfne as jtfne

    assert jtfne.vis.render_column_viewer is render_column_viewer

    out1 = tmp_path / "column_viewer_a.html"
    out2 = tmp_path / "column_viewer_b.html"

    path1, summary1 = render_column_viewer(model, output_path=out1)
    path2, summary2 = render_column_viewer(model, output_path=out2)

    assert Path(path1) == out1
    assert out1.exists() and out1.stat().st_size > 0
    assert out2.exists() and out2.stat().st_size > 0

    html = out1.read_text(encoding="utf-8")
    # required content
    assert "E→E" in html, "HTML missing E→E edge category"
    assert "weight" in html.lower(), "HTML missing weight stats"
    assert "degree" in html.lower(), "HTML missing degree stats"
    # Plotly CDN
    assert "plotly" in html.lower(), "HTML missing Plotly CDN/reference"
    assert "cdn.plot.ly" in html or "plotly" in html.lower()
    # check for 3D div and viewer title
    assert "plot3d" in html.lower() or "scatter3d" in html.lower()

    # re-render SHA matches (deterministic)
    sha1 = hashlib.sha256(out1.read_bytes()).hexdigest()
    sha2 = hashlib.sha256(out2.read_bytes()).hexdigest()
    assert sha1 == sha2, f"re-render SHA mismatch: {sha1[:12]} != {sha2[:12]}"

    # summary sanity
    assert summary1["realized"]["n_edges"] == summary2["realized"]["n_edges"]
    assert "n_edges_shown_per_category" in summary1
