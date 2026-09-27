"""Generate the interactive network pages and stills embedded in the docs.

Outputs (both from the same Plotly figure, so still and page cannot drift):

* ``docs/_static/visuals/<name>.html`` — interactive page (Plotly.js via CDN)
* ``docs/assets/visuals/<name>.png``  — still shown inline next to the tutorial

Figures:

* ``column_network`` — the canonical 1000-neuron column
  (``canonical-v1-column-1000n``, develop seed 0) with a sample of its realized
  edges split by presynaptic class.
* ``area_graph_n20`` — the 20-area hierarchy used by Atlas AT-10-N20
  (``artifacts/atlas/at10_n20_055.build_model``), areas on a circle in
  hierarchy order, one chord per connected area pair.
* ``hdp_h_dynamics``, ``hdp_weights`` — stills of the H and W panels of the
  small full-recording HDP run (``generate_doc_page_atlases.spec_hdp_10``);
  their interactive pages are that atlas's own panels.

Usage:
    python scripts/generate_docs_visuals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "artifacts" / "atlas", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import jaxfne as jtfne  # noqa: E402
from jaxfne.vis.exporters import _write_image  # noqa: E402  (retries Kaleido teardown, P-011)
from jaxfne.vis.plotly.network import plot_area_graph, plot_network_3d  # noqa: E402

HTML_DIR = ROOT / "docs" / "_static" / "visuals"
PNG_DIR = ROOT / "docs" / "assets" / "visuals"
MAX_EDGES = 800
THEME = dict(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
             font=dict(color="#c9d1d9"))


def column_network():
    genome = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")
    tensor = jtfne.develop(genome, seed=0)
    model = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=1, duration_ms=10.0, dt_ms=0.5))
    return plot_network_3d(model, show_edges=True, edge_color_by="source_class",
                           max_edges=MAX_EDGES, point_size=3.0,
                           title=f"Canonical column: 1000 neurons, {MAX_EDGES} sampled edges")


def area_graph_n20():
    import at10_n20_055 as n20

    return plot_area_graph(n20.build_model(), title="G_20 hierarchy: 20 areas, inter-area edges")


def _hdp_10_model():
    import generate_doc_page_atlases as dpa

    _title, model, _signals, _kw = dpa.spec_hdp_10()
    return model


def hdp_h_dynamics():
    from jaxfne.vis.atlas_suite import _h_dynamics_fig

    return _h_dynamics_fig(_hdp_10_model())


def hdp_weights():
    from jaxfne.vis.atlas_suite import _hdp_fig

    return _hdp_fig(_hdp_10_model())


# Stills only: their interactive pages are docs/_static/atlas/hdp_10/{h_dynamics,hdp}.html.
FIGURES = {"column_network": column_network, "area_graph_n20": area_graph_n20,
           "hdp_h_dynamics": hdp_h_dynamics, "hdp_weights": hdp_weights}
STILL_ONLY = {"hdp_h_dynamics", "hdp_weights"}


def main() -> int:
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    for name, make in FIGURES.items():
        fig = make()
        fig.update_layout(**THEME, margin=dict(l=0, r=0, t=50, b=0))
        if name not in STILL_ONLY:  # responsive page: no fixed size
            fig.write_html(str(HTML_DIR / f"{name}.html"), include_plotlyjs="cdn", full_html=True)
        fig.update_layout(width=1000, height=760)
        _write_image(fig, str(PNG_DIR / f"{name}.png"))
        print(f"wrote {name}: {'png' if name in STILL_ONLY else 'html + png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
