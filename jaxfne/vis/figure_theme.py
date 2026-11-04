"""Themed standalone HTML for one figure-site panel: light or dark, from the page that frames it.

``themed_html`` returns the full page of a Plotly figure (plotly.js from the CDN) with one script
injected before ``</body>``. The script starts from the OS colour scheme, so a page opened
directly follows the OS. It switches when its parent posts ``{type: "jaxfne-scheme", scheme}``,
and it reads only messages from ``window.parent``.

Each scheme is a layout template plus explicit layout keys for the values the figure builders
hard-code (page and plot backgrounds, font, and the line, grid and tick colours of every x and y
axis and of every 3D scene), so a builder's own value cannot pin the old scheme. The script only
relayouts: trace data is never changed.
"""
from __future__ import annotations

import json
import re

DIV_ID = "jx-figure"

_XY_AXIS = re.compile(r"[xy]axis\d*")
_SCENE = re.compile(r"scene\d*")


def _axis(ink: str, line: str, grid: str, tick: str) -> dict:
    # "color" sets the tick labels and the axis title, which a figure may have fixed for one scheme.
    return {"color": ink, "linecolor": line, "gridcolor": grid, "tickcolor": tick, "zerolinecolor": grid}


def _palette(paper: str, plot: str, ink: str, line: str, grid: str, tick: str) -> dict:
    axis = _axis(ink, line, grid, tick)
    # A 3D axis also paints its back plane, which the base template colours otherwise.
    plane = {**axis, "backgroundcolor": plot}
    return {
        "paper_bgcolor": paper,
        "plot_bgcolor": plot,
        "font": {"color": ink},
        "title": {"font": {"color": ink}},
        "legend": {"font": {"color": ink}},
        "xaxis": axis,
        "yaxis": axis,
        "scene": {"bgcolor": plot, "xaxis": plane, "yaxis": plane, "zaxis": plane},
    }


# DARK is the slate palette the figure site has always used (paper, plot and font as set by the
# atlas builders; axis colours from plotly_dark's own grid and line). LIGHT is a white page with dark ink.
LIGHT = _palette("#ffffff", "#ffffff", "#1f2328", "#59636e", "#d1d9e0", "#59636e")
DARK = _palette("#0d1117", "#161b22", "#c9d1d9", "#506784", "#283442", "#506784")

_SCRIPT = """(function () {
  var DIV = __DIV__;
  var SCHEMES = __SCHEMES__;
  function apply(scheme) {
    var figure = document.getElementById(DIV);
    if (figure && window.Plotly) {
      Plotly.relayout(figure, SCHEMES[scheme]);
    }
  }
  window.addEventListener("message", function (event) {
    if (event.source !== window.parent || !event.data || event.data.type !== "jaxfne-scheme") {
      return;
    }
    if (event.data.scheme === "light" || event.data.scheme === "dark") {
      apply(event.data.scheme);
    }
  });
  apply(window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
})();
"""


def _flat(prefix: str, node: dict, out: dict) -> dict:
    for key, value in node.items():
        if isinstance(value, dict):
            _flat(f"{prefix}{key}.", value, out)
        else:
            out[f"{prefix}{key}"] = value
    return out


def _axis_names(fig) -> tuple[list[str], list[str]]:
    keys = list(fig.layout.to_plotly_json())
    xy = sorted({"xaxis", "yaxis", *(k for k in keys if _XY_AXIS.fullmatch(k))})
    scenes = sorted(k for k in keys if _SCENE.fullmatch(k))
    return xy, scenes


def _update(palette: dict, xy: list[str], scenes: list[str]) -> dict:
    """The relayout that applies one scheme: its template, then every key by its full path."""
    update: dict = {"template": {"layout": palette}}
    _flat("", {k: v for k, v in palette.items() if k not in ("xaxis", "yaxis", "scene")}, update)
    for axis in xy:
        _flat(f"{axis}.", palette["xaxis"], update)
    for scene in scenes:
        _flat(f"{scene}.", palette["scene"], update)
    return update


def apply_scheme(fig, scheme: str):
    """A copy of ``fig`` in one scheme's palette, through the same keys the page relays.

    The docs stills are written from these copies, so a still and its page share one palette.
    The input figure is not changed.
    """
    palettes = {"light": LIGHT, "dark": DARK}
    if scheme not in palettes:
        raise ValueError(f"scheme must be 'light' or 'dark', not {scheme!r}")
    out = type(fig)(fig.to_dict())
    xy, scenes = _axis_names(out)
    out.update_layout(_update(palettes[scheme], xy, scenes))
    return out


def themed_html(fig, div_id: str = DIV_ID) -> str:
    """The figure's standalone page with the scheme script; the trace data is unchanged."""
    html = fig.to_html(full_html=True, include_plotlyjs="cdn", div_id=div_id)
    if html.count("</body>") != 1:
        raise ValueError("plotly's page has no single </body> to hold the theme script")
    xy, scenes = _axis_names(fig)
    schemes = {"light": _update(LIGHT, xy, scenes), "dark": _update(DARK, xy, scenes)}
    script = _SCRIPT.replace("__DIV__", json.dumps(div_id)).replace("__SCHEMES__", json.dumps(schemes))
    return html.replace("</body>", f"<script>\n{script}</script>\n</body>")
