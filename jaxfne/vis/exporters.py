"""Canonical figure export — replaces save_figure/save_figures/save_png.

Single entry point for writing any jaxfne figure (matplotlib OR plotly) to
disk. Detects the figure type; no caller-side backend branching needed.
"""
from __future__ import annotations

from pathlib import Path


def _is_plotly_figure(fig) -> bool:
    return type(fig).__module__.startswith("plotly.")


def export_figure(fig, path, *, formats: tuple = ("html",), dpi: int = 150) -> dict[str, str]:
    """Export one figure (matplotlib or plotly) to one or more formats.

    Matplotlib figures: ``formats`` should be image formats (``png``, ``svg``,
    ``pdf``) — ``html`` is rejected. Plotly figures: ``html`` writes a
    standalone interactive page; ``png``/``svg``/``pdf`` rasterize via kaleido.

    Returns ``{format: path}``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = {}

    if _is_plotly_figure(fig):
        for fmt in formats:
            out = path.with_suffix(f".{fmt}")
            if fmt == "html":
                fig.write_html(str(out), include_plotlyjs="cdn")
            elif fmt in ("png", "svg", "pdf"):
                fig.write_image(str(out))
            else:
                raise ValueError(f"unsupported plotly export format: {fmt!r}")
            written[fmt] = str(out)
    else:
        for fmt in formats:
            if fmt == "html":
                raise ValueError("matplotlib figures cannot export to html")
            out = path.with_suffix(f".{fmt}")
            fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
            written[fmt] = str(out)

    return written


def export_figures(figures: dict, output_dir, *, formats: tuple = ("html", "png", "svg"), dpi: int = 150) -> dict:
    """Export a ``{name: figure}`` dict (mixed matplotlib/plotly OK).

    Writes to ``output_dir/<format>/<name>.<format>``; skips a format for a
    given figure if that figure's backend doesn't support it (e.g. ``html``
    for matplotlib).
    """
    output_dir = Path(output_dir)
    manifest = {}
    for name, fig in figures.items():
        is_plotly = _is_plotly_figure(fig)
        applicable = formats if is_plotly else tuple(f for f in formats if f != "html")
        manifest[name] = {}
        for fmt in applicable:
            (output_dir / fmt).mkdir(parents=True, exist_ok=True)
            out = output_dir / fmt / f"{name}.{fmt}"
            if is_plotly:
                if fmt == "html":
                    fig.write_html(str(out), include_plotlyjs="cdn")
                else:
                    fig.write_image(str(out))
            else:
                fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
            manifest[name][fmt] = str(out)
    return manifest


class FigureBundle:
    """A named collection of figures (matplotlib and/or plotly) with ``.export(...)``."""

    def __init__(self, figures: dict):
        self.figures = figures

    def __getitem__(self, key):
        return self.figures[key]

    def __iter__(self):
        return iter(self.figures.items())

    def keys(self):
        return self.figures.keys()

    def export(self, output_dir, *, formats: tuple = ("html", "png", "svg"), dpi: int = 150) -> dict:
        return export_figures(self.figures, output_dir, formats=formats, dpi=dpi)
