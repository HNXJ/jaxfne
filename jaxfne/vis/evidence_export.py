"""Matplotlib evidence-figure export via canonical ``jaxfne.vis`` exporters."""
from __future__ import annotations

from pathlib import Path


def save_matplotlib_evidence_figure(
    fig,
    path: str | Path,
    *,
    dpi: int = 150,
    bbox_inches: str = "tight",
    facecolor: str | None = "white",
    close: bool = True,
) -> str:
    """Save a matplotlib figure through ``jaxfne.vis.exporters.export_figure``.

    Returns the written file path as a string.
    """
    from .exporters import close_matplotlib_figure

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = path.suffix.lstrip(".") or "png"
    if fmt != "png":
        raise ValueError(f"evidence export supports png only, got {fmt!r}")
    save_kwargs = {"dpi": dpi, "bbox_inches": bbox_inches}
    if facecolor is not None:
        save_kwargs["facecolor"] = facecolor
    fig.savefig(str(path), **save_kwargs)
    if close:
        close_matplotlib_figure(fig)
    return str(path)
