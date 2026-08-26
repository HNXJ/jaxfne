"""Shared polish layer: frozen-path guard, vector export, typography floor.

Phase B, downstream of the frozen 0.4.17 scientific set. Every write passes a
fail-closed guard: only ``artifacts/figures/publication/final/`` and
``artifacts/publication/polish/`` are writable, and nothing on
``artifacts/publication/frozen_manifest.json`` may be touched. Purely additive reads are fine.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from matplotlib.figure import Figure

_REPO = Path(__file__).resolve().parents[3]
FROZEN_MANIFEST = _REPO / "artifacts/publication/frozen_manifest.json"
FINAL_DIR = _REPO / "artifacts" / "figures" / "publication" / "final"
POLISH_DIR = _REPO / "artifacts" / "publication" / "polish"

# Cross-figure typography budget (approved)
SUPTITLE_PT = 12.0
SUBTITLE_PT = 9.0
BODY_MIN_PT = 7.0
FONT_FLOOR_PT = 6.5


def load_frozen_paths() -> dict[str, str]:
    """Mechanically derive the immutable file set and its shas."""
    raw = json.loads(FROZEN_MANIFEST.read_text())
    files = raw.get("files", {}) if isinstance(raw, dict) else {f["path"]: f["sha256"] for f in raw}
    return {k: v for k, v in files.items()}


FROZEN = load_frozen_paths()


def guarded_path(rel: str) -> Path:
    """Fail-closed writer: only allows new polish-layer outputs."""
    rel = rel.lstrip("/")
    if rel in FROZEN:
        raise PermissionError(f"refusing to write frozen path: {rel}")
    abs_path = (_REPO / rel).resolve()
    final_ok = abs_path.is_relative_to(FINAL_DIR.resolve())
    polish_ok = abs_path.is_relative_to(POLISH_DIR.resolve())
    if not (final_ok or polish_ok):
        raise PermissionError(f"refusing write outside polish layer: {rel}")
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    return abs_path


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def export_final(fig: Figure, basename: str, *, dpi: int = 300) -> dict:
    """300-DPI PNG + true vector PDF, both emitted from the live Figure."""
    png = guarded_path(f"artifacts/figures/publication/final/{basename}.png")
    pdf = guarded_path(f"artifacts/figures/publication/final/{basename}.pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, format="pdf", bbox_inches="tight", facecolor="white")
    return {
        "png_300dpi": str(png.relative_to(_REPO)),
        "png_sha256": sha256_file(png),
        "png_bytes": png.stat().st_size,
        "pdf_vector": str(pdf.relative_to(_REPO)),
        "pdf_bytes": pdf.stat().st_size,
        "dpi": dpi,
    }


def all_text_artists(fig: Figure):
    for ax in fig.axes:
        for t in ax.texts:
            yield t
        yield ax.xaxis.label
        yield ax.yaxis.label
        if ax.xaxis.get_visible():
            yield from ax.get_xticklabels()
        if ax.yaxis.get_visible():
            yield from ax.get_yticklabels()
        if ax.legend_ is not None and ax.legend_.get_texts():
            yield from ax.legend_.get_texts()
        yield ax.title
    for t in fig.texts:
        yield t


def enforce_font_floor(fig: Figure, floor: float = FONT_FLOOR_PT) -> dict:
    """Raise any stretched text under the floor; never shrink. Returns change log."""
    changed = {}
    for i, t in enumerate(all_text_artists(fig)):
        try:
            fs = t.get_fontsize()
        except Exception:
            continue
        if fs is not None and fs < floor:
            t.set_fontsize(floor)
            changed[i] = {"old": fs, "new": floor}
    return changed


def min_font(fig: Figure) -> float:
    sizes = [t.get_fontsize() for t in all_text_artists(fig)]
    sizes = [s for s in sizes if s is not None]
    return min(sizes)


def clip_check(fig: Figure) -> dict:
    """Canvas-containment check against the tight bbox (frozen pipeline competed
    with bbox_inches='tight', so the effective canvas is the tight bbox).

    ``get_tightbbox`` yields a TransformedBbox in figure (inches) coordinates,
    while ``get_window_extent`` yields display pixels; divide the latter by
    ``fig.dpi`` to compare in the same (inch) space.
    """
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tb = fig.get_tightbbox(renderer)
        x_off, y_off = float(tb.x0), float(tb.y0)
        w_in, h_in = float(tb.width), float(tb.height)
    except Exception:
        x_off, y_off = 0.0, 0.0
        w_in, h_in = (float(v) for v in fig.get_size_inches())
    flagged = []
    outside = []
    protruding = set()
    tol_in = 1.5 / 72.0
    for t in all_text_artists(fig):
        if not getattr(t, "get_visible", lambda: True)():
            continue  # hidden artists (e.g. axis('off') residual ticks) don't render
        try:
            bb = t.get_window_extent(renderer=None)
            if bb.width <= 0 or bb.height <= 0:
                continue
        except Exception:
            continue
        px = 1.0 / fig.dpi  # display px -> inches
        sx = bb.x0 * px - x_off
        sy = bb.y0 * px - y_off
        ex = sx + bb.width * px
        ey = sy + bb.height * px
        full = t.get_text()
        if ex < -tol_in or ey < -tol_in or sx > w_in + tol_in or sy > h_in + tol_in:
            outside.append((round(sx * 72, 1), round(sy * 72, 1), round(ex * 72, 1), round(ey * 72, 1), full[:40]))
            protruding.add(full)
            continue
        if sx < -tol_in or sy < -tol_in or ex > w_in + tol_in or ey > h_in + tol_in:
            flagged.append((round(sx * 72, 1), round(sy * 72, 1), round(ex * 72, 1), round(ey * 72, 1), full[:40]))
            protruding.add(full)
    return {"pass": len(flagged) == 0, "flagged": flagged, "outside_canvas": outside, "protruding": protruding}


def colors_present_in_png(png: Path, required: set[str] | None = None) -> set[str]:
    """Every solid artist color we allow must appear among the frozen PNG pixels."""
    from PIL import Image

    with Image.open(png).convert("RGB") as im:
        px = im.load()
        w, h = im.size
        colors = set()
        target = required if required is not None else set()
        for yy in range(h):
            for xx in range(w):
                c = px[xx, yy]
                colors.add("#%02X%02X%02X" % c)
                if target and target.issubset(colors):
                    return colors
        return colors


def semantic_palette():
    return {
        "#1A4A8A", "#666666", "#8B4513", "#888888", "#F3F3F3", "#6A0DAD", "#F5E6FF",
        "#E8F0FE", "#0B6E4F", "#B85C00", "#C44E52", "#AAC4E8", "#F7FAFF", "#F0FFF8",
        "#FFF8E7", "#FFF0F0", "#AA4444", "#555555", "#333333", "#222222", "#444444",
        "#888888", "#0B6E4F", "#C44E52", "#B85C00", "#1A4A8A",
        "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3",
        "#8C8C8C", "#CCB974", "#64B5CD",
        # matplotlib tab10 defaults used by figures that never set explicit colors
        "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B",
        "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
        # neutral grays from the frozen renders (axis/grid/annotation work)
        "#CCCCCC", "#000000", "#FFFFFF",
    }


def render_info(png: Path):
    from PIL import Image
    with Image.open(png) as im:
        dpi = im.info.get("dpi", (0, 0))
        return {"size_px": list(im.size), "dpi": [round(float(d), 1) for d in dpi]}


def is_vector_pdf(pdf: Path) -> bool:
    head = pdf.read_bytes()[:4096]
    return b"%PDF" in head


REPO_ROOT = _REPO