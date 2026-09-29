"""Column-glyph renderer for ``network_hspice(style="columns")``.

Same inputs as the block schematic (``describe`` output, stages, theme), drawn as
cylinders in a serial or staged layout: ``x : A ∘ B ∘ … ∘ Z : y``. One cylinder per
area, one node per cell class (area of the node follows the count), one arrow per
adjacent-stage projection with its edge count. A single-neuron area is drawn as one
node without a cylinder.

The wave glyph in an arrow is a schematic key, not data: a fast wave marks a
feedforward arrow, a slow wave a feedback arrow. Theme changes presentation only.
"""

from __future__ import annotations

import collections
import pathlib
from typing import Any, Mapping, Sequence

import numpy as np

_ACCENT = {"dark": "#e6c15a", "light": "#9a6b00"}
_PORT = "#22e04a"

COL_PITCH = 3.4  # inches between stage centres
ROW_PITCH = 4.0  # inches between areas stacked in one stage
CYL_W, CYL_H, CYL_RY = 1.55, 2.9, 0.26
MARGIN_X, MARGIN_TOP, MARGIN_BOT = 1.3, 1.3, 1.5


def grammar_caption(stages: Sequence[Sequence[str]], x: str, y: str) -> str:
    """``x : A ∘ B ∘ C : y``; areas sharing a stage are grouped as ``(A, B)``."""
    parts = [c[0] if len(c) == 1 else "(" + ", ".join(c) + ")" for c in stages]
    return f"{x} : " + " ∘ ".join(parts) + f" : {y}"


def _glow(artist_factory, th, *, lw):
    """Soft halo on dark themes: the same outline stroked wider at low alpha."""
    if th.name != "dark":
        return
    for k, a in ((5, 0.05), (3, 0.09), (1.6, 0.16)):
        artist_factory(lw * k, a)


def _cylinder(ax, cx, cy, colour, th):
    import matplotlib.patches as mp

    w, h, ry = CYL_W, CYL_H, CYL_RY
    x0, y0 = cx - w / 2, cy - h / 2
    ax.add_patch(mp.Rectangle((x0, y0 + ry), w, h - 2 * ry, facecolor=colour, alpha=0.06,
                              edgecolor="none", zorder=1))

    def draw(lw, alpha):
        ax.plot([x0, x0], [y0 + ry, y0 + h - ry], color=colour, lw=lw, alpha=alpha, zorder=2)
        ax.plot([x0 + w, x0 + w], [y0 + ry, y0 + h - ry], color=colour, lw=lw, alpha=alpha, zorder=2)
        ax.add_patch(mp.Ellipse((cx, y0 + h - ry), w, 2 * ry, fill=False, edgecolor=colour,
                                lw=lw, alpha=alpha, zorder=2))
        ax.add_patch(mp.Arc((cx, y0 + ry), w, 2 * ry, theta1=180, theta2=360, edgecolor=colour,
                            lw=lw, alpha=alpha, zorder=2))

    draw(1.6, 1.0)
    _glow(draw, th, lw=1.6)


def _nodes(ax, cx, cy, classes: Mapping[str, int], cmap, th):
    """One disc per cell class, stacked bottom to top in ``classes`` order."""
    import matplotlib.patches as mp

    names = list(classes)
    if not names:
        return
    top = cy + CYL_H / 2 - CYL_RY - 0.35
    bot = cy - CYL_H / 2 + CYL_RY + 0.35
    ys = np.linspace(top, bot, len(names)) if len(names) > 1 else [cy]
    biggest = max(classes.values())
    for name, y in zip(names, ys):
        r = 0.13 + 0.17 * np.sqrt(classes[name] / biggest)
        ax.add_patch(mp.Circle((cx, y), r, facecolor=cmap[name], edgecolor=th.text,
                               lw=0.6, alpha=0.95, zorder=4))
        ax.text(cx + r + 0.09, y, f"{name} {classes[name]}", ha="left", va="center",
                fontsize=7.5, color=th.text, zorder=5)


def _single_node(ax, cx, cy, classes, cmap, th):
    import matplotlib.patches as mp

    name = next(iter(classes), "")
    ax.add_patch(mp.Circle((cx, cy), 0.32, facecolor=cmap.get(name, th.text), edgecolor=th.text,
                           lw=0.8, zorder=4))


def _wave(ax, x0, x1, y, amp, cycles, colour):
    xs = np.linspace(x0, x1, 200)
    ax.plot(xs, y + amp * np.sin(2 * np.pi * cycles * (xs - x0) / (x1 - x0)), color=colour,
            lw=1.2, zorder=6)


def _block_arrow(ax, xa, xb, y, colour, th, *, forward, cycles, label):
    import matplotlib.patches as mp

    L, hh, hw = abs(xb - xa), 0.38, 0.62
    head = 0.5
    if forward:
        pts = [(xa, y + hh / 2), (xb - head, y + hh / 2), (xb - head, y + hw / 2), (xb, y),
               (xb - head, y - hw / 2), (xb - head, y - hh / 2), (xa, y - hh / 2)]
        w0, w1 = xa + 0.1, xb - head - 0.05
    else:
        pts = [(xb, y + hh / 2), (xa + head, y + hh / 2), (xa + head, y + hw / 2), (xa, y),
               (xa + head, y - hw / 2), (xa + head, y - hh / 2), (xb, y - hh / 2)]
        w0, w1 = xa + head + 0.05, xb - 0.1
    ax.add_patch(mp.Polygon(pts, closed=True, facecolor=colour, alpha=0.10, edgecolor="none",
                            zorder=3))
    ax.add_patch(mp.Polygon(pts, closed=True, fill=False, edgecolor=colour, lw=1.4, zorder=5))
    _wave(ax, w0, w1, y, hh * 0.32, cycles, colour)
    ax.text((xa + xb) / 2, y + (0.62 if forward else -0.62), label, ha="center",
            va="center", fontsize=7.5, color=colour, zorder=7)


def render_columns(d: Mapping[str, Any], stages: Sequence[Sequence[str]], th, cmap, *,
                   x: str, y: str, title: str | None, path, dpi: int, figsize, save) -> dict:
    """Draw ``d`` (``describe`` output) as column glyphs and return what was drawn."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    accent = _ACCENT.get(th.name, th.block_edge)
    n_col = len(stages)
    tallest = max(len(c) for c in stages)
    W = 2 * MARGIN_X + (n_col - 1) * COL_PITCH + CYL_W
    H = MARGIN_TOP + MARGIN_BOT + (tallest - 1) * ROW_PITCH + CYL_H
    fig, ax = plt.subplots(figsize=figsize or (W, H))
    fig.patch.set_facecolor(th.background)
    ax.set_facecolor(th.background)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    centre: dict[str, tuple[float, float]] = {}
    for ci, col in enumerate(stages):
        cx = MARGIN_X + CYL_W / 2 + ci * COL_PITCH
        for ri, a in enumerate(col):
            cy = H - MARGIN_TOP - CYL_H / 2 - ri * ROW_PITCH - (tallest - len(col)) * ROW_PITCH / 2
            centre[a] = (cx, cy)
            classes: dict[str, int] = collections.Counter()
            for layer in d["populations"][a].values():
                for c, n in layer.items():
                    classes[c] += n
            classes = dict(classes)
            half = CYL_H / 2
            if sum(classes.values()) == 1:
                _single_node(ax, cx, cy, classes, cmap, th)
                half = 0.55
            else:
                _cylinder(ax, cx, cy, accent, th)
                _nodes(ax, cx, cy, classes, cmap, th)
            ax.text(cx, cy + half + 0.28, a, ha="center", va="bottom", fontsize=13,
                    fontweight="bold", color=th.text)
            tally = " · ".join(f"{c}={n}" for c, n in classes.items())
            ax.text(cx, cy - half - 0.28, tally, ha="center", va="top", fontsize=8,
                    color=th.muted)

    stage_of = {a: i for i, col in enumerate(stages) for a in col}
    pair_edges: collections.Counter = collections.Counter()
    omitted = []
    for p in d["projections"]:
        s, t = stage_of.get(p["source_area"]), stage_of.get(p["target_area"])
        if s is None or t is None or abs(s - t) != 1:
            omitted.append({k: p[k] for k in ("source", "target", "n_edges")})
            continue
        pair_edges[(p["source_area"], p["target_area"])] += p["n_edges"]

    drawn = []
    for (sa, da), n in sorted(pair_edges.items()):
        forward = stage_of[da] > stage_of[sa]
        (xs, ys), (xd, yd) = centre[sa], centre[da]
        if forward:
            xa, xb = xs + CYL_W / 2 + 0.12, xd - CYL_W / 2 - 0.12
        else:
            xa, xb = xd + CYL_W / 2 + 0.12, xs - CYL_W / 2 - 0.12
        lane = -0.55 if not forward else 0.55
        ya = (ys + yd) / 2 + lane - (0.25 if len(stages[stage_of[sa]]) == 1 else 0.0)
        if abs(ys - yd) > 0.1:
            ax.annotate("", xy=(xb if forward else xa, yd + lane), xytext=(xa if forward else xb, ys + lane),
                        arrowprops=dict(arrowstyle="-|>", color=th.channel_color(
                            "feedforward" if forward else "feedback"), lw=1.6), zorder=6)
            ax.text((xa + xb) / 2, (ys + yd) / 2 + lane + 0.2, f"n={n}", ha="center",
                    fontsize=7.5, color=th.muted)
        else:
            _block_arrow(ax, xa, xb, ya, accent, th, forward=forward,
                         cycles=8 if forward else 2.5, label=f"n={n}")
        drawn.append({"source": sa, "target": da, "n_edges": int(n),
                      "channel": "feedforward" if forward else "feedback"})

    yy = H / 2
    for tag, xt, xa, xb in ((x, 0.35, 0.75, MARGIN_X - 0.1), (y, W - 0.35, W - MARGIN_X + 0.1, W - 0.75)):
        ax.annotate("", xy=(xb, yy), xytext=(xa, yy), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color=_PORT, lw=2.4))
        ax.text(xt, yy + 0.35, tag, ha="center", va="bottom", fontsize=11, color=_PORT,
                fontweight="bold")

    head = title or grammar_caption(stages, "x", "y")
    fig.text(0.5, 0.965, head, ha="center", va="top", fontsize=17, fontweight="bold",
             color=accent)
    foot = (f"{d['n_neurons']} neuron{'s' * (d['n_neurons'] != 1)} in {len(d['areas'])} "
            f"area{'s' * (len(d['areas']) != 1)}; {d['n_edges_long_range']} edges across areas.")
    if drawn:
        foot += " Wave glyphs are a schematic key: fast = feedforward, slow = feedback."
    fig.text(0.5, 0.02, foot, ha="center", va="bottom", fontsize=8, color=th.muted)

    saved = save(fig, path, th, dpi)
    plt.close(fig)
    return {"figure": "network_hspice", "style": "columns", "path": saved, "theme": th.name,
            "stages": [list(c) for c in stages], "areas": d["areas"],
            "n_neurons": d["n_neurons"], "n_edges_total": d["n_edges_total"],
            "n_edges_local": d["n_edges_local"], "n_edges_long_range": d["n_edges_long_range"],
            "n_projections": len(d["projections"]), "arrows": drawn,
            "omitted_projections": omitted}
