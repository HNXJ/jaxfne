"""jaxfne model-inspection renderers: a circuit schematic and a hierarchical raster.

Portable by construction. Only the jaxfne public surface is used —
``model.neuron_table()`` (neuron_id, area, layer, cell_type), ``model.params["edge_list"]``
(pre, post, weight, receptor_index) and a spike array or ``signals.spikes`` — so this file
carries no jomission import and no TFNE dependency. Copy verbatim to
``jaxfne/jaxfne/vis/network_inspect.py``.

Two renderers, answering the two questions a user has before any analysis:

    network_hspice(model)           what did I actually construct?
    network_raster(model, signals)  what does it immediately do?

Nothing here assumes six layers, a particular layer or area naming scheme, a hierarchy
direction, or that any cell class exists. Areas, layers and cell classes are whatever the
neuron table declares. Stage order is inferred from the projection graph and can be given
explicitly. TFNE notation, when a caller has it, is optional metadata passed as ``title``.

Design rule, pinned by ``test_theme_changes_presentation_only``:

    theme changes presentation only, never information or semantics.

Every renderer returns a description of what it drew. That description is identical under
every theme; only the pixels differ.
"""

from __future__ import annotations

import collections
import dataclasses
import pathlib
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

CHANNEL_ABBREV = {"feedforward": "FF", "feedback": "FB", "lateral": "LAT"}

__all__ = ["network_hspice", "network_raster", "describe", "Theme", "THEMES", "resolve_theme"]

# ---------------------------------------------------------------------------- themes


@dataclasses.dataclass(frozen=True)
class Theme:
    """Presentation only. No field here may change what is drawn, only how it looks."""

    name: str
    background: str
    text: str
    muted: str
    block_face: str
    block_edge: str
    grid: str
    label_halo: str
    classes: Mapping[str, str]
    channels: Mapping[str, str]
    fallback_channel: str = "#666666"
    # Cycled for cell classes the palette does not name, so an arbitrary ontology still
    # gets distinguishable colours instead of one shared grey.
    fallback_cycle: tuple = ("#8e7cc3", "#c98b3a", "#4c956c", "#b0556f", "#3d7ea6",
                             "#a3892f", "#7a6f9b")

    def class_colors(self, classes: Sequence[str]) -> dict[str, str]:
        """Colour every class in one pass, so unknown names cannot collide.

        Named classes keep their palette colour; the rest take cycle entries in the order
        given. Assigning per name independently (a hash, say) can map two classes to the
        same colour, which silently merges two populations in the reader's eye.
        """
        out, k = {}, 0
        for c in classes:
            known = self.classes.get(c)
            if known is not None:
                out[c] = known
            else:
                out[c] = self.fallback_cycle[k % len(self.fallback_cycle)]
                k += 1
        return out

    def class_color(self, cls: str) -> str:
        """One class in isolation. Prefer class_colors when rendering a whole model."""
        return self.classes.get(cls, self.fallback_cycle[0])

    def channel_color(self, channel: str) -> str:
        return self.channels.get(channel, self.fallback_channel)


_LIGHT_CLASSES = {"E": "#d1495b", "PV": "#2a9d8f", "SST": "#457b9d", "VIP": "#e9c46a",
                  "I": "#2a9d8f"}
_DARK_CLASSES = {"E": "#ff6b81", "PV": "#3ddbc6", "SST": "#6fb3e0", "VIP": "#f2cf6a",
                 "I": "#3ddbc6"}

THEMES: dict[str, Theme] = {
    "light": Theme(
        name="light", background="#ffffff", text="#1a1a1a", muted="#666666",
        block_face="#f7f7f9", block_edge="#333333", grid="#c9c9cf", label_halo="#ffffff",
        classes=_LIGHT_CLASSES,
        channels={"feedforward": "#2f6fb0", "feedback": "#b5651d", "lateral": "#6a5acd",
                  "input": "#2f6fb0", "output": "#b5651d", "inhibitory": "#4a4a4a"}),
    "dark": Theme(
        name="dark", background="#14161a", text="#ececf0", muted="#9aa0a6",
        block_face="#1e2127", block_edge="#c9ccd3", grid="#3a3f47", label_halo="#14161a",
        classes=_DARK_CLASSES,
        channels={"feedforward": "#6aa9e9", "feedback": "#e8a25c", "lateral": "#a493f5",
                  "input": "#6aa9e9", "output": "#e8a25c", "inhibitory": "#b8bdc6"}),
}


def resolve_theme(theme: str | Theme) -> Theme:
    if isinstance(theme, Theme):
        return theme
    try:
        return THEMES[theme]
    except KeyError:
        raise ValueError(f"unknown theme {theme!r}; known: {sorted(THEMES)} "
                         f"or pass a Theme instance") from None


# ------------------------------------------------------------------------- structure

_NUM = re.compile(r"(\d+)")


def _natural_key(s: str) -> tuple:
    """Order L2 before L10, and leave non-numeric names alphabetical."""
    return tuple(int(p) if p.isdigit() else p.lower() for p in _NUM.split(str(s)))


def _class_key(order: Sequence[str]):
    rank = {c: i for i, c in enumerate(order)}
    return lambda c: (rank.get(c, len(rank)), _natural_key(c))


def _spikes_of(signals: Any) -> np.ndarray:
    if signals is None:
        raise ValueError("signals is required")
    arr = getattr(signals, "spikes", signals)
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"spikes must be (n_time, n_neurons); got shape {arr.shape}")
    return arr


def describe(model: Any, *, inhibitory_receptors: Iterable[int] = (1,),
             class_order: Sequence[str] = ("E", "PV", "SST", "VIP"),
             stages: Sequence[Sequence[str]] | None = None) -> dict:
    """Populations and block-level projections of any jaxfne model.

    Groups come from ``neuron_table()``; no area, layer or cell-class name is assumed.
    Projections are aggregated to (source group -> target group) with edge count, mean
    weight and sign. No individual neuron appears in the result, which is what makes the
    schematic a block diagram rather than a connectivity plot.
    """
    table = model.neuron_table()
    rows = list(table.to_dict("records")) if hasattr(table, "to_dict") else list(table)
    if not rows:
        raise ValueError("neuron_table() is empty; nothing to describe")

    ids = np.asarray([int(r["neuron_id"]) for r in rows])
    area = np.asarray([str(r.get("area", "model")) for r in rows], dtype=object)
    layer = np.asarray([str(r.get("layer", "-")) for r in rows], dtype=object)
    cls = np.asarray([str(r.get("cell_type", "-")) for r in rows], dtype=object)
    n_total = int(ids.max()) + 1
    group = np.empty(n_total, dtype=object)
    for i, a, L, c in zip(ids, area, layer, cls):
        group[i] = (a, L, c)

    pops: dict[str, dict[str, dict[str, int]]] = collections.defaultdict(
        lambda: collections.defaultdict(dict))
    counts = collections.Counter(zip(area, layer, cls))
    for (a, L, c), n in counts.items():
        pops[a][L][c] = int(n)
    areas = sorted(pops, key=_natural_key)
    ckey = _class_key(class_order)
    pops = {a: {L: dict(sorted(pops[a][L].items(), key=lambda kv: ckey(kv[0])))
                for L in sorted(pops[a], key=_natural_key)} for a in areas}

    edges = model.params["edge_list"]
    pre = np.asarray(edges.pre)
    post = np.asarray(edges.post)
    weight = np.asarray(edges.weight, dtype=float)
    receptor = np.asarray(getattr(edges, "receptor_index", np.zeros(pre.shape, dtype=int)))
    inhib = set(int(r) for r in inhibitory_receptors)

    n_edge: collections.Counter = collections.Counter()
    w_sum: collections.Counter = collections.Counter()
    n_local = 0
    for p, q, w, r in zip(pre, post, weight, receptor):
        gp, gq = group[p], group[q]
        if gp is None or gq is None:
            continue
        ap, aq = gp[0], gq[0]
        if ap == aq:
            n_local += 1
            continue
        key = (gp, gq, int(r))
        n_edge[key] += 1
        w_sum[key] += float(w)

    stage_of = {}
    if stages is not None:
        for i, col in enumerate(stages):
            for a in col:
                stage_of[a] = i

    projections = []
    for (gp, gq, r), n in sorted(n_edge.items()):
        sa, sl, sc = gp
        da, dl, dc = gq
        projections.append({
            "source_area": sa, "target_area": da,
            "source": f"{sa}.{sl}.{sc}", "target": f"{da}.{dl}.{dc}",
            "source_layer": sl, "target_layer": dl,
            "n_edges": int(n), "mean_weight": round(w_sum[(gp, gq, r)] / n, 6),
            "receptor": int(r),
            "sign": "inhibitory" if int(r) in inhib else "excitatory",
        })
    if stages is None:
        stage_of = infer_stages(areas, projections)
    for p in projections:
        s, d = stage_of.get(p["source_area"], 0), stage_of.get(p["target_area"], 0)
        # Direction is read from the stage graph, not from any naming convention. A caller
        # with domain labels (a TFNE relation table, say) can relabel these afterwards.
        p["channel"] = "feedforward" if d > s else "feedback" if d < s else "lateral"

    return {"areas": areas, "populations": pops, "projections": projections,
            "stage_of": stage_of,
            "n_neurons": int(len(rows)),
            "n_edges_total": int(pre.size),
            "n_edges_local": int(n_local),
            "n_edges_long_range": int(pre.size) - int(n_local)}


def infer_stages(areas: Sequence[str], projections: Sequence[Mapping]) -> dict[str, int]:
    """Assign each area a column from the projection graph alone.

    For each unordered pair the direction carrying more edges is taken as forward; an exact
    tie carries no evidence of ordering and leaves the two areas at the same stage. Longest
    path over the resulting acyclic graph gives the column. A single area, an isolated model
    or a fully symmetric graph all collapse to one column, which is the honest answer rather
    than an invented hierarchy. Pass ``stages=`` to assert an order the topology does not
    show.
    """
    weight: dict = collections.Counter()
    for p in projections:
        if p["source_area"] != p["target_area"]:
            weight[(p["source_area"], p["target_area"])] += int(p["n_edges"])
    succ: dict[str, set] = {a: set() for a in areas}
    for a, b in list(weight):
        if weight[(a, b)] > weight.get((b, a), 0):
            succ.setdefault(a, set()).add(b)
    depth = {a: 0 for a in areas}
    for _ in range(len(areas) + 1):
        changed = False
        for a in areas:
            for b in succ.get(a, ()):
                if depth.get(b, 0) < depth[a] + 1:
                    depth[b] = depth[a] + 1
                    changed = True
        if not changed:
            break
    return depth


# ------------------------------------------------------------------------- rendering


def _mpl():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    return plt, mpatches


def _save(fig, path, theme, dpi):
    if path is None:
        return None
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=theme.background)
    return str(out)


def network_hspice(model: Any, *, x: str | None = None, y: str | None = None,
                   theme: str | Theme = "light", stages: Sequence[Sequence[str]] | None = None,
                   title: str | None = None, path: str | pathlib.Path | None = None,
                   inhibitory_receptors: Iterable[int] = (1,),
                   class_order: Sequence[str] = ("E", "PV", "SST", "VIP"),
                   figsize: tuple[float, float] | None = None, dpi: int = 150,
                   show_legend: bool = True) -> dict:
    """A block schematic of the constructed circuit: x on the left, y on the right.

    Areas are blocks, layers are rows inside them, cell classes are chips inside a row, and
    every cross-area projection is one annotated arrow carrying its edge count, mean weight
    and sign. Individual neurons are never drawn.

    ``title`` is free metadata: pass a TFNE expression when you have one, otherwise the
    default states the model in generic terms.
    """
    th = resolve_theme(theme)
    d = describe(model, inhibitory_receptors=inhibitory_receptors, class_order=class_order,
                 stages=stages)
    if stages is None:
        by_depth: dict[int, list] = collections.defaultdict(list)
        for a in d["areas"]:
            by_depth[d["stage_of"].get(a, 0)].append(a)
        stages = [sorted(by_depth[k], key=_natural_key) for k in sorted(by_depth)]
    stages = [list(c) for c in stages if c]

    plt, mpatches = _mpl()
    all_classes = sorted({c for a in d["populations"].values() for L in a.values() for c in L},
                         key=_class_key(class_order))
    cmap = th.class_colors(all_classes)
    n_col = len(stages)
    tallest = max(len(c) for c in stages)
    figsize = figsize or (max(9.0, 4.4 * n_col + 4.0), max(5.0, 2.9 * tallest + 2.4))
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(th.background)
    ax.set_facecolor(th.background)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left, right = 0.085, 0.915
    col_w = (right - left) / n_col
    box_w = col_w * 0.62
    rows: dict[str, dict[str, float]] = {}
    cx_of: dict[str, float] = {}
    span: dict[str, tuple[float, float]] = {}
    for ci, col in enumerate(stages):
        cx = left + ci * col_w + (col_w - box_w) / 2
        box_h = 0.68 / len(col) - 0.05
        for ri, a in enumerate(col):
            by = 0.20 + (len(col) - 1 - ri) * (box_h + 0.08)
            rows[a] = _draw_block(ax, cx, by, box_w, box_h, a, d["populations"][a], th, cmap)
            cx_of[a] = cx + box_w / 2
            span[a] = (cx, cx + box_w)

    _draw_port(ax, 0.006, "x", x or "input", th.channel_color("input"), th)
    _draw_port(ax, 0.928, "y", y or "output", th.channel_color("output"), th)
    _first, last = stages[0][0], stages[-1][0]
    for a in stages[0]:
        _arrow(ax, (0.074, 0.52), (span[a][0], np.mean(list(rows[a].values()) or [0.5])),
               th.channel_color("input"), th, "", rad=0.10, lw=2.0)
    ax.annotate("", xy=(0.926, 0.52),
                xytext=(span[last][1], np.mean(list(rows[last].values()) or [0.5])),
                zorder=6, arrowprops=dict(arrowstyle="-|>", color=th.channel_color("output"),
                                          linewidth=2.0, connectionstyle="arc3,rad=-0.12"))

    slot: collections.Counter = collections.Counter()
    FWD = (0.17, 0.29, 0.41, 0.53)
    BACK = (0.64, 0.74, 0.84, 0.94)
    drawn: set = set()
    for p in d["projections"]:
        sa, da, ch = p["source_area"], p["target_area"], p["channel"]
        if sa not in rows or da not in rows:
            continue
        colour = th.channel_color("inhibitory" if p["sign"] == "inhibitory" else ch)
        label = (f"{CHANNEL_ABBREV.get(ch, ch[:3].upper())}  {p['source'].split('.', 1)[1]}"
                 f"→{p['target'].split('.', 1)[1]}\n"
                 f"n={p['n_edges']}  w={p['mean_weight']:.3f}")
        ys = rows[sa].get(p["source_layer"], 0.5)
        yd = rows[da].get(p["target_layer"], 0.5)
        if ch == "lateral":
            key = (p["source"], p["target"])
            if (p["target"], p["source"]) in drawn:
                continue
            drawn.add(key)
            ax.annotate("", xy=(cx_of[da], yd), xytext=(cx_of[sa], ys), zorder=6,
                        arrowprops=dict(arrowstyle="<|-|>", color=colour, linewidth=2.0,
                                        shrinkA=3, shrinkB=3))
            ax.text(cx_of[sa] + 0.004, (ys + yd) / 2, label, ha="left", va="center",
                    fontsize=6.5, color=colour, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor=th.label_halo, alpha=0.9,
                              edgecolor="none"))
            continue
        fwd = ch == "feedforward"
        k = slot[(sa, da, ch)]
        slot[(sa, da, ch)] += 1
        pool = FWD if fwd else BACK
        p0 = (span[sa][1] if fwd else span[sa][0], ys)
        p1 = (span[da][0] if fwd else span[da][1], yd)
        _arrow(ax, p0, p1, colour, th, label, rad=0.16 if sa != da else 0.04,
               style="-|>" if p["sign"] != "inhibitory" else "-[",
               dashed=not fwd, label_pos=pool[k % len(pool)])

    if show_legend:
        handles = [mpatches.Patch(color=cmap[c], label=c) for c in all_classes]
        seen = {p["channel"] for p in d["projections"]}
        for ch, lab, dash in (("feedforward", "feedforward", False),
                              ("feedback", "feedback", True),
                              ("lateral", "lateral / within stage", False)):
            if ch in seen:
                handles.append(plt.Line2D([], [], color=th.channel_color(ch), lw=2,
                                          ls="--" if dash else "-", label=lab))
        if any(p["sign"] == "inhibitory" for p in d["projections"]):
            handles.append(plt.Line2D([], [], color=th.channel_color("inhibitory"), lw=2,
                                      label="inhibitory (bar end)"))
        leg = ax.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5),
                        frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.015))
        for t in leg.get_texts():
            t.set_color(th.text)

    head = title or f"input → model → output   ({len(d['areas'])} areas)"
    foot = (f"{d['n_neurons']} neurons, {d['n_edges_total']} edges: {d['n_edges_local']} "
            f"within-area, {d['n_edges_long_range']} across areas in "
            f"{len(d['projections'])} block projections. No individual neuron is drawn.")
    fig.suptitle(head, fontsize=14, fontweight="bold", color=th.text, y=0.985)
    fig.text(0.5, 0.945, foot, ha="center", va="top", fontsize=8.5, color=th.muted)
    fig.subplots_adjust(top=0.90)

    saved = _save(fig, path, th, dpi)
    plt.close(fig)
    return {"figure": "network_hspice", "path": saved, "theme": th.name,
            "stages": [list(c) for c in stages], "areas": d["areas"],
            "n_neurons": d["n_neurons"], "n_edges_total": d["n_edges_total"],
            "n_edges_local": d["n_edges_local"],
            "n_edges_long_range": d["n_edges_long_range"],
            "n_projections": len(d["projections"])}


def _draw_port(ax, x0, tag, label, colour, th):
    import matplotlib.patches as mpatches
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, 0.47), 0.066, 0.10, boxstyle="round,pad=0.006,rounding_size=0.01",
        facecolor=th.block_face, edgecolor=colour, linewidth=1.6, zorder=2))
    ax.text(x0 + 0.033, 0.545, tag, ha="center", va="center", fontsize=13,
            fontweight="bold", color=th.text)
    ax.text(x0 + 0.033, 0.503, label, ha="center", va="center", fontsize=7.2, color=colour)


def _draw_block(ax, x, y, w, h, area, layers, th, cmap, *, fontsize=7):
    import matplotlib.patches as mpatches
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
        facecolor=th.block_face, edgecolor=th.block_edge, linewidth=1.4, zorder=2))
    ax.text(x + w / 2, y + h + 0.012, area, ha="center", va="bottom",
            fontsize=fontsize + 3, fontweight="bold", color=th.text, zorder=3)
    names = list(layers)
    if not names:
        return {}
    row_h = h / len(names)
    # Reserve the gutter from the longest layer name rather than assuming short ones.
    gutter = min(0.42 * w, 0.010 + 0.0062 * max(len(str(L)) for L in names))
    out = {}
    for i, L in enumerate(names):
        ry = y + h - (i + 1) * row_h
        out[L] = ry + row_h / 2
        ax.add_patch(mpatches.Rectangle((x, ry), w, row_h, facecolor="none",
                                        edgecolor=th.grid, linewidth=0.6, zorder=3))
        ax.text(x + 0.006, ry + row_h / 2, L, ha="left", va="center", fontsize=fontsize,
                color=th.muted, zorder=4)
        chips = list(layers[L].items())
        cw = (w - gutter - 0.006) / max(len(chips), 1)
        for j, (cls, n) in enumerate(chips):
            cx = x + gutter + j * cw
            ax.add_patch(mpatches.Rectangle((cx, ry + row_h * 0.18), cw * 0.88, row_h * 0.64,
                                            facecolor=cmap.get(cls, th.class_color(cls)), alpha=0.9,
                                            edgecolor="none", zorder=4))
            ax.text(cx + cw * 0.44, ry + row_h / 2, f"{cls} {n}", ha="center", va="center",
                    fontsize=fontsize - 0.5, color=th.background, fontweight="bold", zorder=5)
    return out


def _arrow(ax, p0, p1, colour, th, label, *, rad=0.0, lw=1.6, style="-|>", dashed=False,
           label_pos=0.5, fontsize=6.5):
    ax.annotate("", xy=p1, xytext=p0, zorder=6, annotation_clip=False,
                arrowprops=dict(arrowstyle=style, color=colour,
                                linestyle="--" if dashed else "-", linewidth=lw,
                                connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if not label:
        return
    mx = p0[0] + (p1[0] - p0[0]) * label_pos
    my = p0[1] + (p1[1] - p0[1]) * label_pos + rad * 0.35
    ax.text(mx, my, label, ha="center", va="center", fontsize=fontsize, color=colour,
            zorder=7, bbox=dict(boxstyle="round,pad=0.15", facecolor=th.label_halo,
                                alpha=0.88, edgecolor="none"))


def unit_order(model: Any, *, areas: Sequence[str] | None = None,
               class_order: Sequence[str] = ("E", "PV", "SST", "VIP")) -> tuple:
    """Row order for a hierarchical raster: area, then layer, then cell class.

    One function, so a before figure and an after figure are the same figure twice.
    """
    table = model.neuron_table()
    rows = list(table.to_dict("records")) if hasattr(table, "to_dict") else list(table)
    ckey = _class_key(class_order)
    keep = set(areas) if areas is not None else None
    recs = [(str(r.get("area", "model")), str(r.get("layer", "-")),
             str(r.get("cell_type", "-")), int(r["neuron_id"])) for r in rows
            if keep is None or str(r.get("area", "model")) in keep]
    if not recs:
        raise ValueError("no units selected; check the areas argument")
    arank = ({a: i for i, a in enumerate(areas)} if areas is not None
             else {a: i for i, a in enumerate(sorted({r[0] for r in recs}, key=_natural_key))})
    recs.sort(key=lambda r: (arank[r[0]], _natural_key(r[1]), ckey(r[2]), r[3]))
    order = np.asarray([r[3] for r in recs])
    groups = []
    at = 0
    for key, chunk in _runs(recs):
        groups.append({"area": key[0], "layer": key[1], "cell_type": key[2],
                       "start": at, "stop": at + chunk, "n": chunk})
        at += chunk
    return order, groups


def _runs(recs):
    cur, n = None, 0
    for r in recs:
        k = (r[0], r[1], r[2])
        if k != cur:
            if cur is not None:
                yield cur, n
            cur, n = k, 0
        n += 1
    if cur is not None:
        yield cur, n


def network_raster(model: Any, signals: Any, *, window_ms: tuple[float, float] | None = None,
                   theme: str | Theme = "light", dt_ms: float = 1.0,
                   areas: Sequence[str] | None = None, title: str | None = None,
                   subtitle: str = "", path: str | pathlib.Path | None = None,
                   class_order: Sequence[str] = ("E", "PV", "SST", "VIP"),
                   figsize: tuple[float, float] = (16.0, 10.0), dpi: int = 150,
                   max_points: int = 400_000, show_rate: bool = True) -> dict:
    """A spike raster with rows ordered area, then layer, then cell class.

    ``signals`` is a jaxfne Signals or any (n_time, n_neurons) spike array. ``window_ms`` is
    (t0, t1); the default is the whole recording. Call it twice with the same arguments to
    get a before/after pair that can be read side by side.
    """
    th = resolve_theme(theme)
    plt, mpatches = _mpl()
    spikes = _spikes_of(signals)
    order, groups = unit_order(model, areas=areas, class_order=class_order)
    total_ms = spikes.shape[0] * dt_ms
    t0, t1 = window_ms if window_ms is not None else (0.0, total_ms)
    a, b = int(round(t0 / dt_ms)), int(round(t1 / dt_ms))
    if a < 0 or b > spikes.shape[0] or b <= a:
        raise ValueError(f"window {t0}-{t1} ms outside the recorded {total_ms} ms "
                         f"at dt {dt_ms} ms")
    block = np.asarray(spikes[a:b][:, order])
    ti, ui = np.nonzero(block)
    n_spikes = int(ti.size)
    thinned = n_spikes > max_points
    if thinned:
        keep = np.linspace(0, n_spikes - 1, max_points).astype(int)
        ti, ui = ti[keep], ui[keep]

    per_unit_class = np.empty(order.size, dtype=object)
    for g in groups:
        per_unit_class[g["start"]:g["stop"]] = g["cell_type"]
    cmap = th.class_colors(sorted({g["cell_type"] for g in groups},
                                  key=_class_key(class_order)))
    colours = [cmap[c] for c in per_unit_class]

    if show_rate:
        fig, (ax, axr) = plt.subplots(2, 1, figsize=figsize, height_ratios=[4.2, 1.0],
                                      sharex=True, gridspec_kw={"hspace": 0.06})
    else:
        fig, ax = plt.subplots(figsize=figsize)
        axr = None
    fig.patch.set_facecolor(th.background)
    for a_ in (ax, axr):
        if a_ is None:
            continue
        a_.set_facecolor(th.background)
        a_.tick_params(colors=th.muted)
        for sp in a_.spines.values():
            sp.set_color(th.grid)

    ax.scatter(t0 + ti * dt_ms, ui, s=0.45, c=[colours[i] for i in ui], marker="|",
               linewidths=0.45)
    seen = None
    for g in groups:
        if g["area"] != seen:
            ax.axhline(g["start"], color=th.block_edge, linewidth=1.0, alpha=0.85)
            ax.text(t0 - (t1 - t0) * 0.012, (g["start"] + g["stop"]) / 2, g["area"],
                    ha="right", va="center", fontsize=9, fontweight="bold", color=th.text)
            seen = g["area"]
        else:
            ax.axhline(g["start"], color=th.grid, linewidth=0.5, alpha=0.7)
    ax.set_ylim(0, order.size)
    ax.set_yticks([])
    ax.set_ylabel("unit  (area → layer → cell type)", color=th.text, labelpad=34)
    ax.set_title(title or f"{order.size} units, {t1 - t0:.0f} ms", fontsize=14,
                 fontweight="bold", loc="left", color=th.text, pad=22)
    if subtitle:
        ax.text(0.0, 1.008, subtitle, transform=ax.transAxes, fontsize=8.5, color=th.muted)
    classes = sorted(set(per_unit_class.tolist()), key=_class_key(class_order))
    leg = ax.legend(handles=[mpatches.Patch(color=cmap[c], label=c) for c in classes],
                    loc="upper right", ncol=min(len(classes), 4), frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(th.text)

    rate = block.sum(axis=1) / (order.size * dt_ms * 1e-3)
    if axr is not None:
        axr.plot(t0 + np.arange(block.shape[0]) * dt_ms, rate, color=th.text, linewidth=0.8)
        axr.set_xlabel("time (ms)", color=th.text)
        axr.set_ylabel("population\nrate (Hz)", color=th.text)
        axr.set_xlim(t0, t1)
        axr.margins(x=0)
    else:
        ax.set_xlabel("time (ms)", color=th.text)
        ax.set_xlim(t0, t1)

    saved = _save(fig, path, th, dpi)
    plt.close(fig)
    return {"figure": "network_raster", "path": saved, "theme": th.name,
            "n_units": int(order.size), "n_groups": len(groups),
            "areas": [g["area"] for g in groups][:0] or sorted({g["area"] for g in groups},
                                                               key=_natural_key),
            "window_ms": [float(t0), float(t1)], "dt_ms": float(dt_ms),
            "n_spikes": n_spikes, "thinned": bool(thinned),
            "mean_rate_hz": round(float(rate.mean()), 4)}
