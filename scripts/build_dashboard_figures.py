#!/usr/bin/env python3
"""Render each dashboard chart to a static PNG from dashboard/data/replications.json.

The PNGs are the static copies of the page's Plotly charts and carry the same values. Each
chart is written twice, as dashboard/figures/<paper id>-<chart id>.png and
dashboard/figures/<paper id>-<chart id>.dark.png; the page shows the one matching the colour
scheme. dashboard/figures/manifest.json records the SHA-256 of the chart data each PNG pair
was drawn from, receipt digests excluded. This script does not import jaxfne, so the repo-root guard (P-025) is not needed.

  python scripts/build_dashboard_figures.py           render every chart (needs matplotlib)
  python scripts/build_dashboard_figures.py --check   exit 1 if a PNG is missing or drawn from other data

--check compares hashes only, so it runs without matplotlib and does not depend on its version.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dashboard" / "data" / "replications.json"
FIGURES = ROOT / "dashboard" / "figures"
MANIFEST = FIGURES / "manifest.json"
PALETTE = ["#1565c0", "#ff9800", "#00acc1", "#e53935", "#4caf50", "#8e24aa"]
DASHES = ["-", "--", ":", "-."]
GOLD = "#cfb87c"


def drawn(node):
    """The chart without receipt digests ("sha256" keys): a digest is never drawn."""
    if isinstance(node, dict):
        return {k: drawn(v) for k, v in node.items() if k != "sha256"}
    if isinstance(node, list):
        return [drawn(v) for v in node]
    return node


def chart_hash(chart):
    return hashlib.sha256(json.dumps(drawn(chart), sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def charts(doc):
    for paper in doc.get("papers", []):
        for chart in paper.get("charts", []):
            stem = f"{paper['id']}-{chart['id']}"
            yield f"{stem}.png", chart, False
            yield f"{stem}.dark.png", chart, True


def axis_title(axis):
    axis = axis or {}
    label, unit = axis.get("label", ""), axis.get("unit", "")
    return f"{label} ({unit})" if unit else label


def draw(chart, path, dark=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Styles are global in pyplot: reset each draw so a dark figure never restyles the next light one.
    plt.style.use("dark_background" if dark else "default")

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    traces = chart.get("traces", [])
    groups = []
    for t in traces:
        g = t.get("group", t.get("name"))
        if g not in groups:
            groups.append(g)
    if chart.get("kind") == "bars":
        cats = chart.get("categories", [])
        width = 0.8 / max(len(traces), 1)
        for i, t in enumerate(traces):
            xs = [j + (i - (len(traces) - 1) / 2) * width for j in range(len(cats))]
            ys = [float("nan") if y is None else y for y in t.get("y", [])]
            ax.bar(xs, ys, width, label=t["name"], color=PALETTE[groups.index(t.get("group", t["name"])) % len(PALETTE)])
        ax.set_xticks(range(len(cats)), cats, rotation=20, ha="right")
    else:
        seen = {}
        for t in traces:
            g = t.get("group", t["name"])
            n = seen[g] = seen.get(g, -1) + 1
            ys = [float("nan") if y is None else y for y in t.get("y", [])]
            ax.plot(t.get("x", []), ys, marker="o", ms=4, label=t["name"],
                    linestyle="none" if t.get("markers_only") else DASHES[n % len(DASHES)],
                    color=PALETTE[groups.index(g) % len(PALETTE)])
    if (chart.get("y") or {}).get("log"):
        ax.set_yscale("log")
    for ref in chart.get("refs", []):
        if ref["kind"] == "hline":
            ax.axhline(ref["y"], color=GOLD, linestyle="--", linewidth=2, label=ref.get("label"))
        elif ref["kind"] == "band":
            ax.axhspan(ref["y0"], ref["y1"], color=GOLD, alpha=0.3, label=ref.get("label"))
    ax.set_title(chart.get("title", ""))
    ax.set_xlabel(axis_title(chart.get("x")))
    ax.set_ylabel(axis_title(chart.get("y")))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor(), metadata={"Software": None})
    plt.close(fig)


def check(doc):
    try:
        recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"stale: cannot read {MANIFEST} ({exc}); run scripts/build_dashboard_figures.py", file=sys.stderr)
        return 1
    want = {name: chart_hash(chart) for name, chart, _ in charts(doc)}
    bad = sorted(n for n in set(want) | set(recorded) if want.get(n) != recorded.get(n) or not (FIGURES / n).is_file())
    if bad:
        print(f"stale: {', '.join(bad)}; run scripts/build_dashboard_figures.py", file=sys.stderr)
        return 1
    print(f"fresh: {len(want)} figures match the chart data")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render the dashboard charts to static PNGs.")
    parser.add_argument("--check", action="store_true", help="exit 1 if a PNG is missing or stale")
    args = parser.parse_args(argv)
    doc = json.loads(DATA.read_text(encoding="utf-8"))
    if args.check:
        return check(doc)
    FIGURES.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, chart, dark in charts(doc):
        draw(chart, FIGURES / name, dark=dark)
        manifest[name] = chart_hash(chart)
    for old in FIGURES.glob("*.png"):
        if old.name not in manifest:
            old.unlink()
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(manifest)} figures to {FIGURES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
