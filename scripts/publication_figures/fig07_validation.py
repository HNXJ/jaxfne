"""F7, the main-paper validation figure: the scored criteria that pass, each statistic
beside its criterion threshold (a dashed line). Every number is read from a score JSON
and every pass is recomputed here. A plotted criterion that does not pass raises
ValueError (F7 carries no negative results); so does a recomputed pass that disagrees
with the pass the score JSON records.

Outputs F7_validation.png (300 dpi) and F7_validation.json (every plotted value and
threshold) into --out. Usage: python scripts/publication_figures/fig07_validation.py [--out DIR]
Exit codes: 0 rendered; a failing criterion raises ValueError (traceback, exit 1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pub_figure_common import repo_sha  # noqa: E402
from _pub_semantic_styles import DASHED, SOLID  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PCL_SCORE = REPO / "artifacts/etudes/papers/pcl/prereg/scores/fig3_v4_full.json"
VOGELS_SCORE = REPO / "artifacts/etudes/papers/vogels2011/run/p3_v4_score.json"
DEFAULT_OUT = REPO / "artifacts/_temp/figures"
FULL_WIDTH_IN = 7.2  # full text width, as supplement.py
DPI = 300  # final raster, as supplement.py

# Criterion constants, each with the prereg that fixes it.
T1_FLOOR = 2.0  # pcl/prereg/P3_fig3.md: R_ori >= 2, above the 99th percentile of N-label and N-pref
T3_SIMPLE_MIN = 0.15  # pcl/prereg/P3_fig3.md: median_X_simple >= 0.15
# pcl/prereg/P3_fig3.md: median_X_complex <= 0.5 * median_X_simple. The <= direction is the
# one that matches the recorded t3_complex_pass (complex -0.122 against 0.265).
T3_COMPLEX_FRACTION = 0.5
CONV_T_CRIT = 2.101  # vogels2011/prereg/P3_v4_signed.md: CONV |b|/se
CONV_SLOPE_HZ_S = 0.01  # vogels2011/prereg/P3_v4_signed.md: CONV slope floor, Hz/s
O_DROP_RATIO = 2.0  # vogels2011/prereg/P3_v4_signed.md: O-DROP ratio (the score JSON cites P3 v2)


def _row(panel: str, test: str, row_id: str, value: float, threshold: float, op: str,
         rule: str, passed: bool, recorded: object, terms: dict) -> dict:
    """One plotted criterion. Raises when it does not pass or disagrees with the record."""
    if not passed:
        raise ValueError(
            f"{row_id} does not pass ({test}): value {value:.6g}, rule {rule}, "
            f"threshold {threshold:.6g}"
        )
    if recorded is not None and bool(recorded) != bool(passed):
        raise ValueError(
            f"{row_id}: recomputed pass {bool(passed)} disagrees with the recorded pass "
            f"{bool(recorded)}"
        )
    return {
        "panel": panel,
        "test": test,
        "id": row_id,
        "value": float(value),
        "threshold": float(threshold),
        "op": op,
        "rule": rule,
        "pass": bool(passed),
        "recorded_pass": None if recorded is None else bool(recorded),
        "terms": terms,
    }


def pcl_rows(pcl: dict) -> list[dict]:
    rows = []
    for kind in ("simple", "complex"):
        field = f"t1_r_ori_{kind}"
        null99, npref99 = pcl[f"{field}_null99"], pcl[f"{field}_npref99"]
        threshold = max(T1_FLOOR, null99, npref99)
        value = pcl[field]
        rows.append(_row(
            "a", "T1", field, value, threshold, ">=",
            "value >= 2 and value > null99 and value > npref99",
            value >= T1_FLOOR and value > null99 and value > npref99, pcl[f"t1_{kind}_pass"],
            {"floor": T1_FLOOR, "null99": null99, "npref99": npref99},
        ))
    simple = pcl["median_X_simple"]
    rows.append(_row(
        "b", "T3", "median_X_simple", simple, T3_SIMPLE_MIN, ">=", "value >= threshold",
        simple >= T3_SIMPLE_MIN, pcl["t3_simple_pass"], {},
    ))
    complex_x = pcl["median_X_complex"]
    limit = T3_COMPLEX_FRACTION * simple
    rows.append(_row(
        "b", "T3", "median_X_complex", complex_x, limit, "<=", "value <= threshold",
        complex_x <= limit, pcl["t3_complex_pass"],
        {"median_X_simple": simple, "fraction": T3_COMPLEX_FRACTION},
    ))
    # The v5 criterion, not t4_pass (the v4 sign rule, which failed). No recorded v5 flag.
    diff, diff_null = pcl["t4_diff"], pcl["t4_diff_null99"]
    l_simple, l_complex = pcl["median_L_simple"], pcl["median_L_complex"]
    rows.append(_row(
        "c", "T4", "t4_diff", diff, diff_null, ">",
        "value > threshold and median_L_complex > median_L_simple",
        diff > diff_null and l_complex > l_simple, None,
        {"median_L_simple": l_simple, "median_L_complex": l_complex},
    ))
    return rows


def vogels_rows(vog: dict) -> list[dict]:
    rows = []
    for seed in vog["seeds"]:
        conv = vog["CONV"]["per_seed"][str(seed)]
        b, se = float(conv["b_hz_s"]), float(conv["se_hz_s"])
        if se <= 0:
            raise ValueError(f"seed {seed}: se_hz_s must be positive, got {se}")
        t = abs(b) / se
        converged = not (abs(b) >= CONV_SLOPE_HZ_S and t > CONV_T_CRIT)
        rows.append(_row(
            "d", "CONV", f"CONV seed {seed}", t, CONV_T_CRIT, "<=",
            "pass unless |b| >= 0.01 Hz/s and value > threshold",
            converged, conv["converged"],
            {"b_hz_s": b, "se_hz_s": se, "slope_floor_hz_s": CONV_SLOPE_HZ_S},
        ))
    drop = vog["O-DROP"]
    ratio = float(drop["ratio"])
    rows.append(_row(
        "d", "O-DROP", "O-DROP ratio", ratio, O_DROP_RATIO, ">=", "value >= threshold",
        ratio >= O_DROP_RATIO, drop["pass"],
        {"before_hz": drop["before_hz"], "z_hz": drop["z_hz"]},
    ))
    return rows


def _rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else path.name


def write_fig07(out_dir: Path, pcl_path: Path = PCL_SCORE,
                vogels_path: Path = VOGELS_SCORE) -> dict[str, int]:
    pcl = json.loads(pcl_path.read_text(encoding="utf-8"))
    vog = json.loads(vogels_path.read_text(encoding="utf-8"))
    rows = pcl_rows(pcl) + vogels_rows(vog)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_f7(rows, out_dir / "F7_validation.png")
    payload = {
        "figure": "F7_validation",
        "sources": {"pcl": _rel(pcl_path), "vogels": _rel(vogels_path)},
        "repo_head": repo_sha(),
        "criteria": rows,
    }
    (out_dir / "F7_validation.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {"criteria": len(rows), "passing": sum(row["pass"] for row in rows)}


def _tick(row: dict) -> str:
    if row["test"] == "CONV":  # the shared threshold is in the panel title
        return row["id"].removeprefix("CONV ")
    label = TICK_LABELS.get(row["id"], row["id"])
    return f"{label}\n{row['op']} {row['threshold']:.3f}"


TICK_LABELS = {
    "t1_r_ori_simple": "simple", "t1_r_ori_complex": "complex",
    "median_X_simple": "simple", "median_X_complex": "complex",
    "t4_diff": "complex - simple", "O-DROP ratio": "onset / steady",
}


def _bar_panel(ax, rows: list[dict], title: str, ylabel: str, note: str | None = None) -> None:
    """Bars are statistics; each dashed segment is that statistic's criterion threshold."""
    x = list(range(len(rows)))
    ax.bar(x, [row["value"] for row in rows], width=0.6, color=SOLID["color"], zorder=2)
    for i, row in enumerate(rows):
        ax.plot([i - 0.4, i + 0.4], [row["threshold"]] * 2, color=DASHED["color"],
                linestyle=DASHED["linestyle"], linewidth=DASHED["linewidth"], zorder=3)
        va = "bottom" if row["value"] >= 0 else "top"
        ax.text(i, row["value"], f"{row['value']:.3g}", ha="center", va=va, fontsize=6.5)
    ax.axhline(0, color="#333333", linewidth=0.6)
    ax.margins(y=0.2)  # headroom for the value labels and the panel note
    ax.set_xticks(x, [_tick(row) for row in rows], fontsize=6.5)
    ax.set_title(title, loc="left", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    if note:
        ax.text(0.02, 0.97, note, transform=ax.transAxes, va="top", fontsize=6.5)


def plot_f7(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(FULL_WIDTH_IN, 6.2), layout="constrained")
    grid = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    lower = grid[1, 1].subgridspec(1, 2, width_ratios=[5, 2])
    ax_conv = fig.add_subplot(lower[0, 0])
    ax_drop = fig.add_subplot(lower[0, 1])

    by_panel = {p: [r for r in rows if r["panel"] == p] for p in "abc"}
    _bar_panel(ax_a, by_panel["a"], "a  T1, orientation r (PCL)", "r")
    _bar_panel(ax_b, by_panel["b"], "b  T3, median X (PCL)", "median X")
    l_simple = by_panel["c"][0]["terms"]["median_L_simple"]
    l_complex = by_panel["c"][0]["terms"]["median_L_complex"]
    _bar_panel(ax_c, by_panel["c"], "c  T4 (v5), t4_diff (PCL)", "t4_diff",
               note=f"median_L simple {l_simple:.2f}, complex {l_complex:.2f}")
    _bar_panel(ax_conv, [r for r in rows if r["test"] == "CONV"],
               f"d  CONV per seed (Vogels), <= {CONV_T_CRIT:.3f}", "|b|/se")
    _bar_panel(ax_drop, [r for r in rows if r["test"] == "O-DROP"], "O-DROP", "ratio")
    fig.supxlabel("Dashed line: criterion threshold. Passing criteria only.", fontsize=7)
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    counts = write_fig07(args.out)
    print(f"F7 criteria={counts['criteria']} passing={counts['passing']} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
