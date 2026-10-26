"""Supplement items S1 (notation table) and S5 (HDP reduction identity).

S1 parses docs/theory/doctrine/definition_map.md. A row is mapped when its symbol and
test are both named, otherwise open. S5 runs the reduction checks of
tests/test_hdp_reduction_identity.py for every core registered rule, through that
module's own helpers, and writes the numbers its assertions use.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFINITION_MAP = REPO / "docs/theory/doctrine/definition_map.md"
IDENTITY_TEST = REPO / "tests/test_hdp_reduction_identity.py"
DEFAULT_OUT = REPO / "artifacts/_temp/supplement"

FULL_WIDTH_IN = 7.2  # full text width, match-visual-style page box
ROW_HEIGHT_IN = 0.32  # one rule per row in the S5 panel
DPI = 300  # final raster, match-visual-style export default
FALSIFIER_COLOR = "#1565c0"  # series colour one, match-visual-style defaults


def parse_definition_map(path: Path) -> list[dict[str, str]]:
    """Return one dict per `| D<n> |` row with id, definition, symbol, test and status."""
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not re.match(r"^\| D\d+ \|", line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5:
            raise ValueError(f"{path}: row {cells[0]!r} has {len(cells)} cells, expected 5")
        row_id, definition, _source, symbol, test = cells
        status = "mapped" if _is_named(symbol) and _is_named(test) else "open"
        rows.append(
            {
                "id": row_id,
                "definition": definition,
                "symbol": symbol,
                "test": test,
                "status": status,
            }
        )
    if not rows:
        raise ValueError(f"{path}: no D rows found")
    return rows


def _is_named(cell: str) -> bool:
    return cell.strip("`").strip() not in ("", "none")


def write_s1(out_dir: Path) -> dict[str, int]:
    rows = parse_definition_map(DEFINITION_MAP)
    mapped = sum(row["status"] == "mapped" for row in rows)
    open_rows = len(rows) - mapped
    lines = [
        "# S1. Notation table",
        "",
        f"Source: `{DEFINITION_MAP.relative_to(REPO).as_posix()}`. Rows: {len(rows)}. "
        f"Mapped: {mapped}. Open: {open_rows}. A row is open when its symbol or test is `none`.",
        "",
        "| Id | Definition | Symbol | Test | Status |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['id']} | {r['definition']} | {r['symbol']} | {r['test']} | {r['status']} |"
        for r in rows
    ]
    (out_dir / "S1_notation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"rows": len(rows), "mapped": mapped, "open": open_rows}


def load_identity_module() -> ModuleType:
    """Load tests/test_hdp_reduction_identity.py by path and check it runs the repo jaxfne."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location("hdp_reduction_identity", IDENTITY_TEST)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {IDENTITY_TEST}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    import jaxfne

    jaxfne_file = Path(jaxfne.__file__).resolve()
    if not jaxfne_file.is_relative_to(REPO):
        raise ImportError(f"jaxfne imported from {jaxfne_file}, not from {REPO}")
    return module


def identity_rows(mod: ModuleType) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in mod._core_rule_names():
        case = mod._case(name)
        net = mod._net(case.family)
        base = net.run_base()
        mod._assert_cells_active(base, net)
        zero = net.run_rule(name, case.zero(net), None)
        masked = net.run_rule(name, case.nonzero(net), np.zeros(net.n_edges, dtype=bool))
        free = net.run_rule(name, case.nonzero(net), None)
        rows.append(
            {
                "rule": name,
                "family": case.family,
                "spike_mismatches_zero_drive": int(np.sum(zero["spikes"] != base)),
                "max_abs_dw_zero_drive": float(np.max(np.abs(zero["w"] - net.w0))),
                "max_abs_dw_all_false_mask": float(np.max(np.abs(masked["w"] - net.w0))),
                "max_abs_dw_nonzero_params": float(np.max(np.abs(free["w"] - net.w0))),
            }
        )
    return rows


def write_s5(out_dir: Path) -> list[dict[str, object]]:
    import jax

    mod = load_identity_module()
    rows = identity_rows(mod)
    payload = {
        "source": IDENTITY_TEST.relative_to(REPO).as_posix(),
        "versions": {"jax": jax.__version__, "numpy": np.__version__},
        "rules": rows,
    }
    (out_dir / "S5_reduction_identity.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# S5. HDP reduction identity per registered rule",
        "",
        f"Source: `{payload['source']}`. Identity columns must be 0; the falsifier column "
        "must be above 0. Weights are in each kernel's stored unit (AdEx: nS).",
        "",
        "| Rule | Kernel | Spike mismatches, zero drive | max abs dW, zero drive "
        "| max abs dW, all-False mask | max abs dW, nonzero (falsifier) |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['rule']} | {r['family']} | {r['spike_mismatches_zero_drive']} "
        f"| {r['max_abs_dw_zero_drive']:.3g} | {r['max_abs_dw_all_false_mask']:.3g} "
        f"| {r['max_abs_dw_nonzero_params']:.3g} |"
        for r in rows
    ]
    (out_dir / "S5_reduction_identity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_s5(rows, out_dir / "S5_reduction_identity.png")
    return rows


def plot_s5(rows: list[dict[str, object]], path: Path) -> None:
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
    n = len(rows)
    falsifier = np.array([float(r["max_abs_dw_nonzero_params"]) for r in rows])
    fig, ax = plt.subplots(
        figsize=(FULL_WIDTH_IN, ROW_HEIGHT_IN * n + 1.1), layout="constrained"
    )
    y = np.arange(n)
    # Falsifiers span about four decades, so a log axis; the identity values are exact
    # zeros, which a log axis cannot place, so they are printed as a column.
    ax.scatter(falsifier, y, s=22, color=FALSIFIER_COLOR, zorder=3)
    ax.hlines(y, falsifier.min() / 3, falsifier, color=FALSIFIER_COLOR, linewidth=0.8)
    for yi, value in zip(y, falsifier):
        ax.text(value * 1.25, yi, f"{value:.3g}", va="center", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(falsifier.min() / 3, falsifier.max() * 4)
    ax.set_yticks(y, [str(r["rule"]) for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("max |W - W0| with nonzero parameters (kernel weight unit; AdEx in nS)")
    ax.spines[["top", "right"]].set_visible(False)
    identity = [
        f"{r['spike_mismatches_zero_drive']} / {r['max_abs_dw_zero_drive']:.3g} / "
        f"{r['max_abs_dw_all_false_mask']:.3g}"
        for r in rows
    ]
    for yi, text in zip(y, identity):
        ax.annotate(text, (1.02, yi), xycoords=("axes fraction", "data"), va="center", fontsize=7)
    ax.annotate("spikes / dW zero drive /\ndW all-False mask", (1.02, -0.9),
                xycoords=("axes fraction", "data"), va="bottom", fontsize=7)
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    counts = write_s1(args.out)
    rules = write_s5(args.out)
    print(f"S1 rows={counts['rows']} mapped={counts['mapped']} open={counts['open']}")
    print(f"S5 rules={len(rules)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
