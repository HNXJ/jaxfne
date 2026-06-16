#!/usr/bin/env python3
"""Generate Extended Data ED4: optional dependency laziness panel.

Runs isolated subprocess import receipts to verify root ``import jaxfne as jtfne``
does not eagerly load optional visualization/export backends.

Outputs:
  figures/evidence/ed04_optional_dependency_laziness.png
  outputs/evidence/ed04_optional_dependency_laziness_manifest.json
  outputs/evidence/ed04_optional_dependency_laziness_receipt.json
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from _figure_common import (
    ensure_evidence_dirs,
    repo_root,
    repo_sha,
    save_figure_manifest,
    sha256_file,
    truth_gates,
    utc_now_iso,
    write_json_strict,
)


SOURCE_FILES = [
    "scripts/evidence_figures/ed04_optional_dependency_laziness.py",
    "scripts/evidence_figures/_figure_common.py",
    "tests/test_root_import_lightweight.py",
    "tests/test_config_runtime_hardening_v028.py",
    "tests/test_v0321_migration_boundaries.py",
    "jaxfne/__init__.py",
    "jaxfne/vis/core.py",
    "jaxfne/bridges.py",
]

TITLE = "Optional dependency laziness (local subprocess receipts)"

SCOPE_STATUS = (
    "Isolated subprocess receipts for import jaxfne as jtfne; "
    "root import must not eagerly load optional visualization/export backends; "
    "matplotlib may load only after explicit vis.require_matplotlib(); "
    "local environment install state does not fail receipts unless root import is eager"
)

OPTIONAL_ROOT_FORBIDDEN = (
    "matplotlib",
    "plotly",
    "pandas",
    "optax",
    "jaxley",
    "pynwb",
    "h5py",
)

_SUBPROCESS_ENV = {
    **os.environ,
    "PYTHONPATH": str(repo_root()),
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
}

_dirs = ensure_evidence_dirs()
FIGURE_PATH = _dirs["figures"] / "ed04_optional_dependency_laziness.png"
RECEIPT_JSON_PATH = _dirs["outputs"] / "ed04_optional_dependency_laziness_receipt.json"


def _run_subprocess_json(script: str) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=_SUBPROCESS_ENV,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed (exit {result.returncode}): {result.stderr[-500:]}"
        )
    stdout = result.stdout.strip().splitlines()[-1]
    return json.loads(stdout)


def run_root_import_receipt() -> dict[str, Any]:
    script = f"""
import json, sys
import jaxfne as jtfne
roots = {{name.split(".")[0] for name in sys.modules}}
forbidden = {list(OPTIONAL_ROOT_FORBIDDEN)!r}
loaded_forbidden = sorted(roots & set(forbidden))
jaxfne_modules = sorted(name for name in sys.modules if name.startswith("jaxfne."))
print(json.dumps({{
    "check_id": "root_import_lightweight",
    "module_count": len(sys.modules),
    "loaded_forbidden": loaded_forbidden,
    "root_import_lightweight": len(loaded_forbidden) == 0,
    "has_vis_module": hasattr(jtfne, "vis"),
    "vis_imported": "jaxfne.vis" in sys.modules,
    "jaxfne_module_sample": jaxfne_modules[:12],
}}))
"""
    return _run_subprocess_json(script)


def run_vis_lazy_receipt() -> dict[str, Any]:
    script = """
import json, sys
import jaxfne as jtfne
mpl_before = "matplotlib" in sys.modules
_ = jtfne.vis.require_matplotlib
mpl_after_attr = "matplotlib" in sys.modules
try:
    jtfne.vis.require_matplotlib()
    mpl_after_require = "matplotlib" in sys.modules
    matplotlib_available = True
except ImportError:
    mpl_after_require = "matplotlib" in sys.modules
    matplotlib_available = False
print(json.dumps({
    "check_id": "vis_lazy_until_require",
    "matplotlib_before_root": mpl_before,
    "matplotlib_after_attr_access": mpl_after_attr,
    "matplotlib_after_require": mpl_after_require,
    "matplotlib_available": matplotlib_available,
    "lazy_until_require": (not mpl_before) and (not mpl_after_attr),
}))
"""
    return _run_subprocess_json(script)


def run_jaxley_lazy_receipt() -> dict[str, Any]:
    script = """
import json, sys
import jaxfne as jtfne
from jaxfne.bridges import JaxleyEmitterBridge
jaxley_root = "jaxley" in sys.modules
jaxley_after_bridge_import = "jaxley" in sys.modules
_ = JaxleyEmitterBridge().to_spec()
jaxley_after_to_spec = "jaxley" in sys.modules
try:
    jtfne.require_jaxley()
    jaxley_after_require = "jaxley" in sys.modules
    jaxley_available = True
except ImportError:
    jaxley_after_require = "jaxley" in sys.modules
    jaxley_available = False
print(json.dumps({
    "check_id": "jaxley_lazy_until_require",
    "jaxley_after_root_import": jaxley_root,
    "jaxley_after_bridge_import": jaxley_after_bridge_import,
    "jaxley_after_to_spec": jaxley_after_to_spec,
    "jaxley_after_require": jaxley_after_require,
    "jaxley_available": jaxley_available,
    "lazy_until_require": (not jaxley_root) and (not jaxley_after_to_spec),
}))
"""
    return _run_subprocess_json(script)


def run_io_export_receipt() -> dict[str, Any]:
    script = """
import json, sys
import jaxfne as jtfne
pynwb_before = "pynwb" in sys.modules
payload = jtfne.json_safe({"status": "computational_scaffold", "value": 1.0})
_ = jtfne.sha256_text(json.dumps(payload, allow_nan=False))
pynwb_after = "pynwb" in sys.modules
print(json.dumps({
    "check_id": "io_export_no_pynwb",
    "pynwb_before": pynwb_before,
    "pynwb_after_json_safe": pynwb_after,
    "json_safe_keys": sorted(payload.keys()),
    "lazy_io_export": (not pynwb_before) and (not pynwb_after),
}))
"""
    return _run_subprocess_json(script)


def evaluate_fixtures(receipts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for receipt in receipts:
        check_id = receipt["check_id"]
        if check_id == "root_import_lightweight":
            passed = receipt.get("root_import_lightweight") is True
            issue = receipt.get("loaded_forbidden") or []
        elif check_id == "vis_lazy_until_require":
            passed = receipt.get("lazy_until_require") is True
            issue = []
            if receipt.get("matplotlib_before_root"):
                issue.append("matplotlib_eager_at_root")
            if receipt.get("matplotlib_after_attr_access"):
                issue.append("matplotlib_eager_on_attr_access")
        elif check_id == "jaxley_lazy_until_require":
            passed = receipt.get("lazy_until_require") is True
            issue = []
            if receipt.get("jaxley_after_root_import"):
                issue.append("jaxley_eager_at_root")
            if receipt.get("jaxley_after_to_spec"):
                issue.append("jaxley_eager_on_bridge_spec")
        elif check_id == "io_export_no_pynwb":
            passed = receipt.get("lazy_io_export") is True
            issue = []
            if receipt.get("pynwb_after_json_safe"):
                issue.append("pynwb_eager_on_json_safe")
        else:
            passed = False
            issue = ["unknown_check_id"]
        rows.append(
            {
                "check_id": check_id,
                "fixture_pass": passed,
                "issues": issue,
                "receipt": receipt,
            }
        )
    return rows


def _runtime_receipt() -> dict[str, Any]:
    try:
        import jax

        jax_version = str(jax.__version__)
    except Exception:
        jax_version = "unknown"
    try:
        import jaxlib

        jaxlib_version = str(jaxlib.__version__)
    except Exception:
        jaxlib_version = "unknown"
    return {
        "jaxfne_version": __import__("jaxfne").__version__,
        "repo_sha": repo_sha(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "jax_version": jax_version,
        "jaxlib_version": jaxlib_version,
        "subprocess_python": sys.executable,
        "optional_root_forbidden": list(OPTIONAL_ROOT_FORBIDDEN),
    }


def draw_figure(rows: list[dict[str, Any]], runtime: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(14, 9), dpi=150)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(7.0, 8.55, TITLE, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(
        7.0,
        8.05,
        "import jaxfne as jtfne",
        ha="center",
        va="center",
        fontsize=9,
        family="monospace",
        color="#333333",
    )

    table_box = FancyBboxPatch(
        (0.55, 2.85),
        12.9,
        4.75,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0,
        edgecolor="#1A4A8A",
        facecolor="#F8FAFF",
    )
    ax.add_patch(table_box)
    ax.text(0.75, 7.35, "Panel A - subprocess laziness fixtures", fontsize=9, fontweight="bold", va="top")

    headers = ["check", "pass", "detail"]
    col_x = [0.75, 4.8, 5.8]
    y = 6.95
    for x, header in zip(col_x, headers):
        ax.text(x, y, header, fontsize=7, fontweight="bold", va="top", family="monospace")

    y = 6.55
    for row in rows:
        color = "#2E7D32" if row["fixture_pass"] else "#C62828"
        receipt = row["receipt"]
        if row["check_id"] == "root_import_lightweight":
            detail = f"forbidden_loaded={receipt.get('loaded_forbidden', [])}"
        elif row["check_id"] == "vis_lazy_until_require":
            detail = (
                f"mpl_before={receipt.get('matplotlib_before_root')} "
                f"attr={receipt.get('matplotlib_after_attr_access')} "
                f"avail={receipt.get('matplotlib_available')}"
            )
        elif row["check_id"] == "jaxley_lazy_until_require":
            detail = (
                f"root={receipt.get('jaxley_after_root_import')} "
                f"spec={receipt.get('jaxley_after_to_spec')} "
                f"avail={receipt.get('jaxley_available')}"
            )
        else:
            detail = f"pynwb_after={receipt.get('pynwb_after_json_safe')}"
        ax.text(col_x[0], y, row["check_id"], fontsize=6.5, va="top", family="monospace")
        ax.text(
            col_x[1],
            y,
            str(row["fixture_pass"]),
            fontsize=6.5,
            va="top",
            family="monospace",
            color=color,
        )
        ax.text(col_x[2], y, detail[:72], fontsize=6, va="top", family="monospace")
        y -= 0.55

    receipt_box = FancyBboxPatch(
        (0.55, 0.45),
        12.9,
        2.15,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0,
        edgecolor="#8B4513",
        facecolor="#FFF8E7",
    )
    ax.add_patch(receipt_box)
    ax.text(0.75, 2.35, "Panel B - scope / subprocess receipt", fontsize=9, fontweight="bold", va="top")
    ax.text(0.85, 1.95, SCOPE_STATUS, fontsize=7, va="top", color="#333333")
    root = next(r for r in rows if r["check_id"] == "root_import_lightweight")["receipt"]
    lines = [
        f"module_count_after_root_import: {root.get('module_count')}",
        f"optional_dependency_claim_scope: local_subprocess_receipt",
        f"jaxfne: {runtime['jaxfne_version']}  repo_sha: {runtime['repo_sha'][:12]}",
        "physical_amplitude_calibrated: false",
    ]
    for i, line in enumerate(lines):
        ax.text(0.85, 1.55 - i * 0.22, line, fontsize=6.5, va="top", family="monospace", color="#444444")

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.35)
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    runtime = _runtime_receipt()
    raw_receipts = [
        run_root_import_receipt(),
        run_vis_lazy_receipt(),
        run_jaxley_lazy_receipt(),
        run_io_export_receipt(),
    ]
    rows = evaluate_fixtures(raw_receipts)

    if not all(row["fixture_pass"] for row in rows):
        failed = [row["check_id"] for row in rows if not row["fixture_pass"]]
        raise RuntimeError(f"Optional dependency laziness failures: {failed}")

    write_json_strict(
        RECEIPT_JSON_PATH,
        {
            "schema_version": "jaxfne.evidence_ed04_optional_dep_receipt.v0.1.0",
            "generated_at_utc": utc_now_iso(),
            "runtime_receipt": runtime,
            "fixtures": rows,
            "truth_gates": truth_gates(),
            "optional_dependency_claim_scope": "local_subprocess_receipt",
        },
    )

    draw_figure(rows, runtime)

    root = repo_root()
    manifest = save_figure_manifest(
        figure_id="ed04",
        title=TITLE,
        output_path=FIGURE_PATH,
        source_files=SOURCE_FILES,
        extra={
            "scope_status": SCOPE_STATUS,
            "generator_command": "python scripts/evidence_figures/ed04_optional_dependency_laziness.py",
            "claim_boundary": "local_subprocess_optional_dep_receipt",
            "fixtures": rows,
            "runtime_receipt": runtime,
            "receipt_json_path": str(RECEIPT_JSON_PATH.relative_to(root)),
            "receipt_json_sha256": sha256_file(RECEIPT_JSON_PATH),
            "truth_gates": truth_gates(),
            "physical_amplitude_calibrated": False,
            "optional_dependency_claim_scope": "local_subprocess_receipt",
        },
    )

    sha = sha256_file(FIGURE_PATH)
    print(f"wrote: {FIGURE_PATH.relative_to(root)}")
    print(f"wrote: {RECEIPT_JSON_PATH.relative_to(root)}")
    print(f"wrote: outputs/evidence/{FIGURE_PATH.stem}_manifest.json")
    print(f"sha256: {sha}")
    print(f"figure_id: {manifest['figure_id']}")
    print(f"fixtures_passed: {sum(1 for r in rows if r['fixture_pass'])}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
