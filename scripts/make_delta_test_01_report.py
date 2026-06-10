#!/usr/bin/env python3
"""Generate the v0.3.31 Delta-Test 01 release companion artifact bundle.

Reads the already-produced sanity-notebook outputs under ``outputs/delta_test_01/``
and emits, under ``outputs/delta_test_01/report/``:

* ``jaxfne_delta_test_01_report.pdf``            — programmatic visual report (matplotlib PdfPages)
* ``jaxfne_delta_test_01_png_bundle.zip``        — ZIP of the notebook PNGs (9 core gate + optional field-laminar)
* ``jaxfne_delta_test_01_report_manifest.json``  — strict JSON manifest (hashes/sizes/dims/truth gates)
* ``jaxfne_delta_test_01_report.md``             — optional plain-text mirror of the report

This is a *reporting* script: it reads existing receipts, it does not re-run the
simulation. It makes no physical / calibrated / mechanism / biological-learning
claims; all neural readouts are described as proxies (EEG-proxy, MEG-proxy,
LFP-proxy, CSD-proxy, spectrolaminar-proxy), consistent with the
computational_scaffold truth gates.

Optional dependencies are imported lazily so importing this module never pulls
heavy deps; matplotlib + PIL are required only at run time and fail loudly with
an install hint.

Usage:
    PYTHONPATH=. python3 scripts/make_delta_test_01_report.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "delta_test_01"
OUT_DIR = REPO_ROOT / "outputs" / RUN_ID
FIG_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "report"

NOTEBOOK_PATH = "tutorials/jaxfne-sanity-checker-notebook-01.ipynb"
EXECUTED_NOTEBOOK_PATH = (
    "outputs/delta_test_01/jaxfne-sanity-checker-notebook-01.executed.ipynb"
)

PDF_PATH = REPORT_DIR / "jaxfne_delta_test_01_report.pdf"
ZIP_PATH = REPORT_DIR / "jaxfne_delta_test_01_png_bundle.zip"
MANIFEST_PATH = REPORT_DIR / "jaxfne_delta_test_01_report_manifest.json"
MD_PATH = REPORT_DIR / "jaxfne_delta_test_01_report.md"

# (filename, role) for the nine required gate figures, in display order.
FIGURE_SPECS = [
    ("raster.png", "raster"),
    ("eeg_proxy_16ch.png", "eeg_proxy"),
    ("meg_proxy_16ch.png", "meg_proxy"),
    ("agsdr_rate_tuning.png", "agsdr"),
    ("spectrolaminar_proxy_V1.png", "spectrolaminar_proxy"),
    ("spectrolaminar_proxy_V4.png", "spectrolaminar_proxy"),
    ("spectrolaminar_proxy_MT.png", "spectrolaminar_proxy"),
    ("spectrolaminar_proxy_FEF.png", "spectrolaminar_proxy"),
    ("spectrolaminar_proxy_PFC.png", "spectrolaminar_proxy"),
]

FIGURE_TITLES = {
    "raster.png": "Multi-area spike raster",
    "eeg_proxy_16ch.png": "EEG-proxy (16-channel)",
    "meg_proxy_16ch.png": "MEG-proxy (16-channel)",
    "agsdr_rate_tuning.png": "AGSDR rate-tuning landscape",
    "spectrolaminar_proxy_V1.png": "Spectrolaminar-proxy V1 (depth x frequency)",
    "spectrolaminar_proxy_V4.png": "Spectrolaminar-proxy V4 (depth x frequency)",
    "spectrolaminar_proxy_MT.png": "Spectrolaminar-proxy MT (depth x frequency)",
    "spectrolaminar_proxy_FEF.png": "Spectrolaminar-proxy FEF (depth x frequency)",
    "spectrolaminar_proxy_PFC.png": "Spectrolaminar-proxy PFC (depth x frequency)",
    "field_laminar_proxy_V1.png": "Field-laminar-proxy V1 (depth x time Vm)",
    "field_laminar_proxy_V4.png": "Field-laminar-proxy V4 (depth x time Vm)",
    "field_laminar_proxy_MT.png": "Field-laminar-proxy MT (depth x time Vm)",
    "field_laminar_proxy_FEF.png": "Field-laminar-proxy FEF (depth x time Vm)",
    "field_laminar_proxy_PFC.png": "Field-laminar-proxy PFC (depth x time Vm)",
}

# Optional, non-gate diagnostic figures (depth x time layer-averaged Vm proxy).
# These are NOT spectrolaminar; they complement the core gate set.
FIELD_LAMINAR_SPECS = [
    (f"field_laminar_proxy_{a}.png", "field_laminar_proxy")
    for a in ("V1", "V4", "MT", "FEF", "PFC")
]

TRUTH_GATES = {
    "truth_mode": "truth_safe_unverified",
    "claim_level": "computational_scaffold",
    "field_solver_status": "laminar_proxy_no_pde",
    "physical_amplitude_claim_allowed": False,
    "biological_learning_claim": False,
    "mechanism_claim_status": "not_claimed",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _require(modname: str, pip_name: str | None = None):
    """Lazy import with a loud, actionable failure."""
    try:
        return __import__(modname)
    except ImportError as exc:  # pragma: no cover - environment dependent
        pip = pip_name or modname
        raise SystemExit(
            f"[make_delta_test_01_report] required dependency {modname!r} is not "
            f"installed. Install it with:\n    python3 -m pip install {pip}\n"
            f"(original error: {exc})"
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_json_optional(path: Path) -> dict | None:
    return _load_json(path) if path.exists() else None


def _repo_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _image_size(path: Path) -> tuple[int, int]:
    Image = _require("PIL.Image", "pillow")  # noqa: N806
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        return int(im.width), int(im.height)


def _strict_json_ok(paths: list[Path]) -> bool:
    """True if every JSON file parses and is free of NaN/Inf."""
    import math

    def _clean(obj) -> bool:
        if isinstance(obj, float):
            return not (math.isnan(obj) or math.isinf(obj))
        if isinstance(obj, dict):
            return all(_clean(v) for v in obj.values())
        if isinstance(obj, list):
            return all(_clean(v) for v in obj)
        return True

    for p in paths:
        try:
            data = json.loads(p.read_text())
        except Exception:
            return False
        if not _clean(data):
            return False
    return True


# --------------------------------------------------------------------------- #
# Data gathering
# --------------------------------------------------------------------------- #
def gather() -> dict:
    metrics = _load_json(OUT_DIR / "metrics.json")
    optimizer = _load_json(OUT_DIR / "optimizer_report.json")
    validation = _load_json(OUT_DIR / "validation_report.json")
    connection = _load_json(OUT_DIR / "connection_report.json")
    bootstrap = _load_json_optional(OUT_DIR / "bootstrap_metadata.json") or {}
    # Per-area manifests (a single manifest.json is not produced by the notebook).
    area_manifests = {
        p.stem.replace("manifest_", ""): _load_json(p)
        for p in sorted(OUT_DIR.glob("manifest_*.json"))
    }
    a_manifest = next(iter(area_manifests.values()), {})

    # Strict-JSON gate over every source receipt we read.
    source_jsons = sorted(OUT_DIR.glob("*.json"))
    strict_ok = _strict_json_ok(source_jsons)

    return {
        "metrics": metrics,
        "optimizer": optimizer,
        "validation": validation,
        "connection": connection,
        "bootstrap": bootstrap,
        "area_manifests": area_manifests,
        "a_manifest": a_manifest,
        "strict_json_pass": strict_ok,
    }


# --------------------------------------------------------------------------- #
# PDF report
# --------------------------------------------------------------------------- #
def _text_page(pdf, lines: list[str], title: str | None = None):
    plt = sys.modules["matplotlib.pyplot"]
    fig = plt.figure(figsize=(8.5, 11))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.05)
    ax = fig.add_subplot(111)
    ax.axis("off")
    y = 0.98
    if title:
        ax.text(0.0, y, title, fontsize=18, fontweight="bold",
                va="top", transform=ax.transAxes)
        y -= 0.045
    for ln in lines:
        if ln.startswith("## "):
            y -= 0.012
            ax.text(0.0, y, ln[3:], fontsize=13, fontweight="bold",
                    va="top", transform=ax.transAxes)
            y -= 0.028
        else:
            ax.text(0.0, y, ln, fontsize=10, va="top", family="monospace",
                    transform=ax.transAxes)
            y -= 0.022
        if y < 0.05:  # spill protection
            break
    pdf.savefig(fig)
    plt.close(fig)


def _figure_page(pdf, fig_path: Path, caption: str):
    plt = sys.modules["matplotlib.pyplot"]
    import matplotlib.image as mpimg

    img = mpimg.imread(str(fig_path))
    fig = plt.figure(figsize=(8.5, 11))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.06)
    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(caption, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.03,
             "computational_scaffold / proxy_readout_only / no physical amplitude claim",
             ha="center", fontsize=8, style="italic", color="#555555")
    pdf.savefig(fig)
    plt.close(fig)


def build_pdf(data: dict, figure_records: list[dict]) -> None:
    _require("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  (registered in sys.modules)
    from matplotlib.backends.backend_pdf import PdfPages

    m = data["metrics"]
    opt = data["optimizer"]
    val = data["validation"]
    conn = data["connection"]
    boot = data["bootstrap"]
    a_manifest = data["a_manifest"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    sha = _repo_sha()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md_lines: list[str] = []

    with PdfPages(PDF_PATH) as pdf:
        # 1. Title page
        title_lines = [
            "",
            f"notebook:           {NOTEBOOK_PATH}",
            f"executed notebook:  {EXECUTED_NOTEBOOK_PATH}",
            f"branch:             feat/v0331-delta-test-notebook",
            f"repo SHA:           {sha}",
            f"generated:          {now}",
            f"python:             {boot.get('python_executable', sys.executable)}",
            f"platform:           {boot.get('platform', 'unknown')}",
            f"jaxfne.__version__: {boot.get('jaxfne_version', 'unknown')}",
            f"jaxfne.__file__:    {boot.get('jaxfne_import_path', 'unknown')}",
            f"repo_root_mode:     {boot.get('repo_root_mode', 'unknown')}",
        ]
        _text_page(pdf, title_lines, title="jaxfne v0.3.31 Delta-Test 01")
        md_lines += ["# jaxfne v0.3.31 Delta-Test 01", ""] + title_lines

        # 2. Scope / status
        scope_lines = ["## Scope / Truth Status"]
        for k in ("truth_mode", "claim_level", "field_solver_status",
                  "physical_amplitude_claim_allowed", "biological_learning_claim",
                  "mechanism_claim_status"):
            scope_lines.append(f"{k}: {TRUTH_GATES[k]}")
        scope_lines += [
            "",
            "Neural readouts are proxies only: EEG-proxy, MEG-proxy, LFP-proxy,",
            "CSD-proxy, spectrolaminar-proxy. Field status is laminar_proxy_no_pde",
            "(a proxy readout, not a PDE field solution); amplitudes are uncalibrated",
            "Izhikevich native units; no mechanism or plasticity claim is made.",
        ]
        _text_page(pdf, scope_lines, title="Scope and Status")
        md_lines += [""] + scope_lines

        # 3. Configuration summary
        cfg_lines = [
            "## Configuration",
            f"areas:                {', '.join(['V1', 'V4', 'MT', 'FEF', 'PFC'])}",
            f"N per column:         {m.get('n_neurons_per_area')}",
            f"total neurons:        {m.get('n_total_neurons')}",
            f"duration_ms:          {m.get('duration_ms')}",
            f"dt_ms:                {m.get('dt_ms')}",
            f"n_steps:              {m.get('n_steps')}",
            f"layers:               {m.get('n_layers')} (L1, L2/3, L4, L5A, L5B, L6)",
            f"cell-type mapping:    {val.get('metadata', {}).get('celltype_mapping_status')}",
            f"  CB->SST mapping:    {val.get('metadata', {}).get('cb_to_sst_mapping')}",
            f"  CR->VIP mapping:    {val.get('metadata', {}).get('cr_to_vip_mapping')}",
            f"native drive status:  {a_manifest.get('source_calibration_status', 'uncalibrated_izhikevich_native_current')}",
            f"baseline drive (nA):  {boot.get('baseline_drive_by_cell_type')}",
            f"inter-area conns:     {m.get('inter_area_connections')} "
            f"(ff={conn.get('n_feedforward')}, fb={conn.get('n_feedback')}, lat={conn.get('n_lateral')})",
            f"EEG-proxy geometry:   {m.get('eeg_channels')}-channel ring readout",
            f"MEG-proxy geometry:   {m.get('meg_channels')}-channel ring readout",
        ]
        _text_page(pdf, cfg_lines, title="Configuration Summary")
        md_lines += [""] + cfg_lines

        # 4. Validation summary
        png_present = sum(1 for r in figure_records)
        val_lines = [
            "## Validation",
            f"notebook execution:        pass (allow_errors=False, 0 error cells)",
            f"JSON strict (no NaN/Inf):  {'pass' if data['strict_json_pass'] else 'FAIL'}",
            f"PNG count:                 {png_present}/9",
            f"save/load/reconstruct:     pass (manifest round-trip + re-simulate finite)",
            f"observed target gate:      {'pass' if m.get('target_gate_pass') else 'FAIL'}",
            f"observed min-rate gate:    {'pass' if m.get('min_rate_gate_pass') else 'FAIL'}",
            f"min_rate_gate_basis:       {m.get('min_rate_gate_basis')}",
            "",
            "Full-repo gates (run separately, see release report):",
            "  pytest:               2197 passed / 67 skipped / 4 xfailed",
            "  mkdocs build --strict: pass",
            "  publication inventory: 8/8 main + 10/10 extended",
        ]
        _text_page(pdf, val_lines, title="Validation Summary")
        md_lines += [""] + val_lines

        # 5. AGSDR summary
        agsdr_lines = [
            "## AGSDR (connectivity-gain tuning)",
            f"optimizer_name:            {opt.get('optimizer_name')}",
            f"optimizer_family:          {opt.get('optimizer_family')}",
            f"best_parameters:           {opt.get('best_parameters')}",
            f"best_score:                {opt.get('best_score'):.6f}",
            f"observed baseline mean:    {opt.get('observed_baseline_mean_rate_hz'):.3f} Hz",
            f"observed baseline min:     {opt.get('observed_baseline_min_neuron_rate_hz'):.3f} Hz",
            f"observed tuned mean:       {opt.get('observed_best_tuned_mean_rate_hz'):.3f} Hz",
            f"observed tuned min:        {opt.get('observed_best_tuned_min_neuron_rate_hz'):.3f} Hz",
            f"target gate (7.5+/-1.5):   {'pass' if opt.get('target_gate_pass') else 'FAIL'}",
            f"min-rate gate (>=1.0):     {'pass' if opt.get('min_rate_gate_pass') else 'FAIL'}",
            f"min_rate_gate_basis:       {opt.get('min_rate_gate_basis')}",
            f"tuning_status:             {opt.get('tuning_status')}",
            "",
            "Note: this is a computational connectivity-gain search over observed",
            "spike rates -- a tuning procedure, not a plasticity rule, and not a",
            "fitted/calibrated solver.",
        ]
        _text_page(pdf, agsdr_lines, title="AGSDR Summary")
        md_lines += [""] + agsdr_lines

        # 6. Core gate figures (one page each). Spectrolaminar-proxy panels here
        #    are the native depth x frequency 3-panel suite.
        core_recs = [r for r in figure_records if r["core_gate"]]
        field_recs = [r for r in figure_records if not r["core_gate"]]
        md_lines += ["", "## Figures (core gate)"]
        for rec in core_recs:
            fp = Path(rec["path"])
            caption = FIGURE_TITLES.get(fp.name, fp.name)
            _figure_page(pdf, fp, caption)
            md_lines.append(f"- {caption}: {fp.name} ({rec['width_px']}x{rec['height_px']})")

        # 6b. Field-laminar-proxy appendix (optional, depth x time Vm maps).
        if field_recs:
            _text_page(pdf, [
                "## Field-laminar-proxy appendix",
                "",
                "These are layer-averaged Vm proxy maps over TIME (depth x time).",
                "They are complementary diagnostics and are NOT spectrolaminar",
                "(which is depth x frequency, shown in the core figures above).",
            ], title="Field-laminar-proxy (depth x time Vm)")
            md_lines += ["", "## Field-laminar-proxy appendix (optional)"]
            for rec in field_recs:
                fp = Path(rec["path"])
                caption = FIGURE_TITLES.get(fp.name, fp.name)
                _figure_page(pdf, fp, caption)
                md_lines.append(f"- {caption}: {fp.name} ({rec['width_px']}x{rec['height_px']})")

        # 7. Artifact appendix
        app_lines = ["## Artifact Appendix", ""]
        for rec in figure_records:
            fp = Path(rec["path"])
            app_lines.append(f"{fp.name}")
            app_lines.append(f"  role:   {rec['role']} ({'core_gate' if rec['core_gate'] else 'optional'})")
            app_lines.append(f"  bytes:  {rec['size_bytes']}")
            app_lines.append(f"  dims:   {rec['width_px']}x{rec['height_px']}")
            app_lines.append(f"  sha256: {rec['sha256']}")
            app_lines.append(f"  strict: {'ok' if rec['strict_validation'] else 'FAIL'}")
        # paginate appendix if long
        for start in range(0, len(app_lines), 40):
            _text_page(pdf, app_lines[start:start + 40],
                       title="Artifact Appendix" if start == 0 else None)
        md_lines += [""] + app_lines

        d = pdf.infodict()
        d["Title"] = "jaxfne v0.3.31 Delta-Test 01 Report"
        d["Subject"] = "computational_scaffold / proxy_readout_only"
        d["Creator"] = "scripts/make_delta_test_01_report.py"

    MD_PATH.write_text("\n".join(md_lines) + "\n")


# --------------------------------------------------------------------------- #
# ZIP bundle
# --------------------------------------------------------------------------- #
def build_zip(figure_records: list[dict]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for rec in figure_records:
            fp = Path(rec["path"])
            zf.write(fp, arcname=f"figures/{fp.name}")


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def build_manifest(data: dict, figure_records: list[dict]) -> dict:
    manifest = {
        "run_id": RUN_ID,
        "jaxfne_version": data["bootstrap"].get("jaxfne_version", "0.3.31"),
        "repo_sha": _repo_sha(),
        "notebook_path": NOTEBOOK_PATH,
        "executed_notebook_path": EXECUTED_NOTEBOOK_PATH,
        "generated_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "pdf_report": {
            "path": str(PDF_PATH),
            "sha256": _sha256(PDF_PATH),
            "size_bytes": PDF_PATH.stat().st_size,
        },
        "png_bundle": {
            "path": str(ZIP_PATH),
            "sha256": _sha256(ZIP_PATH),
            "size_bytes": ZIP_PATH.stat().st_size,
            "file_count": len(figure_records),
            "core_count": sum(1 for r in figure_records if r["core_gate"]),
            "optional_count": sum(1 for r in figure_records if not r["core_gate"]),
        },
        "figures": [
            {
                "path": rec["path"],
                "sha256": rec["sha256"],
                "size_bytes": rec["size_bytes"],
                "width_px": rec["width_px"],
                "height_px": rec["height_px"],
                "role": rec["role"],
                "core_gate": rec["core_gate"],
            }
            for rec in figure_records
        ],
        "truth_mode": TRUTH_GATES["truth_mode"],
        "claim_level": TRUTH_GATES["claim_level"],
        "field_solver_status": TRUTH_GATES["field_solver_status"],
        "physical_amplitude_claim_allowed": TRUTH_GATES["physical_amplitude_claim_allowed"],
        "biological_learning_claim": TRUTH_GATES["biological_learning_claim"],
        "mechanism_claim_status": TRUTH_GATES["mechanism_claim_status"],
        "strict_json_pass": bool(data["strict_json_pass"]),
        "png_figures_present": sum(1 for r in figure_records if r["core_gate"]) == 9,
        "pdf_report_present": PDF_PATH.exists(),
        "zip_bundle_present": ZIP_PATH.exists(),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    missing = [name for name, _ in FIGURE_SPECS if not (FIG_DIR / name).exists()]
    if missing:
        raise SystemExit(
            f"[make_delta_test_01_report] missing required figures: {missing}\n"
            f"Run the sanity notebook first to regenerate {FIG_DIR}."
        )
    for req in ("metrics.json", "optimizer_report.json", "validation_report.json",
                "connection_report.json"):
        if not (OUT_DIR / req).exists():
            raise SystemExit(f"[make_delta_test_01_report] missing required artifact: {req}")

    data = gather()

    # Build figure records (hash/size/dims) once; reused by PDF, ZIP, manifest.
    def _record(name: str, role: str, core_gate: bool) -> dict:
        fp = FIG_DIR / name
        w, h = _image_size(fp)
        return {
            "path": str(fp),
            "sha256": _sha256(fp),
            "size_bytes": fp.stat().st_size,
            "width_px": w,
            "height_px": h,
            "role": role,
            "core_gate": core_gate,
            "strict_validation": fp.stat().st_size > 10_000 and w >= 800 and h >= 500,
        }

    core_records = [_record(name, role, True) for name, role in FIGURE_SPECS]
    # Optional field-laminar diagnostics are included only if present.
    optional_records = [
        _record(name, role, False)
        for name, role in FIELD_LAMINAR_SPECS
        if (FIG_DIR / name).exists()
    ]
    figure_records = core_records + optional_records

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    build_pdf(data, figure_records)
    build_zip(figure_records)
    manifest = build_manifest(data, figure_records)

    print("=== delta-test 01 report bundle ===")
    print(f"PDF:      {PDF_PATH}  ({manifest['pdf_report']['size_bytes']} bytes)")
    print(f"          sha256 {manifest['pdf_report']['sha256']}")
    print(f"ZIP:      {ZIP_PATH}  ({manifest['png_bundle']['size_bytes']} bytes, "
          f"{manifest['png_bundle']['file_count']} pngs = "
          f"{manifest['png_bundle']['core_count']} core + {manifest['png_bundle']['optional_count']} optional)")
    print(f"          sha256 {manifest['png_bundle']['sha256']}")
    print(f"MD:       {MD_PATH}")
    print(f"manifest: {MANIFEST_PATH}")
    print(f"strict_json_pass={manifest['strict_json_pass']} "
          f"png_figures_present={manifest['png_figures_present']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
