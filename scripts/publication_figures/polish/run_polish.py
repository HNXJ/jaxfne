#!/usr/bin/env python3
"""Run artist-only polish on Figures 1-7, emit per-figure + cross-figure audits.

Phase B, downstream of the frozen 0.4.17 scientific set. Polish is strictly
artist-only: typography, annotation/inset geometry, export. Data arrays, words,
axes limits/scales, image normalization/extent and semantic palettes are
verified unchanged by machine checks. All writes go through guarded_path.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import to_hex
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "publication_figures"))

from _polish_common import (  # noqa: E402
    FROZEN,
    all_text_artists,
    clip_check,
    enforce_font_floor,
    export_final,
    guarded_path,
    min_font,
    semantic_palette,
)
import _pub_figure_common as pfc  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_default(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not serializable: {type(o)}")


def write_json_strict(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")


# --------------------------------------------------------------- data loading
def load_fig02_04():
    from _experiment_a_frozen import load_experiment_a_bundle
    from fig02_04_experiment_a import build_figure2, build_figure3, build_figure4

    return load_experiment_a_bundle(), build_figure2, build_figure3, build_figure4


def load_fig06():
    import fig06_hwd_evidence as m
    from jaxfne.publication.fig06_evidence import (
        d3_classification,
        h3_memory_curves_beta_comparison,
        h4_primary_mx,
        load_fig06_evidence,
        w3b_counts,
    )

    ev = load_fig06_evidence()
    return m, dict(
        ev=ev,
        mx=h4_primary_mx(ev),
        h3=h3_memory_curves_beta_comparison(ev),
        counts=w3b_counts(ev),
        d3=d3_classification(ev),
    )


def load_fig07():
    import fig07_e_integration as m
    from jaxfne.publication.fig07_evidence import (
        e1_hierarchy_summary,
        e2_delay_classes,
        e3_owner,
        e4_observation_semantics,
        e5_arm_definitions,
        e5_null_controls,
        e5_propagation_metrics,
        load_fig07_evidence,
    )

    ev = load_fig07_evidence()
    return m, dict(
        h1=e1_hierarchy_summary(ev),
        delays=e2_delay_classes(ev),
        owner=e3_owner(ev),
        obs=e4_observation_semantics(ev),
        nulls=e5_null_controls(ev),
        arms=e5_arm_definitions(ev),
        prop=e5_propagation_metrics(ev),
    )


# --------------------------------------------------------------- snapshots
def _collect_colors(fig):
    colors = set()
    for ax in fig.axes:
        for line in ax.lines:
            c = line.get_color()
            if c:
                colors.add(to_hex(c, keep_alpha=False))
        for coll in ax.collections:
            fc = coll.get_facecolor()
            if isinstance(fc, np.ndarray) and fc.size:
                colors.update(to_hex(row, keep_alpha=False) for row in np.asarray(fc).reshape(-1, 4))
            ec = coll.get_edgecolor()
            if isinstance(ec, np.ndarray) and ec.size:
                colors.update(to_hex(row, keep_alpha=False) for row in np.asarray(ec).reshape(-1, 4))
        for t in ax.texts:
            c = t.get_color()
            if c:
                colors.add(to_hex(c, keep_alpha=False))
    for t in fig.texts:
        c = t.get_color()
        if c:
            colors.add(to_hex(c, keep_alpha=False))
    return sorted(colors)


def _discrete_colors(fig):
    """Colors of semantically-fixed artists only; colormap-driven (continuous)
    artists are data-backed and excluded from the finite semantic budget."""
    colors = set()
    for ax in fig.axes:
        for line in ax.lines:
            c = line.get_color()
            if c:
                colors.add(to_hex(c, keep_alpha=False))
        for coll in ax.collections:
            cmap = getattr(coll, "get_cmap", lambda: None)()
            if cmap is None:
                fc = coll.get_facecolor()
                if isinstance(fc, np.ndarray) and fc.size:
                    colors.update(to_hex(row, keep_alpha=False) for row in np.asarray(fc).reshape(-1, 4))
        for t in ax.texts:
            c = t.get_color()
            if c:
                colors.add(to_hex(c, keep_alpha=False))
    for t in fig.texts:
        c = t.get_color()
        if c:
            colors.add(to_hex(c, keep_alpha=False))
    return colors


def _data_hash(fig):
    parts = []
    for ax in fig.axes:
        for line in ax.lines:
            parts.append(bytes(line.get_xdata().tobytes()))
            parts.append(bytes(line.get_ydata().tobytes()))
        for coll in ax.collections:
            off = coll.get_offsets()
            if off is not None and len(off):
                parts.append(bytes(np.asarray(off).tobytes()))
        for im in ax.images:
            arr = im.get_array()
            if arr is not None:
                parts.append(bytes(np.asarray(arr).tobytes()))
    return hashlib.sha256(b"|".join(parts)).hexdigest()


def _frame_state(fig):
    return {
        i: {
            "xlim": list(ax.get_xlim()),
            "ylim": list(ax.get_ylim()),
            "xscale": ax.get_xscale(),
            "yscale": ax.get_yscale(),
        }
        for i, ax in enumerate(fig.axes)
    }


def _image_state(fig):
    out = []
    for ax in fig.axes:
        for im in ax.images:
            norm = im.norm
            out.append(
                {
                    "extent": list(im.get_extent()),
                    "clim": list(im.get_clim()),
                    "cmap": im.get_cmap().name,
                    "norm_type": type(norm).__name__,
                    "norm_params": getattr(norm, "__dict__", {}).copy(),
                    "array_hash": hashlib.sha256(np.asarray(im.get_array()).tobytes()).hexdigest(),
                }
            )
    return out


def _text_strings(fig):
    return sorted(t.get_text() for t in all_text_artists(fig))


def _block_rects(fig):
    rects = []
    for ax in fig.axes:
        for p in ax.patches:
            if isinstance(p, FancyBboxPatch):
                rects.append(("bbox", p.get_x(), p.get_y(), p.get_width(), p.get_height()))
            elif isinstance(p, Rectangle):
                rects.append(("rect", p.get_x(), p.get_y(), p.get_width(), p.get_height()))
            elif isinstance(p, FancyArrowPatch):
                a, b = p._posA_posB
                rects.append(("arrow", a[0], a[1], b[0], b[1]))
    return rects


def snapshot(fig):
    out = {
        "frames": _frame_state(fig),
        "data_hash": _data_hash(fig),
        "colors": _collect_colors(fig),
        "images": _image_state(fig),
        "text_strings": _text_strings(fig),
        "blocks": _block_rects(fig),
        "clip_protruding_before": clip_check(fig)["protruding"],
    }
    return out


def log_axis_snapshot(fig):
    for ax in fig.axes:
        if ax.get_xscale() == "log":
            return {
                "ax_index": fig.axes.index(ax),
                "xlim": list(ax.get_xlim()),
                "ylim": list(ax.get_ylim()),
                "xscale": ax.get_xscale(),
            }
    return None


def find_text(fig, substr, *, data=False):
    for t in all_text_artists(fig):
        if substr in t.get_text():
            return t
    return None


def fig_fraction_rect(t):
    bb = t.get_window_extent(renderer=None)
    if bb.width <= 0 or bb.height <= 0:
        return None
    inv = t.figure.transFigure.inverted()
    x0, y0 = inv.transform((bb.x0, bb.y0))
    x1, y1 = inv.transform((bb.x1, bb.y1))
    return [round(float(x0), 4), round(float(y0), 4), round(float(x1), 4), round(float(y1), 4)]


def axis_fig_fraction(fig, idx):
    bb = fig.axes[idx].get_position()
    return [round(bb.x0, 4), round(bb.y0, 4), round(bb.x1, 4), round(bb.y1, 4)]


def rects_intersect(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


# --------------------------------------------------------------- polish transforms
def polish_fig01(fig):
    rec = {}
    # uniform title convention: suptitle ~= 12 pt, subtitle ~= 9 pt
    for t in fig.texts:
        txt = t.get_text()
        if txt.startswith("Figure 1"):
            rec["title_was"] = t.get_fontsize()
            t.set_fontsize(12.0)
            rec["title_now"] = 12.0
        elif "Biology" in txt and "State" in txt:
            rec["subtitle_was"] = t.get_fontsize()
            t.set_fontsize(9.0)
            rec["subtitle_now"] = 9.0
    # geometry intentionally left untouched (no measured cross-box overlap)
    rec["geometry_translation"] = "none (probe showed no cross-box overlap)"
    return rec


def polish_fig02(fig):
    rec = {}
    t = find_text(fig, "neq V_m")
    if t is None:
        rec["issues"] = ["annotation not found"]
        return rec
    rec["old_fig_frac"] = fig_fraction_rect(t)
    # genuine data-free region: top band between subtitle and top spike panel
    t.set_position((0.72, 0.905))
    rec["new_center"] = [0.72, 0.905]
    return rec


def polish_fig03(fig):
    rec = {}
    # colorbar readability only; data/norm/extent/cmap untouched by design.
    for ax in fig.axes:
        cb = getattr(ax, "_colorbar", None)
        if cb is not None:
            rec.setdefault("colorbars", []).append(ax.get_ylabel())
    return rec


def polish_fig04(fig):
    rec = {"analysis_only_boxes": 2}
    return rec


def polish_fig05(fig):
    rec = {}
    t = find_text(fig, "recovered:")
    if t is None:
        rec["issues"] = ["recovered box not found"]
        return rec
    rec["old_fig_frac"] = fig_fraction_rect(t)
    # dedicated margin: inter-panel gutter between A and B (no any heatmap there)
    t.set_transform(fig.transFigure)
    t.set_position((0.505, 0.755))
    t.set_ha("center")
    t.set_va("center")
    rec["new_center"] = [0.505, 0.755]
    return rec


def all_axes(fig):
    def walk(ax):
        yield ax
        for c in getattr(ax, "child_axes", []) or []:
            yield from walk(c)

    for ax in fig.axes:
        yield from walk(ax)


def find_inset(fig, xlabel):
    for ax in all_axes(fig):
        if ax.get_xlabel() == xlabel:
            return ax
    return None


def find_host(fig, inset):
    for h in fig.axes:
        if inset in (getattr(h, "child_axes", None) or []):
            return h
    return None


def polish_fig06(fig):
    rec = {}
    inset = find_inset(fig, r"$\omega$")
    if inset is None:
        rec["issues"] = ["omega inset not found"]
        return rec
    host = find_host(fig, inset)
    if host is None:
        rec["issues"] = ["host axes not found"]
        return rec
    fig.canvas.draw()
    pos = inset.get_position()
    hp = host.get_position()
    Hw, Hh = hp.x1 - hp.x0, hp.y1 - hp.y0
    # current inset rect in HOST-AXES fraction
    old_ax = [
        (pos.x0 - hp.x0) / Hw,
        (pos.y0 - hp.y0) / Hh,
        pos.width / Hw,
        pos.height / Hh,
    ]
    rec["old_ax_frac"] = [round(float(v), 4) for v in old_ax]
    # Enlarge and move the inset into the one genuinely free region of panel E
    # (right column, host-frac x >= 0.64): the original inset already occluded
    # the "dot omega = 0 throughout W2" formula text, and the formula strings sit
    # left of fig-frac x ~0.31, so x0=0.64 keeps clear of them. Growing to the
    # right/down enlarges the omega sweep without covering anything.
    new_ax = [0.64, 0.06, 0.34, 0.88]
    rec["new_ax_frac"] = [round(float(v), 4) for v in new_ax]
    new_fig = [
        hp.x0 + new_ax[0] * Hw,
        hp.y0 + new_ax[1] * Hh,
        new_ax[2] * Hw,
        new_ax[3] * Hh,
    ]
    inset.set_axes_locator(None)
    inset.set_position(new_fig)
    fig.canvas.draw()
    p2 = inset.get_position()
    rec["new_fig_frac"] = [round(float(p2.x0), 4), round(float(p2.y0), 4), round(float(p2.width), 4), round(float(p2.height), 4)]
    rec["area_ratio"] = round(float((new_ax[2] * new_ax[3]) / max(old_ax[2] * old_ax[3], 1e-12)), 3)
    return rec


def polish_fig07(fig):
    rec = {}
    la = log_axis_snapshot(fig)
    rec["log_axis_before"] = la
    return rec


# --------------------------------------------------------------- targeted audits
def audit_fig01(fig, before, rec):
    checks = {}
    title = next((t for t in fig.texts if t.get_text().startswith("Figure 1")), None)
    checks["suptitle_12pt"] = title is not None and abs(title.get_fontsize() - 12.0) < 1e-6
    checks["blocks_geometry_preserved"] = _block_rects(fig) == before["blocks"]
    checks["images_unchanged"] = _image_state(fig) == before["images"]
    return checks


def audit_fig02(fig, before, rec):
    checks = {}
    t = find_text(fig, "neq V_m")
    if t is None:
        checks["annotation_on_data_free_region"] = False
        return checks
    r = fig_fraction_rect(t)
    axes_fr = [axis_fig_fraction(fig, i) for i in range(len(fig.axes))]
    ok = r is not None and not any(rects_intersect(r, a) for a in axes_fr)
    subtitle = next((x for x in fig.texts if "rightarrow Q" in x.get_text()), None)
    if subtitle is not None and r is not None:
        sr = fig_fraction_rect(subtitle)
        if sr is not None and rects_intersect(r, sr):
            ok = False
    checks["annotation_on_data_free_region"] = bool(ok)
    checks["annotation_in_canvas"] = r is not None and 0.02 <= r[0] and r[2] <= 0.98 and 0.02 <= r[1] and r[3] <= 0.98
    return checks


def audit_fig03(fig, before, rec):
    return {"normalization_extent_data_unchanged": _image_state(fig) == before["images"]}


def audit_fig04(fig, before, rec):
    strings = "\n".join(_text_strings(fig))
    checks = {
        "analysis_only_visible": strings.count("ANALYSIS_ONLY") >= 2,
        "images_unchanged": _image_state(fig) == before["images"],
    }
    return checks


def audit_fig05(fig, before, rec):
    checks = {}
    t = find_text(fig, "recovered:")
    r = fig_fraction_rect(t) if t is not None else None
    axes_fr = [axis_fig_fraction(fig, i) for i in range(len(fig.axes))]
    checks["diagnostic_outside_heatmap"] = r is not None and not any(rects_intersect(r, a) for a in axes_fr)
    checks["diagnostic_in_canvas"] = r is not None and 0.02 <= r[0] and r[2] <= 0.98 and 0.02 <= r[1] and r[3] <= 0.98
    # not overlapping the causal banner (xlim [0, 10] top strip) either
    banner = next((ax for ax in fig.axes if ax.figure is fig and list(ax.get_xlim()) == [0, 10]), None)
    if r is not None and banner is not None:
        br = [round(float(v), 4) for v in banner.get_position().bounds]
        br = [br[0], br[1], br[0] + br[2], br[1] + br[3]]
        checks["diagnostic_clear_of_banner"] = not rects_intersect(r, br)
    else:
        checks["diagnostic_clear_of_banner"] = r is None
    return checks


def audit_fig06(fig, before, rec):
    checks = {}
    inset = find_inset(fig, r"$\omega$")
    if inset is None:
        checks.update({"inset_enlarged": False, "inset_no_occlusion": False})
        return checks
    p = rec.get("polish", {})
    checks["inset_enlarged"] = p.get("area_ratio", 1.0) > 1.02
    ip = inset.get_position()
    ir = [float(ip.x0), float(ip.y0), float(ip.x0 + ip.width), float(ip.y0 + ip.height)]
    host = find_host(fig, inset)
    occlusion = []
    if host is not None:
        for t in host.texts:
            tr = fig_fraction_rect(t)
            if tr is not None and rects_intersect(tr, ir):
                occlusion.append(t.get_text()[:30])
        if occlusion:
            checks["inset_fig_rect"] = [round(v, 4) for v in ir]
            checks["occluded_position"] = [
                [round(v, 4) for v in fig_fraction_rect(t)] for t in host.texts
                if fig_fraction_rect(t) is not None and rects_intersect(fig_fraction_rect(t), ir)
            ]
    checks["inset_no_occlusion"] = len(occlusion) == 0
    checks["occluded_artists"] = occlusion
    host_frame = before["frames"].get(fig.axes.index(host)) if host in fig.axes else None
    now_frame = _frame_state(fig).get(fig.axes.index(host)) if host in fig.axes else None
    checks["host_frame_unchanged"] = now_frame == host_frame
    return checks


def audit_fig07(fig, before, rec):
    checks = {}
    la = log_axis_snapshot(fig)
    checks["log_axis_and_limits_unchanged"] = la == rec.get("polish", {}).get("log_axis_before")
    return checks


# --------------------------------------------------------------- generic audit
def generic_checks(fig, before, _rec=None):
    cc = clip_check(fig)
    checks = {
        "font_floor_6_5": round(min_font(fig), 2) >= 6.5 - 1e-9,
        "min_font_pt": round(min_font(fig), 2),
        "palette_unchanged": _collect_colors(fig) == before["colors"],
        "palette_within_semantic": _discrete_colors(fig).issubset({c.lower() for c in semantic_palette()}),
        "data_arrays_unchanged": _data_hash(fig) == before["data_hash"],
        "text_strings_unchanged": _text_strings(fig) == before["text_strings"],
        "axes_limits_scales_unchanged": _frame_state(fig) == before["frames"],
        "clip_no_new_protrusion": cc["protruding"].issubset(before["clip_protruding_before"]),
    }
    return checks, cc


# --------------------------------------------------------------- drivers
def run_figure(fig_num, fig, polish_fn, audit_fn, basename, frozen_upstream):
    fig.canvas.draw()
    before = snapshot(fig)
    p_rec = polish_fn(fig)
    enforce_font_floor(fig)
    # typography budget: axis labels (body*) raised to >= 7 pt where smaller
    for ax in fig.axes:
        for lbl in (ax.xaxis.label, ax.yaxis.label):
            if lbl.get_fontsize() and lbl.get_fontsize() < 7:
                lbl.set_fontsize(7)
    fig.canvas.draw()
    rec = {"figure": fig_num, "polish": p_rec, "checkpoint": f"figure_{fig_num}_polish"}
    rec["targeted"] = audit_fn(fig, before, rec)
    rec["generic"], rec["clip_detail"] = generic_checks(fig, before)
    rec["status"] = "PASSED" if (all(v is True for v in _flatten_bool(rec["targeted"]))
                                 and all(v is True for v in _flatten_bool(rec["generic"]))) else "FAILED"
    rec["export"] = export_final(fig, basename)
    rec["frozen_upstream_sha256"] = frozen_upstream
    rec["audited_at_utc"] = utc_now()
    rec["repo_head"] = pfc.repo_sha()
    return rec


def _flatten_bool(d):
    for v in d.values():
        if isinstance(v, dict):
            yield from _flatten_bool(v)
        elif isinstance(v, (bool, np.bool_)):
            yield bool(v)


def write_figure_artifacts(rec, plot_key):
    n = str(rec["figure"])
    spec = {
        "figure": plot_key,
        "schema": f"jaxfne.publication.fig{n}_polish_spec.v1",
        "status": "FROZEN",
        "artist_only_polish": True,
        "downstream_of_frozen_sha": rec["frozen_upstream_sha256"],
        "checkpoint": rec["checkpoint"],
        "repo_head": rec["repo_head"],
    }
    write_json_strict(guarded_path(f"artifacts/publication/polish/fig{n}_polish_spec.json"), spec)
    write_json_strict(guarded_path(f"artifacts/publication/polish/fig{n}_polish_audit.json"), rec)
    write_json_strict(guarded_path(f"artifacts/publication/polish/fig{n}_polish_receipt.json"), {
        "schema": f"jaxfne.publication.fig{n}_polish_receipt.v1",
        "figure": plot_key,
        "status": rec["status"],
        "png_300dpi": rec["export"]["png_300dpi"],
        "png_sha256": rec["export"]["png_sha256"],
        "pdf_vector": rec["export"]["pdf_vector"],
        "pdf_bytes": rec["export"]["pdf_bytes"],
        "downstream_of_frozen_sha": rec["frozen_upstream_sha256"],
        "checkpoint": rec["checkpoint"],
        "repo_head": rec["repo_head"],
        "audited_at_utc": rec["audited_at_utc"],
    })


def main() -> int:
    import fig01_grammar as m01
    from fig05_protocol_c import build_figure as build_fig05

    plans = {}

    # fig01
    rec = run_figure(1, m01.build_figure(), polish_fig01, audit_fig01, "fig01_tfne_grammar", FROZEN["figures/publication/fig01_tfne_grammar.png"])
    write_figure_artifacts(rec, "Fig01.grammar")
    plans["fig01"] = rec

    bundle, bf2, bf3, bf4 = load_fig02_04()
    rec = run_figure(2, bf2(bundle), polish_fig02, audit_fig02, "fig02_emitter_source", FROZEN["figures/publication/fig02_emitter_source.png"])
    write_figure_artifacts(rec, "Fig02.canonical_Q")
    plans["fig02"] = rec

    rec = run_figure(3, bf3(bundle), polish_fig03, audit_fig03, "fig03_local_observation", FROZEN["figures/publication/fig03_local_observation.png"])
    write_figure_artifacts(rec, "Fig03.lfp_csd_proxy")
    plans["fig03"] = rec

    rec = run_figure(4, bf4(bundle), polish_fig04, audit_fig04, "fig04_multiscale_boundary", FROZEN["figures/publication/fig04_multiscale_boundary.png"])
    write_figure_artifacts(rec, "Fig04.EEG_MEG_analysis_only")
    plans["fig04"] = rec

    spec5 = json.loads((REPO / "artifacts/publication/fig05_wave_spec.json").read_text())
    rec = run_figure(5, build_fig05(spec5), polish_fig05, audit_fig05, "fig05_traveling_wave_no_wave", FROZEN["figures/publication/fig05_traveling_wave_no_wave.png"])
    write_figure_artifacts(rec, "Fig05.protocol_C")
    plans["fig05"] = rec

    m6, d6 = load_fig06()
    rec = run_figure(6, m6.build_figure(d6["ev"], d6["mx"], d6["h3"], d6["counts"], d6["d3"]), polish_fig06, audit_fig06, "fig06_rbs_hdp_ladder", FROZEN["figures/publication/fig06_rbs_hdp_ladder.png"])
    write_figure_artifacts(rec, "Fig06.HWD_ladder")
    plans["fig06"] = rec

    m7, d7 = load_fig07()
    rec = run_figure(7, m7.build_figure(d7["h1"], d7["delays"], d7["owner"], d7["obs"], d7["nulls"], d7["arms"], d7["prop"]), polish_fig07, audit_fig07, "fig07_e_integration", FROZEN["figures/publication/fig07_e_integration.png"])
    write_figure_artifacts(rec, "Fig07.E_integration")
    plans["fig07"] = rec

    summary = {"status": "PASSED", "figures": {}}
    ok = True
    for k, r in plans.items():
        ok = ok and r["status"] == "PASSED"
        summary["figures"][k] = {
            "status": r["status"],
            "png": r["export"]["png_sha256"][:12],
            "pdf_bytes": r["export"]["pdf_bytes"],
        }
        print(f"  {k}: {r['status']}  min_font={r['generic']['min_font_pt']}  "
              f"area_ratio={r.get('polish', {}).get('area_ratio', '-')}")
        if r["status"] != "PASSED":
            print("    targeted:", {kk: vv for kk, vv in r["targeted"].items() if vv is not True})
            print("    generic:", {kk: vv for kk, vv in r["generic"].items() if vv is not True})
    summary["status"] = "PASSED" if ok else "FAILED"
    cross = {
        "schema": "jaxfne.publication.figures_1_7_final_layout_audit.v1",
        "checkpoint": "figures_1_7_final_layout_audit",
        "status": summary["status"],
        "repo_head": pfc.repo_sha(),
        "audited_at_utc": utc_now(),
        "figures": summary["figures"],
        "cross_typography": {
            "font_floor_pt": 6.5,
            "all_figures_min_font_ok": all(r["generic"]["font_floor_6_5"] for r in plans.values()),
            "title_convention": "Figure N - ... suptitle ~12 pt",
        },
    }
    write_json_strict(guarded_path("artifacts/publication/polish/figures_1_7_final_layout_audit.json"), cross)
    print("CROSS_AUDIT_STATUS:", summary["status"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())