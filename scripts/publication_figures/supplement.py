"""Supplement items S1 (notation table), S2 (calibration transforms), S5
(HDP reduction identity), S12 (the scored Vogels P3 v4), S13 (PCL Fig. 3), S14 (the scored GSDR P3 fit)
and S16 (the scored Destexhe 2009 P3 replication).

S1 parses docs/theory/doctrine/definition_map.md. A row is mapped when its symbol and
test are both named, otherwise open. S2 applies every public calibration transform
through the public API and records its numeric check. S5 runs the reduction checks of
tests/test_hdp_reduction_identity.py for every core registered rule, through that
module's own helpers, and writes the numbers its assertions use. S14 tabulates the
scored GSDR P3 fit from its score JSON, through the scorer's own pass rule. S16 tabulates
the scored Destexhe 2009 P3 replication from its score JSON.
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

# The tree under test is this repo: a site-packages jaxfne (if installed) must
# not shadow it when the generator runs as `python scripts/...` from elsewhere.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

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


# --- S2: relative quantities and calibration transforms ----------------------
#
# Each row applies one public calibration authority through the public API to a
# known relative input and records input, output and expected output. A row
# passes when the calibrated values match the relative values within
# rtol 1e-12 (exact equality for integers) and the epistemic level equals the
# transform's target. apply_calibration is a relabel: values are preserved.

S2_Rtol = 1e-12


def _values_match(actual: object, expected: object) -> bool:
    a = np.asarray(actual)
    e = np.asarray(expected)
    if a.shape != e.shape:
        return False
    if a.dtype.kind in "iu" and e.dtype.kind in "iu":
        return bool(np.array_equal(a, e))
    return bool(
        np.allclose(a.astype(np.float64), e.astype(np.float64), rtol=S2_Rtol, atol=0.0)
    )


def _s2_probe_relative():
    from jaxfne import fields as fields_mod

    phi = np.array([[0.5, -1.25, 2.0], [0.125, 0.0, -0.75]])
    return fields_mod.lfp_proxy_probe(phi)


def _s2_field_relative():
    from jaxfne import fields as fields_mod

    phi = np.array([[0.5, -1.25, 2.0], [0.125, 0.0, -0.75]])
    return fields_mod.FieldOutput(
        source_proxy=phi,
        phi_e_proxy=phi,
        csd_proxy=phi,
        lfp_proxy=phi,
        kernel=np.eye(3),
        contact_depths=np.array([0.0, 0.5, 1.0]),
        diagnostics={},
    )


def _s2_specs() -> list[dict[str, object]]:
    return [
        {
            "name": "jaxfne.fields.apply_calibration",
            "code_path": "jaxfne/fields/probes.py:269 (transform jaxfne/fields/probes.py:167)",
            "relative_quantity": "ProbeReadout data at RELATIVE_PROXY "
            "(jaxfne/fields/probes.py:536 lfp_proxy_probe output; dimensionless)",
            "absolute_quantity": "same data relabeled CALIBRATED",
            "unit": "V",
            "boundary": "probe",
            "doctrine": "artifacts/AGENTS.md:32 "
            "('Absolute units arise through explicit calibration transformations "
            "at semantic boundaries'); "
            "docs/theory/doctrine/definition_map.md:41 (D35 calibration boundary "
            "C_S(Q; G, Theta))",
            "build_relative": _s2_probe_relative,
            "extract": lambda out: out.data,
            "units": "V",
            "conductivity": "declared_proxy_conductivity",
            "distance": "declared_probe_geometry",
            "target_level": "CALIBRATED",
        },
        {
            "name": "jaxfne.fields.apply_calibration",
            "code_path": "jaxfne/fields/probes.py:269 (transform jaxfne/fields/probes.py:167)",
            "relative_quantity": "FieldOutput phi_e_proxy at RELATIVE_PROXY "
            "(jaxfne/fields/proxy.py:56 uncalibrated simulation readout; dimensionless)",
            "absolute_quantity": "same data relabeled REDUCED_PHYSICAL",
            "unit": "V",
            "boundary": "field",
            "doctrine": "docs/theory/doctrine/rbs_rbd_hdp.md:290-296 "
            "('Where physical calibration depends on geometry, make the boundary "
            "explicit'; Q_phys = C_S(Q_tilde; G, Theta_cal)); "
            "docs/theory/doctrine/relative_quantity_grammar.md:142 "
            "('calibrated: an explicit calibration transformation has been applied')",
            "build_relative": _s2_field_relative,
            "extract": lambda out: out.phi_e_proxy,
            "units": "V",
            "conductivity": "declared_proxy_conductivity",
            "distance": "declared_field_geometry",
            "target_level": "REDUCED_PHYSICAL",
        },
    ]


def _check_s2_row(spec: dict[str, object]) -> dict[str, object]:
    from jaxfne import fields as fields_mod

    rel = spec["build_relative"]()  # type: ignore[operator]
    values_in = np.asarray(spec["extract"](rel), dtype=np.float64)  # type: ignore[operator]
    transform = fields_mod.CalibrationTransform(
        units=spec["units"],
        conductivity=spec["conductivity"],
        distance=spec["distance"],
        target_level=spec["target_level"],
    )
    try:
        out = fields_mod.apply_calibration(rel, transform)
        values_out = np.asarray(spec["extract"](out), dtype=np.float64)  # type: ignore[operator]
        level_ok = out.epistemic_level == transform.target_level
        values_ok = _values_match(values_out, values_in)
        passed = bool(level_ok and values_ok)
        detail = f"level={out.epistemic_level} level_ok={level_ok} values_ok={values_ok}"
    except Exception as exc:  # a refusal is a failed check, recorded not raised
        values_out = None
        passed = False
        detail = f"{type(exc).__name__}: {exc}"
    return {
        "name": spec["name"],
        "code_path": spec["code_path"],
        "relative_quantity": spec["relative_quantity"],
        "absolute_quantity": spec["absolute_quantity"],
        "unit": spec["units"],
        "boundary": spec["boundary"],
        "doctrine": spec["doctrine"],
        "target_level": spec["target_level"],
        "input": values_in.tolist(),
        "output": None if values_out is None else np.asarray(values_out).tolist(),
        "expected": values_in.tolist(),
        "passed": passed,
        "detail": detail,
    }


def write_s2(out_dir: Path) -> dict[str, int]:
    rows = [_check_s2_row(spec) for spec in _s2_specs()]
    failed = sum(not row["passed"] for row in rows)
    payload = {
        "source": "jaxfne/fields/probes.py",
        "rtol": S2_Rtol,
        "rows": rows,
    }
    (out_dir / "S2_calibration.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# S2. Relative quantities and calibration transforms",
        "",
        "Each row applies a public calibration transform to a known relative input "
        "through the public API. A row passes when the calibrated values match the "
        f"relative values within rtol {S2_Rtol:g} (exact equality for integers) and "
        "the epistemic level equals the transform target.",
        "",
        "| Transform | Relative in | Absolute out | Unit | Boundary | Check | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['name']} | {r['relative_quantity']} | {r['absolute_quantity']} "
        f"| {r['unit']} | {r['boundary']} | {r['detail']} | "
        f"{'pass' if r['passed'] else 'FAIL'} |"
        for r in rows
    ]
    (out_dir / "S2_calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"rows": len(rows), "checked": len(rows), "failed": failed}


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


# --- S12: the scored Vogels P3 v4 (segment B) -----------------------------------
#
# One row per scored seed of the signed P3 v4 (prereg/P3_v4_signed.md): the CONV slope
# over 130-150 s with its t, and the 149-150 s E rate. The lane rows are recomputed from
# the per-seed values with the prereg's rules (CONV: not converged only when |b| >= 0.01
# Hz/s and |b|/se > 2.101; R-E: |z - 3 Hz| / 3 Hz <= 0.20; O-DROP: ratio >= 2) and must
# agree with the verdict the score JSON records.

VOGELS_P3_SCORE = (
    REPO / "artifacts" / "etudes" / "papers" / "vogels2011" / "run" / "p3_v4_score.json"
)
S12_CONV_SLOPE_HZ_S = 0.01
S12_CONV_T_CRIT = 2.101
S12_TARGET_HZ = 3.0
S12_RE_BAND = 0.20
S12_DROP_RATIO = 2.0


def write_s12(out_dir: Path, score_path: Path = VOGELS_P3_SCORE) -> dict[str, object]:
    payload = json.loads(score_path.read_text(encoding="utf-8"))
    rows = []
    for seed in payload["seeds"]:
        conv = payload["CONV"]["per_seed"][str(seed)]
        b, se = float(conv["b_hz_s"]), float(conv["se_hz_s"])
        t = abs(b) / se if se > 0 else float("inf")
        rows.append({
            "seed": seed,
            "b_hz_s": conv["b_hz_s"],
            "se_hz_s": conv["se_hz_s"],
            "t": t,
            "converged": not (abs(b) >= S12_CONV_SLOPE_HZ_S and t > S12_CONV_T_CRIT),
            "re_hz": payload["R-E"]["per_seed_z_hz"][str(seed)],
        })
    z = float(payload["R-E"]["z_hz"])
    e = abs(z - S12_TARGET_HZ) / S12_TARGET_HZ
    ratio = float(payload["O-DROP"]["ratio"])
    holds = {
        "CONV": all(r["converged"] for r in rows),
        "R-E": e <= S12_RE_BAND,
        "O-DROP": ratio >= S12_DROP_RATIO,
    }
    failed = [k for k, ok in holds.items() if not ok]
    verdict = "closes" if not failed else "off (" + ", ".join(failed) + ")"
    if payload.get("verdict") != verdict:
        raise ValueError(
            f"{score_path}: recorded verdict {payload.get('verdict')!r} disagrees with "
            f"the recomputed {verdict!r}"
        )
    out = {
        "source": score_path.relative_to(REPO).as_posix()
        if score_path.is_relative_to(REPO)
        else score_path.name,
        "prereg": payload.get("prereg"),
        "prereg_sha256": payload.get("prereg_sha256"),
        "R-E": {"z_hz": payload["R-E"]["z_hz"], "e": e, "pass": holds["R-E"]},
        "O-DROP": {"before_hz": payload["O-DROP"]["before_hz"], "ratio": payload["O-DROP"]["ratio"],
                   "pass": holds["O-DROP"]},
        "CONV": {"pass": holds["CONV"]},
        "failed": failed,
        "verdict": verdict,
        "rows": rows,
    }
    (out_dir / "S12_vogels_p3.json").write_text(
        json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# S12. Scored Vogels P3 v4, segment B",
        "",
        f"Source: `{out['source']}` (prereg `{out['prereg']}`). The lane closes when every seed "
        f"converges (not converged only when |b| >= {S12_CONV_SLOPE_HZ_S} Hz/s and |b|/se > "
        f"{S12_CONV_T_CRIT}), R-E holds (|z - {S12_TARGET_HZ:g} Hz| / {S12_TARGET_HZ:g} Hz <= "
        f"{S12_RE_BAND}) and O-DROP holds (ratio >= {S12_DROP_RATIO:g}). R-E z {z:.3f} Hz, "
        f"e {e:.3f}; O-DROP ratio {ratio:.1f}. **Verdict: {verdict}**.",
        "",
        "| seed | b Hz/s (se) | t | converged | E rate 149-150 s, Hz |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['seed']} | {r['b_hz_s']:+.4f} ({r['se_hz_s']:.4f}) | {r['t']:.2f} "
        f"| {'yes' if r['converged'] else 'no'} | {r['re_hz']:.3f} |"
        for r in rows
    ]
    (out_dir / "S12_vogels_p3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"rows": len(rows), "verdict": "closes" if not failed else "off"}


# --- S13: PCL Fig. 3, the signed failures and R5-exact -----------------------------
#
# v4 (P3_fig3_result.md) and v5 T2 (P3_fig3_v5_result.md) are reported from their score
# JSONs; R5-exact (signed R4b) is recomputed from its rows: VOID when the negative control
# differs on no file, PASS when every file is equal, FAIL otherwise.

PCL_SCORES = REPO / "artifacts" / "etudes" / "papers" / "pcl" / "prereg" / "scores"
PCL_V4_SCORE = PCL_SCORES / "fig3_v4_full.json"
PCL_V5_T2_SCORE = PCL_SCORES / "fig3_v5_t2_even.json"
PCL_R5_EXACT_SCORE = PCL_SCORES / "fig3_r5_exact.json"
S13_V4_KEYS = ("t1_simple_pass", "t1_complex_pass", "t2_simple_pass", "t2_complex_pass",
               "t3_simple_pass", "t3_complex_pass", "t4_pass", "all_pass")
S13_T2_KEYS = ("frac", "sse", "sse_flat", "pass_frac", "guard", "t2_pass")


def _yn(flag: object) -> str:
    return "pass" if flag else "fail"


def write_s13(out_dir: Path, v4_path: Path = PCL_V4_SCORE, v5_path: Path = PCL_V5_T2_SCORE,
              r5_path: Path = PCL_R5_EXACT_SCORE) -> dict[str, object]:
    v4 = json.loads(v4_path.read_text(encoding="utf-8"))
    v5 = json.loads(v5_path.read_text(encoding="utf-8"))
    r5 = json.loads(r5_path.read_text(encoding="utf-8"))
    rows = r5["rows"]
    equal = sum(bool(row["core_equal"]) for row in rows)
    ctl = sum(bool(row["ctl_differs"]) for row in rows)
    verdict = "VOID" if ctl == 0 else ("PASS" if equal == len(rows) else "FAIL")
    if r5.get("verdict") != verdict:
        raise ValueError(
            f"{r5_path}: recorded verdict {r5.get('verdict')!r} disagrees with the recomputed "
            f"{verdict!r} ({equal} of {len(rows)} files equal, control differs on {ctl})"
        )
    out = {
        "v4": {key: v4[key] for key in S13_V4_KEYS},
        "v5_t2": {kind: {key: v5[kind][key] for key in S13_T2_KEYS} for kind in ("simple", "complex")},
        "r5_exact": {"prereg": r5.get("prereg"), "code_sha": r5.get("code_sha"), "files": len(rows),
                     "core_equal": equal, "ctl_differs": ctl, "verdict": verdict, "rows": rows},
    }
    (out_dir / "S13_pcl_fig3.json").write_text(
        json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    t2 = out["v5_t2"]
    lines = [
        "# S13. PCL Fig. 3: signed failures and R5-exact",
        "",
        f"Signed v4: T1 simple {_yn(v4['t1_simple_pass'])}, complex {_yn(v4['t1_complex_pass'])}; "
        f"T2 simple {_yn(v4['t2_simple_pass'])}, complex {_yn(v4['t2_complex_pass'])}; "
        f"T3 simple {_yn(v4['t3_simple_pass'])}, complex {_yn(v4['t3_complex_pass'])}; "
        f"T4 {_yn(v4['t4_pass'])}. **v4 verdict: {'PASS' if v4['all_pass'] else 'FAIL'}**.",
        "",
        "Signed v5 T2 (even phases): " + "; ".join(
            f"{kind} {t2[kind]['frac']:.3f} of points in tolerance, SSE {t2[kind]['sse']:.3f} "
            f"against flat {t2[kind]['sse_flat']:.3f}, {_yn(t2[kind]['t2_pass'])}"
            for kind in ("simple", "complex")
        ) + f". **v5 T2 verdict: {'PASS' if t2['simple']['t2_pass'] and t2['complex']['t2_pass'] else 'FAIL'}**.",
        "",
        f"R5-exact (`{r5.get('prereg')}`, code {r5.get('code_sha')}): core and numba arms equal in every "
        f"spike count and time on {equal} of {len(rows)} files; the negative control differs on {ctl}. "
        f"**R5-exact verdict: {verdict}**.",
        "",
        "| file (ori/sf/phase/trial) | events | simple numba / core | complex numba / core | equal | control differs |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['file']} | {r['n_events']} | {r['simple_numba']} / {r['simple_core']} "
        f"| {r['complex_numba']} / {r['complex_core']} | {'yes' if r['core_equal'] else 'no'} "
        f"| {'yes' if r['ctl_differs'] else 'no'} |"
        for r in rows
    ]
    (out_dir / "S13_pcl_fig3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"files": len(rows), "core_equal": equal, "verdict": verdict}


# --- S14: the scored GSDR P3 fit ------------------------------------------------
#
# Each row scores one optimizer seed of the signed P3 fit
# (artifacts/etudes/papers/gsdr2026/PREREG_P3_fit_signed.md) through the scorer's own
# rule: a seed passes when G2 (mean kappa_500_1000 over the held-out trials below 0.1)
# and G3 (mean rate within 5-50 Hz) both hold; the fit passes when at least 4 of 5
# seeds pass. Every number comes from run/p3_fit_score.json, never typed; the receipt
# directory the score JSON names stays out (a local warehouse path, not evidence).

P3_FIT_SCORE = (
    REPO / "artifacts" / "etudes" / "papers" / "gsdr2026" / "run" / "p3_fit_score.json"
)
S14_PASS_SEEDS = 4


def _s14_row_scored(row: dict[str, object]) -> dict[str, object]:
    passed = bool(row["G2"]) and bool(row["G3"])
    return {
        "seed": row["seed"],
        "loss_g0": row["loss_g0"],
        "loss_g30": row["loss_g30"],
        "kappa_mean": row["kappa_mean"],
        "kappa_min": row["kappa_min"],
        "kappa_max": row["kappa_max"],
        "rate_mean_hz": row["rate_mean_hz"],
        "G2": bool(row["G2"]),
        "G3": bool(row["G3"]),
        "passed": passed,
    }


def write_s14(out_dir: Path, score_path: Path = P3_FIT_SCORE) -> dict[str, object]:
    payload = json.loads(score_path.read_text(encoding="utf-8"))
    rows = [_s14_row_scored(row) for row in payload["rows"]]
    passing = sum(row["passed"] for row in rows)
    verdict = "PASS" if passing >= S14_PASS_SEEDS else "FAIL"
    recorded = payload.get("verdict")
    if recorded is not None and recorded != verdict:
        raise ValueError(
            f"{score_path}: recorded verdict {recorded!r} disagrees with "
            f"{passing} of {len(rows)} seeds passing (needs {S14_PASS_SEEDS})"
        )
    out = {
        "source": score_path.relative_to(REPO).as_posix()
        if score_path.is_relative_to(REPO)
        else score_path.name,
        "code_sha": payload.get("code_sha"),
        "prereg": payload.get("prereg"),
        "pass_needs": S14_PASS_SEEDS,
        "seeds_passing": passing,
        "verdict": verdict,
        "rows": rows,
    }
    (out_dir / "S14_gsdr_fit.json").write_text(
        json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# S14. Scored GSDR P3 fit",
        "",
        f"Source: `{out['source']}` (code {out['code_sha']}). A seed passes when G2 "
        f"(mean kappa below 0.1 on held-out trials) and G3 (mean rate 5-50 Hz) both "
        f"hold; the fit passes when at least {S14_PASS_SEEDS} of {len(rows)} seeds pass. "
        f"Seeds passing: {passing}. **Verdict: {verdict}**.",
        "",
        "| seed | best loss g0 -> g30 | kappa mean [min, max] | rate Hz | G2 | G3 | pass |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['seed']} | {r['loss_g0']:.2f} -> {r['loss_g30']:.2f} "
        f"| {r['kappa_mean']:.3f} [{r['kappa_min']:.3f}, {r['kappa_max']:.3f}] "
        f"| {r['rate_mean_hz']:.1f} | {'yes' if r['G2'] else 'no'} "
        f"| {'yes' if r['G3'] else 'no'} | {'yes' if r['passed'] else 'no'} |"
        for r in rows
    ]
    (out_dir / "S14_gsdr_fit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"rows": len(rows), "passing": passing, "verdict": verdict}


# --- S16: the scored Destexhe 2009 P3 replication ------------------------------
#
# Each row scores one simulation seed of the signed Destexhe P3 v3 replication
# (artifacts/etudes/papers/destexhe2009/prereg/P3_v3_signed.md). A seed passes
# when E1 (alive at 10 s), E2 (CV_ISI relative error e <= 0.05), and E3 (CC
# absolute error e <= 0.01) all hold; C1 control holds (Fig. 6A control seeds
# dead by 10 s). The replication passes when at least 4 of 5 seeds pass.

DESTEXHE_P3_VERDICT = (
    REPO / "artifacts" / "etudes" / "papers" / "destexhe2009" / "run" / "p3_v3_verdict.json"
)
S16_PASS_SEEDS = 4


def _s16_row_scored(row: dict[str, object]) -> dict[str, object]:
    alive = bool(row.get("alive", False))
    e2_val = float(row["e2"])
    e3_val = float(row["e3"])
    e1 = alive
    e2 = e2_val <= 0.05
    e3 = e3_val <= 0.01
    passed = e1 and e2 and e3
    return {
        "seed": row["seed"],
        "alive": alive,
        "cv_isi": row.get("cv_isi"),
        "e2": e2_val,
        "cc": row.get("cc"),
        "e3": e3_val,
        "rate_hz_all": row.get("rate_hz_all"),
        "E1": e1,
        "E2": e2,
        "E3": e3,
        "pass": passed,
    }


def write_s16(out_dir: Path, score_path: Path = DESTEXHE_P3_VERDICT) -> dict[str, object]:
    payload = json.loads(score_path.read_text(encoding="utf-8"))
    rows = [_s16_row_scored(row) for row in payload["rows"]]
    passing = sum(row["pass"] for row in rows)
    controls = payload.get("controls", [])
    c1_holds = (
        all(not bool(c.get("alive", False)) for c in controls)
        if controls
        else bool(payload.get("C1", True))
    )
    verdict = "PASS" if (passing >= S16_PASS_SEEDS and c1_holds) else "FAIL"
    recorded = payload.get("verdict")
    if recorded is not None and recorded != verdict:
        raise ValueError(
            f"{score_path}: recorded verdict {recorded!r} disagrees with "
            f"recomputed verdict {verdict!r} ({passing} of {len(rows)} seeds passing, "
            f"needs {S16_PASS_SEEDS}, C1={c1_holds})"
        )
    source_str = (
        score_path.relative_to(REPO).as_posix()
        if score_path.is_relative_to(REPO)
        else score_path.name
    )
    out = {
        "source": source_str,
        "prereg": payload.get("prereg"),
        "pass_needs": S16_PASS_SEEDS,
        "seeds_passing": passing,
        "C1": c1_holds,
        "verdict": verdict,
        "rows": rows,
        "controls": controls,
    }
    (out_dir / "S16_destexhe2009.json").write_text(
        json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# S16. Destexhe 2009 P3 replication under signed v3",
        "",
        f"Source: `{out['source']}`. A seed passes when E1 (alive at 10 s), "
        f"E2 (e2 <= 0.05), and E3 (e3 <= 0.01) all hold; C1 control holds "
        f"(control seeds dead by 10 s). The replication passes when at least "
        f"{S16_PASS_SEEDS} of {len(rows)} seeds pass. Seeds passing: {passing}. "
        f"C1 control: {'holds' if c1_holds else 'failed'}. **Verdict: {verdict}**.",
        "",
        "| seed | alive | E2 e | E3 e | rate Hz | E1 | E2 | E3 | pass |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rate_str = f"{r['rate_hz_all']:.1f}" if r["rate_hz_all"] is not None else "n/a"
        lines.append(
            f"| {r['seed']} | {'yes' if r['alive'] else 'no'} | {r['e2']:.3f} | {r['e3']:.4f} "
            f"| {rate_str} | {'yes' if r['E1'] else 'no'} | {'yes' if r['E2'] else 'no'} "
            f"| {'yes' if r['E3'] else 'no'} | {'yes' if r['pass'] else 'no'} |"
        )
    (out_dir / "S16_destexhe2009.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"seeds": len(rows), "rows": len(rows), "passing": passing, "verdict": verdict}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    counts = write_s1(args.out)
    s2 = write_s2(args.out)
    rules = write_s5(args.out)
    s12 = write_s12(args.out)
    s13 = write_s13(args.out)
    s14 = write_s14(args.out)
    s16 = write_s16(args.out)
    print(f"S1 rows={counts['rows']} mapped={counts['mapped']} open={counts['open']}")
    print(f"S2 rows={s2['rows']} checked={s2['checked']} failed={s2['failed']}")
    print(f"S5 rules={len(rules)} out={args.out}")
    print(f"S12 rows={s12['rows']} verdict={s12['verdict']} out={args.out}")
    print(f"S13 files={s13['files']} core_equal={s13['core_equal']} verdict={s13['verdict']} out={args.out}")
    print(f"S14 rows={s14['rows']} passing={s14['passing']} verdict={s14['verdict']} out={args.out}")
    print(f"S16 seeds={s16['seeds']} passing={s16['passing']} verdict={s16['verdict']} out={args.out}")
    return 1 if s2["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
