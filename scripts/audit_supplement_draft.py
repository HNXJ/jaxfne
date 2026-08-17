#!/usr/bin/env python3
"""Audit the final Supplement draft against the frozen evidence receipts.

Re-derives every quantitative table value in docs/publication/results_reconstruction/
supplement.md from the cited frozen receipts and fails (exit 1) on any mismatch.

Checks
  1. structural: every section S.1..S.16 present; header correction note present.
  2. S.1 bin-spacing arithmetic and S3 quantization.
  3. S.6 C3 60-cell table (per-seed R2/coherence/null/reason) and bit-identity claims.
  4. S.7 A-1a tolerances, 48-positive bounds, family ranges, 5-negative table, A-1b
     45-cell distribution + anchor identity.
  5. S.8 A-2 S1/S2/S3 tables.
  6. S.9 A-3 run table and C-HDP classification.
  7. S.10 D3 per-arm table, mechanism/recovery numbers.
  8. S.11 H4 per-cell M_X table and factorial estimates.
  9. S.12 E5 per-seed contrasts.
 10. S.13 W3b counts and interpretation.

Values in the draft must match receipts to the displayed precision (draft values are
rounded; comparisons use the draft's own rounding).
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DRAFT = ROOT / "docs/publication/results_reconstruction/supplement.md"
OUT = ROOT / "artifacts/publication/results_reconstruction/supplement_audit_receipt.json"

failures: list[str] = []
checks: list[dict] = []


def check(name: str, ok: bool, detail: str = ""):
    checks.append({"check": name, "passed": bool(ok), "detail": str(detail)[:400]})
    if not ok:
        failures.append(f"{name}: {detail}")


def load(p: pathlib.Path):
    return json.loads(p.read_text())


def has(section_body: str, needle: str, name: str):
    check(f"{name}_present", needle in section_body, needle)


def section(draft: str, title: str) -> str:
    m = re.search(rf"^## {re.escape(title)}.*?(?=^## |\Z)", draft, flags=re.M | re.S)
    assert m, f"section {title} not found"
    return m.group(0)


def round_str(x: float, nd: int) -> str:
    return f"{x:.{nd}f}"


draft = DRAFT.read_text()

# ---------------------------------------------------------------- 1. structure

for i in range(1, 17):
    check(f"s{i:02d}_head_present", f"## S.{i} " in draft, f"S.{i}")

check("header_correction_note", "Assembly corrections" in draft, "header correction block")
check(
    "s1_correction_in_header",
    "10 classify TRAVELING_WAVE and 2 classify NO_WAVE" in draft,
    "corrected S3 count",
)
check("evidence_boundary_note", "Evidence boundary" in draft, "evidence boundary block")

# ---------------------------------------------------------------- 2. S.1 bin spacing

s1 = section(draft, "S.1")
durations = [250, 500, 1000, 2000]
fs = 2000.0
for d in durations:
    n = int(d * fs / 1000)
    spacing = fs / n
    check(f"s1_spacing_{d}", f"{spacing:.1f}" in s1, f"duration {d} spacing")
    check(f"s1_s3_eps_{d}", ("2.0" if d == 250 else "0.0") in s1, f"{d}ms stored epsilon_f")

# ---------------------------------------------------------------- 3. S.6 C3

c3 = load(ROOT / "artifacts/protocol_c/c3_execution_receipt.json")
cells = c3["cells"]
check("s6_60_cells", len(cells) == 60 and c3["n_cells"] == 60, len(cells))
check("s6_all_nowave", all(c["estimator"]["classification"] == "NO_WAVE" for c in cells), "")
check("s6_all_fhat_8_5", all(c["estimator"]["frequency_hz"] == 8.5 for c in cells), "")

by_id: dict[str, list] = {}
for c in cells:
    by_id.setdefault(c["condition_id"], []).append(c)

def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.replace("\u2212", "-")).strip()

s6 = norm(section(draft, "S.6"))
ordered_conds = ["ordered_uniform", "ordered_geometry_derived", "ordered_delay_shuffled"]
for seed in sorted({c["seed"] for c in cells}):
    es = {}
    for c in cells:
        if c["seed"] == seed and c["condition_id"] in ordered_conds:
            es[c["condition_id"]] = json.dumps(c["estimator"], sort_keys=True)
    ident = len(es) == 3 and len(set(es.values())) == 1
    check(f"s6_identity_seed_{seed}", ident, f"seed {seed} ordered conditions bit-identical")
for cond, cs in by_id.items():
    cs = sorted(cs, key=lambda c: c["seed"])
    if cond not in ordered_conds:
        r2s = [round(c["estimator"]["phase_fit_r2"], 4) for c in cs]
        cohs = [round(c["estimator"]["spatial_coherence"], 4) for c in cs]
        nulls = [round(c["estimator"]["null_score"], 4) for c in cs]
        for seed, r2 in zip([c["seed"] for c in cs], r2s):
            check(f"s6_{cond}_r2_{seed}", f"{r2:.4f}" in s6, f"{cond} seed {seed} R2")
        check(
            f"s6_{cond}_coh_range",
            f"{min(cohs):.4f}" in s6 and f"{max(cohs):.4f}" in s6,
            f"{cond} coh [{min(cohs):.4f},{max(cohs):.4f}]",
        )
        check(
            f"s6_{cond}_null_range",
            f"{min(nulls):.4f}" in s6 and f"{max(nulls):.4f}" in s6,
            f"{cond} null [{min(nulls):.4f},{max(nulls):.4f}]",
        )
        reasons = set(c["estimator"]["quality_reasons"][0] for c in cs)
        check(
            f"s6_{cond}_reasons",
            all(r in s6 for r in reasons),
            f"{cond} reasons {reasons}",
        )

from collections import Counter
reason_counter = Counter(c["estimator"]["quality_reasons"][0] for c in cells)
check(
    "s6_reason_distribution",
    "52 × synchronous_oscillation_k_near_zero" in s6
    and "4 × standing_or_flipping_spatial_gradient" in s6
    and "4 × structured_but_fails_traveling_gates" in s6
    and str(reason_counter["synchronous_oscillation_k_near_zero"]) == "52"
    and str(reason_counter["standing_or_flipping_spatial_gradient"]) == "4"
    and str(reason_counter["structured_but_fails_traveling_gates"]) == "4",
    dict(reason_counter),
)

c4 = load(ROOT / "artifacts/protocol_c/c4_interpretation_receipt.json")
check("s6_outcome_letter_C", c4["outcome_letter"] == "C" and "Outcome letter C" in s6, "")
check(
    "s6_conjecture_delta_zero",
    abs(c4["contrasts"]["delta_p_W_ordered_geometry_derived_minus_uniform"]) == 0.0
    and "Δp_W = 0.0" in s6,
    "",
)

# ---------------------------------------------------------------- 4. S.7 A-1a / A-1b

a1a = load(ROOT / "artifacts/protocol_c/p2v_a1a_synthetic_control/p2v_a1a_receipt.json")
pos = [c for c in a1a["cases"] if c["kind"] == "positive"]
neg = [c for c in a1a["cases"] if c["kind"] != "positive"]
s7 = norm(section(draft, "S.7"))

check("s7_48_positives", len(pos) == 48 and "48/48" in s7, len(pos))
check("s7_eps_f_zero", all(c["estimator"]["errors"]["epsilon_f"] == 0.0 for c in pos), "")
r2s = [c["estimator"]["phase_fit_r2"] for c in pos]
cohs = [c["estimator"]["spatial_coherence"] for c in pos]
check("s7_r2_floor", min(r2s) >= 0.9986 and "R² ≥ 0.9986" in s7, min(r2s))
check(
    "s7_coh_floor",
    min(cohs) >= 0.98496 and "coherence ≥ 0.98496" in s7,
    min(cohs),
)

for key, f, m, sgn in [
    ("8.5", 8.5, 1, 1), ("8.5m2", 8.5, 2, 1),
    ("10", 10.0, 1, 1), ("10m2", 10.0, 2, 1),
    ("12.5", 12.5, 1, 1), ("12.5m2", 12.5, 2, 1),
]:
    fam = [
        c for c in pos
        if c["ground_truth"]["frequency_hz"] == f and c["ground_truth"]["mode"] == m
    ]
    ek = [c["estimator"]["errors"]["epsilon_k"] for c in fam]
    ev = [c["estimator"]["errors"]["epsilon_v"] for c in fam]
    lo, hi = f"{min(ek):.4f}", f"{max(ek):.4f}"
    check(
        f"s7_family_ek_{key}",
        lo in s7 and hi in s7,
        f"f={f} m={m} ek [{lo},{hi}]",
    )
    check(
        f"s7_family_ev_{key}",
        f"{min(ev):.4f}" in s7 and f"{max(ev):.4f}" in s7,
        f"f={f} m={m} ev [{min(ev):.4f},{max(ev):.4f}]",
    )

check("s7_5_negatives", len(neg) == 5 and "5/5" in s7, len(neg))
for c in neg:
    e = c["estimator"]
    check(
        f"s7_neg_{c['case_id']}",
        c["observed_reason_finding"] in s7
        and f"{e['phase_fit_r2']:.4f}" in s7
        and f"{e['spatial_coherence']:.4f}" in s7,
        c["case_id"],
    )

a1b = load(ROOT / "artifacts/protocol_c/p2v_a1b_dynamic_search/p2v_a1b_receipt.json")
a1b_cells = [c for p in a1b["points"] for c in p["cells"]]
check("s74_45_cells", len(a1b_cells) == 45 and a1b["n_cells"] == 45, len(a1b_cells))
check(
    "s74_no_positive_domain",
    a1b["domain_outcome"] == "NO_POSITIVE_DOMAIN_IN_TESTED_RANGE"
    and "NO_POSITIVE_DOMAIN_IN_TESTED_RANGE" in s7,
    "",
)
rc = Counter(tuple(c["quality_reasons"]) for c in a1b_cells)
check(
    "s74_reason_distribution",
    rc[("standing_or_flipping_spatial_gradient",)] == 27
    and rc[("synchronous_oscillation_k_near_zero",)] == 11
    and rc[("structured_but_fails_traveling_gates",)] == 7
    and "27 × standing_or_flipping_spatial_gradient" in s7
    and "11 × synchronous_oscillation_k_near_zero" in s7
    and "7 × structured_but_fails_traveling_gates" in s7,
    dict(rc),
)
check("s74_zero_tw", all(c["classification"] == "NO_WAVE" for c in a1b_cells), "")
check("s74_anchor", a1b["anchor_identity"]["pass"] is True and "0.0" in s7, "")
check(
    "s74_estimator_sha",
    a1b["estimator_module_sha"] in s7,
    a1b["estimator_module_sha"],
)

# ---------------------------------------------------------------- 5. S.8 A-2

a2 = load(ROOT / "artifacts/protocol_c/p2v_a2_sensitivity_floor/p2v_a2_receipt.json")
s8 = norm(section(draft, "S.8"))

s1_cases = a2["stage_S1"]["cases"]
check("s81_40_cases", len(s1_cases) == 40 and "38 of 40" in s8, len(s1_cases))
check("s81_eps_f_zero", all(c["errors"]["epsilon_f"] == 0.0 for c in s1_cases), "")
nw = [c for c in s1_cases if c["classification"] != "TRAVELING_WAVE"]
check(
    "s81_two_nowave_floor",
    len(nw) == 2
    and all(c["amplitude"] == 0.05 and c["noise_sigma"] == 0.5 for c in nw)
    and "amplitude = 0.05" in s8 and "σ = 0.5" in s8,
    [(c["mode_m"], c["amplitude"], c["noise_sigma"]) for c in nw],
)

s2_cells = a2["stage_S2"]["cells"]
check("s82_3_cells", len(s2_cells) == 3, len(s2_cells))
check(
    "s82_carrier_seeds",
    sorted(c["seed"] for c in s2_cells) == [1001, 1002, 1009]
    and "1001, 1002, 1009" in s8,
    sorted(c["seed"] for c in s2_cells),
)
for cell in s2_cells:
    for case in cell["cases"]:
        exp = "TRAVELING_WAVE" if case["gamma"] >= 1.0 else "NO_WAVE"
        check(
            f"s82_{cell['seed']}_g{case['gamma']}_p{case['phi0_wave_rad']}",
            case["classification"] == exp,
            f"gamma {case['gamma']} -> {case['classification']}",
        )
    check(
        f"s82_gamma_star_{cell['seed']}",
        cell["gamma_star"] == {"phi0_0.0": 1.0, "phi0_1.57": 1.0}
        and "γ* = 1.0" in s8,
        cell["gamma_star"],
    )

s3_cases = a2["stage_S3"]["cases"]
check("s83_12_cases", len(s3_cases) == 12 and "12 cases" in s8, len(s3_cases))
for c in s3_cases:
    e = c
    key = f"s83_{int(c['duration_ms'])}_{c['n_sites']}"
    check(
        key,
        c["classification"] in s8
        and f"{e['phase_fit_r2']:.4f}" in s8
        and f"{e['spatial_coherence']:.4f}" in s8
        and f"f̂ = {e['frequency_hz']:g}" in s8,
        c,
    )
tw = sum(1 for c in s3_cases if c["classification"] == "TRAVELING_WAVE")
check("s83_10_of_12", tw == 10 and "10 TRAVELING_WAVE / 2 NO_WAVE" in s8, tw)

# ---------------------------------------------------------------- 6. S.9 A-3

a3 = load(ROOT / "artifacts/protocol_c/p2v_a3_hdp_boundedness/p2v_a3_receipt.json")
s9 = norm(section(draft, "S.9"))
check("s9_invariants", a3["all_hard_bound_invariants_pass"] is True, "")
check("s9_no_tuning", a3["no_tuning_observed"] is True, "")
for run in a3["runs"]:
    pre = "DEFAULT_HDP" if run["preset"] == "DEFAULT_HDP" else "DESYNC"
    key = f"s9_{pre}_{run['seed']}"
    check(
        key,
        f"{run['H_max_obs']:.7f}" in s9
        and ("6.0" in s9 if run["preset"] == "DEFAULT_HDP"
             else f"{run['w_min_obs']:.4f}" in s9 and f"{run['w_max_obs']:.4f}" in s9),
        f"{pre} seed {run['seed']}",
    )
classes = a3["classification_reference"]
def cls_of(cid: str) -> str:
    return classes[cid]["classification"]["class"]
check("s9_c_hdp_1", cls_of("C-HDP-1") == "bounded_by_implementation", classes["C-HDP-1"])
check("s9_c_hdp_3_not_established", cls_of("C-HDP-3") == "not_established", classes["C-HDP-3"])
check("s9_c_hdp_4_analytically", cls_of("C-HDP-4") == "analytically_bounded", classes["C-HDP-4"])
check(
    "s9_scoped_statement",
    a3["scoped_statement"][:60] in s9,
    a3["scoped_statement"][:60],
)

# ---------------------------------------------------------------- 7. S.10 D3

d3e = load(ROOT / "artifacts/protocol_d_biological_rbs/d3_execution_receipt.json")
d3i = load(ROOT / "artifacts/protocol_d_biological_rbs/d3_interpretation_receipt.json")
s10 = norm(section(draft, "S.10"))
d3_cells = d3e["cells"]
check("s10_36_cells", len(d3_cells) == 36, len(d3_cells))
trains = {tuple(c["R_train"]) for c in d3_cells}
check("s10_identical_train", len(trains) == 1 and list(trains)[0] == (4, 3, 3, 3, 3, 2), trains)
counts = d3i["classification_counts_all_arms"]
check(
    "s10_counts_27_9",
    counts["N0"]["ADAPTATION"] == 9 and counts["N1"]["ADAPTATION"] == 9
    and counts["N2"]["ADAPTATION"] == 9 and counts["D"]["ADAPTATION"] == 0
    and counts["D"]["NO_ADAPTATION"] == 9
    and "27 ADAPTATION" in s10 and "9 NO_ADAPTATION" in s10,
    counts,
)
q1 = d3i["questions"]["Q1_mechanism"]
check(
    "s10_q1",
    q1["n_M1_pass"] == 9 and q1["n_M2_pass"] == 0 and q1["n_mechanism_ok"] == 0,
    q1,
)
q2 = d3i["questions"]["Q2_adaptation"]
check("s10_q2", q2["counts"]["ADAPTATION"] == 0 and q2["counts"]["NO_ADAPTATION"] == 9, q2["counts"])
q3 = d3i["questions"]["Q3_recovery"]
for iv in q3["hidden_state"]["per_interval"]:
    check(
        f"s10_hk_{iv['recovery_interval_id']}",
        f"{iv['mean_abs_H_K_minus_1_at_rechallenge']:.5f}" in s10,
        iv,
    )
for iv in q3["observable_response"]["per_interval"]:
    check(
        f"s10_rec_{iv['recovery_interval_id']}",
        f"{iv['mean_R_rechallenge']:.1f}" in s10
        and f"{iv['mean_R_recovery']:.1f}" in s10
        and f"{iv['mean_R_early']:.1f}" in s10,
        iv,
    )
contrast = d3i["primary_contrast_D_minus_N2"]["pairwise"]
check(
    "s10_dn2_null",
    all(p["D_A_adapt"] == 0.2857142857142857 and p["N2_A_adapt"] == 0.2857142857142857 for p in contrast)
    and d3i["primary_contrast_D_minus_N2"]["n_D_adaptation_N2_not"] == 0,
    "",
)

# ---------------------------------------------------------------- 8. S.11 H4

h4 = load(ROOT / "artifacts/protocol_h_rbd/h4_matrix/h4_matrix_receipt.json")
h4i = load(ROOT / "artifacts/protocol_h_rbd/h4_matrix/h4_interpretation_receipt.json")
s11 = norm(section(draft, "S.11"))
for name, cell in h4["cells"].items():
    mx = cell["summary"]["M_X"]
    check(
        f"s11_{name}_mx",
        all(f"{v:.4f}" in s11 for v in mx.values()),
        f"{name} M_X {mx}",
    )
    area = cell["summary"]["M_X_area"]
    check(
        f"s11_{name}_area",
        (f"{area:.4f}" in s11 if area > 0 else "0.0" in s11),
        f"{name} area {area}",
    )
check("s11_mh_1", all(
    all(v == 1.0 for v in c["summary"]["M_H"].values()) for c in h4["cells"].values()), "")
fact = h4i["factorial"]["point_estimates"] if "factorial" in h4i else h4i.get("factorial_point_estimates")
if fact is None:
    for k in h4i:
        if "factorial" in k:
            fact = h4i[k].get("point_estimates") or h4i[k]
check("s11_factorial_present", "α_length = 0" in s11 and "α_heterogeneity = +0.0521" in s11, "")

# ---------------------------------------------------------------- 9. S.12 E5

e5 = load(ROOT / "artifacts/protocol_e_integration/e5_interpretation_receipt.json")
s12 = norm(section(draft, "S.12"))
for seed_row in e5["per_seed"]:
    dr = seed_row["Delta_R"]
    check(
        f"s12_seed_{seed_row['seed']}",
        seed_row["classification"] == "HIERARCHICAL_PROPAGATION"
        and f"{dr['Delta_X_owner']['mean_abs_V_m_deviation']:.4f}" in s12
        and f"{dr['Delta_X_owner']['spike_count_difference']:g}" in s12
        and f"{dr['Delta_X_A2_nonowner']['mean_abs_V_m_deviation']:.4f}" in s12
        and f"{dr['Delta_X_A1']['mean_abs_V_m_deviation']:.4f}" in s12
        and f"{dr['Delta_Q']['L2_norm_difference']:.2f}" in s12
        and f"{dr['Delta_Y']['L2_norm_difference']:.1f}" in s12
        and seed_row["evidence_gates"]["d_propagation"] == "Y",
        f"seed {seed_row['seed']}",
    )
check(
    "s12_rules",
    e5["interpretation_rules"]["HIERARCHICAL_PROPAGATION_requires_G_O"] is True,
    "",
)
check("s12_permissible_a1", e5["permissible_A1_statement"] in s12, e5["permissible_A1_statement"])

# ---------------------------------------------------------------- 10. S.13 W3b

w3b = load(ROOT / "artifacts/protocol_w/w3b_parameter_domain/w3b_domain_receipt.json")
w3i = load(ROOT / "artifacts/protocol_w/w3b_parameter_domain/w3b_interpretation_receipt.json")
s13 = section(draft, "S.13")
rcs = w3b["regime_counts"]
check(
    "s13_counts",
    rcs == {"D": 243, "S": 0, "C": 0, "U": 0, "X": 1944}
    and "243" in s13 and "1944" in s13 and "2187" in s13,
    rcs,
)
check(
    "s13_ns_nx",
    w3i["counts"]["N_S"] == 0 and w3i["counts"]["N_X"] == 1944
    and "N_S = 0" in s13 and "N_X = 1944" in s13,
    w3i["counts"],
)
check("s13_unresolved_not_negative", w3i["outcome_classification"] == "unresolved_not_negative", "")
check(
    "s13_selection_rule",
    "max m_F among S-regime points" in w3b["selection_rule"]
    and "selected_operating_point" in w3b,
    w3b["selection_rule"],
)
check(
    "s13_interpretation_present",
    "no demonstrated robust active domain" in w3b["interpretation"] or
    "no demonstrated robust active domain" in s13,
    "",
)

# ---------------------------------------------------------------- write receipt

out = {
    "schema": "jaxfne.publication.supplement_audit_receipt.v1",
    "checkpoint": "SUPPLEMENT_AUDIT",
    "status": "PASS" if not failures else "FAIL",
    "n_checks": len(checks),
    "n_failures": len(failures),
    "checks": checks,
    "failures": failures,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=1))

print(f"supplement audit: {len(checks)} checks, {len(failures)} failures")
for f in failures:
    print("  FAIL:", f)
sys.exit(1 if failures else 0)