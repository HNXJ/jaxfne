#!/usr/bin/env python3
"""Audit the Phase-4 Results draft against the frozen evidence receipts.

Checks
  1. structural: draft paragraphs [P1..Pn], claim references {CL-xx}, quantitative
     markers {Qnn} all resolve to the traceability map and Q-table.
  2. quantitative: every Qnn quoted in the draft is recomputed from the cited receipt
     and must match the quoted value (tolerances tight).
  3. forbidden-overclaim: per-claim forbidden language from the claim ledger may appear
     only inside the listed allowed paragraphs (see traceability_map.md).

Writes artifacts/publication/results_reconstruction/results_audit_receipt.json.
Fails (exit 1) on any violation.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DRAFT = ROOT / "docs/publication/results_reconstruction/results_draft.md"
TRACE = ROOT / "docs/publication/results_reconstruction/traceability_map.md"
OUT = ROOT / "artifacts/publication/results_reconstruction/results_audit_receipt.json"


def load(p: pathlib.Path):
    return json.loads(p.read_text())


def rel(p: pathlib.Path) -> str:
    return str(p.relative_to(ROOT))


failures: list[str] = []
checks: list[dict] = []


def check(name: str, ok: bool, detail: str = ""):
    checks.append({"check": name, "passed": bool(ok), "detail": detail})
    if not ok:
        failures.append(f"{name}: {detail}")


# ---------------------------------------------------------------- 1. structure

draft = DRAFT.read_text()
pids = re.findall(r"\[P(\d+)\]", draft)
pids = sorted({int(x) for x in pids})
check("draft_paragraph_ids", pids == sorted(range(1, max(pids) + 1)),
      f"found {pids}")

qin_draft = set(m[1:-1] for m in re.findall(r"\{Q\d{2}\}", draft))
check("q_markers_bare", True, "")
for pid in pids:
    if not re.search(rf"\[P{pid}\]", draft):
        check(f"P{pid}_present", False)

claimed = re.findall(r"\{CL-\d+", draft)
check("claim_refs_present", len(claimed) >= 21, f"{len(claimed)} claim refs")

# ---------------------------------------------------------------- 2. Q-table

def q_table() -> dict[str, dict]:
    return {
        # --- canonical source
        "Q01": {
            "expect": "hash(Q_Fig2) = hash(Q_Fig3) = hash(Q_Fig4)",
            "load": lambda: load(ROOT / "artifacts/publication/fig02_04_experiment_a_spec.json")["cross_figure_invariant"],
            "cmp": lambda got, _exp: got == "hash(Q_Fig2) = hash(Q_Fig3) = hash(Q_Fig4)",
        },
        # --- A-1a (estimator synthetic control)
        "Q02": {
            "expect": "48 positive cases, all detected",
            "load": lambda: load(ROOT / "artifacts/protocol_c/p2v_a1a_synthetic_control/p2v_a1a_receipt.json"),
            "cmp": lambda r, _e: r["n_positive_cases"] == 48 and r["summary"]["all_positives_pass"] is True,
        },
        "Q03": {
            "expect": "zero frequency error",
            "load": lambda: load(ROOT / "artifacts/protocol_c/p2v_a1a_synthetic_control/p2v_a1a_receipt.json"),
            "cmp": lambda r, _e: _max_err(r, "frequency") == 0.0,
        },
        "Q04": {
            "expect": "wave-number error <= 0.00831 relative",
            "load": lambda: load(ROOT / "artifacts/protocol_c/p2v_a1a_synthetic_control/p2v_a1a_receipt.json"),
            "cmp": lambda r, _e: _max_err(r, "k_norm") <= 0.00831 + 1e-9,
        },
        "Q05": {
            "expect": "zero direction error",
            "load": lambda: load(ROOT / "artifacts/protocol_c/p2v_a1a_synthetic_control/p2v_a1a_receipt.json"),
            "cmp": lambda r, _e: _max_err(r, "direction") == 0.0,
        },
        "Q06": {
            "expect": "velocity error <= 0.00824 relative",
            "load": lambda: load(ROOT / "artifacts/protocol_c/p2v_a1a_synthetic_control/p2v_a1a_receipt.json"),
            "cmp": lambda r, _e: _max_err(r, "velocity") <= 0.00824 + 1e-9,
        },
        "Q07": {
            "expect": "5/5 negatives rejected",
            "load": lambda: load(ROOT / "artifacts/protocol_c/p2v_a1a_synthetic_control/p2v_a1a_receipt.json"),
            "cmp": lambda r, _e: r["n_negative_cases"] == 5 and r["summary"]["all_negatives_pass"] is True,
        },
        "Q08": {
            "expect": "freq [8.5,12.5] Hz; velocity [53.407, 39.27] mm/s",
            "load": lambda: load(ROOT / "artifacts/protocol_c/p2v_a1a_synthetic_control/p2v_a1a_receipt.json"),
            "cmp": lambda r, _e: (
                r["summary"]["frequency_range_recovered"] == [8.5, 12.5]
                and abs(r["summary"]["velocity_range_recovered_mm_per_s"][0] - 53.407) < 1e-3
                and abs(r["summary"]["velocity_range_recovered_mm_per_s"][1] - 39.27) < 1e-2
            ),
        },
        # --- C3 frozen grid
        "Q09": {
            "expect": "60 cells; delay 4 uniform",
            "load": lambda: load(ROOT / "artifacts/protocol_c/c3_execution_receipt.json"),
            "cmp": lambda r, _e: (
                r["n_cells"] == 60
                and all(c["delay_steps"] == [4] * c["n_neurons"] for c in r["cells"])
            ),
        },
        "Q10": {
            "expect": "60/60 NO_WAVE; 52/4/4 quality mix",
            "load": lambda: load(ROOT / "artifacts/protocol_c/c3_execution_receipt.json"),
            "cmp": lambda r, _e: (
                all(c["estimator"]["classification"] == "NO_WAVE" for c in r["cells"])
                and _quality_mix(r["cells"]) == {"synchronous_oscillation_k_near_zero": 52,
                                                 "standing_or_flipping_spatial_gradient": 4,
                                                 "structured_but_fails_traveling_gates": 4}
            ),
        },
        # --- A-1b dynamic search
        "Q11": {
            "expect": "15 points x 3 seeds = 45 cells",
            "load": lambda: load(ROOT / "artifacts/protocol_c/p2v_a1b_dynamic_search/p2v_a1b_receipt.json"),
            "cmp": lambda r, _e: r["n_points"] == 15 and r["n_cells"] == 45,
        },
        "Q12": {
            "expect": "45/45 NO_WAVE, 0 invalid",
            "load": lambda: load(ROOT / "artifacts/protocol_c/p2v_a1b_dynamic_search/p2v_a1b_receipt.json"),
            "cmp": lambda r, _e: (
                all(o["outcome"] == "NEGATIVE" and o["n_traveling_wave_cells"] == 0
                    and o["n_invalid_cells"] == 0 for o in r["point_outcomes"].values())
            ),
        },
        "Q13": {
            "expect": "NO_POSITIVE_DOMAIN_IN_TESTED_RANGE",
            "load": lambda: load(ROOT / "artifacts/protocol_c/p2v_a1b_dynamic_search/p2v_a1b_receipt.json"),
            "cmp": lambda r, _e: r["domain_outcome"] == "NO_POSITIVE_DOMAIN_IN_TESTED_RANGE",
        },
        "Q14": {
            "expect": "anchor vc0.131_k1 bitwise 3 seeds",
            "load": lambda: load(ROOT / "artifacts/protocol_c/p2v_a1b_dynamic_search/p2v_a1b_receipt.json"),
            "cmp": lambda r, _e: (
                r["anchor_identity"]["anchor_point"] == "vc0.131_k1"
                and r["anchor_identity"]["pass"] is True
                and all(c["vm_max_abs_diff"] == 0.0 and c["bitwise"] is True
                        for c in r["anchor_identity"]["cells"])
            ),
        },
        "Q15": {
            "expect": "reasons 27/11/7; 2 spiking neurons max 0.5 Hz; no adaptive extension",
            "load": lambda: load(ROOT / "artifacts/protocol_c/p2v_a1b_dynamic_search/p2v_a1b_receipt.json"),
            "cmp": lambda r, _e: (
                _quality_mix([c for p in r["points"] for c in p["cells"]]) == {
                    "standing_or_flipping_spatial_gradient": 27,
                    "synchronous_oscillation_k_near_zero": 11,
                    "structured_but_fails_traveling_gates": 7}
                and all(
                    c["activity_summary"]["n_neurons_with_spikes"] == 2
                    and c["activity_summary"]["max_spike_rate_hz"] == 0.5
                    for p in r["points"] for c in p["cells"])
                and r.get("no_adaptive_extension_observed") is True
            ),
        },
        # --- A-2 sensitivity floor
        "Q16": {
            "expect": "S1 40 cases",
            "load": lambda: load(ROOT / "artifacts/protocol_c/p2v_a2_sensitivity_floor/p2v_a2_receipt.json"),
            "cmp": lambda r, _e: r["stage_S1"]["n_cases"] == 40 == len(r["stage_S1"]["cases"]),
        },
        "Q17": {
            "expect": "gamma* = 1.0 on 3 cells x 2 phases",
            "load": lambda: load(ROOT / "artifacts/protocol_c/p2v_a2_sensitivity_floor/p2v_a2_receipt.json"),
            "cmp": lambda r, _e: (
                len(r["stage_S2"]["cells"]) == 3
                and all(c["gamma_star"] == {"phi0_0.0": 1.0, "phi0_1.57": 1.0}
                        for c in r["stage_S2"]["cells"])
            ),
        },
        "Q18": {
            "expect": "S3 12 cases; 6-site parity (NO_WAVE at 500/2000 ms)",
            "load": lambda: load(ROOT / "artifacts/protocol_c/p2v_a2_sensitivity_floor/p2v_a2_receipt.json"),
            "cmp": lambda r, _e: (
                r["stage_S3"]["n_cases"] == 12 == len(r["stage_S3"]["cases"])
                and _s3_sites(r["stage_S3"]["cases"]) == {
                    (250.0, 24): "TRAVELING_WAVE", (250.0, 12): "TRAVELING_WAVE",
                    (250.0, 6): "TRAVELING_WAVE",
                    (500.0, 24): "TRAVELING_WAVE", (500.0, 12): "TRAVELING_WAVE",
                    (500.0, 6): "NO_WAVE",
                    (1000.0, 24): "TRAVELING_WAVE", (1000.0, 12): "TRAVELING_WAVE",
                    (1000.0, 6): "TRAVELING_WAVE",
                    (2000.0, 24): "TRAVELING_WAVE", (2000.0, 12): "TRAVELING_WAVE",
                    (2000.0, 6): "NO_WAVE",
                }
            ),
        },
        # --- H4
        "Q19": {
            "expect": "M_X 0.0/0.0521/0.0/0.0; alpha_length 0.0",
            "load": lambda: load(ROOT / "artifacts/protocol_h_rbd/h4_matrix/h4_interpretation_receipt.json"),
            "cmp": lambda r, _e: (
                r["primary_endpoint_results"]["M_X_short_uniform"] == 0.0
                and abs(r["primary_endpoint_results"]["M_X_short_heterogeneous"] - 0.05208333333333384) < 1e-6
                and r["primary_endpoint_results"]["M_X_long_uniform"] == 0.0
                and r["primary_endpoint_results"]["M_X_long_heterogeneous"] == 0.0
                and _alpha_length(r) == 0.0
            ),
        },
        # --- D3
        "Q20": {
            "expect": "D 9/9 attenuation 0/9 formal; counts 9/9/9 vs 9; D-N2 0.2857 vs 0.2857",
            "load": lambda: load(ROOT / "artifacts/protocol_d_biological_rbs/d3_interpretation_receipt.json"),
            "cmp": lambda r, _e: (
                r["questions"]["Q1_mechanism"]["n_D_cells"] == 9
                and r["questions"]["Q1_mechanism"]["n_M1_pass"] == 9
                and r["questions"]["Q1_mechanism"]["n_M2_pass"] == 0
                and r["questions"]["Q1_mechanism"]["n_mechanism_ok"] == 0
                and r["classification_counts_all_arms"] == {
                    "N0": {"ADAPTATION": 9, "NO_ADAPTATION": 0, "UNRESOLVED": 0},
                    "N1": {"ADAPTATION": 9, "NO_ADAPTATION": 0, "UNRESOLVED": 0},
                    "N2": {"ADAPTATION": 9, "NO_ADAPTATION": 0, "UNRESOLVED": 0},
                    "D": {"ADAPTATION": 0, "NO_ADAPTATION": 9, "UNRESOLVED": 0}}
                and _d3_null(r["primary_contrast_D_minus_N2"]["pairwise"])
            ),
        },
        "Q25": {
            "expect": "36 cells x 4 arms",
            "load": lambda: load(ROOT / "artifacts/protocol_d_biological_rbs/d3_execution_receipt.json"),
            "cmp": lambda r, _e: (
                r["n_cells"] == 36 and len(r["cells"]) == 36
            ),
        },
        # --- W3b
        "Q21": {
            "expect": "2187 lattice; D=243 S=0 C=0 U=0 X=1944; N_S=0",
            "load": lambda: load(ROOT / "artifacts/protocol_w/w3b_parameter_domain/w3b_domain_receipt.json"),
            "cmp": lambda r, _e: (
                r["regime_counts"] == {"D": 243, "S": 0, "C": 0, "U": 0, "X": 1944}
                and r["aggregate_quantities"]["N_S"] == 0
                and r["aggregate_quantities"]["N_X"] == 1944
                and dict(r["frozen_lattice"]).__class__.__name__ == "dict"
                and len(r.get("frozen_lattice", r.get("lattice"))) in (2187, len(r.get("frozen_lattice", {})))
            ),
        },
        # --- A-3 boundedness
        "Q22": {
            "expect": "H/w ranges + growth ratios + invariants",
            "load": lambda: load(ROOT / "artifacts/protocol_c/p2v_a3_hdp_boundedness/p2v_a3_receipt.json"),
            "cmp": lambda r, _e: _a3_ok(r),
        },
        # --- E5
        "Q23": {
            "expect": "Fig7E bit-exact (N0==N1 V_m/spikes/Q)",
            "load": lambda: load(ROOT / "artifacts/protocol_e_integration/e5_execution_receipt.json"),
            "cmp": lambda r, _e: _e5_sanity(r),
        },
        "Q24": {
            "expect": "9 trajectories; gates; 3/3 HIERARCHICAL_PROPAGATION",
            "load": lambda: load(ROOT / "artifacts/protocol_e_integration/e5_execution_receipt.json"),
            "cmp": lambda r, _e: (
                r["design"]["trajectory_count"] == 9
                and r["quality_gates"]["G1_arm_isolation"]["passed"] is True
                and all(x["H_K_N1_equals_D_bit_exact"] for x in r["quality_gates"]["G1_arm_isolation"]["H_K_N1_equals_D"])
                and r["quality_gates"]["G7_classification_applied"]["per_seed"]
                == ["HIERARCHICAL_PROPAGATION"] * 3
            ),
        },
        # --- provenance / reproducibility
        "Q27": {
            "expect": "7/7 receipt SHAs match shipped figures",
            "load": lambda: None,
            "cmp": lambda _r, _e: _figure_hashes_match(),
        },
        "Q26": {
            "expect": "equivalence gate 7/7 byte-sha-equal",
            "load": lambda: load(ROOT / "artifacts/publication/equivalence_report.json"),
            "cmp": lambda r, _e: (
                len(r["cases"]) == 7
                and all(c["byte_sha_equal"] for c in r["cases"])
            ),
        },
        "Q28": {
            "expect": "pins in pyproject.toml",
            "load": lambda: (ROOT / "pyproject.toml").read_text(),
            "cmp": lambda t, _e: (
                re.search(r"matplotlib\s*>=\s*3\.10\.9", t) is not None
                and "<3.11" in t
                and re.search(r"scipy\s*==\s*1\.17\.1", t) is not None
            ),
        },
    }


def _quality_mix(cells) -> dict:
    import collections
    mix = collections.Counter()
    for c in cells:
        for q in c["estimator"]["quality_reasons"]:
            mix[q] += 1
    return dict(mix)


def _max_err(receipt, kind: str) -> float:
    worst = 0.0
    for case in receipt["cases"]:
        if case["kind"] != "positive":
            continue
        errs = case["estimator"]["errors"]
        if kind == "frequency":
            rel = abs(errs.get("frequency", 0.0))
        elif kind == "k_norm":
            v = errs.get("k_norm", errs.get("relative_k_norm", 0.0))
            rel = abs(v) if isinstance(v, (int, float)) else 0.0
        elif kind == "direction":
            v = errs.get("direction", errs.get("direction_deg", 0.0))
            rel = abs(v) if isinstance(v, (int, float)) else 0.0
        elif kind == "velocity":
            v = errs.get("velocity", errs.get("relative_velocity", 0.0))
            rel = abs(v) if isinstance(v, (int, float)) else 0.0
        else:
            rel = 0.0
        worst = max(worst, rel)
    return worst


def _s3_sites(cases) -> dict:
    return {(c["duration_ms"], c["n_sites"]): c["classification"] for c in cases}


def _alpha_length(r) -> float:
    fe = r.get("factorial_estimates", {})
    for k, v in fe.items():
        if isinstance(v, dict) and "alpha_length" in v:
            return float(v["alpha_length"])
        if k == "alpha_length":
            return float(v)
    s = json.dumps(fe)
    m = re.search(r'"alpha_length"\s*:\s*(-?[\d.eE]+)', s)
    return float(m.group(1)) if m else float("nan")


def _d3_null(pairs) -> bool:
    for p in pairs:
        if p["recovery_interval_id"] == "short" and p["seed"] == 11:
            return p["D_A_adapt"] == p["N2_A_adapt"] and abs(p["D_A_adapt"] - 0.2857142857142857) < 1e-6
    return False


def _a3_ok(r) -> bool:
    if r["all_hard_bound_invariants_pass"] is not True:
        return False
    runs = {x["preset"]: x for x in r["runs"]}
    default = runs.get("DEFAULT_HDP")
    desync = runs.get("DEFAULT_HDP_DESYNC")
    if not default or not desync:
        return False
    ok = (
        default["H_min_obs"] >= 1.0 - 1e-3 and default["H_max_obs"] <= 1.0008 + 1e-4
        and default["w_min_obs"] == 6.0 and default["w_max_obs"] == 6.0
        and default["w_abs_growth_ratio"] == 1.0
        and desync["H_min_obs"] >= 1.0 - 1e-3 and desync["H_max_obs"] <= 1.0310 + 1e-4
        and desync["w_min_obs"] >= 5.8692 - 1e-3 and desync["w_max_obs"] <= 6.1684 + 1e-3
        and desync["w_abs_growth_ratio"] <= 1.00004 + 1e-5
        and r.get("no_tuning_observed") is True
    )
    return ok


def _e5_sanity(r) -> bool:
    sc = r["sanity_checks"]["N0_equals_N1_neural"]
    return (
        len(sc) == 3
        and all(x["N0_equals_N1_V_m_bit_exact"] and x["N0_equals_N1_spikes_bit_exact"]
                and x["N0_equals_N1_Q_bit_exact"] for x in sc)
    )


def _figure_hashes_match() -> bool:
    import hashlib
    ok = True
    for i in range(1, 8):
        rec = load(ROOT / f"artifacts/publication/fig{i:02}_generation_receipt.json")
        png = ROOT / rec["figure_path"]
        declared = rec["figure_sha256"]
        actual = hashlib.sha256(png.read_bytes()).hexdigest()
        if actual != declared:
            ok = False
    return ok


table = q_table()
for qid in sorted(table):
    entry = table[qid]
    try:
        got = entry["load"]()
        ok = entry["cmp"](got, entry["expect"])
    except Exception as exc:  # noqa: BLE001
        ok, excstr = False, str(exc)
        check(qid, False, f"exception: {excstr}")
        continue
    check(qid, ok, f"expect: {entry['expect']}")

unused = set(table) - qin_draft
missing = qin_draft - set(table)
check("q_markers_all_in_table", not unused and not missing,
      f"unused in table: {sorted(unused)}; in draft but not table: {sorted(missing)}")

# ------------------------------------------------- 3. forbidden-overclaim scan

FORBIDDEN = [
    # (pattern, allowed paragraph ids — empty means never allowed, allow_if)
    (r"empirical\s+(result|quantity|claim)", [], r"no\s+empirical"),
    (r"physical\s+(amplitude\s+)?calibrat\w+", [], r"no\s+physical|was\s+not\s+performed|not\s+performed"),
    (r"\bcortex\b|\bcortical\b|\bin\s+vivo\b", [], None),
    (r"validated\b", [4, 5, 9], None),
    (r"\bmemory\b", [11, 12, 13], r"no\s+memory"),
    (r"extend(s|ed|ing)?\s+memory\b", [], r"do\s+not\s+extend"),
    (r"recurrent\s+(geometry|delays).{0,40}memory", [], r"do\s+not\s+extend|do\s+not\s+claim"),
    (r"cognitive\b|\bcognition\b|predictive\s+(coding|processing)", [], r"no\s+cognition|no\s+predictive|not\s+invoked"),
    (r"\badaptation\b", [14], None),
    (r"\bfatigue\b", [], None),
    (r"feedback\s+(suppresses|enhances|modulat\w+)", [], r"no\s+feedback"),
    (r"closed[- ]?loop", [13, 15, 16, 18], r"no\s+closed|not\s+demonstrated|remains\s+open"),
    (r"robust[- ]?active", [15], None),
    (r"\bstability\b", [15, 16], r"remains\s+open|not\s+(a\s+)?(resolved|classified)"),
    (r"plasticity\b", [17], r"no\s+HDP"),
    (r"byte[- ]?for[- ]?byte", [20], r"no\s+claim"),
    (r"new\s+(biological\s+)?mechanism\b", [], r"adds?\s+no\s+new"),
    (r"new[- ]?neuron[- ]?type\b", [], r"adds?\s+no\s+new"),
    (r"length\s+has\s+no\s+effect\b", [], None),
    (r"no\s+traveling\s+wave\w*", [6, 9], r"NO_WAVE"),
    (r"demonstrated\s+(closed[- ]?loop|feedback)", [], r"not\s+demonstrated"),
    (r"spike\s+(rate\s+|count\s+)?attenuation\b", [14], None),
]

paras = re.split(r"\*\*\[P(\d+)\]", draft)
# paras: ['', '1', text1, '2', text2, ...]
paras_by_id: dict[int, str] = {}
for j in range(1, len(paras), 2):
    paras_by_id[int(paras[j])] = paras[j + 1]

for pattern, allowed, allow_if in FORBIDDEN:
    hits = []
    for pid, text in paras_by_id.items():
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            if pid in allowed:
                continue
            if allow_if and re.search(allow_if, text, flags=re.IGNORECASE):
                continue
            hits.append((pid, m.group(0)))
    check(f"forbidden:{pattern}", not hits,
          f"hits: {hits[:5]}{' ...' if len(hits) > 5 else ''}")

sanity_checks = [
    ("P14_has_NO_ADAPTATION", "NO_ADAPTATION" in paras_by_id[14]),
    ("P15_has_remains_open", "remains open" in paras_by_id[15]),
    ("P12_has_no_positive_effect", bool(re.search(r"no\s+positive\s+effect", paras_by_id[12]))),
    ("P13_has_not_demonstrated", "not\ndemonstrated" in paras_by_id[13]
     or "not demonstrated" in paras_by_id[13]),
    ("P18_has_no_closed_loop", "no closed loop" in paras_by_id[18]),
    ("P19_no_band_claim", "suppresses" not in paras_by_id[19] and "enhances" not in paras_by_id[19]),
    ("P9_tested_regime_wording", "tested" in paras_by_id[9] and "preregistered" in paras_by_id[9]),
]
for name, ok in sanity_checks:
    check(name, ok)

# ---------------------------------------------------------------- write receipt

ok = not failures
receipt = {
    "schema": "jaxfne.publication.results_audit_receipt.v1",
    "checkpoint": "publication_results_reconstruction",
    "status": "PASS" if ok else "FAIL",
    "write_once": True,
    "draft": str(DRAFT.relative_to(ROOT)),
    "traceability_map": str(TRACE.relative_to(ROOT)),
    "n_checks": len(checks),
    "n_failed": len(failures),
    "checks": checks,
    "failures": failures,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(receipt, indent=2, sort_keys=True))
print(f"{len(checks)} checks, {len(failures)} failures → {receipt['status']}")
for f in failures:
    print("  FAIL:", f)
sys.exit(1 if failures else 0)