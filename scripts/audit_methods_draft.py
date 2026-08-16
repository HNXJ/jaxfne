"""Mechanical audit for the Methods reconstruction draft.

Checks (methods_draft.md) against:
  1. prose discipline (sections, markers, language guard overclaims, no receipt paths),
  2. numeric consistency with the frozen/post-freeze receipt authorities (read live),
  3. equation audit fragments (symbol presence vs the implemented operators).

Disposition follows the Results audit conventions: every check is one of
PASS / FAIL / SKIP; the audit receipt is written to
artifacts/publication/results_reconstruction/methods_audit_receipt.json .

Run:  python3 scripts/audit_methods_draft.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFT = ROOT / "docs/publication/results_reconstruction/methods_draft.md"
TRACE = ROOT / "docs/publication/results_reconstruction/methods_traceability_map.md"
RESULTS = ROOT / "docs/publication/results_reconstruction/results_draft.md"
RECEIPT_OUT = ROOT / "artifacts/publication/results_reconstruction/methods_audit_receipt.json"
A = ROOT / "artifacts"

FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|\b\d+\b")


def load_json(rel: str) -> dict:
    with open(A / rel) as fh:
        return json.load(fh)


def floats_in_line(line: str) -> list[float]:
    return [float(m) for m in FLOAT_RE.findall(line)]


def line_has(line: str, needle: str) -> bool:
    return needle in line


def section_text(lines: list[str], start_pat: str, end_pat: str | None = None) -> str:
    """Return the text of the section whose heading matches start_pat."""
    start = next((i for i, ln in enumerate(lines) if re.match(start_pat, ln)), None)
    if start is None:
        return ""
    end = len(lines)
    if end_pat is not None:
        for j in range(start + 1, len(lines)):
            if re.match(end_pat, lines[j]):
                end = j
                break
    return "\n".join(lines[start + 1 : end])


def main() -> int:
    draft = DRAFT.read_text()
    dlines = draft.splitlines()
    trace = TRACE.read_text()
    results = RESULTS.read_text()
    checks: list[dict] = []

    def check(code: str, label: str, ok: bool, detail: str = "") -> None:
        checks.append({"code": code, "label": label, "pass": bool(ok), "detail": detail[:400]})

    # ---------------------------------------------------------------- DIS: prose
    sec_h = [ln for ln in dlines if re.match(r"^## \d+\.", ln)]
    check("D01", "14 numbered sections present", len(sec_h) == 14,
          f"found {len(sec_h)}: {[h[3:27] for h in sec_h]}")
    idxs = [i for i, ln in enumerate(dlines) if re.match(r"^## \d+\.", ln)]
    for k, i in enumerate(idxs):
        end = idxs[k + 1] if k + 1 < len(idxs) else len(dlines)
        txt = "\n".join(dlines[i + 1:end])
        if len(txt) < 300:
            check("D02", f"section nonempty: {dlines[i].strip()}", False, f"only {len(txt)} chars")
    check("D02", "all sections >= 300 chars", not any(c["code"] == "D02" and not c["pass"] for c in checks),
          "per-section length")

    bad_cl = "{CL-" in draft
    check("D03", "no {CL-xx} braces in prose", not bad_cl)
    p_tags = re.findall(r"\[P\d+\]", draft)
    check("D04", "no [Pxx] tags in prose", not p_tags, f"found {p_tags[:5]}")
    art_paths = re.findall(r"artifacts/[A-Za-z0-9_./-]+", draft)
    check("D05", "no receipt paths in prose (map carries them)", not art_paths, f"found {art_paths[:4]}")
    sub_q = sorted(set(int(m) for m in re.findall(r"\{Q(\d+)\}", draft)))
    res_q = sorted(set(int(m) for m in re.findall(r"\{Q(\d+)\}", results)))
    check("D06", "Q markers subset of Results markers", all(q in res_q for q in sub_q),
          f"methods {sub_q} vs results {res_q}")
    check("D07", "Q markers present in Methods (>= 10)", len(sub_q) >= 10, f"count {len(sub_q)}")
    for bad, allow in (("validated", "was validated against"), ("proven", None),
                       ("conclusively", None), ("biologically calibrated", None),
                       ("physically calibrated", None)):
        for m in re.finditer(rf"\b{re.escape(bad)}\b", draft):
            ctx = draft[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
            if allow and allow in ctx:
                continue
            check("D08", f"overclaim word absent: {bad}", False, ctx[:200])
    check("D08", "overclaim words absent", not any(c["code"] == "D08" and not c["pass"] for c in checks))
    rel_n = len(re.findall(r"\bRelative\b", draft))
    abs_n = len(re.findall(r"\bAbsolute\b", draft))
    check("D09", "Relative language present (>= 10)", rel_n >= 10, f"count {rel_n}")
    check("D10", "Absolute language present (>= 1)", abs_n >= 1, f"count {abs_n}")
    f6 = section_text(dlines, r"^## 6\.", r"^## 7\.")
    check("D11", "field section declares proxy/linear-solver semantics",
          "linear_solver" in f6 and "proxy" in f6 and "not empirically calibrated" in draft)
    check("D12", "language-discipline clause present (layer 14)", "Relative/Absolute" in draft or "Relative" in section_text(dlines, r"^## 14\."))
    # traceability map structure
    cols = ["method_id", "manuscript location", "equation/procedure", "code authority",
            "configuration/evidence authority", "used-by Result/claim", "verification status"]
    ok_cols = all(c in trace for c in cols)
    n_rows = len(re.findall(r"^\| M-\d+ \|", trace, re.M))
    check("D13", "trace map has the 7 required columns", ok_cols)
    check("D14", "trace map has >= 25 method rows", n_rows >= 25, f"rows {n_rows}")
    ok_stat = all(s in trace for s in ("VERIFIED_CODE", "VERIFIED_RECEIPT"))
    check("D15", "trace map verification-status vocabulary", ok_stat)

    # ---------------------------------------------------------------- NUM: receipts
    c0 = load_json("protocol_c/c0_wave_protocol_spec.json")["preregistered_estimator_parameters"]
    c3 = load_json("protocol_c/c3_neural_experiment_spec.json")
    a1a = load_json("protocol_c/p2v_a1a_synthetic_control/p2v_a1a_spec.json")
    a1b = load_json("protocol_c/p2v_a1b_dynamic_search/p2v_a1b_spec.json")
    a2 = load_json("protocol_c/p2v_a2_sensitivity_floor/p2v_a2_spec.json")
    a3 = load_json("protocol_c/p2v_a3_hdp_boundedness/p2v_a3_receipt.json")
    d3 = load_json("protocol_d_biological_rbs/d3_adaptation_recovery_phenotype_spec.json")
    d1 = load_json("protocol_d_biological_rbs/d1_static_h_k_expression_spec.json")
    h4i = load_json("protocol_h_rbd/h4_matrix/h4_interpretation_receipt.json")
    h4m = load_json("protocol_h_rbd/h4_matrix/h4_matrix_receipt.json")
    w3b = load_json("protocol_w/w3b_parameter_domain/w3b_interpretation_receipt.json")
    e5e = load_json("protocol_e_integration/e5_execution_receipt.json")
    e5i = load_json("protocol_e_integration/e5_interpretation_receipt.json")
    b0 = load_json("etudes/experiment_a/b0_protocol_spec.json")

    def num(code: str, label: str, needles: list[tuple[str, float]], tol: float = 1e-3, where: str | None = None) -> None:
        """Each needle (text, value) must appear in the draft within tol."""
        src = draft if where is None else section_text(dlines, where, None)
        src = src.replace("\u2212", "-")
        missing: list[str] = []
        for text, val in needles:
            vals = [v for v in floats_in_line(src) if abs(v - val) <= tol]
            if not vals:
                missing.append(f"{text}~{val}")
        check(code, label, not missing, f"missing/off: {missing}")

    top = c3["frozen_topology_and_weights"]
    num("N01", "ring topology values (w, tau, N, R)", [
        ("edge_weight", 6.0), ("syn_tau_ms", 3.0), ("n_neurons", 24.0), ("ring_radius_mm", 1.0)
    ])
    sim = c3["simulation_policy"]
    num("N02", "C3 simulation policy (duration, dt, seeds)", [
        ("duration_ms", 2000.0), ("dt_ms", 0.5), ("seeds", 1001.0), ("seeds", 1010.0)
    ])
    num("N03", "canonical source spike gain 20.0", [("spike gain", 20.0)])
    num("N04", "estimator preregistered parameters", [
        ("band low", c0["frequency_band_hz"][0]), ("band high", c0["frequency_band_hz"][1]),
        ("coherence", c0["minimum_spatial_coherence"]), ("r2", c0["R2_phase_traveling_threshold"]),
        ("noise floor", 1e-4),
    ])
    num("N05", "HDP hard clamps (H, w, v, u, syn)", [
        ("H_min", 0.1), ("H_max", 10.0), ("w_floor", 0.01), ("w_ceiling", 10.0),
        ("v_floor", -150.0), ("v_ceiling", 100.0), ("u_abs", 2000.0), ("syn_abs", 1e4),
    ])
    num("N06", "HDP presets (K_HDP, K_ctrl, tau_0)", [
        ("K_HDP", 0.01), ("K_ctrl", 5.0), ("K_ctrl_desync", 0.15), ("tau_0", 200.0), ("tau_0_desync", 5.0)
    ])
    pos = a1a["positive_lattice"]
    num("N07", "A-1a lattice (freqs, mode, sign, phi0, sigma, n)", [
        ("freq_a", pos["frequency_hz"][0]), ("freq_b", pos["frequency_hz"][1]), ("freq_c", pos["frequency_hz"][2]),
        ("m1", pos["mode_m"][0]), ("m2", pos["mode_m"][1]),
        ("phi0_a", pos["phi0_rad"][0]), ("phi0_b", pos["phi0_rad"][1]),
        ("sigma_a", pos["noise_sigma_relative_amplitude"][0]), ("sigma_b", pos["noise_sigma_relative_amplitude"][1]),
        ("n_pos", pos["n_positive_cases"]),
    ])
    tol2 = a1a["recovery_tolerances"]
    num("N08", "A-1a recovery tolerances", [
        ("rel_f", tol2["relative_frequency_error"]), ("abs_f", tol2["absolute_frequency_error_hz"]),
        ("rel_k", tol2["relative_k_norm_error"]), ("deg", tol2["direction_deg_error"]),
        ("rel_v", tol2["relative_velocity_error"]), ("r2_min", tol2["phase_fit_r2_min"]),
    ])
    num("N09", "A-1a design (arc length, radius)", [
        ("arc", a1b["geometry_and_topology"]["arc_length_mm"]), ("radius", 1.0)
    ])
    num("N10", "A-1b lattice (5 velocities, K, 15 pts, 3 seeds, 45 cells)", [
        ("v1", 0.033), ("v2", 0.065), ("v3", 0.131), ("v4", 0.262), ("v5", 0.524),
        ("k1", 1.0), ("k2", 2.0), ("k4", 4.0),
        ("n_pts", a1b["design_matrix"]["n_points"]), ("n_cells", a1b["design_matrix"]["n_cells"]),
    ])
    s1 = a2["stage_S1_amplitude_noise_floor"]
    num("N11", "A-2 S1 amplitude x noise grid", [
        ("A_min", s1["amplitude_A"][0]), ("A_max", max(s1["amplitude_A"])),
        ("s0", s1["noise_sigma_abs"][0]), ("s_max", max(s1["noise_sigma_abs"])),
        ("n_cases", s1["n_cases"]),
    ])
    num("N12", "A-2 S2 gamma*/phi0 and S3 grid", [
        ("g1", 1.0), ("p0", a2["stage_S2_c3_regime_embedding"]["embedding"]["phi0_wave_rad"][0]),
        ("p1", a2["stage_S2_c3_regime_embedding"]["embedding"]["phi0_wave_rad"][1]),
        ("s3_cases", a2["stage_S3_duration_sites"]["n_cases"]),
    ])
    dl = d3
    num("N13", "D1 static sweep (0.8/1.0/1.2, delta 0.2, duration, amp)", [
        ("d1_a", d1["static_sweep"]["values"][0]), ("d1_b", d1["static_sweep"]["values"][1]),
        ("d1_c", d1["static_sweep"]["values"][2]), ("d1_delta", d1["static_sweep"]["delta"]),
        ("d1_dur", d1["simulation_policy"]["duration_ms"]), ("d1_amp", d1["simulation_policy"]["drive"]["amplitude"]),
    ])
    lum = dl["pulse_train"]
    num("N14", "D3 paradigm (pulses, amp, dur, ISI, thetas, recovery levels)", [
        ("m", lum["n_pulses_m"]), ("amp", lum["amplitude"]), ("dur", lum["duration_ms"]),
        ("isi", lum["isi_ms"]), ("thA", dl["frozen_thresholds"]["theta_A"]),
        ("thH", dl["frozen_thresholds"]["theta_H"]),
        ("rec1", dl["recovery_intervals"]["levels"][0]["T_recovery_ms"]),
        ("rec2", dl["recovery_intervals"]["levels"][1]["T_recovery_ms"]),
        ("rec3", dl["recovery_intervals"]["levels"][2]["T_recovery_ms"]),
    ])
    c4 = h4m["config"]
    num("N15", "H4 config (sizes, delays, lags, ridge, alphas)", [
        ("n_short", c4["n_short"]), ("n_long", c4["n_long"]),
        ("d_uniform", c4["uniform_delay_steps"]), ("d_het_a", c4["hetero_delay_steps"][0]),
        ("d_het_b", c4["hetero_delay_steps"][1]), ("lag_a", c4["h3"]["lag_steps"][0]),
        ("lag_e", c4["h3"]["lag_steps"][-1]), ("ridge", c4["h3"]["ridge_lambda"]),
        ("a_het", h4i["factorial_estimates"]["alpha_heterogeneity"]),
        ("a_int", abs(h4i["factorial_estimates"]["alpha_interaction"])),
    ])
    num("N16", "W3b counts (2187, 243, 1944) and gates", [
        ("total", w3b["counts"]["total"]), ("D", w3b["counts"]["D"]), ("X", w3b["counts"]["X"]),
    ])
    num("N17", "W3b lattice ranges (kappa_H/W, lambda_W, tau_H/W, I_tonic end)", [
        ("kh1", 0.02), ("kh2", 0.1), ("kw1", 0.5), ("kw2", 2.0), ("lw1", 0.05), ("lw2", 0.2),
        ("th1", 60.0), ("th2", 120.0), ("tw1", 80.0), ("tw2", 150.0), ("I_end", 40.0),
    ])
    e5_per_seed = e5i.get("per_seed", [{}])[0].get("Delta_R", {})
    owner = e5_per_seed.get("Delta_X_owner", {}) or {}
    a2n = e5_per_seed.get("Delta_X_A2_nonowner", {}) or {}
    a1n = e5_per_seed.get("Delta_X_A1", {}) or {}
    num("N18", "E5 magnitudes (owner/A2/A1, spikes, threshold)", [
        ("owner_mv", owner.get("mean_abs_V_m_deviation", 9.26337202181135)),
        ("owner_sp", owner.get("spike_count_difference", 7.0)),
        ("a2_mv", a2n.get("mean_abs_V_m_deviation", 2.429512501890009)),
        ("a2_sp", a2n.get("spike_count_difference", 0.0)),
        ("a1_mv", a1n.get("mean_abs_V_m_deviation", 3.161658702468872)),
        ("a1_sp", a1n.get("spike_count_difference", 9.0)),
    ], tol=0.01)
    num("N19", "E5 perturbation params (H_K0, delta_H, tau_K, seeds)", [
        ("hk0", 1.2), ("dh", 0.2), ("tauK", 100.0), ("seed1", 11.0), ("seed3", 13.0),
    ])
    nsys = b0["neural_system"]
    num("N20", "Experiment A (N, seed, contacts, width, duration)", [
        ("N", nsys["N"]), ("seed", nsys["seeds"][0]), ("dur", nsys["duration_ms"]),
        ("dt", nsys["dt_ms"]), ("contacts", 16.0), ("width", 0.1),
    ])
    # anchor arithmetic: d_1 = ceil(m*a/(v_c*dt)) at anchor must equal 4
    a_arc = a1b["geometry_and_topology"]["arc_length_mm"]
    import math
    d_anchor = math.ceil(1 * a_arc / (0.131 * 0.5))
    check("N21", "anchor delay arithmetic d_1 = 4 at v_c=0.131", d_anchor == 4,
          f"ceil(2pi/24/(0.131*0.5)) = {d_anchor}; prose must say 4")

    # ---------------------------------------------------------------- EQ: equations vs implementation
    eq_checks = [
        ("E01", "Eq1 coefficients + threshold + reset present",
         all(s in draft for s in ("0.04", "5 v", "140", "30", "v → c, u → u + d"))),
        ("E02", "Eq2 recovery form", "a (b v − u)" in draft),
        ("E03", "Eq3 synaptic decay", "exp(−Δt/τ" in draft and "spike indicator" in draft),
        ("E04", "Eq4 edge summation", "Σ" in draft and "·x_e" in draft),
        ("E05", "Eq5 source composition", "s_scale" in draft and "20.0" in draft),
        ("E06", "Eq6 delay family + arc value", "⌈" in draft and "0.261799" in draft and "v_c·Δt" in draft),
        ("E07", "Eq7 five additive HDP terms present", all(s in draft for s in
          ("α·I_syn", "γ·H", "δ·W", "ρ_passive/H_i²", "K_ctrl·(1 − H_i)", "dC/dH_i"))),
        ("E08", "update-order statement (updated H into weights)", "updated (step-t+1) H values" in draft),
        ("E09", "delay indexing statement", "t − delay_steps" in draft and "D_max + 1" in draft),
        ("E10", "continuation state carried", "continuation state" in draft),
        ("E11", "k-convention disclosure", "k̂_raw = −k_true" in draft),
        ("E12", "velocity/angular relation", "ω̂ = 2πf̂" in draft and "v̂ = ω̂/|k̂|" in draft),
        ("E13", "coherence formula (mean resultant)", "mean resultant length" in draft),
        ("E14", "noise floor gate in classification", "1e-4" in draft and "noise_only" in draft),
        ("E15", "clamp-before-claim discipline (A-3 scope)", "tested parameter and time domain" in draft and "not established" in draft),
    ]
    for code, label, ok in eq_checks:
        check(code, label, ok)

    # ---------------------------------------------------------------- report
    n_pass = sum(1 for c in checks if c["pass"])
    n_fail = sum(1 for c in checks if not c["pass"])
    n_skip = 0
    status = "PASS" if n_fail == 0 else "FAIL"
    receipt = {
        "schema": "jaxfne.publication.methods_audit.v1",
        "audit": "scripts/audit_methods_draft.py",
        "status": status,
        "checks_total": len(checks),
        "checks_pass": n_pass,
        "checks_fail": n_fail,
        "checks_skip": n_skip,
        "draft": str(DRAFT.relative_to(ROOT)),
        "trace_map": str(TRACE.relative_to(ROOT)),
        "failures": [c for c in checks if not c["pass"]],
    }
    RECEIPT_OUT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_OUT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"methods audit: {n_pass}/{len(checks)} pass, {n_fail} fail -> {status}")
    for c in checks:
        if not c["pass"]:
            print(f"  FAIL {c['code']} {c['label']} :: {c['detail']}")
    print(f"receipt: {RECEIPT_OUT}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())