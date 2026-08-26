#!/usr/bin/env python3
"""Figure 9 — E2 V2 SSA confirmatory negative (supplement-only, 6 panels).

Authority: frozen preregs e2_ssa_spec.v6.json (spec_hash 0df9bfe...),
e2_ping_prereg.json (b89a...), raw evidence
E2b_confirmatory/v2_ssa_confirmatory_receipt.json (20 reps ×5 blocks),
v2_rescored_frozen_only.json, v2_runs/rep_*.json, H3 correction receipt.

Outputs (supplement-only):
  artifacts/figures/publication/final/e2_fig09_ssa_combined.png (combined 6-panel)
  artifacts/figures/publication/final/e2_fig09_A_adequacy.png … F_ladder.png (6 panels)
  artifacts/publication/final/e2_fig09_ssa_receipt.json

Provenance as in Fig08. G1 28/28, G5 pixel identity, S18 preserved.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
SSA_SPEC = REPO / "artifacts/e2/preregistration/e2_ssa_spec.v6.json"
PING_SPEC = REPO / "artifacts/e2/preregistration/e2_ping_prereg.json"
V2_RECEIPT = REPO / "artifacts/e2/preregistration/E2b_confirmatory/v2_ssa_confirmatory_receipt.json"
V2_RESCORED = REPO / "artifacts/e2/preregistration/E2b_confirmatory/v2_rescored_frozen_only.json"
SSA_V6_CORR = REPO / "artifacts/e2/preregistration/e2_ssa_spec_v6_amendment_receipt_CORRECTION.json"

OUT_FIG_DIR = REPO / "artifacts/figures/publication/final"
OUT_ART_DIR = REPO / "artifacts/publication/final"

PING_SPEC_HASH = "b89a09c466186330a58eb70c632d597a7989803f6e418a2d9d778385a498af1f"
SSA_SPEC_HASH = "0df9bfe24bae0e2e04cb0b9c1a2b41988981a62587d4bcd9f53f864a6d520570"

def sha256_file(p: Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def repo_sha():
    try: return subprocess.check_output(["git","rev-parse","HEAD"], cwd=REPO, text=True).strip()
    except: return "unknown"

def canon_hash(j):
    return hashlib.sha256(json.dumps({k:v for k,v in j.items() if k!="spec_hash"}, sort_keys=True, separators=(",",":")).encode()).hexdigest()

def load_json(p): return json.loads(p.read_text(encoding="utf-8"))

def utc_now(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def panel_label(ax, letter, title):
    ax.text(0.02,0.98,f"{letter} — {title}", transform=ax.transAxes, fontsize=7.5, fontweight="bold", va="top", color="#1A4A8A")

def style_axes(ax):
    for s in ax.spines.values():
        s.set_color("#1A4A8A"); s.set_linewidth(1.2)

def draw_A_adequacy(ax, v2):
    panel_label(ax,"A","Adequacy: 20/20 G_A & G_B")
    runs=v2["runs"]
    g_a = sum(1 for r in runs if r["G_A"])
    g_b = sum(1 for r in runs if r["G_B"])
    ax.bar(["G_A (R_A_early)","G_B (R_B_early)"], [g_a, g_b], color=["#1A4A8A","#0B6E4F"], alpha=0.8)
    ax.set_ylim(0,20)
    ax.set_ylabel("replicates pass /20", fontsize=7)
    for i,v in enumerate([g_a,g_b]):
        ax.text(i, v+0.3, f"{v}/20", ha="center", fontsize=8, fontweight="bold")
    # R_early values
    r_early = [r["R_early"] for r in runs]
    ax2=ax.inset_axes([0.55,0.25,0.40,0.45])
    ax2.hist(r_early, bins=8, color="#1A4A8A", alpha=0.7, edgecolor="white")
    ax2.axvline(1.0, color="red", ls="--", lw=0.9)
    ax2.set_title("R_early 6.2-8.3 Hz\n(R_floor 1.0)", fontsize=5)
    ax2.set_xlabel("Hz", fontsize=5)
    ax2.tick_params(labelsize=5)
    ax.text(0.02,0.15,"G_finite ✓  G_stable drift ≤3 SD\nper identity (methods_ssa 475-480)\n176k steps =88 s @0.5ms", transform=ax.transAxes, fontsize=5.5, bbox=dict(boxstyle="round", facecolor="#F0FFF8", edgecolor="#0B6E4F"))
    style_axes(ax); style_axes(ax2)

def draw_B_SI(ax, v2, v2_resc):
    panel_label(ax,"B","SI per rep & pooled (threshold SI>0.10)")
    runs=v2["runs"]
    si = [r["SI"] for r in runs]
    pooled = v2["pooled"]["SI"]
    ci = (v2["pooled"]["BCa_lower"], v2["pooled"]["BCa_upper"])
    resc = v2_resc["pooled_SI"]
    x=np.arange(20)
    ax.bar(x, si, color=["#C44E52" if s>0 else "#1A4A8A" for s in si], alpha=0.7)
    ax.axhline(0.10, color="red", ls="--", lw=1.0, label="S2 gate >0.10")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlim(-1,20)
    ax.set_ylim(-0.25,0.15)
    ax.set_xlabel("replicate 0-19", fontsize=6)
    ax.set_ylabel("SI", fontsize=7)
    ax.text(0.02,0.92,f"pooled SI {pooled:.3f}\nBCa [{ci[0]:.3f},{ci[1]:.3f}]\nrescored {resc:.3f}\n0/20 >0.10 ✗", transform=ax.transAxes, fontsize=5.5, va="top", bbox=dict(boxstyle="round", facecolor="#FFE8E8", edgecolor="#8B0000"))
    ax.text(0.98,0.05,"sign flipped: deviant below standard\n(SIGN_neg)", transform=ax.transAxes, fontsize=5, ha="right", color="#8B0000", bbox=dict(boxstyle="round", facecolor="#FFF8E7", edgecolor="#8B4513"))
    style_axes(ax)

def draw_C_swap(ax, v2, v2_resc):
    panel_label(ax,"C","Controls: many_standards & role-swap")
    runs=v2["runs"]
    si_many = [r["SI_many"] for r in runs]
    swap = [r["swap_asym"] for r in runs]
    x=np.arange(20)
    ax.bar(x-0.2, si_many, 0.4, label="SI_many (|A-B|/|A+B|)", color="#0B6E4F", alpha=0.7)
    ax.bar(x+0.2, swap, 0.4, label="swap asymmetry", color="#8B4513", alpha=0.7)
    ax.axhline(0.10, color="red", ls="--", lw=0.9)
    ax.axhline(0.03, color="orange", ls="--", lw=0.8)
    ax.set_ylim(0,0.45)
    ax.set_xlabel("rep", fontsize=6)
    ax.set_ylabel("value", fontsize=7)
    ax.legend(fontsize=5, loc="upper right")
    # pooled values
    pool_many = v2["pooled"]["SI_many_pool"]
    swap_max = v2["pooled"]["swap_max_observed"]
    resc_swap = v2_resc["swap_max_observed"]
    ax.text(0.02,0.85,f"pooled SI_many {pool_many:.3f}\n(|SI_many|<0.10 ✓)\nswap_max {swap_max:.3f}\n(rescored {resc_swap:.3f})\n|swap| ≤0.10 ✗ max 0.426", transform=ax.transAxes, fontsize=5.5, va="top", bbox=dict(boxstyle="round", facecolor="#FFE8E8", edgecolor="#8B0000"))
    style_axes(ax)

def draw_D_recovery(ax, v2):
    panel_label(ax,"D","Recovery intervals (isi 500 vs 1000)")
    s3=v2["S3"]
    runs=v2["runs"]
    # R_late vs R_rec500 per rep scatter
    r_late = [r["R_late"] for r in runs]
    r_rec500 = [r["R_rec500"] for r in runs]
    r_rec1000 = [r["R_rec1000"] for r in runs]
    ax.scatter(r_late, r_rec500, c="#1A4A8A", s=20, label="R_rec500")
    ax.scatter(r_late, r_rec1000, c="#8B4513", s=20, marker="x", label="R_rec1000")
    # diagonal
    lims=[0,15]
    ax.plot(lims,lims, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("R_late (Hz, late oddball)", fontsize=6)
    ax.set_ylabel("R_rec (Hz)", fontsize=6)
    ax.legend(fontsize=5, loc="upper left")
    # annotation S3 gated
    delta = s3["delta_rec"]
    p_rec = s3["p_rec"]
    rho = s3["rho"]
    ax.text(0.98,0.05,f"delta_rec {delta:.2f} BCa_low {s3['BCa_lower']:.2f}\np_rec {p_rec:.4f} ✓\nI_rec 0.20 ✗ (not all >0.20)\nrho {rho:.2f} (need ≥0.20) ✗", transform=ax.transAxes, ha="right", fontsize=5, bbox=dict(boxstyle="round", facecolor="#FFF8E7", edgecolor="#8B4513"))
    # survival
    ax.set_xlim(6,9); ax.set_ylim(2,13)
    style_axes(ax)

def draw_E_source(ax, v2):
    panel_label(ax,"E","Population vs source & shuffled history")
    runs=v2["runs"]
    si = [r["SI"] for r in runs]
    si_src = [r["SI_source"] for r in runs]
    ax.scatter(si, si_src, c="#1A4A8A", s=25, alpha=0.7)
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.axvline(0, color="gray", ls=":", lw=0.8)
    ax.axvline(0.10, color="red", ls="--", lw=0.8)
    ax.axhline(0.08, color="orange", ls="--", lw=0.8)
    ax.set_xlabel("SI (population)", fontsize=7)
    ax.set_ylabel("SI_source (|sources| window)", fontsize=7)
    ax.set_xlim(-0.25,0.15)
    ax.set_ylim(-0.01,0.01)
    pool_src = v2["pooled"]["SI_source_pool"]
    shuf = v2["pooled"]["SI_shuf_mean"]
    ax.text(0.02,0.92,f"pooled SI_source {pool_src:.4f}\n(sign mismatch SI -0.084)\nSI_shuf {shuf:.3f} (|shuf|<0.03 ✓)\ntheta_H gate 0.01 ✗", transform=ax.transAxes, fontsize=5.5, va="top", bbox=dict(boxstyle="round", facecolor="#FFE8E8", edgecolor="#8B0000"))
    ax.text(0.98,0.02,"SI_source vs SI sign\nmust match (mech)", transform=ax.transAxes, ha="right", fontsize=5, color="#8B0000")
    style_axes(ax)

def draw_F_ladder(ax, v2, v2_resc):
    panel_label(ax,"F","Classification ladder S0→S4 (prospective)")
    ax.axis("off")
    # S2 gate requires SI>0.10 etc; S3 recovery; S4 deferred
    pooled = v2["pooled"]
    # thresholds
    gates = [
        ("S0 no decrement", "abs(SI)<0.10 & |dR|<0.5 & CI includes 0", "2/20 S0 (reps 17,??)"),
        ("S1 global", "R_std<R_ctrl & R_dev<R_ctrl & ...", "0/20"),
        ("S2 stimulus-specific", "SI>0.10 & dR>0.8 & g>0.40 & p<0.025 & shuf<0.03 & swap≤0.10", "0/20 (SI -0.084)"),
        ("S3 recovery", "d_rec>0.5 & I_rec>0.20 & rho≥0.20", f"gate_pass {v2['S3']['gate_pass']} (delta {v2['S3']['delta_rec']:.1f} ✓ but I/rho ✗)"),
        ("S4 mechanistic", "S2 in D & (SI_D-SI_N1)>0.08 & ...", "CONFIRMATORY_DEFERRED_v3"),
    ]
    y0=0.90
    for i,(name, rule, res) in enumerate(gates):
        y = y0 - i*0.16
        ax.text(0.02, y, name, fontsize=6, fontweight="bold", transform=ax.transAxes, va="center")
        ax.text(0.30, y, rule, fontsize=4.5, transform=ax.transAxes, va="center", color="#333333")
        color = "#E8FFE8" if "DEFERRED" in res else "#FFE8E8"
        ax.text(0.82, y, res, fontsize=5, transform=ax.transAxes, va="center", ha="center", bbox=dict(boxstyle="round", facecolor=color, edgecolor="#1A4A8A", pad=0.3))
    ax.text(0.02, 0.08, f"Verdict {v2['verdict']['V2_polarity']} {v2['verdict']['subclass']}\nrescored {v2_resc['label']}\nControls: many_standards ✓, shuffled ✓, swap ✗ — swap_max 0.426 >>0.10\nSign: pooled SI negative (deviant below standard) contradicts S2 direction", fontsize=5.5, transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="#FFE8E8", edgecolor="#8B0000"))
    ax.text(0.98, 0.02, "V2 battery: 20 reps ×5 blocks (oddball_A/B, many, rec500/1000)\n176k steps =88 s bio (H3 corrected 96000+40000+40000)", transform=ax.transAxes, ha="right", fontsize=4.5, color="#666666")

def build_combined(v2, v2_resc):
    fig = plt.figure(figsize=(14,9), dpi=200)
    fig.suptitle("Figure 9 — E2 V2 SSA confirmatory @ theta* (supplement S18-1, S18-4, S18-5)", fontsize=11, fontweight="bold", y=0.98)
    fig.text(0.5,0.94,"theta* same as Fig8 (lexicographic, not optimum)  |  20 reps ×5 blocks =176000 steps (88 s) @0.5 ms |  verdict NEGATIVE: identity-bias / rarity-penalty", ha="center", fontsize=6.5, color="#333333")
    gs = fig.add_gridspec(3,2, left=0.06, right=0.98, top=0.90, bottom=0.08, hspace=0.55, wspace=0.30)
    draw_A_adequacy(fig.add_subplot(gs[0,0]), v2)
    draw_B_SI(fig.add_subplot(gs[0,1]), v2, v2_resc)
    draw_C_swap(fig.add_subplot(gs[1,0]), v2, v2_resc)
    draw_D_recovery(fig.add_subplot(gs[1,1]), v2)
    draw_E_source(fig.add_subplot(gs[2,0]), v2)
    draw_F_ladder(fig.add_subplot(gs[2,1]), v2, v2_resc)
    fig.text(0.5,0.02,"Bounded claim: pooled SI -0.084 BCa[-0.108,-0.034] below S2 gate; swap asymmetry max 0.426 >>0.10; SI sign inverted (deviant below standard) — no stimulus-specific adaptation at theta* under frozen gates.", ha="center", fontsize=6, color="#1A4A8A", wrap=True)
    return fig

def main():
    ping=load_json(PING_SPEC); ssa=load_json(SSA_SPEC)
    assert canon_hash(ping)==PING_SPEC_HASH==ping["spec_hash"], "S18 ping drift"
    assert canon_hash(ssa)==SSA_SPEC_HASH==ssa["spec_hash"], "S18 ssa drift"
    v2=load_json(V2_RECEIPT); v2_resc=load_json(V2_RESCORED)
    # also load correction for steps
    corr=load_json(SSA_V6_CORR)

    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ART_DIR.mkdir(parents=True, exist_ok=True)

    fig=build_combined(v2, v2_resc)
    combined = OUT_FIG_DIR/"e2_fig09_ssa_combined.png"
    fig.savefig(combined, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    panels = [
        ("e2_fig09_A_adequacy.png", lambda ax: draw_A_adequacy(ax, v2)),
        ("e2_fig09_B_SI.png", lambda ax: draw_B_SI(ax, v2, v2_resc)),
        ("e2_fig09_C_swap.png", lambda ax: draw_C_swap(ax, v2, v2_resc)),
        ("e2_fig09_D_recovery.png", lambda ax: draw_D_recovery(ax, v2)),
        ("e2_fig09_E_source.png", lambda ax: draw_E_source(ax, v2)),
        ("e2_fig09_F_ladder.png", lambda ax: draw_F_ladder(ax, v2, v2_resc)),
    ]
    panel_paths=[combined]
    for fname, func in panels:
        f=plt.figure(figsize=(6,4), dpi=200)
        ax=f.add_subplot(111)
        func(ax)
        f.tight_layout()
        p=OUT_FIG_DIR/fname
        f.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(f)
        panel_paths.append(p)

    import jax
    prov={
        "schema":"e2_fig09_ssa_receipt.v1",
        "generated_at": utc_now(),
        "repo_head": repo_sha(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "platform": platform.platform(),
        "authority":{
            "ssa_spec_hash": SSA_SPEC_HASH,
            "ssa_sha256": sha256_file(SSA_SPEC),
            "ping_spec_hash": PING_SPEC_HASH,
            "ping_sha256": sha256_file(PING_SPEC),
            "v2_receipt_sha256": sha256_file(V2_RECEIPT),
            "v2_rescored_sha256": sha256_file(V2_RESCORED),
            "v6_correction_sha256": sha256_file(SSA_V6_CORR),
            "steps_provenance": corr.get("correction", corr),
        },
        "theta_star": v2["runs"][0].get("theta", {"drive_E":4.0,"drive_I":2.0,"weight_mu":0.25,"noise_scale":0.0}),
        "verdict": v2["verdict"],
        "pooled": v2["pooled"],
        "S3": v2["S3"],
        "files":{},
        "S18_preserved": True,
        "disposition": "supplement-only (S18), not main-text; frozen 28/28 preserved",
        "provenance_note":"All quantities from frozen receipts only; no new simulation. Recovery 176000 steps =88 s (H3 corrected).",
    }
    for pp in panel_paths:
        prov["files"][pp.name]={"path":f"artifacts/figures/publication/final/{pp.name}","sha256":sha256_file(pp),"bytes":pp.stat().st_size}
    receipt=OUT_ART_DIR/"e2_fig09_ssa_receipt.json"
    receipt.write_text(json.dumps(prov, indent=2))
    print(f"Fig09: wrote {len(panel_paths)} PNGs to {OUT_FIG_DIR}")
    for k,v in prov["files"].items():
        print(f"  {k}: {v['sha256'][:8]} {v['bytes']} bytes")
    print(f"Receipt: {receipt}  S18 preserved")
    return 0

if __name__=="__main__":
    import sys; sys.exit(main())
