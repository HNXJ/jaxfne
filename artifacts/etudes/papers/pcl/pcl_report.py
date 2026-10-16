"""Build the PCL report (figures + one self-contained HTML page) from committed results.

Inputs: fig2.json (P0), fig2_hdp_avg.json (K0b), confirm/{c1,k1,c1b,k1h}_seed*.json, the K1h-nf and
O1 json, the K2 confirmatory json/npz, and report_data.npz from pcl_report_data.py. Every number in the page is read from
these files. Figures are SVG with text as text; black and the series colors are rewritten to CSS
variables so the page follows the viewer's light or dark theme.
"""
from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import welch  # noqa: E402

HERE = Path(__file__).resolve().parent
C1, C2, C3, C4 = "#1565c0", "#ff9800", "#00acc1", "#e53935"  # series colors (author defaults)
GREY = "#8a8a8a"
plt.rcParams.update({
    "svg.fonttype": "none", "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.4,
})
NEURONS = ("1 (90 %)", "2 (50 %)", "3 (10 %)")


def load(p):
    return json.loads((HERE / p).read_text())


def svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    s = buf.getvalue()
    s = s[s.index("<svg"):]
    for hexc, var in ((C1, "--c1"), (C2, "--c2"), (C3, "--c3"), (C4, "--c4"), (GREY, "--muted"),
                      ("#000000", "--ink"), ("#ffffff", "--bg")):
        s = s.replace(hexc, f"var({var})")
    s = re.sub(r'width="[\d.]+pt" height="[\d.]+pt"', 'width="100%"', s, count=1)
    return s


def letter(ax, c):
    ax.text(-0.18, 1.06, c, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def fig_network(d, p0, k0b):
    dt = float(d["f2_dt"])
    t0, t1 = 200, 500  # ms window
    sl = slice(int(t0 / dt), int(t1 / dt))
    t = np.arange(sl.start, sl.stop) * dt
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.4), constrained_layout=True)
    ax = axs[0, 0]
    for k in range(4):
        for name, col, off in (("no_inh", GREY, 0.18), ("pcl", C1, -0.18)):
            st = t[d[f"f2_s_{name}"][sl, k] > 0]
            ax.vlines(st, 3 - k + off - 0.14, 3 - k + off + 0.14, color=col, lw=1.0)
    ax.set_yticks([3, 2, 1, 0], ["0 (predictor)", "1 (90 %)", "2 (50 %)", "3 (10 %)"])
    ax.set_xlabel("time (ms)")
    ax.set_title("Spikes, grey without and blue with inhibition", loc="left")
    letter(ax, "A")
    ax = axs[0, 1]
    w0, w1 = int(300 / dt), int(400 / dt)
    tt = np.arange(w0, w1) * dt
    ax.plot(tt, d["f2_v_no_inh"][w0:w1, 1], color=GREY, lw=1.0, label="no inhibition")
    ax.plot(tt, d["f2_v_pcl"][w0:w1, 1], color=C1, lw=1.0, label="learned inhibition")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("v, neuron 1 (mV)")
    ax.legend(loc="upper center")
    ax.set_title("Membrane potential of the most predictable neuron", loc="left")
    letter(ax, "B")
    x = np.arange(3)
    ax = axs[1, 0]
    ax.bar(x - 0.2, p0["suppression_mean"], 0.38, color=C2, label="standalone LIF (P0)")
    ax.bar(x + 0.2, k0b["suppression_mean"], 0.38, color=C1, label="jaxfne kernel (K0b)")
    ax.set_xticks(x, NEURONS)
    ax.set_xlabel("neuron (predictable fraction)")
    ax.set_ylabel("spikes removed (fraction)")
    ax.legend()
    ax.set_title("Suppression, mean of 3 networks", loc="left")
    letter(ax, "C")
    ax = axs[1, 1]
    ax.bar(x - 0.2, p0["w_end_mean"], 0.38, color=C2, label="standalone, final")
    ax.bar(x + 0.2, k0b["w_avg_mean_mV"], 0.38, color=C1, label="kernel, epoch average")
    ax.set_xticks(x, NEURONS)
    ax.set_xlabel("neuron (predictable fraction)")
    ax.set_ylabel("inhibitory weight (mV-equivalent)")
    ax.legend()
    ax.set_title("Learned inhibitory weights", loc="left")
    letter(ax, "D")
    return svg(fig)


def fig_third(k2, tr):
    seed = k2["runs"][0]["seed"]
    w, m = tr[f"w_gated_{seed}"], tr[f"m_gated_{seed}"]
    t = np.arange(len(m)) * 0.01  # s (10 ms resolution)
    k = 100  # 1 s moving average for display
    def sm(x):  # centred mean over the samples that exist, so the edges are not pulled toward zero
        return np.convolve(x, np.ones(k), mode="same") / np.convolve(np.ones_like(x), np.ones(k), mode="same")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True,
                            gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    ax = axs[0]
    for j, col in enumerate((C1, C2, C3)):
        ax.plot(t, sm(w[:, j]), color=col, lw=1.1, label=f"onto {NEURONS[j]}")
    ax.set_xlabel("time after switch (s)")
    ax.set_ylabel("weight (mV-eq., 1 s mean)")
    ax.set_title(f"Gated run, seed {seed}", loc="left")
    ax.legend(fontsize=6, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.08))
    letter(ax, "A")
    ax = axs[1]
    ax.plot(t, m, color=GREY, lw=0.4)
    ax.plot(t, sm(m), color=C4, lw=1.2)
    ax.axhline(1.0, color="#000000", lw=0.5, ls=":")
    ax.set_xlabel("time after switch (s)")
    ax.set_ylabel("surprise m")
    ax.set_title("Third factor", loc="left")
    letter(ax, "B")
    ax = axs[2]
    conds = ("fixed", "matched", "gated")
    for i, r in enumerate(k2["runs"]):
        vals = [min(r[c]["t_rev_s"], 160.0) for c in conds]
        ax.plot(range(3), vals, color=GREY, lw=0.7, marker="o", ms=3)
    ax.set_xticks(range(3), conds)
    ax.set_ylabel("re-learning time t_rev (s)")
    ax.set_title("Seeds " + ", ".join(str(r["seed"]) for r in k2["runs"]), loc="left")
    letter(ax, "C")
    return svg(fig)


def fig_column(d, gates):
    dt = float(d["col_dt"])
    s0, c0 = 512, 768
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True,
                            gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    for ax, name, col, lab in ((axs[0], "no_inh", GREY, "no distant / top-down inhibition"),
                               (axs[1], "pcl", C1, "learned PCL inhibition")):
        s = d[f"col_s_{name}"]
        ti, ni = np.nonzero(s[:, s0:c0] > 0)
        ax.scatter(ti * dt, ni, s=1.2, color=col, linewidths=0, rasterized=True)
        ax.set_xlim(0, s.shape[0] * dt)
        ax.set_ylim(-2, c0 - s0 + 2)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("simple cell")
        ax.set_title(f"{lab}: {int(s[:, s0:c0].sum())} spikes", loc="left", fontsize=8)
    letter(axs[0], "A")
    letter(axs[1], "B")
    ax = axs[2]
    names = list(gates)
    x = np.arange(len(names))
    for j, (key, col, lab) in enumerate((("pcl", C1, "learned inhibition"), ("none", GREY, "no inhibition"),
                                         ("random", C2, "random removal"))):
        ax.bar(x + (j - 1) * 0.26, [gates[n][key] for n in names], 0.25, color=col, label=lab)
    ax.axhline(1 / 8, color="#000000", lw=0.5, ls=":")
    ax.set_xticks(x, names)
    ax.set_ylabel("orientation decoding (8-way)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6, loc="upper right")
    ax.set_title("Mean of seeds 10–12", loc="left")
    letter(ax, "C")
    return svg(fig)


def fig_lfp(d):
    dt = float(d["col_dt"])
    z = d["contact_depths"]
    keep = z <= 0.8
    lp, ln = d["col_lfp_pcl"].mean(0)[:, keep], d["col_lfp_no_inh"].mean(0)[:, keep]
    lim = float(np.abs(np.concatenate([lp, ln])).max())
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.6), constrained_layout=True)
    for ax, x, lab, c in ((axs[0, 0], ln, "no distant / top-down inhibition", "A"),
                          (axs[0, 1], lp, "learned PCL inhibition", "B")):
        im = ax.imshow(x.T, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, origin="upper",
                       extent=(0, x.shape[0] * dt, z[keep][-1], z[keep][0]), interpolation="nearest")
        for depth in (0.2, 0.35, 0.55):
            ax.axhline(depth, color="#000000", lw=0.4, ls=":")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("relative depth")
        ax.set_title(lab, loc="left", fontsize=8)
        letter(ax, c)
    cb = fig.colorbar(im, ax=axs[0, :], shrink=0.85, pad=0.01)
    cb.set_label("LFP proxy (rel.)")
    ax = axs[1, 0]
    x = np.arange(3)
    for j, (name, col, lab) in enumerate((("no_inh", GREY, "no inhibition"), ("pcl", C1, "learned inhibition"))):
        ax.bar(x + (j - 0.5) * 0.38, d[f"col_popsrc_{name}"].mean((0, 1)), 0.36, color=col, label=lab)
    ax.axhline(0, color="#000000", lw=0.5)
    ax.set_xticks(x, ["relays (0.55)", "simple (0.35)", "complex (0.20)"])
    ax.set_ylabel("mean source proxy (rel.)")
    ax.legend(fontsize=6, loc="lower left")
    ax.set_title("Mean source per population and depth", loc="left", fontsize=8)
    letter(ax, "C")
    ax = axs[1, 1]
    i = int(np.argmin(np.abs(z - 0.35)))
    for name, col, lab in (("no_inh", GREY, "no inhibition"), ("pcl", C1, "learned inhibition")):
        f, pxx = welch(d[f"col_lfp_{name}"][:, :, i], fs=1000.0 / dt, nperseg=256, axis=-1)
        sel = (f >= 5) & (f <= 300)
        ax.semilogy(f[sel], pxx.mean(0)[sel], color=col, label=lab)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(f"power, depth {z[i]:.2f} (rel.)")
    ax.legend()
    ax.set_title("LFP-proxy spectrum, simple-cell depth", loc="left", fontsize=8)
    letter(ax, "D")
    return svg(fig)


def gate_numbers():
    out = {}
    for gate, pat, key_rand in (("C1", "confirm/c1_seed{}.json", "both"), ("K1", "confirm/k1_seed{}.json", "simple"),
                                ("C1b", "confirm/c1b_seed{}.json", "both")):
        rs = [load(pat.format(s)) for s in (10, 11, 12)]
        k = "simple" if gate == "K1" else "both"
        out[gate] = {"pcl": np.mean([r["acc"]["pcl"][k] for r in rs]),
                     "none": np.mean([r["acc"]["no_inh"][k] for r in rs]),
                     "random": np.mean([r["acc"]["random"][key_rand] for r in rs]),
                     "kept": np.mean([r["spikes"]["pcl"][0] / r["spikes"]["no_inh"][0] for r in rs]),
                     "osi": np.mean([r["osi_trained_median"] for r in rs])}
    return out


def later_gates():
    """K1h (equality with K1), K1h-nf (noise floor) and control O1 (oriented kernels), as page strings."""
    import k1h_nf_gate as NF
    v = {}
    same = [load(f"confirm/k1h_seed{s}.json")[k] == load(f"confirm/k1_seed{s}.json")[k]
            for s in (10, 11, 12) for k in ("spikes", "acc")]
    v["k1h_equal"] = f"{sum(all(same[i:i + 2]) for i in (0, 2, 4))} of 3"
    runs = {s: {a: load(f"confirm/k1nf_{a}_seed{s}.json") for a in ("base", "nudge", "h")} for s in NF.SEEDS}
    verdict, t, dh, _ = NF.gate(runs)
    v["nf_verdict"], v["nf_cls"], v["nf_t"] = verdict, verdict.lower(), f"{t:.3f}"
    v["nf_diff"] = ("did not differ from the per-edge rule in any decoding value" if dh == 0
                    else f"differed from the per-edge rule by at most {dh:.3f} in decoding")
    o1 = [load(f"confirm/o1_bar_seed{s}.json") for s in (10, 11, 12)]
    c1 = [load(f"confirm/c1_seed{s}.json") for s in (10, 11, 12)]
    osi = [r["oriented"]["osi_median"] for r in o1]
    ratio = [c["osi_trained_median"] / r["oriented"]["osi_median"] for c, r in zip(c1, o1)]
    v["o1_osi"] = f"{min(osi):.3f}–{max(osi):.3f}"
    v["o1_dec"] = f2(min(r["oriented"]["acc_simple"] for r in o1))
    v["o1_ratio"] = f"{min(ratio):.2f}–{max(ratio):.2f}"
    v["o1_seg_resp"] = f"{max(load(f'confirm/o1_seg8_seed{s}.json')['oriented']['n_resp'] for s in (10, 11, 12))}"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "report" / "report_data.npz")
    ap.add_argument("--k2", type=Path, default=HERE / "confirm" / "k2b.json")
    ap.add_argument("--out", type=Path, default=HERE / "report" / "pcl_report.html")
    a = ap.parse_args()
    d = np.load(a.data)
    p0, k0b, k2 = load("fig2.json"), load("fig2_hdp_avg.json"), json.loads(a.k2.read_text())
    tr = np.load(a.k2.with_suffix(".npz"))
    gates = gate_numbers()
    figs = {"network": fig_network(d, p0, k0b), "third": fig_third(k2, tr),
            "column": fig_column(d, gates), "lfp": fig_lfp(d)}
    tmpl = (HERE / "report" / "template.html").read_text(encoding="utf-8")
    vals = context(p0, k0b, k2, gates, d) | later_gates()
    html = tmpl
    for k, v in figs.items():
        html = html.replace("{{fig:" + k + "}}", v)
    for k, v in vals.items():
        html = html.replace("{{" + k + "}}", v)
    missing = re.findall(r"\{\{[^}]+\}\}", html)
    assert not missing, f"unfilled placeholders: {missing}"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(html, encoding="utf-8")
    print(a.out, len(html) // 1024, "KiB")


def f2(x):
    return f"{x:.2f}"


def context(p0, k0b, k2, gates, d):
    v = {}
    v["p0_sup"] = " / ".join(f2(x) for x in p0["suppression_mean"])
    v["p0_w"] = " / ".join(f"{x:.0f}" for x in p0["w_end_mean"])
    v["k0b_sup"] = " / ".join(f2(x) for x in k0b["suppression_mean"])
    v["k0b_w"] = " / ".join(f"{x:.0f}" for x in k0b["w_avg_mean_mV"])
    for g, n in gates.items():
        v[f"{g}_kept"] = f"{n['kept']:.2f}"
        v[f"{g}_removed"] = f"{100 * (1 - n['kept']):.0f}"
        v[f"{g}_pcl"], v[f"{g}_none"], v[f"{g}_rand"] = f2(n["pcl"]), f2(n["none"]), f2(n["random"])
        v[f"{g}_osi"] = f2(n["osi"])
    rows = []
    for r in k2["runs"]:
        cells = "".join(f"<td>{r[c]['t_rev_s']:.1f}</td>" if np.isfinite(r[c]["t_rev_s"]) else "<td>never</td>"
                        for c in ("fixed", "matched", "gated"))
        rows.append(f"<tr><td>{r['seed']}</td>{cells}<td>{r['gated_mult_mean']:.3f}</td></tr>")
    v["k2_rows"] = "\n".join(rows)
    t1 = all(r["gated"]["t_rev_s"] < r["fixed"]["t_rev_s"] for r in k2["runs"])
    t2 = all(r["gated"]["t_rev_s"] < r["matched"]["t_rev_s"] for r in k2["runs"])
    v["k2_t1"], v["k2_t2"] = ("PASS" if t1 else "FAIL"), ("PASS" if t2 else "FAIL")
    v["k2_t1_cls"], v["k2_t2_cls"] = ("pass" if t1 else "fail"), ("pass" if t2 else "fail")
    v["k2_g"] = f"{k2['params']['g']:g}"
    tf = [r["fixed"]["t_rev_s"] for r in k2["runs"]]
    tg = [r["gated"]["t_rev_s"] for r in k2["runs"]]
    tm = [r["matched"]["t_rev_s"] for r in k2["runs"]]
    faster = sum(g < f for g, f in zip(tg, tf))
    mults = [r["gated_mult_mean"] for r in k2["runs"]]
    dg = [g - f for g, f in zip(tg, tf)]
    v["k2_fixed"] = f"{min(tf):.1f}–{max(tf):.1f}"
    v["k2_gated"] = f"{min(tg):.1f}–{max(tg):.1f}"
    v["k2_matched"] = f"{min(tm):.1f}–{max(tm):.1f}"
    v["k2_dmax"] = f"{max(abs(x) for x in dg):.1f}"
    v["k2_faster"] = f"{faster} of {len(tg)}"
    v["k2_mult"] = f"{min(mults):.2f}–{max(mults):.2f}"
    v["k2_seeds"] = str(len(tg))
    z = d["contact_depths"]
    i = int(np.argmin(np.abs(z - 0.35)))
    dt = float(d["col_dt"])
    pw = {}
    for name in ("pcl", "no_inh"):
        f, pxx = welch(d[f"col_lfp_{name}"][:, :, i], fs=1000.0 / dt, nperseg=256, axis=-1)
        sel = (f >= 5) & (f <= 300)
        pw[name] = float(pxx.mean(0)[sel].sum())
    v["lfp_ratio"] = f"{pw['pcl'] / pw['no_inh']:.1f}"
    v["lfp_depth"] = f"{z[i]:.2f}"
    for name in ("pcl", "no_inh"):
        src = d[f"col_popsrc_{name}"].mean((0, 1))
        cnt = d[f"col_counts_{name}"]
        v[f"src_relay_{name}"], v[f"src_simple_{name}"], v[f"src_cx_{name}"] = (f"{x:.0f}" if abs(x) >= 1 else f"{x:.2f}"
                                                                                for x in src)
        v[f"spk_simple_{name}"] = str(int(cnt[:, 512:768].sum()))
        v[f"spk_cx_{name}"] = str(int(cnt[:, 768:].sum()))
    s_p, s_n = d["col_popsrc_pcl"].mean((0, 1)), d["col_popsrc_no_inh"].mean((0, 1))
    v["src_simple_ratio"] = f"{s_p[1] / s_n[1]:.1f}"
    v["spk_simple_ratio"] = f"{d['col_counts_no_inh'][:, 512:768].sum() / d['col_counts_pcl'][:, 512:768].sum():.1f}"
    v["relay_same"] = "identical" if np.array_equal(d["col_popsrc_pcl"][..., 0], d["col_popsrc_no_inh"][..., 0]) \
        else "different"
    v["n_col_pcl"] = str(int(d["col_s_pcl"][:, 512:768].sum()))
    v["n_col_none"] = str(int(d["col_s_no_inh"][:, 512:768].sum()))
    return v


if __name__ == "__main__":
    main()
