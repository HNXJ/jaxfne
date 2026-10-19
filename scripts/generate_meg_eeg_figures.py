"""Generate the PNG figures for docs/guides/readout/meg_eeg_from_one_column.md.

One jaxfne column, simulated for six seeds, copied onto a folded cortex and
forwarded to MEG and EEG with jaxfne.fields.head. Prints the numbers the guide
quotes.

Usage:
  python scripts/generate_meg_eeg_figures.py [--output-dir docs/assets/meg_eeg]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--output-dir", type=Path, default=REPO / "docs" / "assets" / "meg_eeg")
OUT = _ap.parse_args().output_dir.resolve()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import jax  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402
from scipy.signal import butter, sosfiltfilt, welch  # noqa: E402

import jaxfne as jtfne  # noqa: E402
from jaxfne.fields import (  # noqa: E402
    column_dipole,
    eeg_sphere_leadfield,
    folded_sphere_surface,
    meg_sphere_leadfield,
    sphere_sensor_cap,
    tile_column_dipoles,
)

assert "site-packages" not in jtfne.__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)
OUT.mkdir(parents=True, exist_ok=True)

N, DUR, DT, BURN = 160, 2400.0, 0.5, 400.0
FS = 1000.0 / DT
K_SEEDS = 6
M_COLS = 1500
COL_NAM = 1e-9  # declared per-column dipole RMS: 1 nA m
SIGMA = 0.33  # S/m, homogeneous head
SOS = butter(4, [1.0, 100.0], btype="band", fs=FS, output="sos")
RNG = np.random.default_rng(2026)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Cambria", "DejaVu Serif"],
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)
GREEN, ORANGE, PURPLE, GREY = "#1b7f5a", "#d35400", "#7d3c98", "#555555"
FIG = (12.4, 5.0)


# ---------------- one column, K realizations ----------------
def simulate(seed: int):
    cfg = (
        jtfne.configuration()
        .runtime(seed=seed, duration_ms=DUR, dt_ms=DT, dtype="float32", jit=True)
        .population(N, neurons={"E": 0.8, "I": 0.2}, layers=["L2/3", "L4", "L5"], name="col")
        .cell_types({"E": 0.8, "PV": 0.2})
        .geometry(layer_thickness={"L2/3": 0.33, "L4": 0.34, "L5": 0.33})
        .set_emitter("izhikevich", "cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="mean_zero_neumann",
            gauge="mean_zero",
        )
        .probe(name="p", modes=["spikes", "V_m"])
    )
    model = jtfne.construct(cfg)
    sim = jtfne.Simulation(
        duration_ms=DUR,
        dt_ms=DT,
        seed=seed,
        record_sources=True,
        record_fields=True,
        runtime=jtfne.RuntimeConfig(dtype="float32", jit=True, seed=seed),
    )
    s = model.simulate(sim)
    return (
        np.asarray(s.sources, np.float64),
        np.asarray(s.spikes),
        np.asarray(model.params["positions"]),
    )


burn = int(BURN / DT)
P, SPK = [], []
for k in range(K_SEEDS):
    Q, spk, pos = simulate(100 + k)
    Qd = Q - Q[burn:].mean(axis=0, keepdims=True)
    p = sosfiltfilt(SOS, np.asarray(column_dipole(Qd, pos)))[burn:]
    P.append(p / p.std())  # unit RMS; physical scale declared below
    SPK.append(spk)
    if k == 0:
        Q0, SPK0, Z0 = Qd, spk, pos[:, 2]
    print("seed", 100 + k, "spikes", int(spk.sum()))
P = np.asarray(P)  # [K, T]
T = P.shape[1]
tt = BURN + np.arange(T) * DT

jax.config.update("jax_enable_x64", True)  # forward models in float64; the column ran in float32

# ---------------- folded cortex (core) ----------------
XYZ, NRM = (np.asarray(a) for a in folded_sphere_surface(M_COLS))
RAD = np.linalg.norm(XYZ, axis=1)
RADIALITY = np.abs(np.sum(NRM * XYZ / RAD[:, None], axis=1))
th = np.arccos(XYZ[:, 2] / RAD)
ph = np.arctan2(XYZ[:, 1], XYZ[:, 0])

# ---------------- sensors (core) ----------------
MEG_POS, MEG_N = (np.asarray(a) for a in sphere_sensor_cap(160, 0.12))
MTH = np.arccos(MEG_N[:, 2])
MPH = np.arctan2(MEG_N[:, 1], MEG_N[:, 0])
epos, EN = (np.asarray(a) for a in sphere_sensor_cap(64, 0.09, 120.0))
ETH = np.arccos(EN[:, 2])
EPH = np.arctan2(EN[:, 1], EN[:, 0])

L_MEG = np.asarray(meg_sphere_leadfield(MEG_POS, MEG_N, XYZ, NRM))  # T/(A m)
L_EEG = np.asarray(
    eeg_sphere_leadfield(epos, XYZ, NRM, conductivity_s_per_m=SIGMA)
)  # V/(A m), avg ref
assert L_EEG.shape == (len(epos), M_COLS), L_EEG.shape

# ---------------- activity: copies of the one column (core) ----------------
cth, cph = np.radians(40.0), np.radians(20.0)
u_c = np.array([np.sin(cth) * np.cos(cph), np.sin(cth) * np.sin(cph), np.cos(cth)])
center = XYZ[np.argmax(XYZ @ u_c / RAD)]  # the column nearest that direction
d_c = np.linalg.norm(XYZ - center, axis=1)
PATCH = d_c < 0.025
S, assign, shift = tile_column_dipoles(
    P, M_COLS, jax.random.PRNGKey(2026), in_phase=PATCH, in_phase_realization=0
)
S = np.asarray(S) * COL_NAM  # [T, M] A m
print("columns", M_COLS, "patch", int(PATCH.sum()))
MEG = S @ L_MEG.T * 1e15  # fT
EEG = S @ L_EEG.T * 1e6  # uV
MEG_bg = S[:, ~PATCH] @ L_MEG[:, ~PATCH].T * 1e15
EEG_bg = S[:, ~PATCH] @ L_EEG[:, ~PATCH].T * 1e6


def proj(t_, p_):
    return t_ * np.cos(p_), t_ * np.sin(p_)


MX, MY = proj(MTH, MPH)
EX, EY = proj(ETH, EPH)
MTRI, ETRI = Triangulation(MX, MY), Triangulation(EX, EY)


def topo(ax, tri, x, y, vals, vmax, title):
    lv = np.linspace(-vmax, vmax, 21)
    cs = ax.tricontourf(tri, vals, levels=lv, cmap="RdBu_r", extend="both")
    ax.tricontour(tri, vals, levels=lv[::2], colors="k", linewidths=0.3)
    ax.plot(x, y, ".", ms=1.6, color="0.2")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    return cs


# ---- F1: one column
fig, ax = plt.subplots(
    2, 1, figsize=(12.4, 5.7), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.08)
)
t_full = np.arange(SPK0.shape[0]) * DT
order = np.argsort(Z0)
ti, ni = np.nonzero(SPK0[:, order])
zl = Z0[order][ni]
ax[0].scatter(
    t_full[ti],
    ni,
    s=1.5,
    lw=0,
    c=np.where(zl < 0.33, "#1f618d", np.where(zl < 0.67, "#117a65", "#922b21")),
)
ax[0].invert_yaxis()
ax[0].set_ylabel("neuron (superficial → deep)")
for d, lab in [(0.165, "L2/3"), (0.5, "L4"), (0.835, "L5")]:
    ax[0].text(1405, np.searchsorted(Z0[order], d), lab, fontsize=10, va="center")
ax[1].plot(tt, P[0], color=GREEN, lw=0.9)
ax[1].set_ylabel("dipole p(t)\n(unit RMS)")
ax[1].set_xlabel("time (ms)")
ax[0].set_xlim(BURN, 1400)
fig.savefig(OUT / "column.png")
plt.close(fig)

# ---- F2: cortex with copied columns
fig = plt.figure(figsize=FIG)
a3 = fig.add_axes([-0.02, -0.2, 0.56, 1.4], projection="3d")
sc = a3.scatter(*XYZ.T * 1000, c=RADIALITY, cmap="viridis", s=7, vmin=0, vmax=1)
sel = np.arange(0, M_COLS, 9)
a3.quiver(*(XYZ[sel].T * 1000), *(NRM[sel].T * 6), color="0.25", lw=0.5, arrow_length_ratio=0)
a3.scatter(*XYZ[PATCH].T * 1000, s=10, c=ORANGE)
a3.set_box_aspect((1, 1, 0.75), zoom=1.5)
a3.set_axis_off()
a3.view_init(45, 20)
fig.colorbar(sc, cax=fig.add_axes([0.53, 0.2, 0.012, 0.6]), label="|cos| to radial")
a2 = fig.add_axes([0.64, 0.14, 0.35, 0.78])
a2.hist(RADIALITY, bins=30, color=GREEN, alpha=0.85)
a2.set_xlabel("|cos| between column axis and radial")
a2.set_ylabel("columns")
a2.set_title("folding gives every orientation", fontsize=12)
fig.savefig(OUT / "cortex.png")
plt.close(fig)

# ---- F3: activity assignment
fig, ax = plt.subplots(2, 1, figsize=FIG, sharex=True, gridspec_kw=dict(hspace=0.25))
for j, m in enumerate(np.flatnonzero(~PATCH)[:6]):
    ax[0].plot(tt, S[:, m] * 1e9 + 4 * j, lw=0.6, color=GREY)
ax[0].set_title(f"background: {K_SEEDS} realizations, random circular shift per copy", fontsize=12)
ax[0].set_yticks([])
for j, m in enumerate(np.flatnonzero(PATCH)[:6]):
    ax[1].plot(tt, S[:, m] * 1e9 + 4 * j, lw=0.6, color=ORANGE)
ax[1].set_title(
    f"active patch ({int(PATCH.sum())} copies within 25 mm): one realization, in phase", fontsize=12
)
ax[1].set_yticks([])
ax[1].set_xlabel("time (ms)")
ax[0].set_xlim(BURN, 1400)
fig.savefig(OUT / "activity.png")
plt.close(fig)

# ---- F4: sensitivity maps on cortex (MEG vs EEG)
sens_m = np.sqrt((L_MEG**2).mean(0)) * COL_NAM * 1e15
sens_e = np.sqrt((L_EEG**2).mean(0)) * COL_NAM * 1e6
CX, CY = proj(th, ph)
fig, ax = plt.subplots(1, 3, figsize=FIG, gridspec_kw=dict(wspace=0.65, width_ratios=[1, 1, 1.1]))
s0 = ax[0].scatter(CX, CY, c=sens_m, s=10, cmap="magma")
ax[0].set_title("MEG sensitivity")
fig.colorbar(s0, ax=ax[0], fraction=0.045, label="fT per 1 nA·m")
s1 = ax[1].scatter(CX, CY, c=sens_e, s=10, cmap="magma")
ax[1].set_title("EEG sensitivity")
fig.colorbar(s1, ax=ax[1], fraction=0.045, label="µV per 1 nA·m")
for a_ in ax[:2]:
    a_.set_aspect("equal")
    a_.axis("off")
ax[2].scatter(RADIALITY, sens_m / sens_m.max(), s=4, color=ORANGE, label="MEG", alpha=0.6)
ax[2].scatter(RADIALITY, sens_e / sens_e.max(), s=4, color=PURPLE, label="EEG", alpha=0.6)
ax[2].set_xlabel("|cos| to radial")
ax[2].set_ylabel("sensitivity / max")
ax[2].legend(frameon=False, markerscale=3)
ax[2].set_title("MEG falls to zero at radial")
fig.savefig(OUT / "sensitivity.png")
plt.close(fig)

# ---- F5: sensor data MEG + EEG
gm = MEG.std(1)
pk = int(np.argmax(gm[int(200 / DT) :])) + int(200 / DT)
fig = plt.figure(figsize=(12.4, 5.6))
gs = fig.add_gridspec(2, 3, width_ratios=[2.6, 1, 1], hspace=0.38, wspace=0.12)
for row, (Y, Ybg, tri, x, y, unit, name) in enumerate(
    [
        (MEG, MEG_bg, MTRI, MX, MY, "fT", "MEG, 160 magnetometers"),
        (EEG, EEG_bg, ETRI, EX, EY, "µV", f"EEG, {len(epos)} electrodes, average ref."),
    ]
):
    a = fig.add_subplot(gs[row, 0])
    a.plot(tt, Y, color="0.4", lw=0.3, alpha=0.5)
    a.plot(tt, Y.std(1), color="k", lw=1.1, label="GFP")
    a.axvline(tt[pk], color=ORANGE, lw=1)
    a.set_xlim(BURN, DUR)
    a.set_ylabel(unit)
    a.set_title(name, fontsize=12)
    a.legend(frameon=False, loc="upper right")
    if row == 1:
        a.set_xlabel("time (ms)")
    vm = np.abs(Y[pk]).max()
    topo(fig.add_subplot(gs[row, 1]), tri, x, y, Y[pk], vm, f"all copies, {tt[pk]:.0f} ms")
    cs = topo(fig.add_subplot(gs[row, 2]), tri, x, y, Ybg[pk], vm, "patch removed")
    fig.colorbar(cs, ax=fig.axes[-1], fraction=0.05, label=unit)
fig.savefig(OUT / "sensors.png")
plt.close(fig)

# ---- F6: scaling with number of copies (coherent vs incoherent)
near = np.argsort(d_c)
ns = np.unique(np.round(np.logspace(0, np.log10(400), 14)).astype(int))
res = {"coh_meg": [], "inc_meg": [], "coh_eeg": [], "inc_eeg": []}
for n in ns:
    idx = near[:n]
    for kind in ("coh", "inc"):
        if kind == "coh":
            src = np.repeat(P[0][:, None], n, 1)
        else:
            src = np.stack(
                [np.roll(P[RNG.integers(K_SEEDS)], RNG.integers(T)) for _ in range(n)], 1
            )
        src = src * COL_NAM
        res[f"{kind}_meg"].append(np.sqrt(np.mean((src @ L_MEG[:, idx].T * 1e15) ** 2)))
        res[f"{kind}_eeg"].append(np.sqrt(np.mean((src @ L_EEG[:, idx].T * 1e6) ** 2)))
fig, ax = plt.subplots(1, 2, figsize=FIG, gridspec_kw=dict(wspace=0.22))
for a_, mod, unit in [(ax[0], "meg", "fT"), (ax[1], "eeg", "µV")]:
    c, inc = np.array(res[f"coh_{mod}"]), np.array(res[f"inc_{mod}"])
    a_.loglog(ns, c, "o-", color=ORANGE, label="in phase (same realization)")
    a_.loglog(ns, inc, "s-", color=GREY, label="independent copies")
    a_.loglog(ns, c[0] * ns, ":", color=ORANGE, lw=0.8, label="∝ N")
    a_.loglog(ns, inc[0] * np.sqrt(ns), ":", color=GREY, lw=0.8, label="∝ √N")
    a_.set_xlabel("copies N (nearest to patch centre)")
    a_.set_ylabel(f"RMS over sensors and time ({unit})")
    a_.set_title(mod.upper() + (": opposing walls cancel" if mod == "meg" else ""))
    a_.legend(frameon=False, fontsize=10)
fig.savefig(OUT / "scaling.png")
plt.close(fig)

# ---- F7: spectra
fig, ax = plt.subplots(figsize=(12.4, 3.3))
for Y, lab, cc in [
    (P[0][:, None], "one column dipole", GREEN),
    (MEG, "MEG (mean over sensors)", ORANGE),
    (EEG, "EEG (mean over electrodes)", PURPLE),
]:
    f, Pw = welch(Y, fs=FS, nperseg=512, axis=0)
    Pw = Pw.mean(1)
    m = (f >= 2) & (f <= 100)
    ax.semilogy(f[m], Pw[m] / Pw[m].max(), color=cc, lw=1.4, label=lab)
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("PSD (normalized)")
ax.legend(frameon=False)
fig.savefig(OUT / "spectra.png")
plt.close(fig)

summ = dict(
    columns=M_COLS,
    patch=int(PATCH.sum()),
    seeds=K_SEEDS,
    eeg_channels=len(epos),
    sigma=SIGMA,
    radius_mm=[float(RAD.min() * 1e3), float(RAD.max() * 1e3)],
    spikes_per_seed=[int(s.sum()) for s in SPK],
    meg_peak_fT=float(np.abs(MEG).max()),
    meg_gfp_median_fT=float(np.median(MEG.std(1))),
    eeg_peak_uV=float(np.abs(EEG).max()),
    eeg_gfp_median_uV=float(np.median(EEG.std(1))),
    patch_share_meg=float(np.std(MEG[pk] - MEG_bg[pk]) / np.std(MEG[pk])),
    patch_share_eeg=float(np.std(EEG[pk] - EEG_bg[pk]) / np.std(EEG[pk])),
    sens_meg_fT_per_nAm=[float(sens_m.min()), float(np.median(sens_m)), float(sens_m.max())],
    sens_eeg_uV_per_nAm=[float(sens_e.min()), float(np.median(sens_e)), float(sens_e.max())],
    scaling_ns=ns.tolist(),
    **{k: [float(v) for v in vals] for k, vals in res.items()},
    gain_400={k: float(v[-1] / v[0]) for k, v in res.items()},
    peak_time_ms=float(tt[pk]),
)
print(
    json.dumps(
        {k: v for k, v in summ.items() if not k.startswith(("coh", "inc", "scaling"))}, indent=1
    )
)

# trim white margins (3D axes leave wide ones) and store as a 256-colour palette PNG
from PIL import Image, ImageChops  # noqa: E402

for f in sorted(OUT.glob("*.png")):
    im = Image.open(f).convert("RGB")
    box = (
        ImageChops.difference(im, Image.new("RGB", im.size, "white"))
        .point(lambda v: 255 if v > 12 else 0)
        .getbbox()
    )
    if box:
        pad = 12
        im = im.crop(
            (
                max(box[0] - pad, 0),
                max(box[1] - pad, 0),
                min(box[2] + pad, im.width),
                min(box[3] + pad, im.height),
            )
        )
    im.quantize(256).save(f, optimize=True)
