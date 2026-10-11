"""F4: OSI ceiling of the O1 kernel family under the declared drifting-bar stimulus.

Declaration: artifacts/etudes/theory_f4/README.md. Idealized model: expected event rates of
pcl_column.stimulus_step (noise 0, no Poisson sampling), the O1 oriented kernels at receptive
field sizes 7, 9, 11, 13, a linear-threshold unit r = sum_t max(drive - h, 0) with h a fraction
of the unit's maximum drive, OSI as in pcl_column.osi, median over units with sum r > 0,
ceiling = max over h.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # repo root
sys.path.insert(0, str(HERE.parents[0] / "pcl"))  # artifacts/etudes/pcl
import pcl_column as C  # noqa: E402
import pcl_orient_control as O  # noqa: E402

N_C0 = 32  # uniform grid over the c0 drift range, both ends included (linspace)
H_FRACS = np.round(np.arange(39) * 0.025, 3)  # 0 to 0.95 in steps of 0.025
RF_SWEEP = (7, 9, 11, 13)
CRIT = 0.3
O1_SEEDS = (10, 11, 12)
THETA = np.arange(C.N_ORI) * np.pi / C.N_ORI
N_STEPS = int(round(C.STIM["seq_ms"] / C.DT))


def osi_curves(r):
    """OSI |sum_k r_k e^{2 i theta_k}| / sum_k r_k, orientations on axis 0 (as pcl_column.osi)."""
    r = np.asarray(r, float)
    w = np.exp(2j * THETA).reshape((-1,) + (1,) * (r.ndim - 1))
    return np.abs((r * w).sum(0)) / np.maximum(r.sum(0), 1e-12)


def lemma_check(r):
    """(osi, bound) for one nonnegative 8-orientation curve; L1: osi <= 1 - 8 min(r) / sum(r)."""
    r = np.asarray(r, float)
    return float(osi_curves(r)), float(1.0 - C.N_ORI * r.min() / r.sum())


def oriented_kernels(rf):
    """(FS, rf*rf*2) O1 kernels built as pcl_orient_control.oriented_kernels at size rf, normalized as in O1."""
    dy, dx = np.meshgrid(np.arange(rf), np.arange(rf), indexing="ij")
    c = (rf - 1) / 2
    k = np.zeros((C.FS, rf, rf, 2))
    for f in range(C.FS):
        ori, q = divmod(f, 2)
        th = ori * np.pi / C.N_ORI + np.pi / 2  # bar normal, as in stimulus_step
        dn = (dx - c) * np.cos(th) + (dy - c) * np.sin(th)
        ahead = np.exp(-(dn - 1) ** 2 / (2 * O.SIGMA**2))
        behind = np.exp(-(dn + 1) ** 2 / (2 * O.SIGMA**2))
        k[f, ..., 0], k[f, ..., 1] = (ahead, behind) if q == 0 else (behind, ahead)
    k = k.reshape(C.FS, -1)
    lam = C.CONN["s_exc"][3] * k.shape[1]  # normalize(): mean weight times synapse count
    return k * lam / np.maximum(k.sum(-1, keepdims=True), 1e-12)


def patch_index(rf):
    """(positions, rf*rf*2) indices into the flattened event vector, in pcl_column.IN_IDX order
    (dy, dx, polarity). Taps outside the 16x16 grid index a zero pad element (input zero-padded)."""
    pad = C.G * C.G * 2
    rows = []
    for py in range(C.NP_S):
        for px in range(C.NP_S):
            taps = []
            for dy in range(rf):
                for dx in range(rf):
                    for p in range(2):
                        y, x = C.STRIDE * py + dy, C.STRIDE * px + dx
                        taps.append((y * C.G + x) * 2 + p if (y < C.G and x < C.G) else pad)
            rows.append(taps)
    return np.array(rows)


def event_images(ori, direction, c0):
    """Expected event rate per step, (N_STEPS, G*G*2), channel 0 ON (bar enters), 1 OFF (bar leaves).

    Bar test as pcl_column.stimulus_step for kind "bar" with noise 0; no Poisson sampling.
    """
    assert C.STIM["kind"] == "bar", "event_images mirrors the bar branch of stimulus_step only"
    th = ori * np.pi / C.N_ORI + np.pi / 2
    yy, xx = np.meshgrid(np.arange(C.G), np.arange(C.G), indexing="ij")
    proj = ((xx - 7.5) * np.cos(th) + (yy - 7.5) * np.sin(th)).ravel()
    t = np.arange(N_STEPS) * C.DT
    d = proj[None, :] - (c0 + direction * C.STIM["speed"] * t)[:, None]
    inside = np.abs(d) < C.STIM["width"] / 2
    prev = np.vstack([np.zeros((1, inside.shape[1]), bool), inside[:-1]])
    on, off = inside & ~prev, ~inside & prev
    return C.STIM["events"] * np.stack([on, off], -1).reshape(N_STEPS, -1).astype(float)


def drives(ori, kern, idx):
    """Drive for one stimulus orientation, shape (2 directions, N_C0, N_STEPS, units).

    Unit = position * n_features + feature, as pcl_column (position-major).
    """
    n_f = kern.shape[0]
    out = np.empty((2, N_C0, N_STEPS, C.NP_S * C.NP_S * n_f))
    for i, direction in enumerate((-1.0, 1.0)):
        for j, c0 in enumerate(-direction * 13.0 + np.linspace(-1.0, 1.0, N_C0)):
            E = event_images(ori, direction, c0)
            Ep = np.concatenate([E, np.zeros((N_STEPS, 1))], axis=1)
            out[i, j] = np.einsum("tpk,fk->tpf", Ep[:, idx], kern).reshape(N_STEPS, -1)
    return out


def all_drives(kern, idx):
    """Drive for all stimulus orientations, shape (8, 2, N_C0, N_STEPS, units)."""
    return np.stack([drives(o, kern, idx) for o in range(C.N_ORI)])


def curve_sweep(D):
    """Median OSI over units with sum r > 0 at each threshold fraction h of the unit's max drive."""
    maxd = D.max(axis=(0, 1, 2, 3))  # per unit, over the whole protocol
    out = []
    for h in H_FRACS:
        r = np.maximum(D - h * maxd, 0.0).sum(axis=(1, 2, 3))  # (8 orientations, units)
        ok = r.sum(0) > 0
        osi = osi_curves(r)
        med = float(np.median(osi[ok])) if ok.any() else None
        out.append(dict(h=float(h), median_osi=med, n_units=int(ok.sum()), tuning=r))
    return out


def rf_ceiling(rf):
    sweep = curve_sweep(all_drives(oriented_kernels(rf), patch_index(rf)))
    best = max((s for s in sweep if s["median_osi"] is not None), key=lambda s: s["median_osi"])
    return sweep, best


def main():
    raw = np.asarray(C.normalize(O.oriented_kernels(), C.MASKS["s_exc"], "s_exc"))
    assert np.allclose(raw, oriented_kernels(7)), "kernel build differs from pcl_orient_control"
    assert np.array_equal(patch_index(7), C.IN_IDX), "patch index differs from pcl_column.IN_IDX"
    sweeps, bests = {}, {}
    for rf in RF_SWEEP:
        sweeps[rf], bests[rf] = rf_ceiling(rf)
    b7 = bests[7]
    r7 = b7["tuning"]
    osi_ref, _ = C.osi(r7, np.arange(C.N_ORI))  # pcl_column.osi on the ceiling curves
    assert np.allclose(osi_ref, osi_curves(r7)), "OSI differs from pcl_column.osi"
    osi7, ok7 = osi_curves(r7), r7.sum(0) > 0
    units = [dict(position=[int(u // C.FS) // C.NP_S, int(u // C.FS) % C.NP_S], feature=int(u % C.FS),
                  osi=float(osi7[u]), tuning=[float(v) for v in r7[:, u]])
             for u in range(C.NS) if ok7[u]]
    feat_tun = r7.reshape(C.N_ORI, C.NP_S * C.NP_S, C.FS).sum(1)  # (8 orientations, features), summed over positions
    feat_osi, feat_arg = osi_curves(feat_tun), np.argmax(feat_tun, axis=0)
    per_feature = [dict(feature=f, kernel_orientation=f // 2, argmax_orientation=int(feat_arg[f]),
                        osi=float(feat_osi[f]), tuning=[float(v) for v in feat_tun[:, f]])
                   for f in range(C.FS)]
    n_differs = sum(p["argmax_orientation"] != p["kernel_orientation"] for p in per_feature)
    smallest =next((rf for rf in RF_SWEEP if bests[rf]["median_osi"] >= CRIT), "none")
    o1 = {s: json.loads((HERE.parents[0] / "pcl" / "confirm" / f"o1_bar_seed{s}.json").read_text(encoding="utf-8"))
          ["oriented"]["osi_median"] for s in O1_SEEDS}
    ceil7 = b7["median_osi"]
    h41 = "holds" if ceil7 < CRIT else "rejected"
    h42 = "holds" if all(v <= ceil7 for v in o1.values()) else "rejected"
    out = dict(
        stimulus=dict(kind=C.STIM["kind"], width=C.STIM["width"], speed=C.STIM["speed"], events=C.STIM["events"],
                      noise_hz=0.0, seq_ms=C.STIM["seq_ms"], dt=C.DT),
        protocol=dict(n_c0=N_C0, c0="-direction*13 + linspace(-1, 1, 32), both directions",
                      n_steps=N_STEPS, h_fracs=H_FRACS.tolist(),
                      max_drive="per unit, over 8 orientations x 2 directions x 32 offsets x steps",
                      edges="zero-padded: patch taps outside the 16x16 grid read 0; kernel normalized over full RF"),
        ceiling=dict(median_osi=ceil7, h_frac=b7["h"], n_units=b7["n_units"]),
        osi_by_threshold=[dict(h=s["h"], median_osi=s["median_osi"], n_units=s["n_units"]) for s in sweeps[7]],
        units_at_ceiling=units,
        per_feature_at_ceiling=per_feature,
        features_argmax_differs_from_kernel=int(n_differs),
        rf_sweep=[dict(rf=rf, ceiling=bests[rf]["median_osi"], h_frac=bests[rf]["h"], n_units=bests[rf]["n_units"])
                  for rf in RF_SWEEP],
        smallest_rf_reaching_criterion=smallest,
        o1_osi_median=o1,
        verdicts=dict(H4_1=h41, H4_2=h42),
        judgement_calls=[
            "max_drive: per unit, maximum over 8 orientations x 2 directions x 32 offsets x all steps",
            "c0 grid: -direction*13 + linspace(-1, 1, 32), both ends included",
            "edges: zero padding; taps outside the 16x16 grid read 0, kernel normalized over the full RF",
            "event_images: numpy mirror of the bar branch of stimulus_step (equality checked in tests)",
            "thresholds: 39 values, 0 to 0.95 step 0.025 (README corrected from 41)",
            "L2: at h = 0 OSI is 0 by construction; the ceiling is taken over h > 0",
            "per_feature_at_ceiling: tuning summed over the 16 positions of each feature at the ceiling threshold",
            "grid anisotropy: odd-orientation features peak at an adjacent even orientation (test_theory_f4)",
        ],
    )
    print(f"ceiling OSI* = {ceil7:.4f} at h = {b7['h']:.3f} x max drive (RF 7, {b7['n_units']} units)")
    print(f"criterion = {CRIT}")
    for rf in RF_SWEEP:
        print(f"RF {rf}: ceiling {bests[rf]['median_osi']:.4f} at h = {bests[rf]['h']:.3f}, "
              f"{bests[rf]['n_units']} units")
    print(f"per-feature argmax at ceiling (kernel orientation -> argmax, OSI): "
          + ", ".join(f"{p['kernel_orientation']}->{p['argmax_orientation']} ({p['osi']:.3f})" for p in per_feature))
    print(f"features whose argmax differs from kernel orientation: {n_differs} of {C.FS}")
    print(f"smallest RF reaching criterion: {smallest}")
    print(f"O1 oriented osi_median: {o1}")
    print(f"H4.1 (ceiling < {CRIT}): {h41}; H4.2 (O1 medians <= ceiling): {h42}")
    (HERE / "f4_ceiling.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
