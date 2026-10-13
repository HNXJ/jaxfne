"""Digitize the mean frequency tuning curves of Fig. 3c,d (Nat Commun 2025, s41467-025-64234-z).

Usage: python -I digitize_fig3.py FIG3.png OUT.json

Input: the publisher's full-size PNG (URL and SHA-256 below; CC BY-NC-ND 4.0, kept outside
the repo at F:/warehouse/pcl/paper/). Mean curve: centreline of the mean-colour mask in a
3-px column at each of 17 K/k values (0.4..2.0 step 0.1; vertex spacing read off 3x crops).
A column where the example-cell lines cover the mean is null. Band: extent of band-colour
plus mean pixels in the column; se = max(0, half-width - half the panel's narrowest band),
since the narrowest band is the stroke alone. y calibration: tick-label glyph centroids
("1.0", "0.5", "0.0") in the label strip left of each axis; x calibration: the 0.4 and 2.0
vertices. Checks: the mid label must read 0.5, the mean at K/k = 1 must read 1.0 (curves
are normalized per cell at k), and in c the pink and purple minima must read 0.0.
"""
import hashlib
import json
import sys

URL = ("https://media.springernature.com/full/springer-static/image/"
       "art%3A10.1038%2Fs41467-025-64234-z/MediaObjects/41467_2025_64234_Fig3_HTML.png")
SHA256 = "6d61db5ef01b1f6abe14b3ba191db2772fec533bcd6731cf1acc3b041f3e96df"

import numpy as np
from PIL import Image

KK = [round(0.4 + 0.1 * i, 1) for i in range(17)]

# Panel geometry in original pixels: x of the K/k=0.4 and 2.0 vertices, the y window of the
# axes, and the x window of the tick-label strip. From crops at boxes (21,1153) and (1056,1163),
# 3x zoom: c vertices at crop x 205 and 1165, d at 90 and 1050.
PANELS = {
    "c_simple": dict(x04=21 + 205 / 3, x20=21 + 1165 / 3, y_top=1153 + 10, y_bot=1153 + 205,
                     label_x=(21 + 40 / 3, 21 + 145 / 3), mean="black", band="gray"),
    "d_complex": dict(x04=1056 + 90 / 3, x20=1056 + 1050 / 3, y_top=1163 + 10, y_bot=1163 + 205,
                      label_x=(1031, 1068), mean="steelblue", band="lightblue"),
}


def mask(rgb, kind):
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    if kind == "black":
        return (r < 60) & (g < 60) & (b < 60)
    if kind == "gray":
        return (abs(r - g) < 12) & (abs(g - b) < 12) & (r >= 60) & (r < 170)
    if kind == "steelblue":
        return (r < 110) & (g > 100) & (g < 160) & (b > 150) & (b < 210)
    if kind == "lightblue":
        return (r > 100) & (r < 180) & (g > 180) & (b > 230)
    if kind == "pink":
        return (r > 230) & (g > 160) & (g < 215) & (b > 170) & (b < 225)
    if kind == "purple":
        return (r > 110) & (r < 180) & (g < 60) & (b > 180)
    raise ValueError(kind)


def label_centroids(rgb, x0, x1, y0, y1):
    """Row centroids of dark glyph clusters in the label strip, top to bottom."""
    sub = mask(rgb[y0:y1, int(x0):int(x1)], "black")
    rows = np.where(sub.any(axis=1))[0]
    groups, cur = [], [rows[0]]
    for r in rows[1:]:
        if r - cur[-1] > 3:
            groups.append(cur)
            cur = []
        cur.append(r)
    groups.append(cur)
    return [y0 + (g[0] + g[-1]) / 2 for g in groups]


def column_y(m, x, y0, y1):
    xs = slice(int(round(x)) - 1, int(round(x)) + 2)
    ys = np.where(m[y0:y1, xs].any(axis=1))[0]
    return None if ys.size == 0 else (y0 + ys.min(), y0 + ys.max(), y0 + np.median(ys))


def main(src, out):
    with open(src, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"input sha256 {digest} is not the recorded {SHA256}")
    rgb = np.asarray(Image.open(src).convert("RGB"))
    result = {"source": "Fig. 3c,d frequency tuning curves, N'dri et al., Nat Commun 2025, "
                        "doi 10.1038/s41467-025-64234-z",
              "url": URL, "sha256": SHA256, "K_over_k": KK, "panels": {}}
    for name, p in PANELS.items():
        y0, y1 = p["y_top"], p["y_bot"]
        labs = label_centroids(rgb, *p["label_x"], y0 - 10, y1 + 10)
        if len(labs) != 3:
            raise SystemExit(f"{name}: expected 3 tick labels, found {len(labs)} at {labs}")
        y_one, y_half, y_zero = labs
        to_val = lambda y: (y_zero - y) / (y_zero - y_one)
        mm, bm = mask(rgb, p["mean"]), mask(rgb, p["band"])
        mean, lo, hi = [], [], []
        for k in KK:
            x = p["x04"] + (k - 0.4) / 1.6 * (p["x20"] - p["x04"])
            c = column_y(mm, x, y0, y1)
            if c is None:  # mean occluded by the example-cell lines drawn on top
                mean.append(None)
                lo.append(None)
                hi.append(None)
                continue
            band = column_y(mm | bm, x, y0, y1)
            mean.append(round(to_val(c[2]), 3))
            hi.append(round(to_val(band[0]), 3))
            lo.append(round(to_val(band[1]), 3))
        # K/k = 1 is each cell's own normalizer (SE 0); its column cuts the peak tip.
        i1 = KK.index(1.0)
        stroke = min(h - l for j, (h, l) in enumerate(zip(hi, lo)) if h is not None and j != i1)
        se = [None if h is None else 0.0 if j == i1 else round(max(0.0, (h - l - stroke) / 2), 3)
              for j, (h, l) in enumerate(zip(hi, lo))]
        checks = {"label_mid_residual": round(to_val(y_half) - 0.5, 3),
                  "mean_at_1": mean[KK.index(1.0)], "stroke_width": round(stroke, 3)}
        if name == "c_simple":
            for kind in ("pink", "purple"):
                ys = np.where(mask(rgb, kind)[y0:y1, int(p["x04"]):int(p["x20"])].any(axis=1))[0]
                checks[f"{kind}_min"] = round(to_val(y0 + ys.max() - 1.5), 3)
        result["panels"][name] = {"mean": mean, "se": se, "band_lo": lo, "band_hi": hi,
                                  "calibration_px": {"y_one": y_one, "y_half": y_half,
                                                     "y_zero": y_zero, "x04": p["x04"],
                                                     "x20": p["x20"]},
                                  "checks": checks}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    for name, r in result["panels"].items():
        print(name, "checks", r["checks"])
        print("  mean", r["mean"])


if __name__ == "__main__":
    main(*sys.argv[1:3])
