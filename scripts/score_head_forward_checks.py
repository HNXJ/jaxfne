#!/usr/bin/env python3
"""Analytic EEG/MEG forward checks of jaxfne.fields.head.

Each check recomputes one agreement pinned by tests/test_head_forward.py between the
spherical-head forward code and an independent reference, with the test's inputs and
tolerances, and records the worst-case absolute deviation beside its tolerance:

- meg_radial: radial MEG leadfield against the primary-current Biot-Savart field
- meg_full_vector: full-vector MEG leadfield against scalar-potential quadrature
- meg_radial_silence: radial-dipole silence as a ratio against 1e-6
- eeg_neumann: EEG leadfield against the Neumann series (400 terms)
- eeg_mne: EEG leadfield against the MNE single-shell forward (skipped when mne is absent)
- eeg_scaling: conductivity reciprocity (0.33 against 0.66 S/m)
- eeg_reference: average-reference null (column means against zero)

Writes artifacts/etudes/core_checks/head_forward/score.json (one
{error, tolerance, pass, reference} row per check) and README.md beside it:

    python scripts/score_head_forward_checks.py         # write both files
    python scripts/score_head_forward_checks.py --check # exit 1 past rtol 1e-9
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import math

import numpy as np
from scipy.integrate import quad
from scipy.special import eval_legendre

from jaxfne.fields.head import (
    eeg_sphere_leadfield,
    meg_sphere_leadfield,
    sphere_sensor_cap,
)

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "etudes" / "core_checks" / "head_forward"
SCORE_PATH = OUT_DIR / "score.json"
README_PATH = OUT_DIR / "README.md"
RTOL = 1e-9
NOISE_FRACTION = 0.1  # errors below this fraction of the tolerance are compared as rounding residuals

MU0 = 4e-7 * np.pi  # literal, independent of the implementation


def _dipoles(m=7, radius=0.07, seed=3):
    """Dipoles at radii in [0.3, 1.0] x radius; a fixed seed per call keeps checks order-independent."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(m, 3))
    pos = radius * rng.uniform(0.3, 1.0, size=(m, 1)) * pos / np.linalg.norm(pos, axis=1, keepdims=True)
    ori = rng.normal(size=(m, 3))
    return pos, ori / np.linalg.norm(ori, axis=1, keepdims=True)


def _row(error, tolerance, reference, passed):
    return {"error": float(error), "tolerance": float(tolerance),
            "pass": bool(passed), "reference": reference}


def check_meg_radial():
    """Radial MEG leadfield against the primary-current field (rtol 2e-4)."""
    pos, ori = _dipoles()
    sens, nrm = sphere_sensor_cap(40, 0.12)
    lead = np.asarray(meg_sphere_leadfield(sens, nrm, pos, ori), float)
    s, n = np.asarray(sens, float), np.asarray(nrm, float)
    ref = np.empty_like(lead)
    for m in range(pos.shape[0]):
        d = s - pos[m]
        b0 = MU0 / (4 * np.pi) * np.cross(ori[m], d) / np.linalg.norm(d, axis=1, keepdims=True) ** 3
        ref[:, m] = np.sum(b0 * n, axis=1)
    assert np.abs(ref).max() > 0, "fixture must produce a field"
    scale = float(np.abs(ref).max())
    error = float(np.abs(lead - ref).max())
    tolerance = 1e-6 * scale + 2e-4 * scale
    return _row(error, tolerance, "primary-current Biot-Savart field", error <= tolerance)


def _primary_b(p, r0, q):
    d = p - r0
    return MU0 / (4 * np.pi) * np.cross(q, d) / np.linalg.norm(d) ** 3


def _scalar_potential(p, r0, q):
    """U(r) from the primary radial field, as in the pinned test."""
    e = p / np.linalg.norm(p)
    val, _ = quad(lambda t: e @ _primary_b(p + t * e, r0, q), 0.0, np.inf, epsabs=0.0, epsrel=1e-11,
                  limit=200)
    return val / MU0


def check_meg_full_vector():
    """Full-vector MEG leadfield against scalar-potential quadrature (rtol 1e-3)."""
    pos, ori = _dipoles(3, 0.07)
    sens, _ = sphere_sensor_cap(12, 0.12)
    s = np.asarray(sens, float)
    tilt = np.random.default_rng(5).normal(size=s.shape)
    tilt /= np.linalg.norm(tilt, axis=1, keepdims=True)
    lead = np.asarray(meg_sphere_leadfield(s, tilt, pos, ori), float)
    ref = np.empty_like(lead)
    h = 1e-5
    for m in range(pos.shape[0]):
        for k, p in enumerate(s):
            grad = [(_scalar_potential(p + h * ax, pos[m], ori[m])
                     - _scalar_potential(p - h * ax, pos[m], ori[m])) / (2 * h)
                    for ax in np.eye(3)]
            ref[k, m] = -MU0 * np.dot(grad, tilt[k])
    radial_share = np.abs(np.sum(tilt * s / np.linalg.norm(s, axis=1, keepdims=True), axis=1))
    assert radial_share.min() < 0.5, "fixture must include non-radial sensor axes"
    scale = float(np.abs(ref).max())
    error = float(np.abs(lead - ref).max())
    tolerance = 1e-4 * scale + 1e-3 * scale
    return _row(error, tolerance, "scalar-potential quadrature (-mu0 grad U)", error <= tolerance)


def check_meg_radial_silence():
    """Radial-dipole silence as silent/loud against 1e-6."""
    pos, _ = _dipoles()
    sens, nrm = sphere_sensor_cap(30, 0.12)
    radial = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    tang = np.cross(radial, [0.0, 0.0, 1.0])
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    silent = float(np.abs(np.asarray(meg_sphere_leadfield(sens, nrm, pos, radial))).max())
    loud = float(np.abs(np.asarray(meg_sphere_leadfield(sens, nrm, pos, tang))).max())
    assert loud > 1e-9, "fixture must produce a field"
    ratio = silent / loud
    return _row(ratio, 1e-6, "radial-to-tangential amplitude ratio", ratio < 1e-6)


def _eeg_series(e, r0, q, sigma, n_max=400):
    """Neumann-series reference, as in the pinned test."""
    big_r = np.linalg.norm(e[0])

    def phi(p):
        b = np.linalg.norm(p)
        cos_g = (e @ p) / (big_r * b)
        n = np.arange(1, n_max)
        terms = ((2 * n + 1) / n)[None, :] * (b / big_r) ** n[None, :] * eval_legendre(n[None, :],
                                                                                      cos_g[:, None])
        return terms.sum(1) / (4 * np.pi * sigma * big_r)

    h = 1e-7
    return sum(q[k] * (phi(r0 + h * np.eye(3)[k]) - phi(r0 - h * np.eye(3)[k])) / (2 * h)
               for k in range(3))


def check_eeg_neumann():
    """EEG closed form against the Neumann series (rtol 1e-3)."""
    elec, _ = sphere_sensor_cap(25, 0.09, 150.0)
    pos, ori = _dipoles(4, 0.07)
    lead = np.asarray(eeg_sphere_leadfield(elec, pos, ori, conductivity_s_per_m=0.33,
                                           average_reference=False), float)
    e = np.asarray(elec, float)
    ref = np.empty_like(lead)
    for m in range(4):
        ref[:, m] = _eeg_series(e, pos[m], ori[m], 0.33)
    assert np.abs(ref).max() > 0, "fixture must produce a field"
    scale = float(np.abs(ref).max())
    error = float(np.abs(lead - ref).max())
    tolerance = 1e-4 * scale + 1e-3 * scale
    return _row(error, tolerance, "Neumann-series potential (400 terms)", error <= tolerance)


def check_eeg_mne():
    """EEG leadfield against the MNE single-shell forward (rtol 1e-3)."""
    try:
        import mne
    except ImportError:
        return {"error": None, "tolerance": None, "pass": None,
                "reference": "MNE single-shell forward (homogeneous)",
                "skipped": "mne not installed"}
    mne.set_log_level("ERROR")
    elec, _ = sphere_sensor_cap(30, 0.09, 150.0)
    e = np.asarray(elec, float)
    pos, ori = _dipoles(5, 0.07)
    names = [f"E{i}" for i in range(len(e))]
    info = mne.create_info(names, 1000.0, "eeg")
    info.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(names, e)), coord_frame="head"))
    sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=0.09,
                                   relative_radii=(0.99, 1.0), sigmas=(0.33, 0.33))
    dip = mne.Dipole(times=np.arange(5) * 1e-3, pos=pos, amplitude=np.ones(5), ori=ori,
                     gof=np.ones(5))
    fwd, _ = mne.make_forward_dipole(dip, sphere, info)
    ref = fwd["sol"]["data"]
    ref = ref - ref.mean(0, keepdims=True)
    ours = np.asarray(eeg_sphere_leadfield(e, pos, ori, conductivity_s_per_m=0.33), float)
    scale = float(np.abs(ref).max())
    error = float(np.abs(ours - ref).max())
    tolerance = 1e-4 * scale + 1e-3 * scale
    return _row(error, tolerance, "MNE single-shell forward (homogeneous)", error <= tolerance)


def check_eeg_scaling():
    """Conductivity reciprocity: sigma 0.33 against twice sigma 0.66 (rtol 1e-5)."""
    elec, _ = sphere_sensor_cap(20, 0.09, 150.0)
    pos, ori = _dipoles(3, 0.06)
    a = np.asarray(eeg_sphere_leadfield(elec, pos, ori, conductivity_s_per_m=0.33), float)
    b = np.asarray(eeg_sphere_leadfield(elec, pos, ori, conductivity_s_per_m=0.66), float)
    scale = float(np.abs(a).max())
    error = float(np.abs(a - 2 * b).max())
    tolerance = 1e-5 * scale
    return _row(error, tolerance, "conductivity reciprocity (0.33 against 0.66 S/m)",
                error <= tolerance)


def check_eeg_reference():
    """Average-reference null: referenced column means against zero (atol 1e-6 of scale)."""
    elec, _ = sphere_sensor_cap(20, 0.09, 150.0)
    pos, ori = _dipoles(3, 0.06)
    a = np.asarray(eeg_sphere_leadfield(elec, pos, ori, conductivity_s_per_m=0.33), float)
    raw = np.asarray(eeg_sphere_leadfield(elec, pos, ori, conductivity_s_per_m=0.33,
                                          average_reference=False), float)
    assert np.abs(raw.mean(0)).max() > 1e-3 * np.abs(raw).max(), "fixture must leave a mean to remove"
    scale = float(np.abs(a).max())
    error = float(np.abs(a.mean(0)).max())
    tolerance = 1e-6 * scale
    return _row(error, tolerance, "average-reference null", error <= tolerance)


CHECKS = (
    ("meg_radial", check_meg_radial),
    ("meg_full_vector", check_meg_full_vector),
    ("meg_radial_silence", check_meg_radial_silence),
    ("eeg_neumann", check_eeg_neumann),
    ("eeg_mne", check_eeg_mne),
    ("eeg_scaling", check_eeg_scaling),
    ("eeg_reference", check_eeg_reference),
)


def compute_all():
    """One score row per check, in CHECKS order."""
    checks = {name: fn() for name, fn in CHECKS}
    live = [row for row in checks.values() if "skipped" not in row]
    return {"checks": checks,
            "pass": bool(live) and all(row["pass"] for row in live),
            "skipped": sorted(name for name, row in checks.items() if "skipped" in row)}


def compare(old, new, rtol=RTOL):
    """Mismatch strings between a stored document and a fresh one; [] means agreement."""
    if set(old.get("checks", ())) != set(new.get("checks", ())):
        return [f"checks: {sorted(old.get('checks', ()))} != {sorted(new.get('checks', ()))}"]
    bad = []
    for name in old["checks"]:
        o, n = old["checks"][name], new["checks"][name]
        if "skipped" in n and "skipped" not in o:
            # This environment lacks the check's optional dependency; the stored row stands.
            continue
        if ("skipped" in o) != ("skipped" in n):
            bad.append(f"{name}: skipped {('skipped' in o)} != {('skipped' in n)}")
            continue
        if "skipped" in o:
            if o["skipped"] != n["skipped"]:
                bad.append(f"{name}.skipped: {o['skipped']!r} != {n['skipped']!r}")
            continue
        if o["pass"] != n["pass"]:
            bad.append(f"{name}.pass: {o['pass']!r} != {n['pass']!r}")
        for key in ("error", "tolerance"):
            if math.isclose(o[key], n[key], rel_tol=rtol, abs_tol=1e-15):
                continue
            # A rounding-level residual differs across platforms (BLAS, float32); below a tenth
            # of the tolerance in both copies it cannot change the verdict.
            floor = NOISE_FRACTION * min(o["tolerance"], n["tolerance"])
            if key == "error" and o[key] <= floor and n[key] <= floor:
                continue
            bad.append(f"{name}.{key}: {o[key]!r} != {n[key]!r} past rtol {rtol:g}")
    if old.get("pass") != new.get("pass"):
        bad.append(f"pass: {old.get('pass')!r} != {new.get('pass')!r}")
    return bad


README_TEXT = """# Head-forward analytic checks

Worst-case deviations of the spherical-head forward code in `jaxfne.fields.head`
(Sarvas 1987 MEG; Frank 1952 homogeneous-sphere EEG) against independent references,
recomputed by `scripts/score_head_forward_checks.py` with the inputs and tolerances of
`tests/test_head_forward.py`. `score.json` holds one `{error, tolerance, pass, reference}`
row per check; `error` is the worst-case absolute deviation and `tolerance` is
`atol + rtol·max|ref|`, so a passing test implies a passing row. These are analytic
agreements of a core method, not a paper replication: there is no paper value beside ours.

| check | reference | tolerance |
|---|---|---|
| meg_radial | primary-current Biot-Savart field | 1e-6·scale + 2e-4·scale |
| meg_full_vector | scalar-potential quadrature (-mu0 grad U) | 1e-4·scale + 1e-3·scale |
| meg_radial_silence | radial-to-tangential amplitude ratio | ratio below 1e-6 |
| eeg_neumann | Neumann-series potential (400 terms) | 1e-4·scale + 1e-3·scale |
| eeg_mne | MNE single-shell forward (homogeneous) | 1e-4·scale + 1e-3·scale |
| eeg_scaling | conductivity reciprocity (0.33 against 0.66 S/m) | 1e-5·scale |
| eeg_reference | average-reference null | column means below 1e-6·scale |

Rerun:

    python scripts/score_head_forward_checks.py         # rewrite score.json (this README is left as is)
    python scripts/score_head_forward_checks.py --check # recompute and exit 1 past rtol 1e-9

`eeg_mne` needs the optional `mne` package; without it the row keeps nulls with
`"skipped": "mne not installed"` and its dashboard criterion shows pending. The suite runs
under `JAX_PLATFORMS=cpu`.
"""


def write(doc):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCORE_PATH, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    if not README_PATH.is_file():
        README_PATH.write_text(README_TEXT, encoding="utf-8", newline="\n")


def check(doc):
    try:
        stored = json.loads(SCORE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"error: cannot read {SCORE_PATH}: {exc}", file=sys.stderr)
        return 1
    bad = compare(stored, doc)
    if bad:
        for line in bad:
            print(f"mismatch: {line}", file=sys.stderr)
        return 1
    print(f"fresh: {SCORE_PATH} matches a recomputation within rtol {RTOL:g}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score the analytic head-forward checks.")
    parser.add_argument("--check", action="store_true",
                        help="recompute and exit 1 past rtol 1e-9")
    args = parser.parse_args(argv)
    doc = compute_all()
    if args.check:
        return check(doc)
    write(doc)
    print(f"wrote {SCORE_PATH} ({len(doc['checks'])} checks, pass={doc['pass']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
