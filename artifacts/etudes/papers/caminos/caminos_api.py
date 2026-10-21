"""CAMINOS P6 library-path runner (PREREG_P6.md, signed v2).

Builds the CAMINOS E-PV-SOM network only from the jaxfne public API and runs
it through B0 route (c): the model's compiled ``adex_kernel_args`` /
``adex_external_drive_kwargs`` called through the ADVANCED kernel
``jaxfne.emitters_adex.simulate_adex`` with ``record_v=False``. No scored run:
outputs land in the caller's directory, never under a ``scored`` path.

Every biophysical value comes from ``PORT_SPEC.md`` Tables 1-2; nothing is
invented. Units are jaxfne's (mV, ms, pF, nS, pA).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent

_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# PORT_SPEC.md Table 2 and Methods text.
NE = 8000
NPV = 750
NSOM = 750
N = NE + NPV + NSOM
DT_MS = 0.1
DELAY_MS = 2.0  # Table 2, tau_d, all synapses
TAU_E_MS = 2.0  # Table 2, tau_E
TAU_I_MS = 8.0  # Table 2, tau_I
REV_E_MV = 0.0  # Table 2, E_rev (E)
REV_I_MV = -80.0  # Table 2, E_rev (I)
Q_E_E = 2.0  # Table 2, Q E -> E (nS)
Q_E_INH = 3.0  # Table 2, Q E -> PV, SOM (nS)
Q_INH = 5.0  # Table 2, Q PV, SOM -> E, PV (nS)
Q_EXT = 2.5  # Table 2, Q Ext -> E, PV (nS)
EXT_RATE_HZ = 4.0  # Methods, rate per external source
DT_S = 1e-4  # Methods, Euler step in seconds

# The seven realized blocks: PV -> SOM and SOM -> SOM are absent in the
# reference model, so no rules are declared for them.
RULES = (
    ("e_e", "E", "E", Q_E_E, "ampa"),
    ("e_pv", "E", "PV", Q_E_INH, "ampa"),
    ("e_som", "E", "SOM", Q_E_INH, "ampa"),
    ("pv_e", "PV", "E", Q_INH, "gaba"),
    ("pv_pv", "PV", "PV", Q_INH, "gaba"),
    ("som_e", "SOM", "E", Q_INH, "gaba"),
    ("som_pv", "SOM", "PV", Q_INH, "gaba"),
)

POPS = ("E", "PV", "SOM")


EXT_SOURCES_FIXED = 560  # v6 fixed drive, used only by the C1 control (PREREG_P6.md)


def ext_source_count(p, ext_sources="pNE"):
    """Private Poisson sources per E and PV cell: ``round(p * NE)`` (PREREG_P6.md), or the v6
    fixed 560 for the C1 control."""
    if ext_sources == "pNE":
        return int(round(float(p) * NE))
    if ext_sources == "fixed":
        return EXT_SOURCES_FIXED
    raise ValueError("ext_sources must be 'pNE' or 'fixed', got %r" % (ext_sources,))


def _delay(delay_ms):
    """The synaptic and drive delay in ms: the port's 2 ms (Table 2) unless a declared override."""
    return DELAY_MS if delay_ms is None else float(delay_ms)


# True zero delay (core gap G12, closed in T-22 phases 1 and 2): a delay
# resolving to 0 steps at the run's dt runs the intra-step fixed point with
# this round cap on every connection, and the drive with ``"same_step": True``.
# The fixed point runs to quiescence; a step that reaches the cap with undelivered
# events refuses the run (T-22, Hamm 2026-10-18). The cap costs nothing below it.
# The rounds a run used are recorded as ``max_rounds_used`` (CALIBRATION.md section 11).
ZERO_DELAY_ROUNDS = 64


def _validate_rounds(rounds):
    """A declared intra-step round count: a positive int (a bool is refused)."""
    import numbers

    if isinstance(rounds, bool) or not isinstance(rounds, numbers.Integral):
        raise ValueError("intra_step_rounds must be a positive integer; got %r" % (rounds,))
    value = int(rounds)
    if value < 1:
        raise ValueError("intra_step_rounds must be a positive integer; got %r" % (rounds,))
    return value


def _is_zero_step_delay(delay_ms, dt_ms=DT_MS):
    """Whether ``delay_ms`` realizes to 0 steps at ``dt_ms`` (round half up, the
    core's rule): delay 0, or a positive delay inside one step."""
    import math

    ms = float(delay_ms)
    return ms >= 0.0 and int(math.floor(ms / float(dt_ms) + 0.5)) == 0


def _zero_delay_rounds(delay_ms, intra_step_rounds):
    """The fixed-point R for a run at ``delay_ms``: the override when given, the
    module default on the zero-delay path, None on the delay>=1 path. A rounds
    override with a delay of >= 1 step is refused (rounds admit only a delay
    rounding to 0 steps)."""
    if _is_zero_step_delay(delay_ms):
        if intra_step_rounds is None:
            return ZERO_DELAY_ROUNDS
        return _validate_rounds(intra_step_rounds)
    if intra_step_rounds is not None:
        raise ValueError(
            "intra_step_rounds=%r with delay_ms=%r at dt_ms=%g realizes to >= 1 step; "
            "intra_step_rounds admits only a delay rounding to 0 steps -- omit it or "
            "set a sub-step delay" % (intra_step_rounds, delay_ms, DT_MS))
    return None


def drive_spec(p, seed, ext_sources="pNE", gain=None, delay_ms=None, intra_step_rounds=None):
    """The ``Simulation(poisson_drive=...)`` dict for the CAMINOS drive.

    ``gain`` None keeps 4 Hz per source. A gain ``a`` sets the mean drive to
    ``560 + a * (n - 560)`` sources at 4 Hz, ``n`` the source count (RESIDUAL_PROTOCOL.md
    step 3). The core draws one Poisson count per cell with mean
    ``n_sources * rate_hz * dt``, so ``n_sources`` stays at ``n`` and ``rate_hz`` carries the
    scaling. A gain that makes the mean drive non-positive is refused. ``delay_ms`` None keeps
    the port's 2 ms; a value is a declared sensitivity (the core refuses below one step).
    A delay resolving to 0 steps at the run's dt adds ``"same_step": True`` (the count
    drawn for step t acts at step t); ``intra_step_rounds`` only selects the fixed-point
    R for the connections (validated here), and is refused with a delay of >= 1 step.
    """
    n = ext_source_count(p, ext_sources)
    rate = EXT_RATE_HZ
    if gain is not None:
        n_eff = EXT_SOURCES_FIXED + float(gain) * (n - EXT_SOURCES_FIXED)
        if not n_eff > 0:
            raise ValueError("gain %r gives a non-positive drive (%g sources)" % (gain, n_eff))
        rate = EXT_RATE_HZ * n_eff / n
    d = _delay(delay_ms)
    _zero_delay_rounds(d, intra_step_rounds)
    spec = {
        "n_sources": n,
        "rate_hz": rate,
        "weight_nS": Q_EXT,
        "mechanism": "ampa",
        "target_cell_types": ["E", "PV"],
        "delay_ms": d,
        "seed": int(seed),
    }
    if _is_zero_step_delay(d):
        spec["same_step"] = True
    return spec


def build_spec(p, seed, delay_ms=None, intra_step_rounds=None):
    """The CAMINOS network built only from the public API (PREREG_P6.md table).

    ``delay_ms`` None keeps the port's 2 ms on every connection; a value is a declared
    sensitivity applied to all seven rules. A delay resolving to 0 steps at the run's
    dt declares ``intra_step_rounds=R`` on every connection (R the override when given,
    ``ZERO_DELAY_ROUNDS`` otherwise); a delay of >= 1 step takes the historical path,
    and a rounds override with it is refused.
    """
    import jaxfne as jtfne

    d = _delay(delay_ms)
    rounds = _zero_delay_rounds(d, intra_step_rounds)
    cfg = (
        jtfne.Configuration()
        .network(name="caminos", n=N,
                 cell_types={"E": NE / N, "PV": NPV / N, "SOM": NSOM / N})
        .emitter(family="adex", adex_decay="euler", adex_step_order="default")
        .mechanisms(name="ampa", kind="receptor",
                    params={"tau_ms": TAU_E_MS, "reversal_mV": REV_E_MV})
        .mechanisms(name="gaba", kind="receptor",
                    params={"tau_ms": TAU_I_MS, "reversal_mV": REV_I_MV})
    )
    fixed_point = {} if rounds is None else {"intra_step_rounds": rounds}
    for name, pre, post, weight, mechanism in RULES:
        cfg = cfg.connections(
            name=name,
            source={"cell_type": pre},
            target={"cell_type": post},
            probability=float(p),
            weight=weight,
            mechanism=mechanism,
            delay_ms=d,
            sampling="bernoulli",
            **fixed_point,
        )
    return cfg.connectivity(edge_seed=int(seed))


def build_sim(p, seed, T_s, ext_sources="pNE", gain=None, delay_ms=None, intra_step_rounds=None):
    """The ``Simulation`` for a run: dt 0.1 ms, the ``drive_spec`` drive.

    ``Simulation(seed=)`` stays at its default: the route refuses it, and the
    run's randomness comes from ``connectivity(edge_seed=)`` and the drive
    seed (PREREG_P6.md). ``intra_step_rounds`` is the fixed-point override for a
    zero-delay run (refused with a delay of >= 1 step).
    """
    import jaxfne as jtfne

    return jtfne.Simulation(duration_ms=float(T_s) * 1000.0, dt_ms=DT_MS,
                            poisson_drive=drive_spec(p, seed, ext_sources, gain, delay_ms,
                                                     intra_step_rounds))


def spikes_to_counts(spikes, sizes):
    """Per-step population counts ``[steps, 3]`` in [E | PV | SOM] order.

    ``spikes`` is a boolean ``[steps, N]`` raster whose first ``sizes["E"]``
    columns are E, next ``sizes["PV"]`` PV, rest SOM (the ``build_spec``
    label order). Returns int32, the port's count-array format.
    """
    import numpy as np

    arr = np.asarray(spikes, dtype=bool)
    if arr.ndim != 2:
        raise ValueError("spikes must be [steps, N], got shape %r" % (arr.shape,))
    n_e, n_pv, n_som = int(sizes["E"]), int(sizes["PV"]), int(sizes["SOM"])
    if arr.shape[1] != n_e + n_pv + n_som:
        raise ValueError("spikes has %d cells but sizes sum to %d"
                         % (arr.shape[1], n_e + n_pv + n_som))
    out = np.empty((arr.shape[0], 3), dtype=np.int32)
    out[:, 0] = arr[:, :n_e].sum(axis=1)
    out[:, 1] = arr[:, n_e:n_e + n_pv].sum(axis=1)
    out[:, 2] = arr[:, n_e + n_pv:].sum(axis=1)
    return out


# Ref. 42 phase method (Tahvili, Vinck, di Volo, bioRxiv 10.1101/2025.02.23.639719v1, Methods
# "Measuring spike phase differences" and "Population activity"; dispatcher ruling 2026-10-13).
# Declared here, not stated in the paper: the filter type and band, and the reading of the
# Gaussian kernel width as sigma.
REF42_DROP_S = 2.0  # the first 2 s are dropped
REF42_KEEP_S = 10.0  # 10 s are analysed
REF42_LFP_SIGMA_MS = 1.0  # "1 ms Gaussian kernel", read as sigma
REF42_BUTTER_ORDER = 4  # zero-phase Butterworth, applied with sosfiltfilt


def _ref42_window(n_steps, dt_s):
    """Row bounds (first, end) of the analysed window."""
    fs = 1.0 / float(dt_s)
    i0 = int(round(REF42_DROP_S * fs))
    i1 = i0 + int(round(REF42_KEEP_S * fs))
    if n_steps < i1:
        raise ValueError("run has %d steps; the ref. 42 window needs %d" % (n_steps, i1))
    return i0, i1


def ref42_lfp(counts, dt_s):
    """LFP proxy of the ref. 42 method: E spikes per bin, Gaussian-smoothed, analysed window.

    The smoothing runs on the whole run, so the window edges are not truncated kernels.
    """
    import numpy as np

    c = np.asarray(counts)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError("counts must be [steps, 3], got shape %r" % (c.shape,))
    fs = 1.0 / float(dt_s)
    sigma = REF42_LFP_SIGMA_MS * 1e-3 * fs
    half = int(np.ceil(4.0 * sigma))
    t = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()
    lfp = np.convolve(c[:, 0].astype(float), kernel, mode="same")
    i0, i1 = _ref42_window(c.shape[0], dt_s)
    return lfp[i0:i1]


def ref42_phase_delay(counts, dt_s, f_peak_hz, half_band_hz=5.0):
    """PV->SOM delay by the ref. 42 phase method, from population counts in [E | PV | SOM] order.

    ``counts`` is the per-step group sums of the core raster (``spikes_to_counts``). A circular
    mean over a group's spikes equals the mean over its per-step counts, so no per-cell raster is
    needed. The LFP proxy (``ref42_lfp``) is band-passed zero-phase with a 4th-order Butterworth
    filter at f_peak +- half_band_hz, then Hilbert-transformed. Each group's mean phase is the
    circular mean over its spikes. The delay is the wrapped difference (SOM minus PV) divided by
    2 pi f_peak, in ms; it is positive when SOM fires after PV. PV or SOM with no spikes in the
    window gives ``delay_ms`` None.

    Returns a dict: delay_ms, phase_PV_rad, phase_SOM_rad, n_spikes_PV, n_spikes_SOM,
    f_peak_Hz, half_band_Hz.
    """
    import numpy as np
    from scipy import signal

    c = np.asarray(counts)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError("counts must be [steps, 3], got shape %r" % (c.shape,))
    f_peak = float(f_peak_hz)
    half = float(half_band_hz)
    fs = 1.0 / float(dt_s)
    if not (0.0 < half < f_peak and f_peak + half < fs / 2.0):
        raise ValueError("band must satisfy 0 < half_band_hz < f_peak_hz and "
                         "f_peak_hz + half_band_hz < Nyquist (got half=%r, f=%r)" % (half, f_peak))
    x = ref42_lfp(c, dt_s)
    sos = signal.butter(REF42_BUTTER_ORDER, [(f_peak - half) / (fs / 2.0), (f_peak + half) / (fs / 2.0)],
                        btype="bandpass", output="sos")
    phase = np.angle(signal.hilbert(signal.sosfiltfilt(sos, x)))
    i0, i1 = _ref42_window(c.shape[0], dt_s)
    w = c[i0:i1].astype(float)

    def group_phase(col):
        n = w[:, col].sum()
        if n <= 0:
            return None, 0
        return float(np.angle(np.sum(w[:, col] * np.exp(1j * phase)))), int(round(n))

    pv_phi, n_pv = group_phase(1)
    som_phi, n_som = group_phase(2)
    delay_ms = None
    if pv_phi is not None and som_phi is not None:
        d = float(np.angle(np.exp(1j * (som_phi - pv_phi))))
        delay_ms = d / (2.0 * np.pi * f_peak) * 1e3
    return {"delay_ms": delay_ms, "phase_PV_rad": pv_phi, "phase_SOM_rad": som_phi,
            "n_spikes_PV": n_pv, "n_spikes_SOM": n_som,
            "f_peak_Hz": f_peak, "half_band_Hz": half}


def _refuse_scored_dir(out_dir, signed=False):
    if signed:
        return
    parts = Path(out_dir).resolve().parts
    if any("scored" in part.lower() for part in parts):
        raise SystemExit("error: refusing to write under a directory containing "
                         "'scored': %s" % out_dir)


def git_sha():
    """Git SHA via ``git rev-parse HEAD`` at the repository root."""
    out = subprocess.run(["git", "-C", str(_repo_root), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return "unknown"
    return out.stdout.strip()


def run_filename(seed, p):
    """Count-array file name, in the ``run_caminos.py`` style."""
    return "counts_seed%d_p%g_float32_dt1e-04.npy" % (int(seed), float(p))


def _refuse_existing(out_dir, fname, sha):
    """Refuse a directory holding another code SHA or this run's file; return its manifest."""
    out = Path(out_dir)
    prior = None
    if (out / "manifest.json").exists():
        prior = json.loads((out / "manifest.json").read_text())
        if prior.get("git_sha") != sha:
            raise SystemExit("error: %s holds runs from code %s; this run is %s (one code SHA "
                             "per directory)" % (out, prior.get("git_sha"), sha))
        if any(e["file"] == fname for e in prior["runs"]):
            raise SystemExit("error: refusing to overwrite %s" % (out / fname))
    if (out / fname).exists():
        raise SystemExit("error: refusing to overwrite %s" % (out / fname))
    return prior


def write_run(out_dir, *, p, seed, T_s, counts, sizes, wall_s, start_time, sha,
              ext_sources="pNE", delay_ms=None, intra_step_rounds=None,
              max_rounds_used=None):
    """Write the count array plus the manifest ``caminos_analysis.score`` reads."""
    import numpy as np

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = run_filename(seed, p)
    prior = _refuse_existing(out, fname, sha)
    entries = {e["file"]: e for e in prior["runs"]} if prior else {}
    counts = np.asarray(counts, dtype=np.int32)
    if counts.ndim != 2 or counts.shape[1] != 3:
        raise ValueError("counts must be [steps, 3], got shape %r" % (counts.shape,))
    tmp = out / (fname + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, counts)
    os.replace(tmp, out / fname)
    d = _delay(delay_ms)
    rounds = _zero_delay_rounds(d, intra_step_rounds)
    entry = {"seed": int(seed), "p": float(p), "file": fname,
             "dtype": "float32", "dt": DT_S, "T": float(T_s),
             "pilot": False, "sizes": dict(sizes),
             "indegree": "bernoulli", "ext": "private",
             "step_order": "default", "ext_sources": ext_sources,
             "ext_sources_n": ext_source_count(p, ext_sources),
             "delay_ms": d}
    if rounds is not None:
        entry["intra_step_rounds"] = rounds
        if max_rounds_used is not None:
            entry["max_rounds_used"] = int(max_rounds_used)
    entry.update({"backend": "edge_list",
                  "git_sha": sha, "start_time": start_time,
                  "wall_s": float(wall_s), "steps": int(counts.shape[0])})
    entries[fname] = entry
    manifest = {"git_sha": sha, "start_time": prior["start_time"] if prior else start_time,
                "runs": list(entries.values())}
    try:
        import jax

        manifest["jax_version"] = jax.__version__
        manifest["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception:
        pass
    mtmp = out / "manifest.json.tmp"
    mtmp.write_text(json.dumps(manifest, indent=2))
    os.replace(mtmp, out / "manifest.json")
    return out / fname


def run(p, seed, T_s, out_dir, signed=False, ext_sources="pNE", delay_ms=None,
        intra_step_rounds=None):
    """B0 route (c): construct the model, run the kernel with ``record_v=False``.

    Writes the port's per-step population count array and ``manifest.json``
    exactly as ``caminos_analysis.score`` reads them. Refuses any ``out_dir``
    with a path part containing "scored" unless ``signed``. ``delay_ms`` None is
    the port's 2 ms; a value is a declared sensitivity and is recorded in the manifest.
    A delay resolving to 0 steps runs the intra-step fixed point (``intra_step_rounds``
    selects R, default ``ZERO_DELAY_ROUNDS``; recorded in the manifest next to
    ``delay_ms``) with a same-step drive; a rounds override with a delay of >= 1
    step is refused.
    """
    import numpy as np

    _refuse_scored_dir(out_dir, signed)
    _refuse_existing(out_dir, run_filename(seed, p), git_sha())  # before the compute
    import jaxfne as jtfne
    from jaxfne._construct_adex import adex_external_drive_kwargs, adex_kernel_args
    from jaxfne.emitters_adex import simulate_adex

    start_time = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    model = jtfne.construct(build_spec(p, seed, delay_ms, intra_step_rounds))
    sim = build_sim(p, seed, T_s, ext_sources, delay_ms=delay_ms,
                    intra_step_rounds=intra_step_rounds)
    runtime_cfg = sim.resolved_runtime
    if str(runtime_cfg.actual_dtype) != "float32":
        raise SystemExit("error: P6 runs are float32, runtime resolved %s"
                         % runtime_cfg.actual_dtype)
    kwargs = adex_kernel_args(model, sim.dt_ms, dtype=runtime_cfg.actual_dtype)
    drive_kwargs = adex_external_drive_kwargs(model, sim)
    out = simulate_adex(**kwargs, **drive_kwargs, n_steps=sim.n_steps,
                        dt_ms=sim.dt_ms, record_v=False)
    import jax

    jax.block_until_ready(out["spikes"])
    labels = tuple(model.params["emitter"].labels)
    sizes = {t: labels.count(t) for t in POPS}
    counts = spikes_to_counts(np.asarray(out["spikes"]), sizes)
    wall_s = time.time() - t0
    write_run(out_dir, p=p, seed=seed, T_s=T_s, counts=counts, sizes=sizes,
              wall_s=wall_s, start_time=start_time, sha=git_sha(),
              ext_sources=ext_sources, delay_ms=delay_ms,
              intra_step_rounds=intra_step_rounds,
              max_rounds_used=(int(np.asarray(out["fixed_point_rounds"]).max())
                               if "fixed_point_rounds" in out else None))
    return {"counts": counts, "sizes": sizes, "steps": sim.n_steps,
            "wall_s": wall_s}


def build_parser():
    ap = argparse.ArgumentParser(description="Run the CAMINOS network through the "
                                 "jaxfne public API (PREREG_P6.md B0 route (c)).")
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--T", type=float, required=True,
                    help="duration in seconds")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ext-sources", choices=("pNE", "fixed"), default="pNE",
                    help="drive: round(p*NE) sources, or the v6 fixed 560 (C1 control only)")
    ap.add_argument("--signed", action="store_true",
                    help="allow a path containing 'scored' (signed P6 runs only)")
    ap.add_argument("--delay-ms", type=float, default=None,
                    help="declared sensitivity: synaptic and drive delay in ms (default 2, the port)")
    ap.add_argument("--intra-step-rounds", type=int, default=None,
                    help="fixed-point rounds R for a zero-delay run (default ZERO_DELAY_ROUNDS); "
                    "refused with a delay of >= 1 step")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args.p, args.seed, args.T, args.out, signed=args.signed, ext_sources=args.ext_sources,
        delay_ms=args.delay_ms, intra_step_rounds=args.intra_step_rounds)


if __name__ == "__main__":
    main()
