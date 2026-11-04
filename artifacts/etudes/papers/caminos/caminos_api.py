"""CAMINOS P6 library-path runner (PREREG_P6.md, signed v2).

Builds the CAMINOS E-PV-SOM network only from the jaxfne public API and runs
it through B0 route (c): the constructed model simulated through
``Model.simulate`` (the adex path runs the ``simulate_adex`` kernel; the
recorded ``V_m`` is never read). No scored run:
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

import jaxfne as jtfne  # noqa: E402

assert "site-packages" not in jtfne.__file__, "caminos_api must run the checkout's jaxfne"

from jaxfne.analysis import ref42_lfp, ref42_phase_delay  # noqa: E402

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

# ref42_lfp / ref42_phase_delay are the jaxfne.analysis core homes imported
# above (same callables; lane constants above hold the declared protocol values).


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
                "runs": list(entries.values()),
                "backend": "edge_list",
                "dtype": "float32",
                "seeds": sorted(e["seed"] for e in entries.values()),
                "activity_regime": {
                    "kind": "declared",
                    "source": "CALIBRATION.md: primary 2 ms run, seeds 1-3 and 4-8",
                    "scope": "valid for the calibrated seed set 1-8 only; other "
                             "seeds or backends must re-declare from their runs",
                    "oscillates": True,
                    "f_pk_hz": 34.0,
                    "lag_ref42_ms": {"seeds_1_3": 6.47, "seeds_4_8": 6.32},
                    "mean_rates_hz": {"E": 1.08, "PV": 11.24, "SOM": 6.74}}}
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
    """B0 route (c): construct the model and simulate it through ``Model.simulate``.

    Writes the port's per-step population count array and ``manifest.json``
    exactly as ``caminos_analysis.score`` reads them. Refuses any ``out_dir``
    with a path part containing "scored" unless ``signed``. ``delay_ms`` None is
    the port's 2 ms; a value is a declared sensitivity and is recorded in the manifest.
    A delay resolving to 0 steps runs the intra-step fixed point (``intra_step_rounds``
    selects R, default ``ZERO_DELAY_ROUNDS``; recorded in the manifest next to
    ``delay_ms``) with a same-step drive; a rounds override with a delay of >= 1
    step is refused.
    The adex ``Model.simulate`` path runs the same ``simulate_adex`` kernel the
    direct route ran, so the spikes are bit-identical; the recorded ``V_m`` is
    never read here.
    """
    import numpy as np

    _refuse_scored_dir(out_dir, signed)
    _refuse_existing(out_dir, run_filename(seed, p), git_sha())  # before the compute
    import jaxfne as jtfne

    start_time = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    model = jtfne.construct(build_spec(p, seed, delay_ms, intra_step_rounds))
    sim = build_sim(p, seed, T_s, ext_sources, delay_ms=delay_ms,
                    intra_step_rounds=intra_step_rounds)
    runtime_cfg = sim.resolved_runtime
    if str(runtime_cfg.actual_dtype) != "float32":
        raise SystemExit("error: P6 runs are float32, runtime resolved %s"
                         % runtime_cfg.actual_dtype)
    signals = model.simulate(sim)
    import jax

    jax.block_until_ready(signals.spikes)
    labels = tuple(model.params["emitter"].labels)
    sizes = {t: labels.count(t) for t in POPS}
    counts = spikes_to_counts(np.asarray(signals.spikes), sizes)
    wall_s = time.time() - t0
    fixed_point = signals.metadata.get("intra_step_fixed_point")
    write_run(out_dir, p=p, seed=seed, T_s=T_s, counts=counts, sizes=sizes,
              wall_s=wall_s, start_time=start_time, sha=git_sha(),
              ext_sources=ext_sources, delay_ms=delay_ms,
              intra_step_rounds=intra_step_rounds,
              max_rounds_used=(int(fixed_point["max_rounds_used"])
                               if fixed_point is not None else None))
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


# --- Cell Reports 2025 cellrep set on the core route (no oracle) ---

# Cell Reports 2025 notebook cell 1 (caminos_jax.py CELLREP_PARAMS, SI units
# converted): AdEx cells (E b 130 pA, EL -60 mV; PV EL -60 mV; SOM Delta 1.5
# ms, b 25 pA; all else the core AdEx defaults), per-target excitatory decays
# (5/1/2 ms), GABA decay 5 ms, weights in nS, zero synaptic delay, p 0.017
# (E -> PV at twice p), 8000/800/600 cells, round(p*NE) = 136 private Poisson
# sources at 4 Hz to E and PV (none to SOM).
CELLREP_NE = 8000
CELLREP_NPV = 800
CELLREP_NSOM = 600
CELLREP_P = 0.017
CELLREP_E_PV_MULT = 2.0
CELLREP_EXT_FIXED = 136
CELLREP_EXT_RATE_HZ = 4.0
CELLREP_Q_EXT_NS = 1.25
CELLREP_Q_E_E_NS = 1.25
CELLREP_Q_E_PV_NS = 3.75
CELLREP_Q_E_SOM_NS = 2.5
CELLREP_Q_PV_E_NS = 6.0
CELLREP_Q_PV_PV_NS = 6.0
CELLREP_Q_SOM_E_NS = 5.0
CELLREP_Q_SOM_PV_NS = 5.0
CELLREP_TAU_E_E_MS = 5.0
CELLREP_TAU_E_PV_MS = 1.0
CELLREP_TAU_E_SOM_MS = 2.0
CELLREP_TAU_I_MS = 5.0
CELLREP_REV_E_MV = 0.0
CELLREP_REV_I_MV = -80.0

# (name, pre, post, weight nS, receptor, probability multiplier on p).
CELLREP_RULES = (
    ("cellrep_e_e", "E", "E", CELLREP_Q_E_E_NS, "ampa_e", 1.0),
    ("cellrep_e_pv", "E", "PV", CELLREP_Q_E_PV_NS, "ampa_pv", CELLREP_E_PV_MULT),
    ("cellrep_e_som", "E", "SOM", CELLREP_Q_E_SOM_NS, "ampa_som", 1.0),
    ("cellrep_pv_e", "PV", "E", CELLREP_Q_PV_E_NS, "gaba", 1.0),
    ("cellrep_pv_pv", "PV", "PV", CELLREP_Q_PV_PV_NS, "gaba", 1.0),
    ("cellrep_som_e", "SOM", "E", CELLREP_Q_SOM_E_NS, "gaba", 1.0),
    ("cellrep_som_pv", "SOM", "PV", CELLREP_Q_SOM_PV_NS, "gaba", 1.0),
)


def _cellrep_sizes(sizes):
    """Cell counts from a ``{"NE", "NPV", "NSOM"}`` dict, or the cellrep defaults."""
    if sizes is None:
        return {"E": CELLREP_NE, "PV": CELLREP_NPV, "SOM": CELLREP_NSOM}
    missing = {"NE", "NPV", "NSOM"} - set(sizes)
    if missing:
        raise ValueError("sizes must have the keys NE, NPV, NSOM; missing %s"
                         % sorted(missing))
    out = {"E": int(sizes["NE"]), "PV": int(sizes["NPV"]), "SOM": int(sizes["NSOM"])}
    if any(v <= 0 for v in out.values()):
        raise ValueError("sizes must hold positive cell counts, got %r" % (sizes,))
    return out


def _cellrep_cell_overrides(swap):
    """Per-type AdEx overrides against the core defaults (only differing fields).

    The core defaults already carry the PLoS CB Table 1 values, so the cellrep
    set overrides E (EL -60 mV, b 130 pA), PV (EL -60 mV) and SOM (Delta
    1.5 ms, b 25 pA). With ``swap`` the PV and SOM neuronal values are
    exchanged (the oracle ``swap``: connectivity, decays, weights and drive
    untouched, the 4 Hz drive stays on the PV indices).
    """
    ov = {"E": {"EL_mV": -60.0, "b_pA": 130.0}}
    if not swap:
        ov["PV"] = {"EL_mV": -60.0}
        ov["SOM"] = {"Delta_mV": 1.5, "b_pA": 25.0}
        return ov
    ov["PV"] = {"EL_mV": -55.0, "Delta_mV": 1.5, "a_nS": 4.0, "b_pA": 25.0}
    ov["SOM"] = {"EL_mV": -60.0, "Delta_mV": 0.5, "a_nS": 0.0, "b_pA": 0.0}
    return ov


def _cellrep_coupling_nS(coupling):
    """Coupling weights in nS: the oracle takes S per connection (a dict with
    keys ``"PV_SOM"`` and/or ``"SOM_SOM"``); unknown keys and negative values
    are refused."""
    if coupling is None:
        return {}
    if not isinstance(coupling, dict):
        raise ValueError("coupling must be a dict with keys 'PV_SOM' and/or "
                         "'SOM_SOM', got %r" % (coupling,))
    unknown = sorted(set(coupling) - {"PV_SOM", "SOM_SOM"})
    if unknown:
        raise ValueError("coupling keys must be 'PV_SOM' and/or 'SOM_SOM', got %r"
                         % (unknown,))
    out = {}
    for key in ("PV_SOM", "SOM_SOM"):
        if key in coupling:
            try:
                val = float(coupling[key])
            except (TypeError, ValueError):
                raise ValueError("coupling weights must be numbers (S per connection), "
                                 "got %r" % (coupling,)) from None
            import math

            if not math.isfinite(val) or val < 0.0:
                raise ValueError("coupling weights must be finite and >= 0, got %r"
                                 % (coupling,))
            out[key] = val * 1e9
    return out


def _cellrep_check_p(p):
    """The connection probability: in (0, 0.5] so E -> PV at twice p stays a probability."""
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        raise ValueError("cellrep p must be in (0, 0.5], got %r" % (p,))
    if not 0.0 < float(p) <= 0.5:
        raise ValueError("cellrep p must be in (0, 0.5] (E -> PV connects at "
                         "twice p), got %r" % (p,))
    return float(p)


def build_cellrep_spec(p=CELLREP_P, seed=0, sizes=None, swap=False, coupling=None):
    """The cellrep network built only from the public API.

    Seven Bernoulli rules (PV -> SOM and SOM -> SOM absent, as in the
    reference model) plus one rule per declared ``coupling`` entry
    (``{"PV_SOM": S, "SOM_SOM": S}``, weights in S per connection). The
    oracle's zero synaptic delay maps to one step (``delay_ms`` = the run's
    ``dt_ms``): the one-step ring without the intra-step fixed point.
    """
    import jaxfne as jtfne

    pp = _cellrep_check_p(p)
    coup = _cellrep_coupling_nS(coupling)
    sz = _cellrep_sizes(sizes)
    n = sz["E"] + sz["PV"] + sz["SOM"]
    cfg = (
        jtfne.Configuration()
        .network(name="cellrep", n=n,
                 cell_types={"E": sz["E"] / n, "PV": sz["PV"] / n, "SOM": sz["SOM"] / n})
        .emitter(family="adex", adex_decay="euler", adex_step_order="default")
        .mechanisms(name="ampa_e", kind="receptor",
                    params={"tau_ms": CELLREP_TAU_E_E_MS, "reversal_mV": CELLREP_REV_E_MV})
        .mechanisms(name="ampa_pv", kind="receptor",
                    params={"tau_ms": CELLREP_TAU_E_PV_MS, "reversal_mV": CELLREP_REV_E_MV})
        .mechanisms(name="ampa_som", kind="receptor",
                    params={"tau_ms": CELLREP_TAU_E_SOM_MS, "reversal_mV": CELLREP_REV_E_MV})
        .mechanisms(name="gaba", kind="receptor",
                    params={"tau_ms": CELLREP_TAU_I_MS, "reversal_mV": CELLREP_REV_I_MV})
    )
    for ctype, fields in _cellrep_cell_overrides(swap).items():
        cfg = cfg.cell_params({"cell_type": ctype}, fields)
    for name, pre, post, weight, mechanism, mult in CELLREP_RULES:
        cfg = cfg.connections(
            name=name,
            source={"cell_type": pre},
            target={"cell_type": post},
            probability=pp * mult,
            weight=weight,
            mechanism=mechanism,
            delay_ms=DT_MS,
            sampling="bernoulli",
        )
    for cname, pair, key in (("cellrep_pv_som", ("PV", "SOM"), "PV_SOM"),
                             ("cellrep_som_som", ("SOM", "SOM"), "SOM_SOM")):
        if key in coup:
            cfg = cfg.connections(
                name=cname,
                source={"cell_type": pair[0]},
                target={"cell_type": pair[1]},
                probability=pp,
                weight=coup[key],
                mechanism="gaba",
                delay_ms=DT_MS,
                sampling="bernoulli",
            )
    return cfg.connectivity(edge_seed=int(seed))


def cellrep_ext_source_count(p, NE, ext_sources="pNE"):
    """External source count for the cellrep drive: the set's fixed 136 by
    default, ``round(p * NE)`` under ``ext_sources="pNE"``, or a positive int
    as given."""
    if ext_sources == "fixed":
        return CELLREP_EXT_FIXED
    if ext_sources == "pNE":
        return int(round(float(p) * int(NE)))
    if isinstance(ext_sources, bool) or not isinstance(ext_sources, int) \
            or int(ext_sources) <= 0:
        raise ValueError("ext_sources must be 'fixed', 'pNE' or a positive int, got %r"
                         % (ext_sources,))
    return int(ext_sources)


def cellrep_drive_entries(p, NE, seed, ext_sources="pNE", som_ext_rate_hz=0.0,
                          ext_rate_schedule=None, som_ext_rate_schedule=None):
    """The cellrep ``Simulation(poisson_drive=...)`` list: one entry per driven
    group (E on ``ampa_e``, PV on ``ampa_pv``) under its own seed stream
    (``seed``, ``seed + 1``), plus SOM on ``ampa_som`` (``seed + 2``) only
    when the SOM drive is on. ``ext_rate_schedule`` replaces the constant 4 Hz
    E/PV rate with a per-step Hz array; ``som_ext_rate_schedule`` is the
    per-step counterpart of ``som_ext_rate_hz`` (refused together with a
    nonzero constant rate, as in the oracle). A zero synaptic delay is one
    step here, so no ``same_step`` entry is needed.
    """
    import math

    n = cellrep_ext_source_count(p, NE, ext_sources)
    try:
        som_rate = float(som_ext_rate_hz)
    except (TypeError, ValueError):
        raise ValueError("som_ext_rate_hz must be a number (Hz), got %r"
                         % (som_ext_rate_hz,)) from None
    if not math.isfinite(som_rate) or som_rate < 0.0:
        raise ValueError("som_ext_rate_hz must be finite and >= 0, got %r"
                         % (som_ext_rate_hz,))
    if som_ext_rate_schedule is not None and som_rate != 0.0:
        raise ValueError("pass only one SOM drive: som_ext_rate_hz=%r with "
                         "som_ext_rate_schedule set" % (som_ext_rate_hz,))
    entries = [
        {"n_sources": n, "rate_hz": CELLREP_EXT_RATE_HZ, "weight_nS": CELLREP_Q_EXT_NS,
         "mechanism": "ampa_e", "target_cell_types": ["E"],
         "delay_ms": DT_MS, "seed": int(seed)},
        {"n_sources": n, "rate_hz": CELLREP_EXT_RATE_HZ, "weight_nS": CELLREP_Q_EXT_NS,
         "mechanism": "ampa_pv", "target_cell_types": ["PV"],
         "delay_ms": DT_MS, "seed": int(seed) + 1},
    ]
    if ext_rate_schedule is not None:
        for entry in entries:
            del entry["rate_hz"]
            entry["rate_schedule_hz"] = ext_rate_schedule
    if som_rate != 0.0 or som_ext_rate_schedule is not None:
        som_entry = {"n_sources": n, "rate_hz": som_rate, "weight_nS": CELLREP_Q_EXT_NS,
                     "mechanism": "ampa_som", "target_cell_types": ["SOM"],
                     "delay_ms": DT_MS, "seed": int(seed) + 2}
        if som_ext_rate_schedule is not None:
            del som_entry["rate_hz"]
            som_entry["rate_schedule_hz"] = som_ext_rate_schedule
        entries.append(som_entry)
    return entries


def build_cellrep_sim(seed, T_s, drive_entries, silence=None, replace=None,
                      replace_rate_hz=None, silence_onset_s=None,
                      ee_gain_schedule=None, dt_ms=DT_MS):
    """The ``Simulation`` for a cellrep core run: dt 0.1 ms, the drive list,
    and the optional ``silence`` / ``replace`` / ``ee_gain_schedule`` (E -> E)
    manipulations. ``silence_onset_s`` without ``silence`` is refused, as in
    the oracle; ``replace`` needs ``replace_rate_hz`` (a dict with a rate per
    replaced type)."""
    import jaxfne as jtfne

    silence_spec = None
    if silence is not None:
        if silence_onset_s is None:
            onset_ms = 0.0
        else:
            try:
                onset_ms = float(silence_onset_s) * 1000.0
            except (TypeError, ValueError):
                raise ValueError("silence_onset_s must be seconds as a number, got %r"
                                 % (silence_onset_s,)) from None
        silence_spec = {"fractions": dict(silence), "seed": int(seed),
                        "onset_ms": onset_ms}
    elif silence_onset_s is not None:
        raise ValueError("silence_onset_s requires silence to be set, got %r"
                         % (silence_onset_s,))
    replace_spec = None
    if replace is not None:
        if not isinstance(replace_rate_hz, dict) or not replace_rate_hz:
            raise ValueError("replace_rate_hz (dict of mean rate in Hz per type) "
                             "is required when replace is set, got %r"
                             % (replace_rate_hz,))
        replace_spec = {"fractions": dict(replace), "rate_hz": dict(replace_rate_hz),
                        "seed": int(seed)}
    edge_gain = None
    if ee_gain_schedule is not None:
        edge_gain = {"pre": "E", "post": "E", "schedule": ee_gain_schedule}
    return jtfne.Simulation(duration_ms=float(T_s) * 1000.0, dt_ms=float(dt_ms),
                            poisson_drive=list(drive_entries),
                            silence=silence_spec, replace=replace_spec,
                            edge_gain=edge_gain)


def run_core_raster(seed, T_s=12.0, sizes=None, p=CELLREP_P, silence=None,
                    replace=None, replace_rate_hz=None, silence_onset_s=None,
                    coupling=None, swap=False, ext_sources="pNE",
                    som_ext_rate_hz=0.0, ext_rate_schedule=None,
                    som_ext_rate_schedule=None, ee_gain_schedule=None,
                    dt_ms=DT_MS, **unknown):
    """Run the cellrep network through the jaxfne core; return the boolean
    spike raster ``[steps, N]`` in E | PV | SOM order.

    The network (``build_cellrep_spec``), drive (``cellrep_drive_entries``)
    and manipulations (``Simulation`` silence / replace / edge_gain) run on
    the public API only; no oracle code is reached. ``sizes`` is
    ``{"NE", "NPV", "NSOM"}`` or None for the cellrep defaults; ``p`` the
    connection probability (E -> PV connects at twice p); any other keyword
    is refused with ``ValueError`` naming it.

    Core-vs-oracle differences (all spike-changing; the core path is not tuned
    to match the oracle): connectivity draws (the oracle splits ``PRNGKey``
    per block; the core compiles from ``edge_seed`` through its own stream),
    external-drive draws (the oracle draws Poisson counts inside its scan;
    the core draws one stream per drive entry under ``seed``, ``seed + 1``,
    ``seed + 2``), and replacement-spike draws (distinct streams). The
    silenced/replaced subsets coincide for the same seed (both use the
    fold-in-911/417 ``manip_mask`` rule over E | PV | SOM order). The oracle's
    zero synaptic delay is one step on both paths (the oracle's one-step
    floor; the core's ``delay_ms`` = dt).
    """
    if unknown:
        raise ValueError("run_core_raster got unknown keyword(s) %s; realized: "
                         "T_s, sizes, p, silence, replace, replace_rate_hz, "
                         "silence_onset_s, coupling, swap, ext_sources, "
                         "som_ext_rate_hz, ext_rate_schedule, "
                         "som_ext_rate_schedule, ee_gain_schedule, dt_ms"
                         % sorted(unknown))
    import numpy as np

    import jaxfne as jtfne

    sz = _cellrep_sizes(sizes)
    model = jtfne.construct(build_cellrep_spec(p=p, seed=seed, sizes=sizes,
                                               swap=swap, coupling=coupling))
    entries = cellrep_drive_entries(p, sz["E"], seed, ext_sources,
                                    som_ext_rate_hz, ext_rate_schedule,
                                    som_ext_rate_schedule)
    sim = build_cellrep_sim(seed, T_s, entries, silence, replace,
                            replace_rate_hz, silence_onset_s,
                            ee_gain_schedule, dt_ms)
    signals = model.simulate(sim)
    import jax

    jax.block_until_ready(signals.spikes)
    raster = np.asarray(signals.spikes, dtype=bool)
    assert raster.shape[1] == sz["E"] + sz["PV"] + sz["SOM"]
    return raster
