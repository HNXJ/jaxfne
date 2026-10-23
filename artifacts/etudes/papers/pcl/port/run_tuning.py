"""Fig. 3 tuning run: the trained PCL network (cSTDP3) over the counterphase grating test set (test7).

Each input file test7/<ori>/<sf>/<phase>/<trial>.npz is fed through the numba run path with learning
off, from a fresh copy of the loaded network. --engine core routes the same feed through the
jaxfne core kernel instead (run_core_events below: the feed is transcribed into event-scan
tables and replayed with construct -> Model -> simulate). Output per file, in the port's flat index order:
simple_counts (int32, 5184 cells), complex_counts (int32, 1152 cells), n_events (camera-0 events fed),
wall_s, and the spike times of every cell: simple_times_us (int64) with simple_times_cell (int32), and
complex_times_us (int64) with complex_times_cell (int32). Times are in microseconds, sorted by time then
cell; a cell id is the flat index used by the counts.

Resumable: a file whose output exists is skipped. Writes are atomic (tmp, then rename).
Feed convention: raw file timestamps and camera 0 only, as test_pcl_trained_load.py does.
"""

from __future__ import annotations

import os

# One process per worker: BLAS and numba threads pinned to 1 before numpy is imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402


def _repo_root(start: Path) -> Path:
    """Nearest ancestor that holds pyproject.toml."""
    for cand in (start, *start.parents):
        if (cand / "pyproject.toml").is_file():
            return cand
    raise RuntimeError(f"no pyproject.toml above {start}")


_HERE = Path(__file__).resolve().parent
_ROOT = str(_repo_root(_HERE))
if sys.path[:1] != [_ROOT]:
    sys.path.insert(0, _ROOT)
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402

import pcl_event_ref as ref  # noqa: E402
import pcl_event_ref_numba as nb  # noqa: E402

NETWORK = Path("F:/warehouse/pcl/data/PCL_datasets_and_networks/add_simulations_cSTDP/cSTDP3")
TEST_SET = Path("F:/warehouse/pcl/data/PCL_datasets_and_networks/grating/counterphase_gratingComplex/test7")
OUT_DIR = Path("F:/warehouse/pcl/runs/fig3_cSTDP3")
OUT_S1 = Path("F:/warehouse/pcl/runs/fig3_cSTDP3_S1")  # S1 (P3_fig3.md): local inhibition only
OUT_NOLOCAL = Path("F:/warehouse/pcl/diag_runs/fig3_cSTDP3_nolocal")  # diagnostic: no local inhibition, not scored
OUT_NOLOCAL_SIMPLE = Path("F:/warehouse/pcl/diag_runs/fig3_cSTDP3_nolocal_simple")  # diagnostic, not scored
MODE_DIRS = {"full": OUT_DIR, "local-only": OUT_S1, "no-local": OUT_NOLOCAL,
             "no-local-simple": OUT_NOLOCAL_SIMPLE}  # default out dir per --inhibition mode
ORIS = ("0", "23", "45", "68", "90", "113", "135", "158")  # checked against the test set listing
WORKERS = 12
CENTRAL_COL = (4, 4)  # recorded in the manifest only; no output is filtered by it
CENTRAL_COL_NOTE = (
    "Chosen by the dispatcher: the authors' tuningCurves.ipynb (PCL_Python_Analysis) analyses simple "
    "neuron_id 2560 with neuron_z 0. 2560 = 40 * 64 + 0, and 40 is column (4, 4) on the 9x9 grid with "
    "column = i * 9 + j. Assumes neuron id = column * 64 + depth, the port's flat index. The notebook's "
    "complex comment '#240' does not fit column * 32 + 0, so the complex mapping is not checked here."
)
AXES = "ori/sf_index/phase_index/trial"
AXES_NOTE = (
    "Measured from the events at ori 0 on the 29 px x-profile (x 164-192). Level 2 (sf_index 0..17) is "
    "spatial frequency: the period shrinks with index, 9.67 px at 0, 5.8 px at 9, 2.9 px at 17 (only "
    "under 3 bins per cycle at 17, so its phase is coarse). Level 3 (phase_index 0..10) is phase: phase 10 "
    "equals phase 0 (checked at sf 0, 9 and 17). The shift per step is about one tenth of the period (0.97 px "
    "at sf 0; 0.5 to 0.6 px at sf 9), not a fixed 1 px. Level 4 (trial) is the event realization, 0 or 1."
)

# Process-level state, set by _init_worker.
_PRISTINE = None
_INHIBITION = "full"
_ENGINE = "numba"
# Tensor-scan replay grid: the oracle integrates integer-microsecond event times, so the
# 0.5 ms grid is binary-exact and every event lands on it (500 us bins).
CORE_DT_MS = 0.5
CORE_DT_US = 500
# Honest boundary of the core replay (run_core_events): the scan kernel returns binned
# spikes with counts and first-spike times only (jaxfne/_model_simulate.py), so a file in
# which any cell spikes twice has unrecoverable per-spike times; and the kernel runs frozen
# thresholds, so a file reaching past the network's measurement interval would cross the
# oracle's rate/threshold adaptation (pcl_event_ref.py), which has no scan expression.
# Either file is refused for that file, never approximated.
CORE_MULTISPIKE = ("engine core cannot report this file: a cell spiked twice, and the scan "
                   "replay returns counts with first-spike times only "
                   "(jaxfne/_model_simulate.py); per-spike times are exact only when "
                   "each cell spikes at most once")
CORE_ADAPTATION = ("engine core cannot report this file: its events reach past the "
                   "measurement interval, where the oracle adapts rates and thresholds "
                   "(pcl_event_ref.py) and the frozen-threshold scan has no expression")


def apply_inhibition_mode(net, inhibition: str):
    """Switch off one inhibition family in place; "full" leaves the trained network as it is.

    local-only (S1): zero the lateral and top-down weight maps. Both run paths read the same
    arrays: the pure path uses LAT[m, n] and TD[s, c] (pcl_event_ref.py), and run_numba passes
    net.LAT and net.TD (.T) straight to the kernel (pcl_event_ref_numba.py). The flags, masks
    and local weights are untouched.
    no-local (diagnostic): clear local_s and local_c. The pure path gates on them at
    pcl_event_ref.py:686 and :725; run_numba packs them at pcl_event_ref_numba.py:233-235 and
    the kernel reads them as G_LOCS and G_LOCC (:123, :148, :173). Lateral and top-down maps stay.
    no-local-simple (diagnostic): clear local_s only; complex local inhibition stays.
    """
    if inhibition == "full":
        return net
    if inhibition in ("no-local", "no-local-simple"):
        net.local_s = False
        net.local_c = inhibition == "no-local-simple" and net.local_c
        return net
    if inhibition != "local-only":
        raise ValueError(f"unknown inhibition {inhibition!r}")
    if net.lat_on and net.LAT is not None:
        net.LAT[:] = 0.0
    if net.td_on and net.TD is not None:
        net.TD[:] = 0.0
    return net


def resolve_out_dir(out: str | None, inhibition: str) -> Path:
    """Default output directory for the inhibition mode; an explicit --out wins."""
    if out is not None:
        return Path(out)
    if inhibition == "local-only":
        return OUT_S1
    return MODE_DIRS[inhibition]


def check_out_dir(out: Path, inhibition: str) -> None:
    """Refuse an out dir whose manifest records the other inhibition mode.

    A manifest without an "inhibition" key predates the flag and means the full run.
    No mode is ever written into another mode's default dir, manifest or not.
    """
    out = Path(out)
    for owner, d in MODE_DIRS.items():
        if owner != inhibition and out.resolve() == Path(d).resolve():
            raise ValueError(f"refusing {inhibition} into the {owner} run's dir {out}")
    path = out / "manifest.json"
    if not path.is_file():
        return
    found = json.loads(path.read_text(encoding="utf-8")).get("inhibition", "full")
    if found != inhibition:
        raise ValueError(f"refusing --out {out}: manifest says inhibition={found!r}, "
                         f"asked {inhibition!r}")


def tasks_for(subset: str) -> list[tuple[str, str, str, str]]:
    """(ori, sf_index, phase_index, trial) as directory and file names."""
    if subset == "pilot":
        sfs, phases, trials = (0, 9), (0,), (0,)
    elif subset == "all":
        sfs, phases, trials = range(18), range(11), (0, 1)
    else:
        raise ValueError(f"unknown subset {subset!r}")
    return [(o, str(f), str(p), str(s)) for o in ORIS for f in sfs for p in phases for s in trials]


def _gather(cells: list) -> tuple[np.ndarray, np.ndarray]:
    """Spike times (us) and flat cell ids of all cells, sorted by time then cell id."""
    times, owners = [], []
    for n, cell in enumerate(cells):
        sp = cell.spikes
        times.append(np.asarray(sp, dtype=np.int64))
        owners.append(np.full(len(sp), n, dtype=np.int32))
    t = np.concatenate(times) if times else np.zeros(0, dtype=np.int64)
    o = np.concatenate(owners) if owners else np.zeros(0, dtype=np.int32)
    order = np.lexsort((o, t))
    return t[order], o[order]


def _core_tables(net):
    """Transcribe the loaded network's wiring into event-scan tables, blocks and groups.

    Reads the same arrays the numba path consumes (net.LAT rows-as-targets, net.TD
    transposed to rows-as-sources, pcl_event_ref_numba.py:258-267): lateral and top-down
    pass through untouched, so an inhibition mode's zeroed maps (apply_inhibition_mode)
    replay as zero-weight calls, exactly as the oracle executes them. A layer whose local
    flag is off gets singleton groups with its triplets dropped, which is the oracle's
    skip: with no group-mates the cascade inhibits nobody and stamps nothing.
    """
    from jaxfne.pcl_topology import (event_scan_tables, local_inhibition_blocks,
                                     scan_inhibition_triplets)

    ns, nc = int(net.NS), int(net.NC)
    tables = event_scan_tables(
        nx=net.NX, ny=net.NY, nd=net.ND, cx=net.CX, cy=net.CY, cd=net.CD,
        origins_x=[float(v) for v in net.s_ox], origins_y=[float(v) for v in net.s_oy],
        size=(net.SX, net.SY), weights=np.asarray(net.w_s, dtype=np.float64),
        c_cover=net.c_cover,
        c_ox=[int(v) for v in net.c_ox], c_oy=[int(v) for v in net.c_oy],
        c_size=(net.CSX, net.CSY), c_weights=np.asarray(net.w_c, dtype=np.float64),
        vw=net.VW, vh=net.VH,
        inhib_range=((net.Rx, net.Ry) if net.lat_on else (0, 0)),
        lateral_on=bool(net.lat_on),
        lateral_weights=(np.ascontiguousarray(net.LAT, dtype=np.float64)
                         if net.lat_on else None),
        topdown_on=bool(net.td_on),
        topdown_weights=(np.ascontiguousarray(net.TD.T, dtype=np.float64)
                         if net.td_on else None),
        activation=bool(net.activation))
    groups = np.zeros(ns + nc, dtype=np.int64)
    for i in range(net.NX):
        for j in range(net.NY):
            groups[(i * net.NY + j) * net.ND:(i * net.NY + j + 1) * net.ND] = i * net.NY + j
    for cx in range(net.CX):
        for cy in range(net.CY):
            groups[ns + (cx * net.CY + cy) * net.CD:ns + (cx * net.CY + cy + 1) * net.CD] = (
                net.NX * net.NY + cx * net.CY + cy)
    # Singleton ids ride above every position id, so a disabled layer never shares a
    # group with an enabled one.
    if not net.local_s:
        groups[:ns] = ns + nc + np.arange(ns, dtype=np.int64)
    if not net.local_c:
        groups[ns:] = ns + nc + np.arange(nc, dtype=np.int64)
    pre, post, mag = scan_inhibition_triplets(
        net.NX, net.NY, net.ND, net.CX, net.CY, net.CD, ns,
        np.asarray(net.wi_s, dtype=np.float64), np.asarray(net.wi_c, dtype=np.float64))
    keep = np.ones(pre.shape[0], dtype=bool)
    if not net.local_s:
        keep &= pre >= ns
    if not net.local_c:
        keep &= pre < ns
    blocks = local_inhibition_blocks(groups, li_pre=pre[keep], li_post=post[keep],
                                     li_weight=mag[keep])
    return tables, blocks, groups


def _core_lif_params(net):
    """Per-type LIF parameters from the loaded network's own cell configs.

    The scan kernel defaults to the paper's Table 2 values (simple
    threshold 10 mV), but the replayed oracle integrates whatever the
    network's configs carry (the synthetic default network: 30 mV). Each
    side reads the same scalars: taus back to ms, voltages as mV. Complex
    cells carry no adaptation term, so ``tau_sra_ms`` stays undeclared
    there (its default rides) and ``delta_sra_mV`` is pinned at the
    oracle's 0.
    """
    simple = {
        "tau_m_ms": float(net.scfg.TAU_M) / 1000.0,
        "v_thresh_mV": float(net.scfg.VTHRESH),
        "v_reset_mV": float(net.scfg.VRESET),
        "v_min_mV": float(ref.NEG_LIMIT),
        "eta_rp_mV": float(net.scfg.DELTA_RP),
        "tau_rp_ms": float(net.scfg.TAU_RP) / 1000.0,
        "tau_sra_ms": float(net.scfg.TAU_SRA) / 1000.0,
        "delta_sra_mV": float(net.scfg.DELTA_SRA),
    }
    complex_ = {
        "tau_m_ms": float(net.ccfg.TAU_M) / 1000.0,
        "v_thresh_mV": float(net.ccfg.VTHRESH),
        "v_reset_mV": float(net.ccfg.VRESET),
        "v_min_mV": float(ref.NEG_LIMIT),
        "eta_rp_mV": float(net.ccfg.DELTA_RP),
        "tau_rp_ms": float(net.ccfg.TAU_RP) / 1000.0,
        "delta_sra_mV": float(net.ccfg.DELTA_SRA),
    }
    return simple, complex_


def run_core_events(net, t, x, y, p):
    """Replay one camera-0 event slice through the jaxfne event-scan kernel.

    Returns the same per-file arrays _run_one writes (simple/complex_counts int32,
    times_us int64 with times_cell int32 in the port's flat index order, n_events),
    rebased back onto the file's raw timestamps. The network is read, never mutated.
    Refusals, before any heavy work: unsorted times (the scan sorts; the oracle follows
    file order), an empty slice, events past the measurement interval from the network's
    last update (CORE_ADAPTATION), and any cell spiking twice (CORE_MULTISPIKE).
    """
    import jaxfne as jtfne
    jtfne.enable_x64()  # the scan kernel integrates in float64 (as the oracle does)
    from jaxfne import neuronal_tensor as nt
    from jaxfne.pcl_topology import (EventScanDrive, build_s1_scan_tensor,
                                     event_scan_events)

    t = np.asarray(t, dtype=np.int64).ravel()
    x = np.asarray(x, dtype=np.int64).ravel()
    y = np.asarray(y, dtype=np.int64).ravel()
    p = (np.asarray(p) != 0).astype(np.int64).ravel()  # as run_numba coerces polarity
    if not (t.shape == x.shape == y.shape == p.shape):
        raise ValueError(f"core replay needs one (t, x, y, p) tuple; got {t.shape}")
    if t.size == 0:
        raise ValueError("core replay needs at least one event to anchor t0")
    if bool((np.diff(t) < 0).any()):
        raise ValueError("core replay needs time-ordered events (the oracle follows file order)")
    if int(t.max()) - int(net._last_update) > int(net.measurement_interval):
        raise NotImplementedError(CORE_ADAPTATION)
    ns, nc = int(net.NS), int(net.NC)
    tables, blocks, groups = _core_tables(net)
    t0 = int(t[0])
    tr = t - t0
    t_last = int(tr.max())
    n_steps = t_last // CORE_DT_US + 1 + 3
    events = event_scan_events(tr, x, y, p, dt_ms=CORE_DT_MS, n_steps=n_steps,
                               vw=net.VW, vh=net.VH)
    drive = EventScanDrive(events, tables, blocks, groups,
                           dt_ms=CORE_DT_MS, n_steps=n_steps)
    dur = float(n_steps) * CORE_DT_MS
    lif_simple, lif_complex = _core_lif_params(net)
    model = jtfne.construct(
        build_s1_scan_tensor(ns=ns, nc=nc, lif_simple=lif_simple,
                             lif_complex=lif_complex),
        nt.RuntimeConfiguration(seed=0, duration_ms=dur, dt_ms=CORE_DT_MS, emitter="lif"))
    signals = model.simulate(
        jtfne.Simulation(duration_ms=dur, dt_ms=CORE_DT_MS, seed=1,
                         runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list")),
        paradigm=drive)
    counts = np.asarray(signals.metadata["scan_counts"], dtype=np.int64)
    first = np.asarray(signals.metadata["scan_first_spike_us"], dtype=np.int64)
    over = np.flatnonzero(counts > 1)
    if over.size:
        raise NotImplementedError(
            f"{CORE_MULTISPIKE} (first {over.size} cells: {over[:8].tolist()})")
    fired = np.flatnonzero(counts == 1)
    times = (first[fired] + t0).astype(np.int64)
    order = np.lexsort((fired.astype(np.int64), times))  # time then cell, as _gather
    times, owners = times[order], fired[order]
    sm = owners < ns
    return {
        "simple_counts": counts[:ns].astype(np.int32),
        "complex_counts": counts[ns:].astype(np.int32),
        "simple_times_us": times[sm],
        "simple_times_cell": owners[sm].astype(np.int32),
        "complex_times_us": times[~sm],
        "complex_times_cell": (owners[~sm] - ns).astype(np.int32),
        "n_events": np.int64(int(t.shape[0])),
    }


def _init_worker(inhibition: str = "full", engine: str = "numba") -> None:
    """Load the trained network once per process and compile the kernel outside the timed region."""
    global _PRISTINE, _INHIBITION, _ENGINE
    _INHIBITION = inhibition
    _ENGINE = engine
    _PRISTINE = ref.PCLEventRef(NETWORK, learning=False)  # as test_pcl_trained_load.py
    apply_inhibition_mode(_PRISTINE, inhibition)
    if engine == "core":
        return  # the scan compiles on first simulate; nothing to warm up
    nb.warm_up()


def _run_one(job: tuple[str, str]) -> float:
    """Run one input file from a fresh copy of the loaded network; write its output atomically."""
    if _ENGINE == "core":
        return _run_one_core(job)
    src, dst = Path(job[0]), Path(job[1])
    t_start = time.perf_counter()
    with np.load(src, allow_pickle=False) as z:
        t = z["arr_0"].astype(np.int64)
        x = z["arr_1"].astype(np.int64)
        y = z["arr_2"].astype(np.int64)
        p = z["arr_3"].astype(np.int64)
        c = z["arr_4"]
    keep = c == 0  # one camera network: camera 0 only (NetworkHandle.cpp:511)
    net = copy.deepcopy(_PRISTINE)
    nb.run_numba(net, t[keep], x[keep], y[keep], p[keep])
    arrays = {
        "simple_counts": np.array([len(cell.spikes) for cell in net.s_cells], dtype=np.int32),
        "complex_counts": np.array([len(cell.spikes) for cell in net.c_cells], dtype=np.int32),
        "n_events": np.int64(int(keep.sum())),
    }
    arrays["simple_times_us"], arrays["simple_times_cell"] = _gather(net.s_cells)
    arrays["complex_times_us"], arrays["complex_times_cell"] = _gather(net.c_cells)
    arrays["wall_s"] = np.float64(time.perf_counter() - t_start)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.npz")  # np.savez appends .npz unless the name ends with it
    np.savez(str(tmp), **arrays)
    os.replace(tmp, dst)
    return float(arrays["wall_s"])


def _run_one_core(job: tuple[str, str]) -> float:
    """Run one input file through the tensor-scan replay; write its output atomically.

    The replay is functional (the network is read, never mutated), so no per-file copy
    is needed. An empty camera-0 slice writes zeros, exactly what a fresh network holds
    with no events. Files outside the run_core_events boundary fail that file, never
    the run: the summary and exit code report them.
    """
    src, dst = Path(job[0]), Path(job[1])
    t_start = time.perf_counter()
    with np.load(src, allow_pickle=False) as z:
        t = z["arr_0"].astype(np.int64)
        x = z["arr_1"].astype(np.int64)
        y = z["arr_2"].astype(np.int64)
        p = z["arr_3"].astype(np.int64)
        c = z["arr_4"]
    keep = c == 0  # one camera network: camera 0 only (NetworkHandle.cpp:511)
    ns, nc = int(_PRISTINE.NS), int(_PRISTINE.NC)
    if int(keep.sum()) == 0:
        arrays = {
            "simple_counts": np.zeros(ns, dtype=np.int32),
            "complex_counts": np.zeros(nc, dtype=np.int32),
            "n_events": np.int64(0),
            "simple_times_us": np.zeros(0, dtype=np.int64),
            "simple_times_cell": np.zeros(0, dtype=np.int32),
            "complex_times_us": np.zeros(0, dtype=np.int64),
            "complex_times_cell": np.zeros(0, dtype=np.int32),
        }
    else:
        arrays = run_core_events(_PRISTINE, t[keep], x[keep], y[keep], p[keep])
    arrays["wall_s"] = np.float64(time.perf_counter() - t_start)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.npz")  # np.savez appends .npz unless the name ends with it
    np.savez(str(tmp), **arrays)
    os.replace(tmp, dst)
    return float(arrays["wall_s"])


def _baseline_sha() -> str:
    res = subprocess.run(["git", "-C", _ROOT, "rev-parse", "HEAD"], capture_output=True, text=True,
                         stdin=subprocess.DEVNULL, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"git rev-parse failed: {res.stderr.strip()}")
    return res.stdout.strip()


def _realized_backend(engine="numba") -> str:
    """Realized compute backend of the tuning run path.

    The scored path runs the numba oracle (``--engine numba``); engine ``core``
    replays the same feed through the jaxfne event-scan kernel, so the jax branch
    below names the device platform that replay ran on. Numba threads are pinned
    to one per worker process (module top).
    """
    if engine == "core":
        import jax

        return "jax " + jax.devices()[0].platform
    try:
        import numba

        try:
            layer = numba.threading_layer()
        except Exception:
            layer = "unknown"
        try:
            threads = numba.get_num_threads()
        except Exception:
            threads = 1
        return f"numba {layer}/{threads} threads"
    except ImportError:
        return "numba"


def _write_manifest(out: Path, record: dict, static: dict) -> None:
    path = out / "manifest.json"
    man = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"invocations": []}
    man.update(static)
    man["invocations"].append(record)
    tmp = out / "manifest.tmp.json"
    tmp.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subset", choices=("pilot", "all"), required=True)
    ap.add_argument("--inhibition", choices=("full", "local-only", "no-local", "no-local-simple"),
                    default="full",
                    help="S1 (P3_fig3.md): local-only zeroes the lateral and top-down maps; "
                         "no-local clears the local flags; no-local-simple clears the simple "
                         "layer's only (diagnostics)")
    ap.add_argument("--engine", choices=("numba", "core"), default="numba",
                    help="numba runs the oracle path; core replays the same feed through "
                         "the jaxfne event-scan kernel (run_core_events: exact only inside "
                         "its stated boundary; default: numba, behaviour unchanged)")
    ap.add_argument("--out", default=None,
                    help="default: the mode's dir in MODE_DIRS (full, fig3_cSTDP3_S1, "
                         "diag_runs/fig3_cSTDP3_nolocal, diag_runs/fig3_cSTDP3_nolocal_simple)")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--start-utc", default=None,
                    help="start time for the manifest, from `date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ`")
    return ap


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    cfg = json.loads((NETWORK / "configs" / "network_config.json").read_text(encoding="utf-8"))
    _, ny, nd = (int(v) for v in cfg["layerSizes"][0])  # simple layer grid: nx, ny, depth
    out = resolve_out_dir(args.out, args.inhibition)
    try:
        check_out_dir(out, args.inhibition)
    except ValueError as exc:
        print(f"error: {exc}", flush=True)
        return 2
    jobs = []
    for o, f, p, s in tasks_for(args.subset):
        src = TEST_SET / o / f / p / f"{s}.npz"
        if not src.is_file():
            print(f"missing input {src}", flush=True)
            return 2
        jobs.append((str(src), str(out / o / f / p / f"{s}.npz")))
    pending = [job for job in jobs if not Path(job[1]).is_file()]

    start = args.start_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    static = {
        "axes": AXES,
        "axes_note": AXES_NOTE,
        "baseline_sha": _baseline_sha(),
        "network": str(NETWORK),
        "config_layer_inhibitions": cfg["layerInhibitions"],
        "config_note": "'static' is read as local inhibition (pcl_event_ref.py docstring)",
        "test_set": str(TEST_SET),
        "learning": False,
        "inhibition": args.inhibition,
        "engine": args.engine,
        "feed": "camera 0 only, raw file timestamps (not rebased), as test_pcl_trained_load.py",
        "workers": args.workers,
        "output_fields": "simple_counts int32 (5184), complex_counts int32 (1152), n_events int64, wall_s; "
                         "simple_times_us int64 with simple_times_cell int32, complex_times_us int64 with "
                         "complex_times_cell int32; all cells; times sorted by time then cell; cell = flat index",
        "cell_id_convention": "flat index = (i * NY + j) * ND + k for simple, (cx * CY + cy) * CD + cz for complex",
        "central_col": {"i": CENTRAL_COL[0], "j": CENTRAL_COL[1],
                        "column": CENTRAL_COL[0] * ny + CENTRAL_COL[1],
                        "flat_base": (CENTRAL_COL[0] * ny + CENTRAL_COL[1]) * nd,
                        "note": CENTRAL_COL_NOTE},
        "backend": _realized_backend(args.engine),
        "dtype": "float64",
        "dtype_note": ("compute float64 (weights, V/A/thr/rate per "
                       "pcl_event_ref_numba.py); events int64; outputs "
                       "simple_counts/complex_counts int32, times int64, wall_s float64"),
        "seeds": [],
        "seeds_note": ("event replay over the recorded test7 streams; no RNG seeds. "
                       "The network weights are the trained cSTDP3 values loaded over "
                       "the seed-0 init."),
        "activity_regime": {
            "kind": "declared",
            "source": "prereg/P3_fig3.md C1-C2",
            "simple": "tuned for orientation, spatial frequency and spatial phase",
            "complex": ("tuned for orientation and spatial frequency, "
                        "approximately phase invariant")},
    }
    record = {"start_utc": start, "argv": sys.argv[1:], "subset": args.subset,
              "n_tasks": len(jobs), "n_pending": len(pending), "out": str(out)}
    out.mkdir(parents=True, exist_ok=True)
    _write_manifest(out, record, static)
    print(f"tasks {len(jobs)} pending {len(pending)} out {out} start {start}", flush=True)
    if not pending:
        print("summary done=0 failed=0 pending_at_start=0", flush=True)
        return 0

    done = failed = 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.inhibition, args.engine)) as ex:
        futs = {ex.submit(_run_one, job): job for job in pending}
        for fut in as_completed(futs):
            src, dst = futs[fut]
            try:
                wall = fut.result()
            except Exception as exc:  # keep the others running; the summary and exit code report it
                failed += 1
                print(f"FAIL {src}: {exc!r}", flush=True)
                continue
            done += 1
            rel = Path(dst).relative_to(out).as_posix()
            print(f"done {done}/{len(pending)} {rel} wall_s={wall:.2f} elapsed_s={time.perf_counter() - t0:.0f}",
                  flush=True)
    print(f"summary done={done} failed={failed} pending_at_start={len(pending)}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
