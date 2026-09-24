#!/usr/bin/env python3
"""0.5.1 ENGINE item 1b-1e: benchmark matrix harness.

Declares nothing; measures per artifacts/perf/matrix_051_spec.json (frozen).
Reuses the phase timing of scripts/benchmark_050_baseline.py
(construct | simulate_first_call | simulate_warm | probe | manifest) and the
RSS method of scripts/profile_050_phases.py, plus a 10ms peak-hold sampler
for measured process peak. Declared array bytes reported alongside.

Writes artifacts/perf/matrix_051.json (tracked). NEVER writes
artifacts/perf/baseline_050.json.

Usage:
    python scripts/benchmark_051_matrix.py [--only-missing]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SPEC = ROOT / "artifacts" / "perf" / "matrix_051_spec.json"
OUT = ROOT / "artifacts" / "perf" / "matrix_051.json"

MEMORY_METHOD = (
    "psutil process RSS: boundary samples at every phase edge plus a "
    "10ms-interval peak-hold sampler thread during construct+simulate; "
    "cell peak = max sample inside the cell window. Declared array bytes "
    "(spikes/V_m/sources, HDP traces where present) reported alongside. "
    "Every timed simulate() is closed with jax.block_until_ready so phase "
    "timers measure completion, not async dispatch. Each cell runs in a "
    "fresh process (one --cells invocation per cell); construct_ms therefore "
    "includes interpreter+JAX init on top of the first-of-shape build cost."
)

_warm_repeats = 3


def _rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 1e6


class _PeakSampler:
    """10ms peak-hold RSS sampler. samples: list of (epoch_s, rss_mb)."""

    def __init__(self) -> None:
        self.samples: list = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append((time.time(), _rss_mb()))
            except Exception:
                pass
            time.sleep(0.01)

    def start(self) -> None:
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=5)

    def peak_between(self, t0: float, t1: float) -> float | None:
        vals = [r for (t, r) in self.samples if t0 <= t <= t1]
        return max(vals) if vals else None


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _env() -> dict:
    import jax
    import jaxfne as J

    try:
        import importlib.metadata as md

        jaxlib_v = md.version("jaxlib")
    except Exception:
        jaxlib_v = "unknown"
    return {
        "jax": jax.__version__,
        "jaxlib": jaxlib_v,
        "platform": platform.platform(),
        "device": [str(d) for d in jax.devices()],
        "jaxfne_version": J.__version__,
    }


def _sync(signals) -> None:
    """Block until simulate outputs are materialized (defeats async-dispatch attribution)."""
    import jax

    arrs = [signals.V_m, signals.spikes]
    if getattr(signals, "sources", None) is not None:
        arrs.append(signals.sources)
    jax.block_until_ready(arrs)


def _nbytes(a) -> int | None:
    try:
        import numpy as np

        return int(np.ascontiguousarray(np.asarray(a)).nbytes)
    except Exception:
        return None


def _model_size(model) -> tuple[int | None, int | None]:
    try:
        n_edges = int(model.params["edge_list"].n_edges)
    except Exception:
        n_edges = None
    try:
        n_neurons = len(model.neuron_table())
    except Exception:
        n_neurons = None
    return n_neurons, n_edges


def _delay_max(model) -> int | None:
    for key in ("delay_steps_max",):
        try:
            diag = model.last_hdp_diagnostics()
            if isinstance(diag, dict) and key in diag:
                import numpy as np

                return int(np.asarray(diag[key]).max())
        except Exception:
            pass
    try:
        import numpy as np

        ds = np.asarray(model.params["edge_list"].delay_steps)
        return int(ds.max())
    except Exception:
        return None


# ---------------------------------------------------------------- builders
def build_sparse(n: int, tag: str):
    import jaxfne as J

    return J.construct(
        J.configuration()
        .network(
            name=f"network_{tag}_ei",
            kind="balanced_ei_population",
            n=n,
            cell_types={"E": 0.75, "PV": 0.25},
        )
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="declared_proxy",
            gauge="mean_zero",
        )
        .probe(name=f"multimodal_{tag}_ei", modes=["spikes", "V_m", "source", "LFP"])
    )


def build_1n():
    import jaxfne as J

    return J.construct(
        J.configuration()
        .network(n=1)
        .emitter(family="izhikevich", preset="regular_spiking")
        .field(domain="point")
        .probe(name="single_neuron", modes=["spikes", "V_m"])
    )


def build_1n_field():
    """0.5.2 item 7: field-capable variant of build_1n (Q+Φ recording).

    The frozen n1 cell keeps build_1n untouched; only field_n1 uses this.
    """
    import jaxfne as J

    return J.construct(
        J.configuration()
        .network(n=1)
        .emitter(family="izhikevich", preset="regular_spiking")
        .field(domain="point")
        .probe(name="single_neuron_field", modes=["spikes", "V_m", "source", "LFP"])
    )


def _hdp_runtime():
    import benchmark_050_baseline as B

    return B._hdp_runtime()


def _edge_runtime():
    import jaxfne as J

    return J.RuntimeConfig(recurrent_backend="edge_list", jit=False)


def _sim(duration_ms, dt_ms, seed, record, runtime=None):
    import jaxfne as J

    kw: dict = {
        "duration_ms": duration_ms,
        "dt_ms": dt_ms,
        "seed": seed,
    }
    if record == "off":
        kw.update(record_sources=False, record_fields=False)
    elif record == "minimal":
        kw.update(record_sources=True, record_fields=False)
    elif record == "full":
        kw.update(record_sources=True, record_fields=True)
    if runtime is not None:
        kw["runtime"] = runtime
    return J.Simulation(**kw)


_PROBE_MODES = {
    "off": ["spikes", "V_m"],
    "minimal": ["spikes", "V_m"],
    "full": ["spikes", "V_m", "LFP"],
}


# ---------------------------------------------------------------- cell run
class BudgetOut(Exception):
    pass


def _check_budget(deadline: float | None) -> None:
    if deadline is not None and time.time() > deadline:
        raise BudgetOut()


def run_config_cell(cell: dict, sampler: _PeakSampler) -> dict:
    """Run one Configuration-path cell. Returns the cell result dict."""
    import numpy as np

    cid = cell["id"]
    budget = float(cell.get("wall_budget_s", 600))
    deadline = time.time() + budget
    t_cell = time.time()
    res: dict = {
        "id": cid,
        "status": "MEASURED",
        "params": {
            k: cell.get(k)
            for k in (
                "n_neurons",
                "duration_ms",
                "dt_ms",
                "recording",
                "mechanism",
                "chunks",
                "seed",
            )
        },
    }
    notes: list[str] = []
    try:
        _check_budget(deadline)
        r0 = _rss_mb()
        import warnings as _w

        t0 = time.perf_counter()
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            n = int(cell["n_neurons"])
            if cid == "n1":
                model = build_1n()
            elif cid == "field_n1":
                # 0.5.2 item 7: field-path cell; frozen n1 builder untouched.
                model = build_1n_field()
            elif cid in ("field_n2", "field_n1000"):
                model = build_sparse(n, cid)
            else:
                model = build_sparse(n, cid.replace("10000", "10k"))
        t_construct = _ms(t0)
        for x in caught:
            notes.append(f"build warning {x.category.__name__}: {x.message}")
        r1 = _rss_mb()
        n_neurons, n_edges = _model_size(model)

        runtime = None
        if cell.get("mechanism") == "HDP":
            runtime = _hdp_runtime()
            notes.append("HDP runtime mirrored from benchmark_050_baseline._hdp_runtime")
        elif cid in ("chunk_ref", "chunk_k4"):
            runtime = _edge_runtime()
            notes.append("edge_list backend required by the continuation path")
        sim = _sim(
            cell["duration_ms"], cell["dt_ms"], int(cell["seed"]), cell["recording"], runtime
        )

        if cid == "chunk_k4":
            _check_budget(deadline)
            # 4 continuation segments of T/4.
            import jaxfne as J

            seg_ms = float(cell["duration_ms"]) / 4
            segs = [
                J.Simulation(
                    duration_ms=seg_ms,
                    dt_ms=cell["dt_ms"],
                    seed=int(cell["seed"]),
                    record_sources=True,
                    record_fields=False,
                    runtime=runtime,
                )
                for _ in range(4)
            ]
            t0 = time.perf_counter()
            out0 = model.simulate(segs[0], return_state=True)
            signals0, state = out0[0], out0[1]
            _sync(signals0)
            parts = [
                [
                    np.asarray(signals0.V_m),
                    np.asarray(signals0.spikes),
                    np.asarray(signals0.sources),
                ]
            ]
            for s in segs[1:]:
                _check_budget(deadline)
                out = model.simulate(s, continuation=state, return_state=True)
                sg, state = out[0], out[1]
                _sync(sg)
                parts.append([np.asarray(sg.V_m), np.asarray(sg.spikes), np.asarray(sg.sources)])
            t_first = _ms(t0)
            r2 = _rss_mb()
            signals = signals0  # representative for probe/manifest phases
            chunk_spikes = np.concatenate([p[1] for p in parts], axis=0)
            res["chunked"] = {
                "segments": 4,
                "spike_rows": int(chunk_spikes.shape[0]),
            }
            notes.append("chunk_k4 first_call = full 4-segment chained run (compile+run)")
            # equivalence vs chunk_ref is computed in post (needs chunk_ref spikes);
            # store a hash-free checksum instead: total spike count + V sum.
            res["chunked"]["spike_count_total"] = int(chunk_spikes.sum())
            res["chunked"]["V_sum"] = float(np.concatenate([p[0] for p in parts], axis=0).sum())
        else:
            _check_budget(deadline)
            t0 = time.perf_counter()
            signals = model.simulate(sim)
            _sync(signals)
            t_first = _ms(t0)
            r2 = _rss_mb()

        _check_budget(deadline)
        warms: list[float] = []
        if cid == "chunk_k4":
            # warm = re-run of the chained 4-segment run would quadruple cost;
            # warm repeats are single-segment-equivalent: re-run last segment
            # continuation is stateful, so warm = full single-call on chunk_ref
            # backend instead. Record honestly: warm repeats of one segment.
            import jaxfne as J

            seg_ms = float(cell["duration_ms"]) / 4
            wseg = J.Simulation(
                duration_ms=seg_ms,
                dt_ms=cell["dt_ms"],
                seed=int(cell["seed"]),
                record_sources=True,
                record_fields=False,
                runtime=runtime,
            )
            for _ in range(_warm_repeats):
                _check_budget(deadline)
                t0 = time.perf_counter()
                _ws = model.simulate(wseg)
                _sync(_ws)
                warms.append(_ms(t0))
            notes.append(
                "warm repeats are single-segment (T/4) runs; first_call covers the chained run"
            )
        else:
            for _ in range(_warm_repeats):
                _check_budget(deadline)
                t0 = time.perf_counter()
                signals_w = model.simulate(sim)
                _sync(signals_w)
                warms.append(_ms(t0))
            signals = signals_w
        r3 = _rss_mb()
        warm_med = float(np.median(np.asarray(warms)))
        warm_range = [float(min(warms)), float(max(warms))]
        try:
            res["output_checksum"] = {
                "spike_count_total": int(np.asarray(signals.spikes).sum()),
                "V_sum": float(np.asarray(signals.V_m).sum()),
            }
        except Exception:
            pass

        t0 = time.perf_counter()
        try:
            model.probe(signals, modes=_PROBE_MODES[cell["recording"]])
        except Exception as e:
            notes.append(f"probe raised: {type(e).__name__}: {e}")
        t_probe = _ms(t0)
        t0 = time.perf_counter()
        try:
            model.manifest(signals)
        except Exception as e:
            notes.append(f"manifest raised: {type(e).__name__}: {e}")
        t_manifest = _ms(t0)

        recording: dict = {}
        for key in ("spikes", "V_m", "sources"):
            try:
                recording[key] = {"bytes": _nbytes(getattr(signals, key))}
            except Exception:
                recording[key] = {"bytes": None}
        # 0.5.2 item 7: Φ (field) bytes beside Q (sources) for field-path
        # cells. Guarded: cells without a field record None, as before.
        # FieldOutput is a pytree container, so sum its array leaves.
        try:
            _field = getattr(signals, "field", None)
            if _field is None:
                recording["field"] = {"bytes": None}
            else:
                import jax as _jax

                _leaves = _jax.tree.leaves(_field)
                _total = 0
                for _leaf in _leaves:
                    _b = _nbytes(_leaf)
                    if _b is not None:
                        _total += _b
                recording["field"] = {"bytes": _total}
        except Exception:
            recording["field"] = {"bytes": None}
        try:
            diag = model.last_hdp_diagnostics()
        except Exception:
            diag = None
        if isinstance(diag, dict):
            for key in ("H_trace", "w_trace"):
                v = diag.get(key)
                recording[key] = {
                    "bytes": _nbytes(v) if v is not None else None,
                    "recorded": v is not None,
                }

        peak = sampler.peak_between(t_cell, time.time())
        res.update(
            {
                "n_neurons_realized": n_neurons,
                "n_edges": n_edges,
                "delay_steps_max": _delay_max(model),
                "phases_ms": {
                    "construct_ms": t_construct,
                    "simulate_first_call_ms": t_first,
                    "simulate_warm_ms_median": warm_med,
                    "simulate_warm_ms_range": warm_range,
                    "simulate_warm_ms_all": warms,
                    "compile_est_ms": max(0.0, t_first - warm_med),
                    "probe_observe_ms": t_probe,
                    "manifest_ms": t_manifest,
                },
                "rss_mb": {
                    "start": r0,
                    "after_construct": r1,
                    "after_first_sim": r2,
                    "after_warm": r3,
                    "cell_peak": peak,
                },
                "recording": recording,
                "notes": notes,
            }
        )
    except BudgetOut:
        res["status"] = "SKIPPED_BUDGET"
        res["notes"] = notes + [f"cell wall budget {budget}s exceeded; partial phases kept"]
        peak = sampler.peak_between(t_cell, time.time())
        res["rss_mb"] = {"cell_peak": peak}
    except Exception as e:
        res["status"] = "UNSUPPORTED"
        res["reason"] = f"{type(e).__name__}: {e}"
        res["notes"] = notes
        peak = sampler.peak_between(t_cell, time.time())
        res["rss_mb"] = {"cell_peak": peak}
    res["wall_s"] = time.time() - t_cell
    return res


def run_at10_cell(cell: dict, sampler: _PeakSampler) -> dict:
    """AT-10 attempt chain: coupled (merged + ring AreaConnections) else uncoupled."""
    import dataclasses

    import numpy as np

    import jaxfne as J
    from jaxfne.neuronal_tensor import AreaConnection, merge_neuronal_tensors

    cid = cell["id"]
    budget = float(cell.get("wall_budget_s", 1800))
    deadline = time.time() + budget
    t_cell = time.time()
    res: dict = {
        "id": cid,
        "status": "MEASURED",
        "params": {
            k: cell.get(k)
            for k in (
                "n_neurons",
                "duration_ms",
                "dt_ms",
                "recording",
                "mechanism",
                "chunks",
                "seed",
            )
        },
    }
    notes: list[str] = []
    try:
        _check_budget(deadline)
        t0 = time.perf_counter()
        t0_single = J.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
        # Scaffolding probe (not a cell): single-column edge count + pop sizes.
        single = J.construct(
            t0_single, J.RuntimeConfiguration(seed=0, duration_ms=100.0, dt_ms=0.5)
        )
        _, e_single = _model_size(single)
        table = single.neuron_table()
        import collections

        pops = collections.Counter((r.get("layer"), r.get("cell_type")) for r in table)
        notes.append(f"scaffold single column n_edges={e_single}; pop sizes: {dict(pops)}")
        t_scaffold = _ms(t0)

        # Choose ring pair (src_layer, src_type)->(tgt_layer, tgt_type) with
        # S*T ~= 10% of single-column outgoing edges.
        target = 0.10 * float(e_single or 0)
        best = None
        keys = [k for k in pops if k[0] is not None]
        for s in keys:
            for tg in keys:
                e = pops[s] * pops[tg]
                if best is None or abs(e - target) < abs(best[0] - target):
                    best = (e, s, tg)
        assert best is not None
        _, (sl, st), (tl, tt) = best
        notes.append(
            f"ring pair {(sl, st)}->{(tl, tt)} gives {best[0]} edges/area vs 10% target {target:.0f}"
        )
        notes.append(
            "tensor path fixes AreaConnection density at probability=1.0 (full bipartite); random-target thinning is not expressible — counts approximate ~10%, placement is bipartite (recorded deviation)"
        )
        notes.append(
            "tensor path carries no per-edge delay (TFNE delay arrives 0.5.2); delay_steps=0 (recorded)"
        )

        _check_budget(deadline)
        t0 = time.perf_counter()
        merged = merge_neuronal_tensors([t0_single] * 20, name="at10-20area")
        area_names = [a.name for a in merged.areas]
        ring = []
        for i in range(20):
            ring.append(
                AreaConnection(
                    source_area=area_names[i],
                    source_layer=sl,
                    source_neuron_type=st,
                    target_area=area_names[(i + 1) % 20],
                    target_layer=tl,
                    target_neuron_type=tt,
                    mechanism="AMPA",
                )
            )
        merged = dataclasses.replace(merged, area_connections=tuple(ring))
        model = J.construct(merged, J.RuntimeConfiguration(seed=0, duration_ms=100.0, dt_ms=0.5))
        t_construct = _ms(t0)
        r1 = _rss_mb()
        n_neurons, n_edges = _model_size(model)
        inter = (n_edges - 20 * (e_single or 0)) if (n_edges and e_single) else None
        frac = (inter / n_edges) if (inter is not None and n_edges) else None
        notes.append(f"coupled: n={n_neurons} edges={n_edges} inter_est={inter} frac={frac}")
        if not inter or inter <= 0:
            raise RuntimeError("AreaConnections realized zero inter-area edges")
        res["coupling"] = {
            "mode": "coupled_ring",
            "inter_edges_est": inter,
            "inter_fraction": frac,
            "delay": "zero (tensor path carries none)",
        }

        _check_budget(deadline)
        sim = J.Simulation(
            duration_ms=100.0,
            dt_ms=0.5,
            seed=0,
            runtime=J.RuntimeConfig(recurrent_backend="edge_list", jit=False),
        )
        notes.append("simulate backend edge_list (dense 20k^2 infeasible)")
        t0 = time.perf_counter()
        signals = model.simulate(sim)
        _sync(signals)
        t_first = _ms(t0)
        r2 = _rss_mb()

        warms = []
        for _ in range(_warm_repeats):
            _check_budget(deadline)
            t0 = time.perf_counter()
            signals_w = model.simulate(sim)
            _sync(signals_w)
            warms.append(_ms(t0))
        signals = signals_w
        r3 = _rss_mb()
        warm_med = float(np.median(np.asarray(warms)))

        t0 = time.perf_counter()
        try:
            model.probe(signals, modes=["spikes", "V_m"])
        except Exception as e:
            notes.append(f"probe raised: {type(e).__name__}: {e}")
        t_probe = _ms(t0)
        t0 = time.perf_counter()
        try:
            model.manifest(signals)
        except Exception as e:
            notes.append(f"manifest raised: {type(e).__name__}: {e}")
        t_manifest = _ms(t0)

        recording = {}
        for key in ("spikes", "V_m", "sources"):
            try:
                recording[key] = {"bytes": _nbytes(getattr(signals, key))}
            except Exception:
                recording[key] = {"bytes": None}
        peak = sampler.peak_between(t_cell, time.time())
        res.update(
            {
                "n_neurons_realized": n_neurons,
                "n_edges": n_edges,
                "delay_steps_max": _delay_max(model),
                "scaffold_ms": {"single_column_probe_ms": t_scaffold},
                "phases_ms": {
                    "construct_ms": t_construct,
                    "simulate_first_call_ms": t_first,
                    "simulate_warm_ms_median": warm_med,
                    "simulate_warm_ms_range": [float(min(warms)), float(max(warms))],
                    "simulate_warm_ms_all": warms,
                    "compile_est_ms": max(0.0, t_first - warm_med),
                    "probe_observe_ms": t_probe,
                    "manifest_ms": t_manifest,
                },
                "rss_mb": {
                    "after_construct": r1,
                    "after_first_sim": r2,
                    "after_warm": r3,
                    "cell_peak": peak,
                },
                "recording": recording,
                "notes": notes,
            }
        )
    except BudgetOut:
        res["status"] = "SKIPPED_BUDGET"
        res["notes"] = notes + [f"cell wall budget {budget}s exceeded; partial phases kept"]
        res["rss_mb"] = {"cell_peak": sampler.peak_between(t_cell, time.time())}
    except Exception as e:
        # Fallback per spec: measure 20 uncoupled columns, mark coupled UNSUPPORTED.
        notes.append(f"coupled attempt failed: {type(e).__name__}: {e}")
        try:
            import jaxfne as J  # noqa: F811
            from jaxfne.neuronal_tensor import merge_neuronal_tensors  # noqa: F811

            _check_budget(deadline)
            t0_single = J.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
            t0 = time.perf_counter()
            merged = merge_neuronal_tensors([t0_single] * 20, name="at10-20area-uncoupled")
            model = J.construct(
                merged, J.RuntimeConfiguration(seed=0, duration_ms=100.0, dt_ms=0.5)
            )
            t_construct = _ms(t0)
            n_neurons, n_edges = _model_size(model)
            sim = J.Simulation(
                duration_ms=100.0,
                dt_ms=0.5,
                seed=0,
                runtime=J.RuntimeConfig(recurrent_backend="edge_list", jit=False),
            )
            t0 = time.perf_counter()
            signals = model.simulate(sim)
            _sync(signals)
            t_first = _ms(t0)
            res["uncoupled_fallback"] = {
                "n_neurons_realized": n_neurons,
                "n_edges": n_edges,
                "construct_ms": t_construct,
                "simulate_first_call_ms": t_first,
            }
            notes.append(f"uncoupled fallback measured: n={n_neurons} edges={n_edges}")
        except Exception as e2:
            notes.append(f"uncoupled fallback also failed: {type(e2).__name__}: {e2}")
        res["status"] = "UNSUPPORTED"
        res["reason"] = f"coupled AT-10 unreachable on existing paths: {notes[0] if notes else '?'}"
        res["notes"] = notes
        res["rss_mb"] = {"cell_peak": sampler.peak_between(t_cell, time.time())}
    res["wall_s"] = time.time() - t_cell
    return res


def _merged_cells(spec: dict, prior_cells: dict, ran: dict) -> list:
    full = json.loads(SPEC.read_text(encoding="utf-8"))["cells"]
    out = []
    for c in full:
        if c["id"] in ran:
            out.append(ran[c["id"]])
        elif c["id"] in prior_cells:
            out.append(prior_cells[c["id"]])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only-missing", action="store_true", help="skip cells already MEASURED in matrix_051.json"
    )
    ap.add_argument(
        "--cells", default="", help="comma-separated cell ids to run (default: all spec cells)"
    )
    args = ap.parse_args()

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    want = [c.strip() for c in args.cells.split(",") if c.strip()]
    if want:
        spec["cells"] = [c for c in spec["cells"] if c["id"] in want]
    # merge with prior OUT so partial runs accumulate across invocations
    prior_cells: dict = {}
    if OUT.exists():
        try:
            for c in json.loads(OUT.read_text(encoding="utf-8")).get("cells", []):
                prior_cells[c["id"]] = c
        except Exception:
            pass

    sampler = _PeakSampler()
    sampler.start()
    ran: dict = {}
    try:
        for cell in spec["cells"]:
            if (
                args.only_missing
                and cell["id"] in prior_cells
                and prior_cells[cell["id"]].get("status") == "MEASURED"
            ):
                print(f"{cell['id']}: keeping prior MEASURED result", flush=True)
                continue
            print(f"--- {cell['id']} (budget {cell.get('wall_budget_s')}s) ---", flush=True)
            if cell["id"] == "at10_20area":
                res = run_at10_cell(cell, sampler)
            else:
                res = run_config_cell(cell, sampler)
            print(
                f"{cell['id']}: {res['status']} wall={res['wall_s']:.1f}s "
                f"phases={res.get('phases_ms', res.get('uncoupled_fallback', {}))}",
                flush=True,
            )
            ran[cell["id"]] = res
            cells_out = _merged_cells(spec, prior_cells, ran)
            OUT.write_text(
                json.dumps(
                    {
                        "series": "matrix_051",
                        "spec": "artifacts/perf/matrix_051_spec.json (frozen, committed before measuring)",
                        "env": _env(),
                        "memory_method": MEMORY_METHOD,
                        "claim": "local_environment_receipt_only",
                        "partial": True,
                        "cells": cells_out,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    finally:
        sampler.stop()

    cells_out = _merged_cells(spec, prior_cells, ran)
    # post: chunk_k4 vs chunk_ref spike equivalence (both in this run if present)
    by_id = {c["id"]: c for c in cells_out}
    if "chunk_k4" in by_id and "chunk_ref" in by_id:
        kr, kk = by_id["chunk_ref"], by_id["chunk_k4"]
        if kr.get("status") == "MEASURED" and kk.get("status") == "MEASURED":
            ck, cr = kk.get("chunked", {}), kr.get("output_checksum", {})
            if ck.get("spike_count_total") is not None and cr.get("spike_count_total") is not None:
                ck["vs_chunk_ref"] = {
                    "spike_count_equal": bool(ck["spike_count_total"] == cr["spike_count_total"]),
                    "V_sum_abs_diff": abs(float(ck["V_sum"]) - float(cr["V_sum"])),
                    "chunk_k4_spikes": ck["spike_count_total"],
                    "chunk_ref_spikes": cr["spike_count_total"],
                }
    OUT.write_text(
        json.dumps(
            {
                "series": "matrix_051",
                "spec": "artifacts/perf/matrix_051_spec.json (frozen, committed before measuring)",
                "env": _env(),
                "memory_method": MEMORY_METHOD,
                "claim": "local_environment_receipt_only",
                "cells": cells_out,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
