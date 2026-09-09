#!/usr/bin/env python3
"""W10: tensor-allocation and usage map for a JaxFNE configuration.

For every significant array A_k reachable from a constructed model, report

    A_k = (name, shape, dtype, bytes, |unique|, lifetime, owner, U_k, R_k)

where

    R_k = materialized elements / independent values      (redundancy ratio)
    U_k = 1[array is actually consumed by canonical execution]

U_k is measured, not inferred from call sites: the array is perturbed (floats to
NaN, integer arrays rolled by one) and ``simulate`` is re-run. If the output is
unchanged, canonical execution does not consume the array. A static reader count
cannot establish this -- a value can be read on a branch that never executes, and
33 textual references to ``emitter.W`` say nothing about whether the edge_list
backend touches it.

Memory is reported in four separate figures, because a representation change can
remove persistent bytes while creating an equally large temporary inside a JIT
step:

    M_persistent  arrays reachable from the constructed model
    M_recording   arrays returned by simulate
    M_peak        maximum resident set size observed during construct + simulate
    M_temporary   M_peak - (M_persistent + M_recording), derived

Usage:
    python scripts/perf/w10_allocation_map.py --n 1000 --p-connect 0.0 --max-in-degree 100
    python scripts/perf/w10_allocation_map.py --n 1000 --json out.json --no-usage
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import fields, is_dataclass, replace
from typing import Any

import numpy as np

MIN_BYTES_DEFAULT = 4096
_NAN_SENTINEL = -9.87654321e30


# -- array discovery ---------------------------------------------------------
def _is_array(obj: Any) -> bool:
    return hasattr(obj, "shape") and hasattr(obj, "dtype") and not isinstance(obj, type)


def walk_arrays(obj: Any, path: str = "model", depth: int = 0, seen: set[int] | None = None):
    """Yield (path, array) for every array reachable from obj."""
    if seen is None:
        seen = set()
    if id(obj) in seen or depth > 12:
        return
    seen.add(id(obj))
    if _is_array(obj):
        yield path, obj
        return
    if is_dataclass(obj) and not isinstance(obj, type):
        for f in fields(obj):
            yield from walk_arrays(getattr(obj, f.name, None), f"{path}.{f.name}", depth + 1, seen)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_arrays(v, f"{path}[{k!r}]", depth + 1, seen)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj[:64]):
            yield from walk_arrays(v, f"{path}[{i}]", depth + 1, seen)
    elif hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            yield from walk_arrays(v, f"{path}.{k}", depth + 1, seen)


def describe(path: str, array: Any) -> dict[str, Any]:
    a = np.asarray(array)
    n_unique = int(np.unique(a).size) if a.size else 0
    return {
        "name": path,
        "owner": path.split(".")[0] if "." in path else path,
        "shape": tuple(int(d) for d in a.shape),
        "dtype": str(a.dtype),
        "bytes": int(a.nbytes),
        "n_unique": n_unique,
        "R_k": (float(a.size) / n_unique) if n_unique else None,
    }


# -- U_k: perturb and observe ------------------------------------------------
def perturb(array: Any) -> Any:
    """Return a same-shape, same-dtype array that must change any consumer's output."""
    import jax.numpy as jnp

    a = np.asarray(array)
    if a.size == 0:
        return None
    if np.issubdtype(a.dtype, np.floating):
        return jnp.full(a.shape, jnp.nan, dtype=array.dtype)
    distinct = np.unique(a)
    if distinct.size < 2:
        # Rolling a constant array is a no-op. Offset it instead -- these constant
        # integer arrays (delay_steps, receptor_index) are precisely the
        # class-shared candidates, so leaving their U_k undecided would hide the
        # answer the map exists to produce.
        return jnp.asarray(a + 1, dtype=array.dtype)
    if distinct.size <= 16:
        # Low-cardinality label arrays are usually laid out in blocks, so rolling by
        # one changes only a couple of elements and a real consumer can easily show
        # no output difference. Cyclically remap the distinct values instead, which
        # changes (almost) every element while staying inside the valid label set.
        remapped = np.roll(distinct, 1)
        lookup = dict(zip(distinct.tolist(), remapped.tolist()))
        return jnp.asarray(
            np.vectorize(lookup.__getitem__, otypes=[a.dtype])(a), dtype=array.dtype
        )
    return jnp.asarray(np.roll(a, 1, axis=0), dtype=array.dtype)


def set_param_array(model, path: str, value):
    """Replace model.params[key] or model.params[key].field; None if unsupported."""
    prefix = "model.params["
    if not path.startswith(prefix):
        return None
    key, _, tail = path[len(prefix):].partition("]")
    key = key.strip("'\"")
    container = model.params.get(key)
    if container is None:
        return None
    if not tail:
        return replace(model, params={**model.params, key: value})
    field_name = tail.lstrip(".")
    if "." in field_name or not is_dataclass(container):
        return None
    return replace(model, params={**model.params, key: replace(container, **{field_name: value})})


def measure_usage(model, records, run_fn, baseline) -> None:
    """Set U_k on each record by perturbing the array and re-running."""
    for rec in records:
        rec["U_k"] = None
        rec["U_k_method"] = "not_attempted"
        target = rec.pop("_array")
        bad = perturb(target)
        if bad is None:
            rec["U_k_method"] = "undecidable_constant_or_empty"
            continue
        mutated = set_param_array(model, rec["name"], bad)
        if mutated is None:
            rec["U_k_method"] = "unreachable_for_substitution"
            continue
        try:
            out = run_fn(mutated)
        except Exception as exc:
            # A raise proves the value is *read*, but not that dynamics consume it:
            # a validation check rejecting an out-of-range perturbation reads the
            # array without the solver ever using it. Record it as read-but-
            # unclassified rather than silently counting it as dynamics-critical.
            rec["U_k"] = None
            rec["U_k_method"] = f"read_but_unclassified_raised:{type(exc).__name__}"
            rec["U_k_detail"] = str(exc)[:200]
            continue
        changed = not np.array_equal(
            np.nan_to_num(baseline, nan=_NAN_SENTINEL),
            np.nan_to_num(out, nan=_NAN_SENTINEL),
        )
        rec["U_k"] = 1 if changed else 0
        rec["U_k_method"] = "perturb_and_observe"


# -- memory ------------------------------------------------------------------
class RSSSampler:
    """Sample resident set size on a background thread to capture M_peak."""

    def __init__(self, interval_s: float = 0.02):
        import psutil

        self._proc = psutil.Process()
        self._interval = interval_s
        self._stop = threading.Event()
        self.baseline = self._proc.memory_info().rss
        self.peak = self.baseline
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._proc.memory_info().rss)
            time.sleep(self._interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.peak = max(self.peak, self._proc.memory_info().rss)


def total_bytes(obj) -> int:
    return sum(int(np.asarray(a).nbytes) for _, a in walk_arrays(obj))


# -- driver ------------------------------------------------------------------
def build_config(n, p_connect, max_in_degree, duration_ms, dt_ms, seed):
    import jaxfne as jtfne

    cfg = (
        jtfne.Configuration()
        .runtime(seed=seed, dtype="float32", duration_ms=duration_ms, dt_ms=dt_ms)
        .column(name="c", layers=["L4"], n=n)
        .cell_types({"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03})
    )
    if p_connect is not None:
        cfg = cfg.connectivity(p_connect=p_connect)
    cfg = (
        cfg.set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes"])
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    )
    if max_in_degree:
        cfg = cfg.mechanisms(
            name="ampa", kind="exponential", params={"tau_ms": 2.0, "receptor": "AMPA"}
        ).connections(
            name="rec",
            source={},
            target={},
            mechanism="ampa",
            weight=0.03,
            max_in_degree=max_in_degree,
            spatial_sigma=0.1,
        )
    return cfg


def main(argv=None) -> int:
    import jaxfne as jtfne

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--p-connect", type=float, default=None)
    ap.add_argument("--max-in-degree", type=int, default=None)
    ap.add_argument("--duration-ms", type=float, default=40.0)
    ap.add_argument("--dt-ms", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min-bytes", type=int, default=MIN_BYTES_DEFAULT,
                    help="omit arrays smaller than this from the map")
    ap.add_argument("--no-usage", action="store_true",
                    help="skip U_k measurement (much faster)")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)

    cfg = build_config(args.n, args.p_connect, args.max_in_degree,
                       args.duration_ms, args.dt_ms, args.seed)

    def run_fn(model):
        sig = jtfne.simulate(model, duration_ms=args.duration_ms, dt_ms=args.dt_ms, seed=0)
        return np.asarray(sig.V_m)

    with RSSSampler() as rss:
        model = jtfne.construct(cfg)
        signals = jtfne.simulate(model, duration_ms=args.duration_ms, dt_ms=args.dt_ms, seed=0)
        baseline = np.asarray(signals.V_m)

    m_persistent = total_bytes(model)
    m_recording = total_bytes(signals)
    m_peak_delta = rss.peak - rss.baseline
    m_temporary = m_peak_delta - m_persistent - m_recording

    records = []
    for path, array in walk_arrays(model):
        if np.asarray(array).nbytes < args.min_bytes:
            continue
        rec = describe(path, array)
        rec["lifetime"] = "persistent"
        rec["_array"] = array
        records.append(rec)
    records.sort(key=lambda r: -r["bytes"])

    if args.no_usage:
        for rec in records:
            rec.pop("_array")
            rec["U_k"] = None
            rec["U_k_method"] = "skipped"
    else:
        measure_usage(model, records, run_fn, baseline)

    edge_list = model.params.get("edge_list")
    report = {
        "schema": "jaxfne.w10.allocation_map.v1",
        "config": {
            "n": args.n,
            "p_connect": args.p_connect,
            "max_in_degree": args.max_in_degree,
            "duration_ms": args.duration_ms,
            "dt_ms": args.dt_ms,
        },
        "realized": {
            "n_edges": int(edge_list.n_edges) if edge_list is not None else 0,
            "recurrent_backend": signals.metadata.get("recurrent_backend"),
        },
        "memory_bytes": {
            "M_persistent": m_persistent,
            "M_recording": m_recording,
            "M_peak_rss_delta": m_peak_delta,
            "M_temporary_derived": m_temporary,
            "note": ("M_temporary is derived from RSS and includes interpreter and "
                     "allocator overhead; treat it as an upper bound, not an exact figure."),
        },
        "arrays": records,
    }

    print(f"N={args.n}  p_connect={args.p_connect}  K_max={args.max_in_degree}  "
          f"E={report['realized']['n_edges']:,}  "
          f"backend={report['realized']['recurrent_backend']}")
    print(f"  M_persistent {m_persistent / 1e6:9.3f} MB   "
          f"M_recording {m_recording / 1e6:9.3f} MB")
    print(f"  M_peak(RSS)  {m_peak_delta / 1e6:9.3f} MB   "
          f"M_temporary {m_temporary / 1e6:9.3f} MB (derived)")
    print(f"\n{'bytes':>12} {'U_k':>4} {'R_k':>12}  {'shape':<16}{'dtype':<9}name")
    for rec in records:
        u = "-" if rec["U_k"] is None else str(rec["U_k"])
        r = "-" if rec["R_k"] is None else f"{rec['R_k']:,.0f}"
        print(f"{rec['bytes']:>12,} {u:>4} {r:>12}  "
              f"{str(rec['shape']):<16}{rec['dtype']:<9}{rec['name']}")

    dead = [r for r in records if r["U_k"] == 0]
    if dead:
        total_dead = sum(r["bytes"] for r in dead)
        print(f"\nNot consumed by canonical execution: {len(dead)} array(s), "
              f"{total_dead / 1e6:.3f} MB "
              f"({100 * total_dead / max(m_persistent, 1):.1f}% of M_persistent)")
        for r in dead:
            print(f"  {r['name']}  {r['shape']}  {r['bytes'] / 1e6:.3f} MB")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nreport: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
