#!/usr/bin/env python3
"""0.5.0 item 3: independent phase profile on frozen representative models.

Per regime: N/E, construct ms, first-call (compile+run) vs warm-call ms,
RSS deltas, recording volume, observe ms. No optimization, measurement only.
Writes artifacts/perf/profile_050.json (tracked). Local receipt only.

Usage:
    python scripts/profile_050_phases.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "artifacts" / "perf" / "profile_050.json"


def _rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 1e6


def _timed(fn):  # type: ignore[no-untyped-def]
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000.0


def _nbytes(a) -> int | None:
    try:
        import numpy as np

        b = np.ascontiguousarray(np.asarray(a))
        return int(b.nbytes)
    except Exception:
        return None


def profile(name: str, build, duration_ms: float, dt_ms: float, seed: int, runtime=None) -> dict:
    import jax
    import jaxfne as J

    r0 = _rss_mb()
    model, t_construct = _timed(build)
    r1 = _rss_mb()
    try:
        n_edges = int(model.params["edge_list"].n_edges)
    except Exception:
        n_edges = None
    try:
        n_neurons = len(model.neuron_table())
    except Exception:
        n_neurons = None

    if runtime is None:
        sim = J.simulation(duration_ms=duration_ms, dt_ms=dt_ms, seed=seed)
    else:
        sim = J.Simulation(duration_ms=duration_ms, dt_ms=dt_ms, seed=seed, runtime=runtime)
    signals, t_first = _timed(lambda: model.simulate(sim))
    r2 = _rss_mb()
    signals2, t_warm = _timed(lambda: model.simulate(sim))
    r3 = _rss_mb()

    recording: dict = {}
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
    for key in ("spikes", "V_m", "sources"):
        try:
            recording[key] = {"bytes": _nbytes(getattr(signals2, key))}
        except Exception:
            recording[key] = {"bytes": None}

    _, t_probe = _timed(lambda: model.probe(signals2, modes=["spikes", "V_m"]))
    _, t_manifest = _timed(lambda: model.manifest(signals2))

    devices = [str(d) for d in jax.devices()]
    return {
        "regime": name,
        "n_neurons": n_neurons,
        "n_edges": n_edges,
        "duration_ms": duration_ms,
        "dt_ms": dt_ms,
        "seed": seed,
        "construct_ms": t_construct,
        "simulate_first_call_ms": t_first,
        "simulate_warm_call_ms": t_warm,
        "probe_observe_ms": t_probe,
        "manifest_ms": t_manifest,
        "rss_mb": {
            "start": r0,
            "after_construct": r1,
            "after_first_sim": r2,
            "after_warm_sim": r3,
            "construct_delta": r1 - r0,
            "first_sim_delta": r2 - r1,
            "warm_sim_delta": r3 - r2,
        },
        "recording": recording,
        "device": devices,
        "backend": devices[0].split(":")[0] if devices else "unknown",
    }


def _hdp_pair():
    sys.path.insert(0, str(ROOT / "scripts"))
    import benchmark_050_baseline as B

    return B._hdp_model, B._hdp_runtime()


def main() -> int:
    import jaxfne as J

    out: list = []

    def build_1n():
        return J.construct(
            J.configuration()
            .network(n=1)
            .emitter(family="izhikevich", preset="regular_spiking")
            .field(domain="point")
            .probe(name="single_neuron", modes=["spikes", "V_m"])
        )

    def build_100n():
        return J.construct(
            J.configuration()
            .network(
                name="network_100_ei",
                kind="balanced_ei_population",
                n=100,
                cell_types={"E": 0.75, "PV": 0.25},
            )
            .emitter(family="izhikevich", preset="cortical_eig")
            .field(
                domain="laminar_column",
                conductivity="proxy",
                boundary="declared_proxy",
                gauge="mean_zero",
            )
            .probe(name="multimodal_100_ei", modes=["spikes", "V_m", "source", "LFP"])
        )

    def build_1000n():
        return J.construct(
            J.load_canonical_neuronal_tensor("canonical-v1-column-1000n"),
            J.RuntimeConfiguration(seed=0, duration_ms=200.0, dt_ms=0.5),
        )

    out.append(profile("dispatch_1n", build_1n, 100.0, 0.1, 0))
    _hdp_build, hdp_runtime = _hdp_pair()
    out.append(profile("hdp_10n", _hdp_build, 100.0, 0.5, 0, runtime=hdp_runtime))
    out.append(profile("sparse_100n", build_100n, 100.0, 0.1, 42))
    out.append(profile("column_1000n", build_1000n, 200.0, 0.5, 0))

    report = {
        "series": "profile_050",
        "jaxfne_version": J.__version__,
        "platform": platform.platform(),
        "claim": "local_environment_receipt_only",
        "memory_note": "process RSS deltas (CPU backend); device-peak not separately measured",
        "regimes": out,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in out:
        print(
            f"{r['regime']}: N={r['n_neurons']} E={r['n_edges']} "
            f"construct={r['construct_ms']:.0f}ms "
            f"first={r['simulate_first_call_ms']:.0f}ms "
            f"warm={r['simulate_warm_call_ms']:.0f}ms "
            f"rss_d1={r['rss_mb']['first_sim_delta']:.1f}MB",
            flush=True,
        )
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
