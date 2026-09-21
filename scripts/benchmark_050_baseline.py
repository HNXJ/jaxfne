#!/usr/bin/env python3
"""0.5.0 baseline: phased wall-clock timings across representative regimes.

Regimes: 1n dispatch, 10n HDP, 100n sparse, 1000n column. Phases per regime:
construct | simulate_first_call (compile+run) | simulate_second_call (run)
| probe | manifest. Memory is reported as declared array bytes (wall-clock
only; device-peak measurement is a listed limitation, not claimed here).

Writes artifacts/perf/baseline_050.json (tracked). Local-environment
receipt only; no universal performance claims.

Usage:
    python scripts/benchmark_050_baseline.py
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

OUT = ROOT / "artifacts" / "perf" / "baseline_050.json"


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _bytes_of(signals, n_neurons: int, n_steps: int) -> dict:
    import numpy as np

    out = {}
    for name in ("spikes", "V_m", "sources"):
        try:
            a = np.asarray(getattr(signals, name))
            out[name] = {"shape": list(a.shape), "bytes": int(a.nbytes)}
        except Exception:
            out[name] = None
    out["expected_TxN_float32"] = n_steps * n_neurons * 4
    return out


def run_regime(
    name: str,
    build,
    n_neurons: int,
    duration_ms: float,
    dt_ms: float,
    seed: int,
    hdp: bool = False,
    runtime=None,
) -> dict:
    import jaxfne as J

    phases: dict = {}
    t0 = time.perf_counter()
    model = build()
    phases["construct_ms"] = _ms(t0)

    if runtime is None:
        sim = J.simulation(duration_ms=duration_ms, dt_ms=dt_ms, seed=seed)
    else:
        sim = J.Simulation(duration_ms=duration_ms, dt_ms=dt_ms, seed=seed, runtime=runtime)
    t0 = time.perf_counter()
    signals = model.simulate(sim)
    phases["simulate_first_call_ms"] = _ms(t0)

    t0 = time.perf_counter()
    signals2 = model.simulate(sim)
    phases["simulate_second_call_ms"] = _ms(t0)

    t0 = time.perf_counter()
    try:
        model.probe(signals, modes=["spikes", "V_m"])
    except Exception:
        pass
    phases["probe_ms"] = _ms(t0)

    t0 = time.perf_counter()
    try:
        model.manifest(signals)
    except Exception:
        pass
    phases["manifest_ms"] = _ms(t0)

    n_steps = int(round(duration_ms / dt_ms))
    return {
        "regime": name,
        "n_neurons": n_neurons,
        "duration_ms": duration_ms,
        "dt_ms": dt_ms,
        "seed": seed,
        "hdp": hdp,
        "phases_ms": phases,
        "arrays": _bytes_of(signals2, n_neurons, n_steps),
    }


def main() -> int:
    import jaxfne as J

    regimes = []
    regimes.append(
        run_regime(
            "dispatch_1n",
            lambda: J.construct(
                J.configuration()
                .network(n=1)
                .emitter(family="izhikevich", preset="regular_spiking")
                .field(domain="point")
                .probe(name="single_neuron", modes=["spikes", "V_m"])
            ),
            1,
            100.0,
            0.1,
            0,
        )
    )
    regimes.append(
        run_regime(
            "hdp_10n", lambda: _hdp_model(), 10, 100.0, 0.5, 0, hdp=True, runtime=_hdp_runtime()
        )
    )
    regimes.append(
        run_regime(
            "sparse_100n",
            lambda: J.construct(
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
            ),
            100,
            100.0,
            0.1,
            42,
        )
    )
    regimes.append(
        run_regime(
            "column_1000n",
            lambda: J.construct(
                J.load_canonical_neuronal_tensor("canonical-v1-column-1000n"),
                J.RuntimeConfiguration(seed=0, duration_ms=200.0, dt_ms=0.5),
            ),
            1000,
            200.0,
            0.5,
            0,
        )
    )

    report = {
        "series": "baseline_050",
        "jaxfne_version": J.__version__,
        "platform": platform.platform(),
        "claim": "local_environment_receipt_only",
        "memory_note": "declared array bytes only; device-peak not measured",
        "regimes": regimes,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in regimes:
        p = r["phases_ms"]
        print(
            f"{r['regime']}: construct={p['construct_ms']:.0f}ms "
            f"sim1={p['simulate_first_call_ms']:.0f}ms "
            f"sim2={p['simulate_second_call_ms']:.0f}ms",
            flush=True,
        )
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


def _hdp_model():
    import jaxfne as J

    cfg = (
        J.configuration()
        .network(name="V1", kind="cortical_column", n=10, cell_types={"E": 0.5, "PV": 0.5})
        .cell_type_drives({"E": 8.0, "PV": 8.0})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="mean_zero_neumann",
            gauge="mean_zero",
        )
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    model = J.construct(cfg)
    return model


def _hdp_runtime():
    import jaxfne as J

    return J.RuntimeConfig(
        enable_hdp=True,
        recurrent_backend="edge_list",
        jit=False,
        hdp_params={
            "K_HDP": 0.01,
            "tau_0_ms": 200.0,
            "K_ctrl": 5.0,
            "barrier_c": 0.01,
            "barrier_d": 0.01,
            "H_min": 0.1,
            "H_max": 10.0,
            "w_min": -10.0,
            "w_max": 10.0,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
