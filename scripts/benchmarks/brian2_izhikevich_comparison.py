"""Real cross-tool comparison: Brian2 vs jaxfne on a matched Izhikevich
sparse-network task. Same N, same sparse connectivity density (max in-degree
~100 per neuron via a fixed connection probability), same duration, CPU only,
default backend settings for both (no hand-tuning either side).

Dimensionless v/u state (matching jaxfne's own convention: v is "internal
units, uncalibrated relative to mV", not literal SI volts) -- this is also
the standard way Brian2's own official Izhikevich example is written
(brian2.readthedocs.io/en/stable/examples/frompapers.Izhikevich_2003.html).

NOT part of jaxfne's own dependencies -- Brian2 is intentionally NOT added to
pyproject.toml (this is a one-off comparison, not a feature). Run this in a
separate, throwaway environment:
    python3 -m venv /tmp/brian2_bench_venv
    /tmp/brian2_bench_venv/bin/pip install brian2
    /tmp/brian2_bench_venv/bin/python3 scripts/benchmarks/brian2_izhikevich_comparison.py

The matched jaxfne-side script is
scripts/benchmarks/jaxfne_izhikevich_comparison.py (runs in jaxfne's normal
environment, no extra install needed). Recorded receipt + caveats:
docs/notes/brian2_benchmark_receipt.md.
"""
import time
import numpy as np
from brian2 import NeuronGroup, Synapses, SpikeMonitor, run, ms, prefs, start_scope

def run_brian2(n, duration_ms, seed=0):
    start_scope()
    np.random.seed(seed)
    t0 = time.time()

    eqs = '''
    dv/dt = (0.04*v**2 + 5*v + 140 - u + I)/ms : 1
    du/dt = a*(b*v - u)/ms : 1
    a : 1
    b : 1
    c : 1
    d : 1
    I : 1
    '''
    G = NeuronGroup(n, eqs, threshold='v>=30', reset='v=c; u+=d', method='euler', dt=0.5*ms)
    G.v = -65.0
    G.u = -13.0
    G.a = 0.02
    G.b = 0.2
    G.c = -65.0
    G.d = 8.0
    G.I = 6.0

    p_connect = min(1.0, 100.0 / n)  # ~100 in-degree, matching jaxfne's max_in_degree=100
    S = Synapses(G, G, on_pre='v_post += 0.3')
    S.connect(condition='i != j', p=p_connect)
    n_synapses = len(S)

    t_construct = time.time() - t0

    M = SpikeMonitor(G)
    t0 = time.time()
    run(duration_ms * ms)
    t_sim = time.time() - t0

    return {
        "n": n, "n_synapses": n_synapses,
        "t_construct": t_construct, "t_sim": t_sim,
        "n_spikes": M.num_spikes,
        "backend": str(prefs.codegen.target),
    }

if __name__ == "__main__":
    for n in (1000, 5000):
        r = run_brian2(n, duration_ms=200.0)
        print(f"BRIAN2 N={r['n']:6d} backend={r['backend']:>10s} construct={r['t_construct']:7.3f}s "
              f"sim={r['t_sim']:7.3f}s n_synapses={r['n_synapses']:8d} n_spikes={r['n_spikes']}")
