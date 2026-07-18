"""Real cross-tool comparison: jaxfne side, matched to
scripts/benchmarks/brian2_izhikevich_comparison.py's task (same N, same
~100-in-degree sparse connectivity via max_in_degree, same 200ms duration
at dt=0.5ms, CPU only). Runs in jaxfne's normal environment -- no extra
install needed.

Recorded receipt + caveats: docs/notes/brian2_benchmark_receipt.md.
"""
import time
import jax
import jax.numpy as jnp
import jaxfne as jtfne
from scripts.cortical_column_localized_workflow import build_config


def run_jaxfne(n, duration_ms=200.0, dt_ms=0.5, max_in_degree=100, seed=0):
    t0 = time.time()
    cfg = build_config(n=n, duration_ms=duration_ms, dt_ms=dt_ms, max_in_degree=max_in_degree)
    model = jtfne.construct(cfg)
    t_construct = time.time() - t0
    n_edges = model.params["edge_list"].n_edges

    sim = jtfne.simulation(duration_ms=duration_ms, dt_ms=dt_ms, seed=seed)
    t0 = time.time()
    sig = model.simulate(sim)
    sig.V_m.block_until_ready()
    t_sim = time.time() - t0

    n_spikes = int(jnp.sum(sig.spikes))
    return {
        "n": n, "n_edges": n_edges, "t_construct": t_construct, "t_sim": t_sim,
        "n_spikes": n_spikes, "backend": jax.default_backend(),
    }


if __name__ == "__main__":
    for n in (1000, 5000):
        r = run_jaxfne(n)
        print(f"JAXFNE N={r['n']:6d} backend={r['backend']:>10s} construct={r['t_construct']:7.3f}s "
              f"sim={r['t_sim']:7.3f}s n_edges={r['n_edges']:8d} n_spikes={r['n_spikes']}")
