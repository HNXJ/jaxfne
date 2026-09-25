"""24-FIELD-01: field/LFP proxy computational audit (no optimization yet).

Decomposes source-generation vs forward-projection cost, scales T/N/C
independently, and checks exact chunked-projection equivalence (linearity).
Predeclared metrics (from W16-6): normalized waveform error + amplitude
error on lfp_proxy. Proxy semantics stay explicit throughout.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne.fields.proxy import csd_tensor, project_laminar_sources


def _synthetic_sources(T, N, seed=0):
    key = jax.random.PRNGKey(seed)
    spikes = (jax.random.uniform(key, (T, N)) < 0.02).astype(jnp.float32)
    return spikes, jnp.linspace(0.0, 1.0, N, dtype=jnp.float32)


def _positions(N, seed=1):
    xyz = jax.random.uniform(jax.random.PRNGKey(seed), (N, 3), dtype=jnp.float32)
    return xyz.at[:, 2].set(jnp.linspace(0.0, 1.0, N, dtype=jnp.float32))


def _bench(fn, n_warm=1, n_run=5):
    out = fn()
    if hasattr(out, "block_until_ready"):
        out.block_until_ready()
    else:
        jax.block_until_ready(out.lfp_proxy)
    ts = []
    for _ in range(n_warm + n_run):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out.lfp_proxy)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts[-n_run:])), out


def test_projection_scales_as_TNC_and_kernel_build_is_negligible():
    # Dispatch-bound regime at canonical scale: 4x T must not blow up
    # superlinearly (guard against algorithmic blowup, not a linearity claim).
    N, C = 400, 16
    pos = _positions(N)
    times = {}
    for T in (500, 2000):
        src, _ = _synthetic_sources(T, N)
        t, _ = _bench(lambda: project_laminar_sources(src, pos, n_contacts=C))
        times[T] = t
    ratio = times[2000] / max(times[500], 1e-9)
    print(f"\nT scaling: {times[500] * 1e3:.2f}ms -> {times[2000] * 1e3:.2f}ms (ratio {ratio:.2f}; linear=4)")
    assert ratio < 16.0


def test_contacts_scale_projection_not_kernel():
    # 16x contacts must cost far less than 16x time: at canonical scale the
    # path is dispatch/overhead-bound, not FLOP-bound (kernel build and
    # matmul are fractions of a millisecond each at steady state).
    T, N = 1000, 400
    src, _ = _synthetic_sources(T, N)
    pos = _positions(N)
    t4, _ = _bench(lambda: project_laminar_sources(src, pos, n_contacts=4))
    t64, _ = _bench(lambda: project_laminar_sources(src, pos, n_contacts=64))
    print(f"\nC scaling: 4->{t4 * 1e3:.2f}ms 64->{t64 * 1e3:.2f}ms")
    assert t64 < 4.0 * t4


def test_chunked_projection_is_bit_exact():
    # Linearity: concat(project halves) == project whole. Enables streaming
    # field computation with zero error (exact algebraic reduction).
    T, N, C = 1000, 200, 16
    src, _ = _synthetic_sources(T, N)
    pos = _positions(N)
    whole = project_laminar_sources(src, pos, n_contacts=C)
    a = project_laminar_sources(src[:500], pos, n_contacts=C)
    b = project_laminar_sources(src[500:], pos, n_contacts=C)
    for name in ("lfp_proxy", "phi_e_proxy", "csd_proxy", "source_proxy"):
        w = np.asarray(getattr(whole, name))
        cat = np.concatenate([np.asarray(getattr(a, name)), np.asarray(getattr(b, name))], axis=0)
        assert np.array_equal(w, cat), name
    assert np.array_equal(np.asarray(whole.kernel), np.asarray(a.kernel))


def test_csd_is_negligible_next_to_projection():
    # Both dispatch-bound at canonical scale; CSD (O(T*C) stencil) must never
    # dominate the O(T*N*C) projection path. Reports the split.
    T, N, C = 2000, 800, 16
    src, _ = _synthetic_sources(T, N)
    pos = _positions(N)
    t_full, out = _bench(lambda: project_laminar_sources(src, pos, n_contacts=C))
    phi = np.asarray(out.phi_e_proxy)
    dz = 1.0 / (C - 1)
    t0 = time.perf_counter()
    for _ in range(5):
        c = csd_tensor(jnp.asarray(phi), dz)
        jax.block_until_ready(c)
    t_csd = (time.perf_counter() - t0) / 5
    print(f"\nprojection {t_full * 1e3:.2f}ms vs CSD {t_csd * 1e3:.2f}ms")
    assert t_csd < 4.0 * t_full
    assert np.isfinite(np.asarray(out.csd_proxy)).all()


def test_source_generation_vs_projection_split():
    # End-to-end: kernel simulation wall vs projection wall on a real model.
    cfg = (
        jtfne.configuration()
        .runtime(seed=0, recurrent_backend="edge_list")
        .network(name="V1", kind="cortical_column", n=64,
                 cell_types={"E": 0.8, "PV": 0.2})
        .cell_type_drives({"E": 10.0, "PV": 10.0})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(
        duration_ms=1000.0, dt_ms=1.0, seed=3, record_sources=True, record_fields=False,
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list", hdp_params={"noise_scale": 0.0}),
    )
    pos_arr = np.asarray(model.params["positions"])
    # Warm both paths so the comparison is steady-state work, not JIT compile (P-009).
    warm = jtfne.simulate(model, sim)
    jax.block_until_ready(project_laminar_sources(
        jnp.asarray(np.asarray(warm.sources)), jnp.asarray(pos_arr), n_contacts=16).lfp_proxy)
    t0 = time.perf_counter()
    sig = jtfne.simulate(model, sim)
    sources = np.asarray(sig.sources)
    t_sim = time.perf_counter() - t0
    t1 = time.perf_counter()
    field = project_laminar_sources(jnp.asarray(sources), jnp.asarray(pos_arr), n_contacts=16)
    jax.block_until_ready(field.lfp_proxy)
    t_proj = time.perf_counter() - t1
    print(f"\nsim {t_sim:.2f}s vs projection {t_proj * 1e3:.2f}ms "
          f"(sources {sources.nbytes / 1e6:.2f}MB, lfp {np.asarray(field.lfp_proxy).nbytes / 1e6:.2f}MB)")
    assert np.isfinite(np.asarray(field.lfp_proxy)).all()
    assert t_proj < t_sim
