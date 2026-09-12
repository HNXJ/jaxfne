"""24-W16-6 bounded classification experiment: spatial decimation map.

Operator: keep every k-th neuron in x-position order, remap edges through
existing selector semantics (no new reduction engine; experiment-local only).
Primary observable: spike_rate_hz_mean (self-averaging).
Criterion (experiment-specific, human-set): eps_r = max(0.5 Hz, 0.1 * SD_full).
Negative control: laminar LFP proxy with preserved probe geometry
(normalized waveform error + amplitude error; no epsilon gate).
Verdict lives in the receipt; tests pin mechanics + determinism only.
"""

from __future__ import annotations

import time
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne

SEEDS = (11, 12, 13, 14)
STRIDES = (1, 2, 4)
DURATION_MS, DT_MS = 800.0, 1.0


def _base_model(n=48):
    cfg = (
        jtfne.configuration()
        .runtime(seed=0, recurrent_backend="edge_list")
        .network(name="V1", kind="cortical_column", n=n,
                 cell_types={"E": 0.8, "PV": 0.2})
        .cell_type_drives({"E": 10.0, "PV": 10.0})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    return jtfne.construct(cfg)


def _decimate_model(model, stride):
    """Experiment-local spatial decimation (stride keep in x order + remap)."""
    assert int(stride) >= 1
    params = model.params
    emitter, edges = params["emitter"], params["edge_list"]
    n = int(emitter.n_neurons)
    pos = np.asarray(params["positions"])
    order = np.argsort(pos[:, 0], kind="stable")
    keep = np.sort(order[:: int(stride)])
    # product selector resolves the same neuron set (EDGE-01 semantics)
    sel = np.asarray(model.select(ids=tuple(int(i) for i in keep)))
    assert np.array_equal(sel, keep)
    remap = np.full(n, -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    pre = np.asarray(edges.pre)
    post = np.asarray(edges.post)
    emask = np.isin(pre, keep) & np.isin(post, keep)

    def _take_per_edge(arr):
        # Slice full-length (E,) arrays; compact placeholders (size 0) and
        # class tables pass through untouched regardless of storage name.
        a = np.asarray(arr)
        return a[emask] if a.shape[0] == len(pre) else arr

    def _take_neuron(arr):
        a = np.asarray(arr)
        return arr[keep] if a.ndim > 0 else jnp.asarray(a)

    new_emitter = replace(
        emitter,
        a=emitter.a[keep],
        b=emitter.b[keep],
        c=emitter.c[keep],
        d=emitter.d[keep],
        drive=_take_neuron(emitter.drive),
        sign=_take_neuron(emitter.sign),
        W=emitter.W[np.ix_(keep, keep)]
        if getattr(emitter.W, "shape", (0,))[0] == n
        else emitter.W,
        v0=emitter.v0[keep],
        u0=emitter.u0[keep],
        source_scale=_take_neuron(emitter.source_scale),
        labels=tuple(emitter.labels[i] for i in keep),
        layer_labels=tuple(emitter.layer_labels[i] for i in keep)
        if emitter.layer_labels is not None
        else None,
    )
    new_edges = replace(
        edges,
        pre=jnp.asarray(remap[pre[emask]], dtype=jnp.int32),
        post=jnp.asarray(remap[post[emask]], dtype=jnp.int32),
        weight=jnp.asarray(_take_per_edge(edges.weight)),
        tau_ms=jnp.asarray(_take_per_edge(edges.tau_ms)),
        delay_steps=jnp.asarray(
            _take_per_edge(edges.delay_steps), dtype=jnp.int32
        ),
        receptor_index=jnp.asarray(
            _take_per_edge(edges.receptor_index),
            dtype=jnp.int32,
        ),
    )
    new_params = dict(params)
    new_params["emitter"] = new_emitter
    new_params["edge_list"] = new_edges
    new_params["positions"] = jnp.asarray(pos[keep])
    new_static = dict(model.static)
    if isinstance(new_static.get("neuron_metadata"), list) and len(
        new_static["neuron_metadata"]
    ) == n:
        new_static["neuron_metadata"] = [new_static["neuron_metadata"][i] for i in keep]
    from jaxfne._model import Model

    return Model(cfg=model.cfg, params=new_params, static=new_static)


def _run(model, seed):
    sim = jtfne.simulation(
        duration_ms=DURATION_MS,
        dt_ms=DT_MS,
        seed=seed,
        record_sources=True,
        record_fields=True,
        runtime=jtfne.RuntimeConfig(
            recurrent_backend="edge_list",
            hdp_params={"noise_scale": 0.0},
        ),
    )
    t0 = time.perf_counter()
    sig = jtfne.simulate(model, sim)
    dt = time.perf_counter() - t0
    rate = float(np.mean(np.asarray(sig.spikes)) * (1000.0 / DT_MS))
    lfp = np.asarray(sig.field.lfp_proxy)
    return rate, lfp, dt, int(np.asarray(sig.V_m).nbytes)


def _lfp_errors(full, dec):
    assert full.shape[1] == dec.shape[1]  # probe geometry preserved
    num = np.linalg.norm(full - dec, axis=0)
    den = np.linalg.norm(full, axis=0) + np.linalg.norm(dec, axis=0)
    nwe = float(np.mean(np.divide(num, den, out=np.zeros_like(num), where=den > 0)))
    rms_f = float(np.sqrt(np.mean(full ** 2)))
    rms_d = float(np.sqrt(np.mean(dec ** 2)))
    amp = abs(rms_f - rms_d) / rms_f if rms_f > 0 else 0.0
    return nwe, float(amp)


def test_decimation_keeps_valid_executable_models():
    model = _base_model()
    n0 = int(model.params["emitter"].n_neurons)
    e0 = int(model.params["edge_list"].n_edges)
    for k in STRIDES:
        dec = _decimate_model(model, k)
        n = int(dec.params["emitter"].n_neurons)
        assert n == (n0 + k - 1) // k
        assert int(dec.params["edge_list"].n_edges) <= e0
        pre = np.asarray(dec.params["edge_list"].pre)
        post = np.asarray(dec.params["edge_list"].post)
        assert pre.shape == post.shape
        assert bool(((pre >= 0) & (pre < n) & (post >= 0) & (post < n)).all())
        sig = _run(dec, SEEDS[0])[0:1]
        assert np.isfinite(sig[0])


def test_stride_one_is_identity_remap():
    model = _base_model()
    dec = _decimate_model(model, 1)
    assert int(dec.params["emitter"].n_neurons) == int(model.params["emitter"].n_neurons)
    assert int(dec.params["edge_list"].n_edges) == int(model.params["edge_list"].n_edges)
    r0, _, _, _ = _run(model, SEEDS[0])
    r1, _, _, _ = _run(dec, SEEDS[0])
    assert r0 == r1


def test_rate_and_lfp_error_curves_reported():
    model = _base_model()
    full_rates, full_lfps = {}, {}
    for s in SEEDS:
        r, lfp, _, _ = _run(model, s)
        full_rates[s], full_lfps[s] = r, lfp
    r_full = np.array([full_rates[s] for s in SEEDS])
    s_r = float(np.std(r_full))
    eps = max(0.5, 0.1 * s_r)
    assert s_r < 5.0, f"full-model rate unstable across seeds (s={s_r}); experiment degenerate"
    print(f"\nfull rates: {r_full}, s_r={s_r:.3f} Hz, eps_r={eps:.3f} Hz")
    for k in STRIDES[1:]:
        dec = _decimate_model(model, k)
        errs, nwes, amps = [], [], []
        for s in SEEDS:
            r, lfp, _, _ = _run(dec, s)
            errs.append(r - full_rates[s])
            nwe, amp = _lfp_errors(full_lfps[s], lfp)
            nwes.append(nwe)
            amps.append(amp)
        errs = np.array(errs)
        print(
            f"k={k}: bias={np.mean(errs):+.3f} Hz var={np.var(errs):.4f} "
            f"abs={np.mean(np.abs(errs)):.3f} Hz pass_eps={bool(np.mean(np.abs(errs)) <= eps)} "
            f"LFP_nwe={np.mean(nwes):.3f} LFP_amp={np.mean(amps):.3f}"
        )
        assert np.isfinite(errs).all()


def test_decimation_reduces_compute_and_recording():
    model = _base_model()
    _, _, t_full, b_full = _run(model, SEEDS[0])
    dec = _decimate_model(model, 4)
    _, _, t_dec, b_dec = _run(dec, SEEDS[0])
    print(f"\nwall full={t_full:.2f}s dec={t_dec:.2f}s bytes full={b_full} dec={b_dec}")
    assert b_dec < b_full
