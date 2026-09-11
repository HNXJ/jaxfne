"""Registered-rule HDP edge-list kernel (zero-delay + finite-delay paths).

23-DELAY-01 owns the finite-delay path: per-edge ``delay_steps`` select a
Protocol D spike ring ``delay_state`` (``(D_max+1, N)``), and registered
rules observe delayed arrivals as per-edge ``pre_sp`` so the qualified
chain holds::

    event_{t-d} -> H_t -> P -> Theta_t -> I_t

Zero-delay behavior is unchanged and bit-exact with HDP-01 qualification.
No history compaction is claimed here: the dense ring matches the legacy
HDP delayed kernel's layout so chunked continuation stays bit-exact.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp

from .emitters import (
    EdgeList,
    IzhikevichParams,
    _delayed_presynaptic_spikes,
    _dtype_from_policy,
    _edge_delays_any_positive,
    _edge_max_delay_steps,
    _izhikevich_dv_du,
    _rbd_continuation_step_offset_array,
    _rbd_delay_state_from_init,
    _segment_sum,
    _source_proxy_from_components,
    _validate_delayed_init_state,
    _validate_edge_delays_nonnegative_eager,
    resolve_edge_delay_steps,
    resolve_edge_tau_ms,
    resolve_receptor_index,
)
from .emitters import _resolved_edge_weight
from .hdp_rule import HDPRuleContext, get_hdp_rule


def simulate_edge_recurrent_izhikevich_hdp_registered(
    params: IzhikevichParams,
    edges: EdgeList,
    n_steps: int,
    dt_ms: float,
    key: jax.Array,
    *,
    dtype: str = "float32",
    drive_schedule: jax.Array | None = None,
    silence_mask: jax.Array | None = None,
    noise_scale: jax.Array | float | None = None,
    init_state: dict | None = None,
    hdp_rule: str,
    hdp_rule_params: Mapping[str, Any] | None = None,
    record_weight_trace: bool = True,
    step_indices: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
    """Edge-list Izhikevich simulation with a registered HDP rule."""
    _validate_edge_delays_nonnegative_eager(edges)
    has_delay = _edge_delays_any_positive(edges)

    descriptor, rule_step = get_hdp_rule(hdp_rule)
    rule_params = {**descriptor.default_params, **(hdp_rule_params or {})}
    h_min, h_max = descriptor.h_bounds
    w_floor, w_ceiling = descriptor.w_bounds

    jdtype = _dtype_from_policy(dtype)
    a = params.a.astype(jdtype)
    b = params.b.astype(jdtype)
    c = params.c.astype(jdtype)
    d = params.d.astype(jdtype)
    drive = params.drive.astype(jdtype)
    source_scale = params.source_scale.astype(jdtype)
    dt = jnp.asarray(dt_ms, dtype=jdtype)
    noise_coef = (
        jnp.asarray(0.5, dtype=jdtype)
        if noise_scale is None
        else jnp.asarray(noise_scale, dtype=jdtype)
    )
    pre = edges.pre.astype(jnp.int32)
    post = edges.post.astype(jnp.int32)
    w_baseline = _resolved_edge_weight(edges, jdtype, params)
    ri = resolve_receptor_index(edges)
    exc_mask = ri == 0
    tau_ms = jnp.maximum(resolve_edge_tau_ms(edges, jdtype), jnp.asarray(1e-6, dtype=jdtype))
    decay = jnp.exp(-dt / tau_ms)
    n_neurons = int(params.v0.shape[0])

    if silence_mask is not None:
        s_mask = silence_mask.astype(jdtype)
    else:
        s_mask = jnp.ones((n_neurons,), dtype=jdtype)

    key, noise_key = jax.random.split(key)
    bulk_noise = jax.random.normal(
        noise_key, shape=(int(n_steps), n_neurons), dtype=jdtype
    )
    sched = (
        jnp.zeros((int(n_steps), n_neurons), dtype=jdtype)
        if drive_schedule is None
        else drive_schedule.astype(jdtype)
    )

    h_min_arr = jnp.asarray(h_min, dtype=jdtype)
    h_max_arr = jnp.asarray(h_max, dtype=jdtype)
    w_floor_arr = jnp.asarray(w_floor, dtype=jdtype)
    w_ceiling_arr = jnp.asarray(w_ceiling, dtype=jdtype)

    def _apply_rule(H, aux, v, u, spikes, prev_spikes, syn_state, w, pre_sp):
        ctx = HDPRuleContext(
            H=H,
            aux=aux,
            v=v,
            u=u,
            spikes=spikes,
            prev_spikes=prev_spikes,
            syn_state=syn_state,
            w=w,
            pre=pre,
            post=post,
            exc_mask=exc_mask,
            dt=dt,
            n_neurons=n_neurons,
            rule_params=rule_params,
            pre_sp=pre_sp,
        )
        upd = rule_step(ctx)
        H_next = jnp.clip(H + dt * upd.dH, h_min_arr, h_max_arr)
        if upd.d_aux is not None:
            aux_next = aux + dt * upd.d_aux
        else:
            aux_next = aux
        if upd.d_theta and "edge_weight" in upd.d_theta:
            wmag = jnp.abs(w)
            dw = upd.d_theta["edge_weight"]
            wmag_next = jnp.clip(wmag + dt * dw, w_floor_arr, w_ceiling_arr)
            w_next = jnp.where(exc_mask, wmag_next, -wmag_next)
        else:
            w_next = w
        return H_next, w_next, aux_next

    if not has_delay:
        if init_state is not None:
            H0 = jnp.asarray(
                init_state.get("H_final", jnp.ones((n_neurons,), dtype=jdtype)),
                dtype=jdtype,
            )
            w0 = jnp.asarray(init_state.get("w_final", w_baseline), dtype=jdtype)
            aux0 = jnp.asarray(
                init_state.get("aux_final", jnp.zeros((0,), dtype=jdtype)),
                dtype=jdtype,
            )
            init = (
                jnp.asarray(init_state["v"], dtype=jdtype),
                jnp.asarray(init_state["u"], dtype=jdtype),
                jnp.asarray(init_state["prev_spikes"], dtype=jdtype),
                jnp.asarray(init_state["syn_state"], dtype=jdtype),
                H0,
                w0,
                aux0,
            )
        else:
            init = (
                params.v0.astype(jdtype),
                params.u0.astype(jdtype),
                jnp.zeros((n_neurons,), dtype=jdtype),
                jnp.zeros((edges.n_edges,), dtype=jdtype),
                jnp.ones((n_neurons,), dtype=jdtype),
                w_baseline,
                jnp.zeros((0,), dtype=jdtype),
            )

        def step(carry, xs):
            v, u, prev_spikes, syn_state, H, w, aux = carry
            sched_t, noise_t = xs
            edge_current = w * syn_state
            syn = _segment_sum(edge_current, post, n_neurons)
            current_native = drive + sched_t + syn + noise_coef * noise_t
            dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
            v_next = v + dt * dv
            u_next = u + dt * du
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            H_next, w_next, aux_next = _apply_rule(
                H, aux, v, u, spikes, prev_spikes, syn_state, w, None
            )
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            syn_next = syn_state * decay + spikes[pre]
            source_proxy = _source_proxy_from_components(
                current_native, spikes, source_scale, dtype=jdtype
            )
            carry_out = (v_reset, u_reset, spikes, syn_next, H_next, w_next, aux_next)
            if record_weight_trace:
                outputs = (v_reset, spikes, source_proxy, H_next, w_next, aux_next)
            else:
                outputs = (v_reset, spikes, source_proxy, H_next, aux_next)
            return carry_out, outputs

        final, scan_outputs = jax.lax.scan(step, init, xs=(sched, bulk_noise))
        if record_weight_trace:
            voltages, spikes, sources, H_trace, w_trace, aux_trace = scan_outputs
        else:
            voltages, spikes, sources, H_trace, aux_trace = scan_outputs
            w_trace = None

        diagnostics = {
            "v": final[0],
            "u": final[1],
            "prev_spikes": final[2],
            "syn_state": final[3],
            "H_final": final[4],
            "w_final": final[5],
            "aux_final": final[6],
            "H_trace": H_trace,
            "w_trace": w_trace,
            "aux_trace": aux_trace,
            "hdp_rule": hdp_rule,
            "hdp_rule_registered": True,
        }
        return voltages, spikes, sources, diagnostics

    # Finite-delay path (Protocol D ring, same layout as legacy HDP kernel).
    delay_steps_arr = resolve_edge_delay_steps(edges)
    max_delay = _edge_max_delay_steps(edges)
    bufsize = max_delay + 1
    time_step_offset = _rbd_continuation_step_offset_array(init_state)
    if init_state is not None and "v" in init_state:
        if "delay_state" in init_state or "spike_history" in init_state:
            _validate_delayed_init_state(
                init_state, bufsize=bufsize, n_neurons=n_neurons, n_edges=edges.n_edges
            )
        spike_hist0 = _rbd_delay_state_from_init(
            init_state, bufsize=bufsize, n_neurons=n_neurons, jdtype=jdtype
        )
    else:
        spike_hist0 = _rbd_delay_state_from_init(
            None, bufsize=bufsize, n_neurons=n_neurons, jdtype=jdtype
        )
    if step_indices is not None:
        step_indices_arr = jnp.asarray(step_indices, dtype=jnp.int32).reshape(-1)
        if int(step_indices_arr.shape[0]) != int(n_steps):
            raise ValueError(
                "step_indices must have shape (n_steps,) when provided; got "
                f"{step_indices_arr.shape} for n_steps={n_steps}"
            )
    else:
        off = time_step_offset
        if isinstance(off, int):
            step_indices_arr = jnp.arange(off, off + int(n_steps), dtype=jnp.int32)
        else:
            step_indices_arr = jnp.arange(
                off,
                off + jnp.asarray(int(n_steps), dtype=jnp.int32),
                dtype=jnp.int32,
            )

    if init_state is not None and "v" in init_state:
        H0 = jnp.asarray(
            init_state.get("H_final", jnp.ones((n_neurons,), dtype=jdtype)),
            dtype=jdtype,
        )
        w0 = jnp.asarray(init_state.get("w_final", w_baseline), dtype=jdtype)
        aux0 = jnp.asarray(
            init_state.get("aux_final", jnp.zeros((0,), dtype=jdtype)),
            dtype=jdtype,
        )
        init = (
            jnp.asarray(init_state["v"], dtype=jdtype),
            jnp.asarray(init_state["u"], dtype=jdtype),
            jnp.asarray(init_state["prev_spikes"], dtype=jdtype),
            jnp.asarray(init_state["syn_state"], dtype=jdtype),
            H0,
            w0,
            aux0,
            spike_hist0,
        )
    else:
        init = (
            params.v0.astype(jdtype),
            params.u0.astype(jdtype),
            jnp.zeros((n_neurons,), dtype=jdtype),
            jnp.zeros((edges.n_edges,), dtype=jdtype),
            jnp.ones((n_neurons,), dtype=jdtype),
            w_baseline,
            jnp.zeros((0,), dtype=jdtype),
            spike_hist0,
        )

    def step_delayed(carry, xs_t):
        t_idx, sched_t, noise_t = xs_t
        v, u, prev_spikes, syn_state, H, w, aux, spike_hist = carry
        edge_current = w * syn_state
        syn = _segment_sum(edge_current, post, n_neurons)
        current_native = drive + sched_t + syn + noise_coef * noise_t
        dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
        v_next = v + dt * dv
        u_next = u + dt * du
        v_next = jnp.where(s_mask > 0.5, v_next, c)
        spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
        spikes = spikes_bool.astype(jdtype)
        presyn = _delayed_presynaptic_spikes(spikes, spike_hist, t_idx, pre, delay_steps_arr)
        H_next, w_next, aux_next = _apply_rule(
            H, aux, v, u, spikes, prev_spikes, syn_state, w, presyn
        )
        v_reset = jnp.where(spikes_bool, c, v_next)
        u_reset = jnp.where(spikes_bool, u_next + d, u_next)
        syn_next = syn_state * decay + presyn
        slot = jnp.mod(t_idx, bufsize)
        spike_hist_next = spike_hist.at[slot].set(spikes)
        source_proxy = _source_proxy_from_components(
            current_native, spikes, source_scale, dtype=jdtype
        )
        carry_out = (v_reset, u_reset, spikes, syn_next, H_next, w_next, aux_next, spike_hist_next)
        if record_weight_trace:
            outputs = (v_reset, spikes, source_proxy, H_next, w_next, aux_next)
        else:
            outputs = (v_reset, spikes, source_proxy, H_next, aux_next)
        return carry_out, outputs

    final, scan_outputs = jax.lax.scan(
        step_delayed, init, xs=(step_indices_arr, sched, bulk_noise)
    )
    if record_weight_trace:
        voltages, spikes, sources, H_trace, w_trace, aux_trace = scan_outputs
    else:
        voltages, spikes, sources, H_trace, aux_trace = scan_outputs
        w_trace = None

    diagnostics = {
        "v": final[0],
        "u": final[1],
        "prev_spikes": final[2],
        "syn_state": final[3],
        "H_final": final[4],
        "w_final": final[5],
        "aux_final": final[6],
        "delay_state": final[7],
        "spike_history": final[7],
        "delay_steps_max": jnp.asarray(max_delay, dtype=jnp.int32),
        "continuation_step_offset": step_indices_arr[-1] + jnp.asarray(1, dtype=jnp.int32),
        "H_trace": H_trace,
        "w_trace": w_trace,
        "aux_trace": aux_trace,
        "hdp_rule": hdp_rule,
        "hdp_rule_registered": True,
    }
    return voltages, spikes, sources, diagnostics
