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
import numpy as np

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
    _validate_recording_budget,
    _decimate_hw_traces,
    resolve_edge_delay_steps,
    resolve_edge_tau_ms,
    resolve_receptor_index,
)
from .emitters import _resolved_edge_weight
from .hdp_rule import (
    HDPRuleContext,
    check_hdp_rule_params,
    expected_aux_shape,
    get_hdp_rule,
)


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
    noise_schedule: jax.Array | None = None,
    init_state: dict | None = None,
    hdp_rule: str,
    hdp_rule_params: Mapping[str, Any] | None = None,
    record_weight_trace: bool = True,
    record_stride: int = 1,
    record_h_subset: jax.Array | None = None,
    record_w_subset: jax.Array | None = None,
    step_indices: jax.Array | None = None,
    plasticity_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
    """Edge-list Izhikevich simulation with a registered HDP rule.

    ``record_stride`` / ``record_h_subset`` / ``record_w_subset`` are the
    0.5.3 item 2 declared H/W recording budgets (defaults = full
    recording); kept frames equal full-trace frames exactly.
    ``noise_schedule`` optionally supplies the exact per-step unit-noise
    draws; the Model plain path passes the continuation-chain schedule so
    chunked == continuous (membrane and per-rule streams), while ``None``
    keeps the legacy bulk draw bit-identical for direct kernel callers.
    """
    _validate_edge_delays_nonnegative_eager(edges)
    has_delay = _edge_delays_any_positive(edges)

    descriptor, rule_step = get_hdp_rule(hdp_rule)
    # 0.5.3 item 7b: unknown rule-param keys fail closed (typo'd gains must
    # never simulate as defaults); identical merge for valid inputs.
    rule_params = check_hdp_rule_params(hdp_rule, hdp_rule_params)
    # 0.5.3 item 5: per-projection gate; None selects the bit-exact path.
    if plasticity_mask is not None:
        _mask_raw = np.asarray(plasticity_mask)
        if _mask_raw.ndim != 1 or _mask_raw.shape[0] != int(edges.n_edges):
            raise ValueError(
                "plasticity_mask must have shape "
                f"({int(edges.n_edges)},), got {_mask_raw.shape}"
            )
        if not bool(np.all(np.isfinite(_mask_raw))):
            raise ValueError("plasticity_mask must be finite (got NaN/inf)")
        plastic_edge = jnp.asarray(_mask_raw > 0.5)
    else:
        plastic_edge = None
    h_min, h_max = descriptor.h_bounds
    w_floor, w_ceiling = descriptor.w_bounds
    b_min, b_max = descriptor.b_bounds
    use_b = "drive_bias" in descriptor.theta_targets

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
    h_shape = (int(n_neurons),) + tuple(int(d) for d in descriptor.h_shape)

    # 0.5.3 item 2: declared H/W recording budgets (fail closed; defaults
    # keep full recording).
    record_stride_n, record_h_idx, record_w_idx = _validate_recording_budget(
        record_stride, record_h_subset, record_w_subset,
        n_neurons=n_neurons, n_edges=int(edges.n_edges),
        record_weight_trace=record_weight_trace,
    )

    if silence_mask is not None:
        s_mask = silence_mask.astype(jdtype)
    else:
        s_mask = jnp.ones((n_neurons,), dtype=jdtype)

    if noise_schedule is None:
        key, noise_key = jax.random.split(key)
        # split[0] is otherwise unused: it seeds the rule-noise stream, leaving
        # the membrane-noise stream (split[1]) bit-identical to previous builds.
        rule_base_key = key
        bulk_noise = jax.random.normal(
            noise_key, shape=(int(n_steps), n_neurons), dtype=jdtype
        )
        rule_bases = None
    else:
        # 0.5.3 item 3 (P-010): chain-consistent draws — the same per-step
        # membrane keys as the continuation path, so chunked == continuous.
        from ._pipeline import _advance_prng_key  # lazy: _pipeline owns the chain

        bulk_noise = jnp.asarray(noise_schedule, dtype=jdtype)
        if bulk_noise.shape != (int(n_steps), int(n_neurons)):
            raise ValueError(
                "noise_schedule must have shape "
                f"({int(n_steps)}, {int(n_neurons)}), got {bulk_noise.shape}"
            )
        _, _step_keys = _advance_prng_key(key, int(n_steps))
        rule_base_key = key
        rule_bases = jax.vmap(lambda step_key: jax.random.split(step_key)[0])(_step_keys)
    sched = (
        jnp.zeros((int(n_steps), n_neurons), dtype=jdtype)
        if drive_schedule is None
        else drive_schedule.astype(jdtype)
    )

    h_min_arr = jnp.asarray(h_min, dtype=jdtype)
    h_max_arr = jnp.asarray(h_max, dtype=jdtype)
    w_floor_arr = jnp.asarray(w_floor, dtype=jdtype)
    w_ceiling_arr = jnp.asarray(w_ceiling, dtype=jdtype)

    def _apply_rule(H, aux, bias, v, u, spikes, prev_spikes, syn_state, w, pre_sp, rkey):
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
            key=rkey,
            pre_sp=pre_sp,
        )
        upd = rule_step(ctx)
        undeclared = [k for k in (upd.d_theta or {}) if k not in descriptor.theta_targets]
        if undeclared:
            raise ValueError(
                f"hdp_rule {hdp_rule!r} returned undeclared theta_targets "
                f"{undeclared!r}; declared: {list(descriptor.theta_targets)}"
            )
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
            if plastic_edge is not None:
                w_next = jnp.where(plastic_edge, w_next, w)
        else:
            w_next = w
        if use_b:
            db = (upd.d_theta or {}).get("drive_bias")
            if db is None:
                b_next = bias
            else:
                b_next = jnp.clip(
                    bias + dt * db,
                    jnp.asarray(b_min, dtype=jdtype),
                    jnp.asarray(b_max, dtype=jdtype),
                )
        else:
            b_next = bias
        return H_next, w_next, aux_next, b_next

    aux_shape = expected_aux_shape(
        descriptor, n_neurons=n_neurons, n_edges=int(edges.n_edges)
    )

    def _checked_h0(from_init: bool) -> jax.Array:
        default = jnp.ones(h_shape, dtype=jdtype)
        if not from_init:
            return default
        got = jnp.asarray(init_state["H_final"], dtype=jdtype)  # type: ignore[index]
        if tuple(got.shape) != tuple(default.shape):
            raise ValueError(
                f"H_final must have shape {tuple(default.shape)} for "
                f"hdp_rule {hdp_rule!r}, got {tuple(got.shape)}"
            )
        return got

    def _checked_aux0(from_init: bool) -> jax.Array:
        default = jnp.zeros(aux_shape, dtype=jdtype)
        if not from_init:
            return default
        got = jnp.asarray(init_state["aux_final"], dtype=jdtype)  # type: ignore[index]
        if tuple(got.shape) != tuple(default.shape):
            raise ValueError(
                f"aux_final must have shape {tuple(default.shape)} for "
                f"hdp_rule {hdp_rule!r}, got {tuple(got.shape)}"
            )
        return got

    def _checked_b0(from_init: bool) -> jax.Array:
        default = jnp.zeros((n_neurons,), dtype=jdtype)
        if not from_init:
            return default
        got = jnp.asarray(init_state["b_final"], dtype=jdtype)  # type: ignore[index]
        if tuple(got.shape) != tuple(default.shape):
            raise ValueError(
                f"b_final must have shape {tuple(default.shape)} for "
                f"hdp_rule {hdp_rule!r}, got {tuple(got.shape)}"
            )
        return got

    if not has_delay:
        if init_state is not None:
            H0 = _checked_h0("H_final" in init_state)
            w0 = jnp.asarray(init_state.get("w_final", w_baseline), dtype=jdtype)
            aux0 = _checked_aux0("aux_final" in init_state)
            b0 = _checked_b0("b_final" in init_state)
            init = (
                jnp.asarray(init_state["v"], dtype=jdtype),
                jnp.asarray(init_state["u"], dtype=jdtype),
                jnp.asarray(init_state["prev_spikes"], dtype=jdtype),
                jnp.asarray(init_state["syn_state"], dtype=jdtype),
                H0,
                w0,
                aux0,
                b0,
            )
        else:
            init = (
                params.v0.astype(jdtype),
                params.u0.astype(jdtype),
                jnp.zeros((n_neurons,), dtype=jdtype),
                jnp.zeros((edges.n_edges,), dtype=jdtype),
                _checked_h0(False),
                w_baseline,
                jnp.zeros(aux_shape, dtype=jdtype),
                jnp.zeros((n_neurons,), dtype=jdtype),
            )

        # The drive-bias coordinate rides the carry uniformly (zeros when the
        # rule declares no "drive_bias" target); the current expression is
        # branched statically so undeclared paths stay bit-exact.
        if step_indices is not None:
            step_ids = jnp.asarray(step_indices, dtype=jnp.int32).reshape(-1)
            if int(step_ids.shape[0]) != int(n_steps):
                raise ValueError(
                    "step_indices must have shape (n_steps,) when provided; got "
                    f"{step_ids.shape} for n_steps={n_steps}"
                )
        else:
            off = _rbd_continuation_step_offset_array(init_state)
            if isinstance(off, int):
                step_ids = jnp.arange(off, off + int(n_steps), dtype=jnp.int32)
            else:
                step_ids = off + jnp.arange(int(n_steps), dtype=jnp.int32)

        def step(carry, xs):
            (sched_t, noise_t, _, rkey_t) = xs
            v, u, prev_spikes, syn_state, H, w, aux, bias = carry
            edge_current = w * syn_state
            syn = _segment_sum(edge_current, post, n_neurons)
            if use_b:
                current_native = (drive + sched_t + bias + syn
                                  + noise_coef * noise_t)
            else:
                current_native = drive + sched_t + syn + noise_coef * noise_t
            dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
            v_next = v + dt * dv
            u_next = u + dt * du
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            H_next, w_next, aux_next, b_next = _apply_rule(
                H, aux, bias, v, u, spikes, prev_spikes, syn_state, w, None, rkey_t
            )
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            syn_next = syn_state * decay + spikes[pre]
            source_proxy = _source_proxy_from_components(
                current_native, spikes, source_scale, dtype=jdtype
            )
            carry_out = (v_reset, u_reset, spikes, syn_next, H_next, w_next, aux_next, b_next)
            if record_weight_trace:
                outputs = (v_reset, spikes, source_proxy, H_next, w_next, aux_next, b_next)
            else:
                outputs = (v_reset, spikes, source_proxy, H_next, aux_next, b_next)
            return carry_out, outputs

        if rule_bases is None:
            rule_step_keys = jax.vmap(lambda t: jax.random.fold_in(rule_base_key, t))(step_ids)
        else:
            rule_step_keys = jax.vmap(lambda rb, t: jax.random.fold_in(rb, t))(rule_bases, step_ids)

        final, scan_outputs = jax.lax.scan(
            step, init, xs=(sched, bulk_noise, step_ids, rule_step_keys)
        )
        if record_weight_trace:
            voltages, spikes, sources, H_trace, w_trace, aux_trace, b_trace = scan_outputs
        else:
            voltages, spikes, sources, H_trace, aux_trace, b_trace = scan_outputs
            w_trace = None
        H_trace, w_trace = _decimate_hw_traces(
            H_trace, w_trace, record_stride_n, record_h_idx, record_w_idx
        )

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
        if use_b:
            diagnostics["b_final"] = final[7]
            diagnostics["b_trace"] = b_trace
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
        H0 = _checked_h0("H_final" in init_state)
        w0 = jnp.asarray(init_state.get("w_final", w_baseline), dtype=jdtype)
        aux0 = _checked_aux0("aux_final" in init_state)
        b0 = _checked_b0("b_final" in init_state)
        init = (
            jnp.asarray(init_state["v"], dtype=jdtype),
            jnp.asarray(init_state["u"], dtype=jdtype),
            jnp.asarray(init_state["prev_spikes"], dtype=jdtype),
            jnp.asarray(init_state["syn_state"], dtype=jdtype),
            H0,
            w0,
            aux0,
            b0,
            spike_hist0,
        )
    else:
        init = (
            params.v0.astype(jdtype),
            params.u0.astype(jdtype),
            jnp.zeros((n_neurons,), dtype=jdtype),
            jnp.zeros((edges.n_edges,), dtype=jdtype),
            _checked_h0(False),
            w_baseline,
            jnp.zeros(aux_shape, dtype=jdtype),
            jnp.zeros((n_neurons,), dtype=jdtype),
            spike_hist0,
        )

    def step_delayed(carry, xs_t):
        t_idx, sched_t, noise_t, rkey_t = xs_t
        v, u, prev_spikes, syn_state, H, w, aux, bias, spike_hist = carry
        edge_current = w * syn_state
        syn = _segment_sum(edge_current, post, n_neurons)
        if use_b:
            current_native = (drive + sched_t + bias + syn
                              + noise_coef * noise_t)
        else:
            current_native = drive + sched_t + syn + noise_coef * noise_t
        dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
        v_next = v + dt * dv
        u_next = u + dt * du
        v_next = jnp.where(s_mask > 0.5, v_next, c)
        spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
        spikes = spikes_bool.astype(jdtype)
        presyn = _delayed_presynaptic_spikes(spikes, spike_hist, t_idx, pre, delay_steps_arr)
        H_next, w_next, aux_next, b_next = _apply_rule(
            H, aux, bias, v, u, spikes, prev_spikes, syn_state, w, presyn, rkey_t
        )
        v_reset = jnp.where(spikes_bool, c, v_next)
        u_reset = jnp.where(spikes_bool, u_next + d, u_next)
        syn_next = syn_state * decay + presyn
        slot = jnp.mod(t_idx, bufsize)
        spike_hist_next = spike_hist.at[slot].set(spikes)
        source_proxy = _source_proxy_from_components(
            current_native, spikes, source_scale, dtype=jdtype
        )
        carry_out = (v_reset, u_reset, spikes, syn_next, H_next, w_next, aux_next, b_next, spike_hist_next)
        if record_weight_trace:
            outputs = (v_reset, spikes, source_proxy, H_next, w_next, aux_next, b_next)
        else:
            outputs = (v_reset, spikes, source_proxy, H_next, aux_next, b_next)
        return carry_out, outputs

    if rule_bases is None:
        rule_step_keys = jax.vmap(lambda t: jax.random.fold_in(rule_base_key, t))(step_indices_arr)
    else:
        rule_step_keys = jax.vmap(lambda rb, t: jax.random.fold_in(rb, t))(
            rule_bases, step_indices_arr
        )

    final, scan_outputs = jax.lax.scan(
        step_delayed, init, xs=(step_indices_arr, sched, bulk_noise, rule_step_keys)
    )
    if record_weight_trace:
        voltages, spikes, sources, H_trace, w_trace, aux_trace, b_trace = scan_outputs
    else:
        voltages, spikes, sources, H_trace, aux_trace, b_trace = scan_outputs
        w_trace = None
    H_trace, w_trace = _decimate_hw_traces(
        H_trace, w_trace, record_stride_n, record_h_idx, record_w_idx
    )

    diagnostics = {
        "v": final[0],
        "u": final[1],
        "prev_spikes": final[2],
        "syn_state": final[3],
        "H_final": final[4],
        "w_final": final[5],
        "aux_final": final[6],
        "delay_state": final[8],
        "spike_history": final[8],
        "delay_steps_max": jnp.asarray(max_delay, dtype=jnp.int32),
        "continuation_step_offset": step_indices_arr[-1] + jnp.asarray(1, dtype=jnp.int32),
        "H_trace": H_trace,
        "w_trace": w_trace,
        "aux_trace": aux_trace,
        "hdp_rule": hdp_rule,
        "hdp_rule_registered": True,
    }
    if use_b:
        diagnostics["b_final"] = final[7]
        diagnostics["b_trace"] = b_trace
    return voltages, spikes, sources, diagnostics
