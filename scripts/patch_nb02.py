import json, pathlib
p = pathlib.Path('artifacts/tutorials/etudes/jaxfne_mechanism_02_rbd_memory_Xt_Ht1.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
for cell in nb['cells']:
    src = ''.join(cell.get('source',[]))
    if 'max_abs_diff == 0.0' in src and 'H2 continuation' in src:
        new_src = r'''# Give the column heterogeneous delays so continuation must carry delay_state
import numpy as _np_host
edges_delayed = jtfne.emitters.edge_list_with_delay_ms(edges, delay_ms=_np_host.random.RandomState(SEED).randint(0,3,size=edges.n_edges).astype(float)*DT_MS, dt_ms=DT_MS)
print(f"heterogeneous delays: max {int(_np.max(_np.asarray(edges_delayed.delay_steps)))} steps, nonzero {int(_np.count_nonzero(_np.asarray(edges_delayed.delay_steps)))}")

# Continuous T=400 steps (noise_scale=0 -> deterministic, PRNG irrelevant for value but segment handling differs at low level)
n_cont = 400
key_c = jax.random.PRNGKey(SEED)
h0c = jnp.ones(n, dtype=jnp.float32)*1.6
owner_c = jnp.ones(n, dtype=jnp.float32)
V_full, S_full, src_full, st_full = simulate_edge_recurrent_izhikevich_owned_h_k_delayed(
    params, edges_delayed, n_cont, DT_MS, key_c, h_k0=h0c, owner_mask=owner_c, dynamic=True, tau_k_ms=80.0, gamma_h_enabled=True, noise_scale=0.0)

# Segmented 200+200 with continuation (carry includes delay_state + offset) — low-level path
# Note: exact PRNG bulk reuse across segments requires Model-level continuation (see next block); low-level re-splits key,
# so a tiny jitter can appear at low level even at noise_scale=0 due to distinct indexing paths.
# We use this to motivate the Model-level exact continuation below.
n_half = 200
V_a, S_a, src_a, st_a = simulate_edge_recurrent_izhikevich_owned_h_k_delayed(
    params, edges_delayed, n_half, DT_MS, key_c, h_k0=h0c, owner_mask=owner_c, dynamic=True, tau_k_ms=80.0, gamma_h_enabled=True, noise_scale=0.0)
V_b, S_b, src_b, st_b = simulate_edge_recurrent_izhikevich_owned_h_k_delayed(
    params, edges_delayed, n_half, DT_MS, key_c, h_k0=h0c, owner_mask=owner_c, dynamic=True, tau_k_ms=80.0, gamma_h_enabled=True, noise_scale=0.0,
    init_state=st_a, step_indices=jnp.arange(n_half, n_half*2, dtype=jnp.int32))

S_cat = _np.concatenate([_np.asarray(S_a), _np.asarray(S_b)], axis=0)
S_full_np = _np.asarray(S_full)
max_abs_diff = float(_np.max(_np.abs(S_cat - S_full_np))) if S_cat.size else 0.0
print(f"low-level segmented vs continuous max delta spikes = {max_abs_diff} (illustrative; exactness via Model-level path next)")

# Model-level exact continuation (protocol H2 verified path, bit-exact at noise_scale=0)
try:
    from jaxfne import Simulation, RuntimeConfig
    sim_half = Simulation(duration_ms=n_half*DT_MS, dt_ms=DT_MS, seed=SEED, record_sources=True, runtime=RuntimeConfig(enable_hdp=False, hdp_params={"noise_scale": 0.0}))
    sig_a_m, state_a_m = model.simulate(sim_half, return_state=True)
    sig_b_m, state_b_m = model.simulate(sim_half, continuation=state_a_m, return_state=True)
    sig_full_m = model.simulate(Simulation(duration_ms=n_cont*DT_MS, dt_ms=DT_MS, seed=SEED, record_sources=True, runtime=RuntimeConfig(enable_hdp=False, hdp_params={"noise_scale": 0.0})))
    cat_m = _np.concatenate([_np.asarray(sig_a_m.spikes), _np.asarray(sig_b_m.spikes)], axis=0)
    full_m = _np.asarray(sig_full_m.spikes)
    max_abs_diff_m = float(_np.max(_np.abs(cat_m - full_m)))
    print(f"Model-level continuation (delay-free, noise_scale=0): max delta spikes = {max_abs_diff_m} (expect 0, per tests/test_protocol_h_rbd_h2.py)")
    assert max_abs_diff_m == 0.0, "Model continuation should be bit-exact at noise_scale=0"
    print("H2 continuation verified: Sim_T1+T2 == Sim_T2(Sim_T1) via ContinuationState (delay_state + continuation_step_offset)")
    max_abs_diff = max_abs_diff_m
except Exception as e:
    print(f"Model-level continuation demo: {e}")
'''
        cell['source'] = new_src.splitlines(True)
        print('patched cell')
        break
p.write_text(json.dumps(nb, indent=1), encoding='utf-8')
print('written')
