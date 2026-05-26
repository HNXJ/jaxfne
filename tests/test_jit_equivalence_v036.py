import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jaxfne as jtfne
from jaxfne.core import make_eig_network, EIGNetwork
from jaxfne.emitters import simulate_eig_izhikevich

def test_jit_eager_equivalence_high_level():
    """Verify high-level eager vs. JIT equivalence through jtfne.simulate interface."""
    cfg = (
        jtfne.configuration()
        .network(n=8, cell_types={"E": 0.75, "PV": 0.25})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy")
        .probe(name="test_probe", n_contacts=4)
    )
    
    model = jtfne.construct(cfg)
    
    # 1. Eager simulation
    signals_eager = jtfne.simulate(
        model, 
        duration_ms=50.0, 
        dt_ms=0.1, 
        seed=42, 
        runtime=jtfne.RuntimeConfig(jit=False)
    )
    
    # 2. JIT-compiled simulation
    signals_jit = jtfne.simulate(
        model, 
        duration_ms=50.0, 
        dt_ms=0.1, 
        seed=42, 
        runtime=jtfne.RuntimeConfig(jit=True)
    )
    
    # 3. Shape and Dtype checks
    assert signals_eager.V_m.shape == signals_jit.V_m.shape
    assert signals_eager.spikes.shape == signals_jit.spikes.shape
    assert signals_eager.V_m.dtype == jnp.float32
    assert signals_jit.V_m.dtype == jnp.float32
    
    # 4. Finite outputs checks
    assert jnp.all(jnp.isfinite(signals_eager.V_m))
    assert jnp.all(jnp.isfinite(signals_jit.V_m))
    assert jnp.all(jnp.isfinite(signals_eager.spikes))
    assert jnp.all(jnp.isfinite(signals_jit.spikes))
    
    # 5. Output Equivalence
    np.testing.assert_allclose(
        np.array(signals_eager.V_m), 
        np.array(signals_jit.V_m), 
        rtol=1e-5, 
        atol=1e-5,
        err_msg="Voltage trace mismatch between eager and JIT-compiled runs"
    )
    
    np.testing.assert_array_equal(
        np.array(signals_eager.spikes),
        np.array(signals_jit.spikes),
        err_msg="Spike matrix mismatch between eager and JIT-compiled runs"
    )
    
    if signals_eager.sources is not None:
        np.testing.assert_allclose(
            np.array(signals_eager.sources),
            np.array(signals_jit.sources),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Source trace mismatch between eager and JIT-compiled runs"
        )

def test_jit_equivalence_pure_kernel():
    """Verify JIT equivalence on the pure numerical emitter kernel (simulate_eig_izhikevich)."""
    cfg = (
        jtfne.configuration()
        .network(n=4, cell_types={"E": 0.75, "PV": 0.25})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column")
        .probe(name="pure_probe")
    )
    
    model = jtfne.construct(cfg)
    emitter = model.params["emitter"]
    
    n_steps = 100
    dt_ms = 0.1
    key = jax.random.PRNGKey(12345)
    
    # Eager execution of pure kernel
    v_eager, s_eager, src_eager = simulate_eig_izhikevich(
        emitter, n_steps, dt_ms, key, dtype=jnp.float32
    )
    
    # JIT compiled execution
    @jax.jit
    def compiled_kernel(k):
        return simulate_eig_izhikevich(emitter, n_steps, dt_ms, k, dtype=jnp.float32)
        
    v_jit, s_jit, src_jit = compiled_kernel(key)
    
    # Assert equivalence
    np.testing.assert_allclose(v_eager, v_jit, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(s_eager, s_jit)
    np.testing.assert_allclose(src_eager, src_jit, rtol=1e-5, atol=1e-5)
