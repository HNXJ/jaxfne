"""P-052: the sigmoid_beta surrogate carries the gradient across neurons on the dense Izhikevich kernel."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.emitters import IzhikevichParams, simulate_eig_izhikevich, spike_sigmoid_beta

N_STEPS, DT = 400, 0.1


def _pair(a0):
    """Neuron 0 fires under strong drive; neuron 1 sits below threshold and hears it through W[1, 0]."""
    f = a0.dtype
    return IzhikevichParams(
        a=jnp.stack([a0, jnp.asarray(0.02, f)]), b=jnp.full(2, 0.2, f), c=jnp.full(2, -65.0, f),
        d=jnp.full(2, 8.0, f), drive=jnp.asarray([15.0, 0.0], f), sign=jnp.ones(2, f),
        W=jnp.asarray([[0.0, 0.0], [20.0, 0.0]], f), v0=jnp.full(2, -65.0, f), u0=jnp.full(2, -13.0, f),
        source_scale=jnp.ones(2, f), labels=("E", "E"),
    )


def _post_mean_v(a0, beta):
    v, _, _ = simulate_eig_izhikevich(_pair(a0), N_STEPS, DT, jax.random.PRNGKey(0), dtype="float64",
                                      noise_scale=0.0, surrogate_beta=beta)
    return jnp.mean(v[:, 1])


@pytest.mark.x64
def test_forward_is_bit_identical_with_the_surrogate():
    params = _pair(jnp.asarray(0.02, jnp.float64))
    key = jax.random.PRNGKey(3)
    plain = simulate_eig_izhikevich(params, N_STEPS, DT, key, dtype="float64")
    surr = simulate_eig_izhikevich(params, N_STEPS, DT, key, dtype="float64", surrogate_beta=0.5)
    assert float(plain[1][:, 0].sum()) > 0, "fixture must spike"
    for x, y in zip(plain, surr):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


@pytest.mark.x64
def test_surrogate_derivative_is_the_scaled_sigmoid_slope():
    beta = 0.7
    x = jnp.linspace(-6.0, 6.0, 25, dtype=jnp.float64)
    got = jax.vmap(jax.grad(lambda z: spike_sigmoid_beta(z, beta)))(x)
    s = jax.nn.sigmoid(beta * x)
    np.testing.assert_allclose(np.asarray(got), np.asarray(beta * s * (1.0 - s)), rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(spike_sigmoid_beta(x, beta)), np.asarray(x >= 0.0, np.float64))


@pytest.mark.x64
def test_gradient_crosses_neurons_only_with_the_surrogate():
    a0 = jnp.asarray(0.02, jnp.float64)
    hard = float(jax.grad(_post_mean_v)(a0, None))
    soft = float(jax.grad(_post_mean_v)(a0, 0.5))
    assert hard == 0.0, "without a surrogate no gradient reaches the postsynaptic neuron (P-052)"
    assert np.isfinite(soft) and soft != 0.0


@pytest.mark.x64
def test_public_route_carries_the_surrogate():
    cfg = (jtfne.configuration().network(n=12, cell_types={"E": 0.75, "PV": 0.25})
           .emitter(family="izhikevich", preset="cortical_eig").field(domain="laminar_column", conductivity="proxy")
           .probe(name="p", n_contacts=4))
    model = jtfne.construct(cfg)
    handle = "cells.izhikevich.a[all]"
    theta = jnp.asarray(np.asarray(jtfne.get_parameter(model, handle)), jnp.float64)

    def mean_v(th, surrogate):
        rt = jtfne.RuntimeConfig(dtype="float64", jit=True, surrogate=surrogate)
        sig = jtfne.simulate(jtfne.set_parameter(model, handle, th), duration_ms=100.0, dt_ms=0.1, seed=0,
                             runtime=rt, record_fields=False)
        return jnp.mean(sig.V_m, axis=0)

    sg = jtfne.SurrogateConfig(method="sigmoid_beta", beta=0.5)
    base = jtfne.simulate(model, duration_ms=100.0, dt_ms=0.1, seed=0,
                          runtime=jtfne.RuntimeConfig(dtype="float64", jit=True), record_fields=False)
    with_s = jtfne.simulate(model, duration_ms=100.0, dt_ms=0.1, seed=0,
                            runtime=jtfne.RuntimeConfig(dtype="float64", jit=True, surrogate=sg), record_fields=False)
    assert float(np.asarray(base.spikes).sum()) > 0, "fixture must spike"
    np.testing.assert_array_equal(np.asarray(base.V_m), np.asarray(with_s.V_m))
    assert with_s.metadata["runtime"]["surrogate"]["method"] == "sigmoid_beta"
    off = lambda J: np.abs(J - np.diag(np.diag(J))).max()  # noqa: E731
    assert off(np.asarray(jax.jacrev(mean_v)(theta, None))) == 0.0
    assert off(np.asarray(jax.jacrev(mean_v)(theta, sg))) > 0.0


def test_unrealized_surrogates_and_routes_refuse():
    with pytest.raises(ValueError, match="straight_through"):
        jtfne.RuntimeConfig(surrogate=jtfne.SurrogateConfig(method="straight_through"))
    with pytest.raises(ValueError, match="'none'"):
        jtfne.RuntimeConfig(surrogate=jtfne.SurrogateConfig())
    for beta in (0.0, -5.0):
        with pytest.raises(ValueError, match="beta must be > 0"):
            jtfne.RuntimeConfig(surrogate=jtfne.SurrogateConfig(method="sigmoid_beta", beta=beta))
    with pytest.raises(TypeError, match="SurrogateConfig"):
        jtfne.RuntimeConfig(surrogate="sigmoid_beta")
    params = _pair(jnp.asarray(0.02, jnp.float32))
    with pytest.raises(ValueError, match="surrogate"):
        simulate_eig_izhikevich(params, 10, 1.0, jax.random.PRNGKey(0), update_order="net_m",
                                noise_scale=0.0, surrogate_beta=0.5)
    cfg = (jtfne.configuration().network(n=8, cell_types={"E": 0.75, "PV": 0.25})
           .emitter(family="izhikevich", preset="cortical_eig").field(domain="laminar_column", conductivity="proxy")
           .probe(name="p", n_contacts=4))
    sg = jtfne.SurrogateConfig(method="sigmoid_beta", beta=0.5)
    with pytest.raises(ValueError, match="surrogate"):
        jtfne.simulate(jtfne.construct(cfg), duration_ms=5.0, dt_ms=0.1, seed=0,
                       runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list", surrogate=sg))


SG = jtfne.SurrogateConfig(method="sigmoid_beta", beta=0.5)


def _public_model():
    cfg = (jtfne.configuration().network(n=12, cell_types={"E": 0.75, "PV": 0.25})
           .emitter(family="izhikevich", preset="cortical_eig").field(domain="laminar_column", conductivity="proxy")
           .probe(name="p", n_contacts=4))
    return jtfne.construct(cfg)


@pytest.mark.x64
def test_batch_forward_is_bit_identical_with_the_surrogate():
    """Fail-first on the base tree (simulate_batch refuses the surrogate).

    The batch keys seed i by a split of PRNGKey(seed) and simulate keys it by
    PRNGKey(seed), so the noise is declared zero: the run is then key-free and
    each batch member must equal the looped simulate bit for bit.
    """
    model = _public_model()
    hdp = {"noise_scale": 0.0}
    rt = jtfne.RuntimeConfig(dtype="float64", jit=True, vmap=True, surrogate=SG, hdp_params=hdp)
    batch = model.simulate_batch(jtfne.Simulation(duration_ms=100.0, dt_ms=0.1, seed=0, runtime=rt), n_seeds=2)
    assert float(np.asarray(batch["spikes"]).sum()) > 0, "fixture must spike"
    for i, seed in enumerate((0, 1)):
        looped = jtfne.simulate(model, duration_ms=100.0, dt_ms=0.1, seed=seed, record_fields=False,
                                runtime=jtfne.RuntimeConfig(dtype="float64", jit=True, surrogate=SG,
                                                            hdp_params=hdp))
        np.testing.assert_array_equal(np.asarray(batch["V_m"])[i], np.asarray(looped.V_m))
        np.testing.assert_array_equal(np.asarray(batch["spikes"])[i], np.asarray(looped.spikes))


@pytest.mark.x64
def test_batch_gradient_crosses_neurons_only_with_the_surrogate():
    """Fail-first on the base tree: the batch route carries the gradient across neurons, as the single route does."""
    model = _public_model()
    handle = "cells.izhikevich.a[all]"
    theta = jnp.asarray(np.asarray(jtfne.get_parameter(model, handle)), jnp.float64)

    def mean_v(th, surrogate):
        rt = jtfne.RuntimeConfig(dtype="float64", jit=True, vmap=True, surrogate=surrogate)
        sim = jtfne.Simulation(duration_ms=100.0, dt_ms=0.1, seed=0, runtime=rt)
        out = jtfne.set_parameter(model, handle, th).simulate_batch(sim, n_seeds=2)
        return jnp.mean(out["V_m"], axis=(0, 1))

    off = lambda J: np.abs(J - np.diag(np.diag(J))).max()  # noqa: E731
    assert off(np.asarray(jax.jacrev(mean_v)(theta, None))) == 0.0
    assert off(np.asarray(jax.jacrev(mean_v)(theta, SG))) > 0.0


@pytest.mark.parametrize("runtime_kw", [
    {"recurrent_backend": "edge_list"},
    {"enable_homeostasis": True},
    {"enable_hdp": True},
])
def test_batch_still_refuses_the_surrogate_off_the_dense_route(runtime_kw):
    """PIN: passes on the base implementation; the batch refuses the surrogate on every route but dense."""
    model = _public_model()
    rt = jtfne.RuntimeConfig(surrogate=SG, **runtime_kw)
    sim = jtfne.Simulation(duration_ms=10.0, dt_ms=0.1, seed=0, runtime=rt)
    with pytest.raises(ValueError, match="surrogate"):
        model.simulate_batch(sim, n_seeds=2)
