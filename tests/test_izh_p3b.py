"""P3b arm: net.m-order kernel variant and injected-thal reference (PREREG_P3b.md v4).

Each test loads its module by path inside the function, so on the baseline
(no kernel_net_m.py, no thal/return_v on simulate) every test fails.
"""

from __future__ import annotations

import importlib.util
import inspect
import pathlib
import hashlib
import types

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ETUDE = REPO_ROOT / "artifacts" / "etudes" / "izhikevich2003"
NETREF_PATH = ETUDE / "net_reference.py"
KERNEL_PATH = ETUDE / "kernel_net_m.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_netref() -> types.ModuleType:
    return _load(NETREF_PATH, "p3b_net_reference")


def _load_kernel() -> types.ModuleType:
    return _load(KERNEL_PATH, "p3b_kernel_net_m")


class _ExplodingRng:
    def __getattr__(self, name):
        raise AssertionError(f"rng must not be read when thal is given ({name})")


def _small_net(nref, seed=20261013, n_e=8, n_i=2):
    return nref.build_network(np.random.default_rng(seed), n_e=n_e, n_i=n_i)


def _kernel_run(km, net, thal, n, dtype):
    import jax.numpy as jnp

    params = {k: jnp.asarray(net[k], dtype=dtype) for k in ("a", "b", "c", "d")}
    v0 = jnp.full((n,), -65.0, dtype=dtype)
    return km.simulate_net_m(params, jnp.asarray(net["S"], dtype=dtype),
                             jnp.asarray(thal, dtype=dtype), v0,
                             params["b"] * -65.0, return_v=True)


def test_p3b_variant_matches_reference_10cell_200ms():
    import jax

    nref, km = _load_netref(), _load_kernel()
    was = bool(jax.config.read("jax_enable_x64"))
    try:
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        net = _small_net(nref)
        n, steps = 10, 200
        thal = np.full((steps, n), 6.0, dtype=np.float64)
        ref = nref.simulate(net, steps, None, thal=thal, return_v=True)
        ker = _kernel_run(km, net, thal, n, jnp.float64)
        assert int(np.asarray(ref["spikes"]).sum()) > 0, "fixture must spike"
        assert np.array_equal(np.asarray(ker["spikes"]), np.asarray(ref["spikes"]))
        assert np.asarray(ker["spikes"]).dtype == bool
        assert np.asarray(ker["spikes"]).shape == (steps, n)
    finally:
        if not was:
            jax.config.update("jax_enable_x64", False)


def test_p3b_thal_injection_leaves_rng_unread():
    nref = _load_netref()
    net = _small_net(nref)
    n, steps = 10, 60
    thal = np.full((steps, n), 6.0, dtype=np.float64)
    got = nref.simulate(net, steps, _ExplodingRng(), thal=thal)
    null = nref.simulate(net, steps, None, thal=thal)
    assert np.array_equal(got["spikes"], null["spikes"])
    assert int(got["spikes"].sum()) > 0, "fixture must spike"


def test_p3b_default_path_unchanged_vs_baseline():
    # Pins computed from net_reference.py at 9ff85e7f (before thal/return_v), same call.
    nref = _load_netref()
    assert "thal" in inspect.signature(nref.simulate).parameters
    assert "return_v" in inspect.signature(nref.simulate).parameters
    net = _small_net(nref, seed=7, n_e=10, n_i=3)
    new = nref.simulate(net, 60, np.random.default_rng(3), noise_on=True)
    assert set(new) == {"spikes", "extra_u", "mean_extra_u_per_spike"}
    assert int(new["spikes"].sum()) == 6, "fixture must spike"
    digest = hashlib.sha256(np.packbits(new["spikes"]).tobytes()).hexdigest()
    assert digest == "dc076c73b5a2c79509250a1ac3c55ae2ff67cfa3b70716e6d7a335b6971c466f"
    np.testing.assert_allclose(float(new["extra_u"].sum()), 6.471066277787681, rtol=1e-12)
    np.testing.assert_allclose(new["mean_extra_u_per_spike"], 2.1570220925958936, rtol=1e-12)


def test_p3b_return_v_shape_and_first_row():
    nref = _load_netref()
    net = _small_net(nref)
    n, steps = 10, 60
    thal = np.full((steps, n), 6.0, dtype=np.float64)
    got = nref.simulate(net, steps, None, thal=thal, return_v=True)
    assert got["v_test"].shape == (steps, n)
    assert np.array_equal(got["v_test"][0], np.full(n, -65.0))  # v before any update
    assert "v_test" not in nref.simulate(net, steps, None, thal=thal)
