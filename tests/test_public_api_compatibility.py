def test_net_alias_points_to_model():
    import jaxfne as jtfne
    assert hasattr(jtfne, "Model")
    assert hasattr(jtfne, "Net")
    assert jtfne.Net is jtfne.Model

def test_optimizer_not_root_alias_until_facade_exists():
    import jaxfne as jtfne
    assert not hasattr(jtfne, "Optimizer")
    assert "Optimizer" not in getattr(jtfne, "__all__", [])


def test_signature_defaults_objective_field_helpers():
    """Override-worthy knobs in objective/field helpers are now defaulted."""
    import jax.numpy as jnp
    import jaxfne as jtfne

    # rate_synchrony_targets: canonical balanced/async defaults (10 Hz, kappa 0).
    obj = jtfne.rate_synchrony_targets()
    assert obj.name == "rate_synchrony_targets"

    # construct_source_tensor: default mode is the standard membrane-current proxy.
    src, meta = jtfne.construct_source_tensor(total_membrane_current=jnp.ones((6, 4)))
    assert meta["mode"] == "total_membrane_current_proxy"
    assert src.shape == (6, 4)

    # probe_laminar_modes: default extracts the full proxy set.
    positions = jnp.linspace(0.0, 1.0, 8)[:, None] * jnp.ones((8, 3))
    fo = jtfne.project_laminar_sources(jnp.ones((12, 8)), positions)
    out = jtfne.probe_laminar_modes(fo)
    proxy_keys = {k for k in out if k.endswith("_proxy")}
    assert {"source_proxy", "phi_e_proxy", "csd_proxy", "lfp_proxy"} <= proxy_keys
