def test_net_alias_points_to_model():
    import jaxfne as jtfne
    assert hasattr(jtfne, "Model")
    assert hasattr(jtfne, "Net")
    assert jtfne.Net is jtfne.Model

def test_optimizer_not_root_alias_until_facade_exists():
    import jaxfne as jtfne
    assert not hasattr(jtfne, "Optimizer")
    assert "Optimizer" not in getattr(jtfne, "__all__", [])
