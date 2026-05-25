import sys
from unittest.mock import patch
import pytest
import jaxfne as jtfne


def _cfg(n=8):
    return (
        jtfne.configuration()
        .network(n=n, cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="p", n_contacts=4)
    )


def test_compact_grammar_execution():
    """Verify that the standard compact grammar works perfectly."""
    cfg = _cfg(8)
    model = jtfne.construct(cfg)
    signals = jtfne.simulate(model)
    assert signals.V_m.shape[1] == 8
    assert signals.field is not None

    # Test visualization
    fig = jtfne.vis.spectrolaminar(signals)
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_simulate_with_kwargs():
    """Verify that simulate accepts custom simulation parameters as kwargs."""
    cfg = _cfg(6)
    model = jtfne.construct(cfg)
    signals = jtfne.simulate(model, duration_ms=15.0, dt_ms=0.5, seed=123)
    assert signals.metadata["duration_ms"] == 15.0
    assert signals.metadata["dt_ms"] == 0.5
    assert signals.metadata["dt_ms"] * signals.V_m.shape[0] == 15.0


def test_simulate_collision_raises_value_error():
    """Verify that specifying both sim object and kwargs raises ValueError."""
    cfg = _cfg(6)
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=10.0)
    with pytest.raises(ValueError, match="Cannot specify both a Simulation object and individual simulation parameters"):
        jtfne.simulate(model, sim=sim, duration_ms=10.0)


def test_spectrolaminar_no_field_raises_value_error():
    """Verify that calling spectrolaminar with signals lacking field arrays raises ValueError."""
    cfg = _cfg(6)
    model = jtfne.construct(cfg)
    signals = jtfne.simulate(model, record_fields=False)
    assert signals.field is None
    with pytest.raises(ValueError, match="Cannot generate spectrolaminar profile: signals.field is None"):
        jtfne.vis.spectrolaminar(signals)


def test_matplotlib_import_guard():
    """Verify that the require_matplotlib helper correctly raises ImportError if matplotlib is missing."""
    with patch.dict(sys.modules, {"matplotlib": None}):
        with pytest.raises(ImportError, match="The visualization features require the optional dependency 'matplotlib'"):
            jtfne.vis.require_matplotlib()


def test_chainable_dsl_facade():
    """Verify that the chainable DSL configuration facade works exactly as requested."""
    cfg = jtfne.Configuration()
    cfg2 = cfg.set_runtime(seed=7, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
    cfg3 = cfg2.add_column("V1", layers=["L2/3", "L4", "L5", "L6"], n=80)
    cfg4 = cfg3.set_probes(["MUA-proxy", "LFP-proxy", "CSD-proxy"])

    # Assertions
    assert cfg.metadata.get("seed") is None  # original is not corrupted
    assert cfg4.metadata["seed"] == 7
    assert cfg4.metadata["dtype"] == "float32"
    assert len(cfg4.networks) == 1
    assert cfg4.networks[0]["n"] == 80
    assert len(cfg4.probes) == 1
    assert cfg4.probes[0]["modes"] == ["MUA-proxy", "LFP-proxy", "CSD-proxy"]

    # Test complete construct/simulate path
    model = jtfne.construct(cfg4)
    signals = jtfne.simulate(model, duration_ms=10.0, dt_ms=0.1, seed=7)
    assert signals.V_m.shape[1] == 80

