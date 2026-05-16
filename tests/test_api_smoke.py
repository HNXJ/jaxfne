import jaxfne as jtfne


def test_minimal_api_smoke():
    cfg = jtfne.configuration()
    cfg = cfg.network(name="V1", kind="cortical_column", n=12, cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
    cfg = cfg.emitter(family="izhikevich", preset="cortical_eig")
    cfg = cfg.field(domain="laminar_column", conductivity="proxy", boundary="declared_proxy", gauge="mean_zero")
    cfg = cfg.probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0)
    signals = model.simulate(sim)
    readout = model.record(signals, modes=["spikes", "V_m", "CSD", "LFP"])
    assert signals.V_m.shape[1] == 12
    assert "spikes" in readout
    assert "CSD" in readout
    assert model.manifest(signals)["truth_mode"] == "truth_safe_unverified"
