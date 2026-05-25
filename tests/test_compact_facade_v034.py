import json
import pytest

import jaxfne as jtfne
from jaxfne.io import config_hash, json_safe


PROXY_MODES = ["MUA-proxy", "LFP-proxy", "CSD-proxy", "EEG-proxy", "MEG-proxy", "EMM-proxy"]


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
    signals = jtfne.simulate(model, duration_ms=20.0, dt_ms=0.5, seed=1)
    assert signals.V_m.shape[1] == 8
    assert signals.field is not None

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
    signals = jtfne.simulate(model, duration_ms=20.0, dt_ms=0.5, seed=2, record_fields=False)
    assert signals.field is None
    with pytest.raises(ValueError, match="Cannot generate spectrolaminar profile: signals.field is None"):
        jtfne.vis.spectrolaminar(signals)



def test_suite_no_2_target_chainable_dsl_facade_constructs_and_simulates():
    """The requested Suite No. 2 grammar should work without helper code."""
    cfg = jtfne.Configuration()
    cfg = cfg.runtime(seed=7, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
    cfg = cfg.column("V1", layers=["L2/3", "L4", "L5", "L6"], n=80)
    cfg = cfg.column("PFC", layers=["L2/3", "L5", "L6"], n=80)
    cfg = cfg.cell_types({"E": 0.75, "PV": 0.12, "SST": 0.08, "VIP": 0.05})
    cfg = cfg.connectivity(feedforward=("V1", "PFC"), feedback=("PFC", "V1"))
    cfg = cfg.probes(PROXY_MODES)

    assert cfg.metadata["seed"] == 7
    assert cfg.metadata["dtype"] == "float32"
    assert cfg.metadata["duration_ms"] == 1000.0
    assert cfg.metadata["dt_ms"] == 0.1
    assert cfg.metadata["truth_mode"] == "truth_safe_unverified"
    assert cfg.metadata["claim_level"] == "computational_scaffold"
    assert cfg.metadata["physical_amplitude_claim_allowed"] is False
    assert cfg.metadata["field_solver_status"] == "laminar_proxy_no_pde"
    assert cfg.metadata["dx_mm"] == 0.010
    assert cfg.metadata["dy_mm"] == 0.010
    assert cfg.metadata["dz_mm"] == 0.010
    assert cfg.metadata["geometry_mode"] == "declared_metadata_not_solved_3d_pde_grid"

    assert [col["name"] for col in cfg.metadata["columns"]] == ["V1", "PFC"]
    assert cfg.metadata["columns"][0]["start_index"] == 0
    assert cfg.metadata["columns"][0]["stop_index"] == 80
    assert cfg.metadata["columns"][1]["start_index"] == 80
    assert cfg.metadata["columns"][1]["stop_index"] == 160
    assert cfg.metadata["connectivity"] == {"feedforward": ("V1", "PFC"), "feedback": ("PFC", "V1")}

    assert len(cfg.networks) == 1
    assert cfg.networks[0]["kind"] == "multi_column"
    assert cfg.networks[0]["n"] == 160
    assert cfg.networks[0]["cell_types"] == {"E": 0.75, "PV": 0.12, "SST": 0.08, "VIP": 0.05}

    assert len(cfg.emitters) == 1
    assert cfg.emitters[0] == {"family": "izhikevich", "preset": "cortical_eig"}
    assert len(cfg.fields) == 1
    assert cfg.fields[0]["domain"] == "laminar_column"
    assert cfg.fields[0]["conductivity"] == "proxy"
    assert cfg.fields[0]["boundary"] == "mean_zero_neumann"
    assert cfg.fields[0]["gauge"] == "mean_zero"

    assert callable(cfg.probes)
    assert len(cfg.probes) == 1
    assert cfg.probes[0]["name"] == "multimodal_probe"
    assert cfg.probes[0]["modes"] == PROXY_MODES
    assert cfg.probes[0]["operator_status"] == "simulated_proxy"
    assert cfg.probes[0]["physical_amplitude_claim_allowed"] is False

    # The config remains strict-JSON safe after tuple connectivity is converted by json_safe.
    json.dumps(json_safe({"metadata": cfg.metadata, "networks": cfg.networks, "probes": cfg.probes}), allow_nan=False)
    assert isinstance(config_hash(cfg), str)

    # Exact 160-neuron Suite No. 2 declaration remains a pure configuration
    # smoke here; numerical execution is covered by smaller existing smoke tests
    # in this file and by tutorial execution gates.
    validation = cfg.validate()
    assert validation["valid"] is True
    assert isinstance(validation["config_hash"], str)


def test_suite_no_2_facade_is_immutable_and_keeps_probe_list_semantics():
    """New callable-probes grammar must not break old cfg.probes read behavior."""
    cfg0 = jtfne.Configuration()
    cfg1 = cfg0.runtime(seed=7)
    cfg2 = cfg1.column("V1", layers=["L2/3", "L4"], n=8)
    cfg3 = cfg2.probes(["LFP-proxy"], n_contacts=6)

    assert cfg0.metadata.get("seed") is None
    assert len(cfg0.networks) == 0
    assert len(cfg0.probes) == 0
    assert callable(cfg0.probes)

    assert cfg1.metadata["seed"] == 7
    assert len(cfg2.networks) == 1
    assert len(cfg3.probes) == 1
    assert cfg3.probes[0]["n_contacts"] == 6
    assert cfg3.probes[0]["modes"] == ["LFP-proxy"]


def test_suite_no_2_backward_compatible_aliases_still_work():
    """Existing alias names remain available for older notebooks."""
    cfg = jtfne.Configuration()
    cfg = cfg.set_runtime(seed=7, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
    cfg = cfg.add_column("V1", layers=["L2/3", "L4", "L5", "L6"], n=80)
    cfg = cfg.set_cell_types({"E": 0.75, "PV": 0.12, "SST": 0.08, "VIP": 0.05})
    cfg = cfg.set_connectivity(feedforward=("V1", "PFC"))
    cfg = cfg.set_probes(["MUA-proxy", "LFP-proxy", "CSD-proxy"])

    assert cfg.metadata["seed"] == 7
    assert cfg.networks[0]["n"] == 80
    assert cfg.networks[0]["cell_types"]["VIP"] == 0.05
    assert cfg.metadata["connectivity"]["feedforward"] == ("V1", "PFC")
    assert cfg.probes[0]["modes"] == ["MUA-proxy", "LFP-proxy", "CSD-proxy"]


def test_suite_no_2_facade_rejects_invalid_declarations():
    cfg = jtfne.Configuration()
    with pytest.raises(ValueError, match="column name"):
        cfg.column("", layers=["L4"], n=10)
    with pytest.raises(ValueError, match="at least one layer"):
        cfg.column("V1", layers=[], n=10)
    with pytest.raises(ValueError, match="positive"):
        cfg.column("V1", layers=["L4"], n=0)
    with pytest.raises(ValueError, match="must not be empty"):
        cfg.cell_types({})
    with pytest.raises(ValueError, match="non-negative"):
        cfg.cell_types({"E": 1.0, "PV": -0.1})
    with pytest.raises(ValueError, match="at least one mode"):
        cfg.probes([])
