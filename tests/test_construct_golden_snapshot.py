"""Golden-output regression for construct(), captured BEFORE splitting it
into staged helpers (see core.construct's stages: validate config -> build
network (suite2-style vs plain dense-W) -> resolve edge_list -> geometry
override -> canonical biophysics -> compile .connections() rules -> build
static).

These assertions are deliberately concrete so a refactor that drops or
mis-threads a stage's local state is caught. Written against the CURRENT
(pre-split) construct() implementation; must keep passing unchanged after
the split. The geometry-override branch is already thoroughly covered by
tests/test_laminar_geometry_v013.py (success/backward-compat/mismatch-raise/
manifest/truth-gates) -- not duplicated here.
"""

import pytest

import jaxfne as jtfne


def test_construct_suite2_style_path_via_laminar_cortex_config():
    """columns/layer_cell_types/uniform_3d metadata -> suite2 population path,
    prebuilt sparse edges, forced edge_list backend, geometry_meta populated."""
    cfg = jtfne.laminar_cortex_config(
        seed=0, duration_ms=20.0, dt_ms=0.5, areas=("V1",), layers=("L4",),
        cell_types={"E": 0.5, "PV": 0.5}, n=10, emitter="izhikevich",
    )
    assert cfg.metadata.get("columns") is not None
    model = jtfne.construct(cfg)

    assert model.params["emitter"].n_neurons == 10
    assert model.params["edge_list"].n_edges > 0
    assert "geometry" in model.static
    assert "neuron_metadata" in model.static
    assert len(model.static["neuron_metadata"]) == 10
    assert model.static["neuron_metadata_summary"]["n_rows"] == 10


def test_construct_plain_path_via_bare_network_builder():
    """No columns/layer_cell_types/uniform_3d -> make_eig_network, dense W,
    geometry_meta stays None, no 'geometry' key in static."""
    cfg = (
        jtfne.configuration()
        .network(n=8)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="p", n_contacts=4)
    )
    assert cfg.metadata.get("columns") is None
    model = jtfne.construct(cfg)

    assert model.params["emitter"].n_neurons == 8
    assert model.params["emitter"].W.shape == (8, 8)  # real dense W, not the (0,0) placeholder
    assert model.params["edge_list"].n_edges >= 0
    assert "geometry" not in model.static
    assert "neuron_metadata" not in model.static


def test_construct_n_contacts_from_first_probe():
    cfg = (
        jtfne.configuration()
        .network(n=6)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="p", n_contacts=12)
    )
    model = jtfne.construct(cfg)
    assert model.static["n_contacts"] == 12


def test_construct_n_contacts_below_minimum_raises():
    cfg = (
        jtfne.configuration()
        .network(n=6)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="p", n_contacts=1)
    )
    with pytest.raises(ValueError, match="n_contacts must be"):
        jtfne.construct(cfg)


def test_construct_defaults_n_contacts_to_16_when_probe_omits_it():
    cfg = (
        jtfne.configuration()
        .network(n=6)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="p")  # no n_contacts declared
    )
    model = jtfne.construct(cfg)
    assert model.static["n_contacts"] == 16


def test_construct_invalid_config_raises():
    cfg = jtfne.Configuration()  # no emitters/field/probes declared
    with pytest.raises(ValueError, match="Invalid jaxfne configuration"):
        jtfne.construct(cfg)


def test_construct_unsupported_emitter_family_raises():
    cfg = (
        jtfne.configuration()
        .network(n=6)
        .emitter(family="hodgkin_huxley_not_implemented")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="p", n_contacts=4)
    )
    with pytest.raises(ValueError, match="Unsupported emitter family"):
        jtfne.construct(cfg)


def test_construct_compiles_declared_connections_into_edge_list():
    cfg = (
        jtfne.laminar_cortex_config(
            seed=0, duration_ms=20.0, dt_ms=0.5, areas=("V1",), layers=("L4",),
            cell_types={"E": 0.5, "PV": 0.5}, n=10, emitter="izhikevich",
        )
        .connections(name="e_to_i", source={"cell_type": "E"}, target={"cell_type": "PV"},
                     probability=1.0, weight=0.4, sign="excitatory")
    )
    model = jtfne.construct(cfg)

    compiled = [c for c in model.cfg.metadata["circuit"]["connections"] if c["name"] == "e_to_i"]
    assert len(compiled) == 1
    assert compiled[0]["status"] == "compiled"
    assert compiled[0]["compiled_n_edges"] > 0


def test_construct_returns_model_with_expected_param_keys():
    cfg = jtfne.laminar_cortex_config(
        seed=0, duration_ms=20.0, dt_ms=0.5, areas=("V1",), layers=("L4",),
        cell_types={"E": 0.5, "PV": 0.5}, n=6, emitter="izhikevich",
    )
    model = jtfne.construct(cfg)
    assert set(model.params.keys()) == {"emitter", "positions", "edge_list"}
    assert "operator_status" in model.static
