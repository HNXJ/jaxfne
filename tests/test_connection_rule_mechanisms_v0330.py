"""v0.3.30 connectivity compiler: mechanism resolution + metadata."""

import numpy as np
import pytest

import jaxfne as jtfne

NEURONS = [
    {"neuron_id": 0, "area": "V1", "layer": "L4", "cell_type": "E"},
    {"neuron_id": 1, "area": "V1", "layer": "L4", "cell_type": "PV"},
]
MECHS = [
    {"name": "ampa", "kind": "synapse", "params": {"tau_ms": 2.0, "sign": 1.0}},
    {"name": "gabaa", "kind": "synapse", "params": {"tau_ms": 10.0, "sign": -1.0}},
]


def _compile(conns, mechs=MECHS, **kw):
    return jtfne.compile_connection_rules(NEURONS, conns, mechs, **kw)


def test_mechanism_index_maps_to_edges():
    r = _compile([
        {"name": "exc", "source": {"cell_type": "E"}, "target": {"cell_type": "PV"}, "weight": 0.1, "mechanism": "ampa"},
        {"name": "inh", "source": {"cell_type": "PV"}, "target": {"cell_type": "E"}, "weight": 0.1, "mechanism": "gabaa"},
    ])
    # edge_mechanism is the receptor index into mechanism_table
    mech = np.asarray(r.edge_mechanism)
    names = [r.mechanism_table[int(i)]["name"] for i in mech]
    assert "ampa" in names and "gabaa" in names


def test_mechanism_metadata_preserved():
    r = _compile([{"name": "exc", "source": {"cell_type": "E"}, "target": {"cell_type": "PV"},
                   "weight": 0.1, "mechanism": "ampa"}])
    ampa = next(m for m in r.mechanism_table if m["name"] == "ampa")
    assert ampa["tau_ms"] == 2.0
    assert ampa["sign"] == 1.0
    assert ampa["kind"] == "synapse"


def test_unknown_mechanism_hard_error():
    with pytest.raises(ValueError, match="unknown mechanism"):
        _compile([{"name": "x", "source": {"cell_type": "E"}, "target": {"cell_type": "PV"},
                   "weight": 0.1, "mechanism": "nmda"}])


def test_missing_mechanism_hard_error():
    with pytest.raises(ValueError, match="no mechanism"):
        _compile([{"name": "x", "source": {"cell_type": "E"}, "target": {"cell_type": "PV"}, "weight": 0.1}])


def test_negative_tau_rejected():
    with pytest.raises(ValueError, match="tau_ms"):
        _compile(
            [{"name": "x", "source": {"cell_type": "E"}, "target": {"cell_type": "PV"}, "weight": 0.1, "mechanism": "bad"}],
            mechs=[{"name": "bad", "kind": "synapse", "params": {"tau_ms": -1.0}}],
        )
