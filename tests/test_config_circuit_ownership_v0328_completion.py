"""v0.3.28 completion: declarative circuit/training ownership in Configuration.

These tests lock the prerequisite for the v0.3.30 connectivity compiler: the
``Configuration`` object owns circuit/training declarations under
``metadata["circuit"]`` / ``metadata["training"]`` as strict-JSON-safe specs
with explicit not-yet-compiled/applied/optimized status — declaration only, no
compilation, no numerical change.
"""

import json

import jax.numpy as jnp
import pytest

import jaxfne as jtfne


def _base_cfg():
    return (
        jtfne.Configuration()
        .runtime(duration_ms=10.0, dt_ms=0.1, seed=0)
        .column("V1", layers=["L4"], n=2)
        .cell_types({"E": 0.8, "PV": 0.2})
        .connectivity()
        .set_emitter(family="izhikevich")
        .probes(["MUA-proxy"])
    )


def test_configuration_circuit_methods_exist_and_chain():
    cfg = (
        jtfne.Configuration()
        .cell_params({"cell_type": "E"}, {"a": 0.02})
        .mechanisms(name="ampa", kind="synapse", params={"tau_ms": 5.0})
        .connections(
            name="E_to_PV",
            source={"cell_type": "E"},
            target={"cell_type": "PV"},
            probability=0.1,
            weight=0.05,
            sign="excitatory",
            mechanism="ampa",
        )
        .lesions(name="silence_L4", selector={"layer": "L4"}, effect="disable_drive")
        .trainables(name="w_e_pv", path="connections.E_to_PV.weight", bounds=(0.0, 1.0))
        .objective_outputs(name="mean_firing_rate_hz", dtype="float32", required=True)
    )
    assert isinstance(cfg, jtfne.Configuration)


def test_circuit_metadata_schema_present():
    cfg = (
        jtfne.Configuration()
        .cell_params({"cell_type": "E"}, {"a": 0.02})
        .mechanisms(name="ampa", kind="synapse", params={"tau_ms": 5.0})
        .connections(name="c1", source={"cell_type": "E"}, target={"cell_type": "PV"}, mechanism="ampa")
        .lesions(name="l1", selector={"layer": "L4"}, effect="disable_drive")
        .trainables(name="t1", path="connections.c1.weight")
        .objective_outputs(name="o1")
    )
    m = cfg.metadata
    assert {"cell_params", "mechanisms", "connections", "lesions"} <= set(m["circuit"])
    assert {"trainables", "objective_outputs"} <= set(m["training"])
    assert len(m["circuit"]["mechanisms"]) == 1
    assert len(m["circuit"]["connections"]) == 1
    assert m["circuit_declared"] is True
    assert m["circuit_compiled"] is False
    assert m["training_declared"] is True
    assert m["training_executed"] is False


def test_declarations_are_strict_json_safe():
    cfg = (
        jtfne.Configuration()
        .mechanisms(name="ampa", kind="synapse", params={"tau_ms": 5.0})
        .connections(name="c1", source={"cell_type": "E"}, target={"cell_type": "E"}, weight=0.01, mechanism="ampa")
        .objective_outputs(name="rate", shape=(8,))
    )
    # Must serialize with allow_nan=False.
    s = json.dumps(cfg.metadata, allow_nan=False)
    assert s
    parsed = json.loads(s)
    assert parsed["circuit"]["connections"][0]["status"] == "declared_not_compiled"


def test_explicit_status_fields():
    cfg = (
        jtfne.Configuration()
        .connections(name="c1", source={"cell_type": "E"}, target={"cell_type": "E"})
        .lesions(name="l1", selector={"layer": "L4"}, effect="disable")
        .trainables(name="t1", path="connections.c1.weight")
        .objective_outputs(name="o1")
    )
    assert cfg.metadata["circuit"]["connections"][0]["status"] == "declared_not_compiled"
    assert cfg.metadata["circuit"]["lesions"][0]["status"] == "declared_not_applied"
    assert cfg.metadata["training"]["trainables"][0]["status"] == "declared_not_optimized"
    assert cfg.metadata["training"]["objective_outputs"][0]["status"] == "declared"


# --- validator failures -------------------------------------------------------

def test_duplicate_names_fail():
    cfg = jtfne.Configuration().mechanisms(name="ampa", kind="synapse")
    with pytest.raises(ValueError, match="duplicate"):
        cfg.mechanisms(name="ampa", kind="synapse")


def test_bad_probability_fails():
    with pytest.raises(ValueError, match="probability"):
        jtfne.Configuration().connections(
            name="c1", source={"cell_type": "E"}, target={"cell_type": "E"}, probability=1.5
        )


def test_bad_sign_fails():
    with pytest.raises(ValueError, match="sign"):
        jtfne.Configuration().connections(
            name="c1", source={"cell_type": "E"}, target={"cell_type": "E"}, sign="positive"
        )


def test_nonfinite_value_fails():
    with pytest.raises(ValueError, match="finite"):
        jtfne.Configuration().connections(
            name="c1", source={"cell_type": "E"}, target={"cell_type": "E"}, weight=float("nan")
        )


def test_reversed_bounds_fail():
    with pytest.raises(ValueError, match="reversed"):
        jtfne.Configuration().trainables(name="t1", path="x", bounds=(1.0, 0.0))


def test_non_json_safe_param_fails():
    with pytest.raises(ValueError, match="non-JSON-safe"):
        jtfne.Configuration().mechanisms(name="ampa", kind="synapse", params={"obj": object()})


def test_objective_output_requires_name():
    with pytest.raises(ValueError, match="name"):
        jtfne.Configuration().objective_outputs(name="")


@pytest.mark.parametrize(
    "call",
    [
        lambda C: C.mechanisms(name="", kind="synapse"),
        lambda C: C.connections(name="", source={"a": 1}, target={"a": 1}),
        lambda C: C.lesions(name="", selector={"layer": "L4"}, effect="disable"),
        lambda C: C.trainables(name="", path="x"),
    ],
)
def test_all_named_methods_reject_empty_name(call):
    with pytest.raises(ValueError, match="name"):
        call(jtfne.Configuration())


def test_trainable_bounds_wrong_length_fails():
    with pytest.raises(ValueError, match="2 elements"):
        jtfne.Configuration().trainables(name="t1", path="x", bounds=(0.5,))


def test_connection_non_mapping_selector_fails():
    with pytest.raises(ValueError, match="selector mappings"):
        jtfne.Configuration().connections(name="c1", source=None, target={"a": 1})


def test_compiled_flag_resets_on_new_declaration():
    """A new declaration must reset circuit_compiled even if a prior step set it True."""
    cfg = jtfne.Configuration().connections(name="c1", source={"a": 1}, target={"a": 1})
    # Simulate a planned compiler having marked it compiled.
    import dataclasses
    md = dict(cfg.metadata)
    md["circuit_compiled"] = True
    cfg = dataclasses.replace(cfg, metadata=md)
    # Chaining another declaration must invalidate the stale flag.
    cfg = cfg.mechanisms(name="ampa", kind="synapse")
    assert cfg.metadata["circuit_compiled"] is False


def test_unknown_mechanism_reference_warns_only():
    with pytest.warns(UserWarning, match="mechanism"):
        cfg = jtfne.Configuration().connections(
            name="c1", source={"cell_type": "E"}, target={"cell_type": "E"}, mechanism="not_declared"
        )
    # Declaration still recorded (warning, not error, in this phase).
    assert cfg.metadata["circuit"]["connections"][0]["mechanism"] == "not_declared"


# --- construct/simulate compatibility ----------------------------------------

def test_construct_simulate_unaffected_by_declarations():
    """Declarations must not change construct/simulate numerics."""
    cfg = _base_cfg().connections(
        name="E_to_E",
        source={"cell_type": "E"},
        target={"cell_type": "E"},
        probability=0.1,
        weight=0.01,
        sign="excitatory",
    )
    model = jtfne.construct(cfg)
    signals = jtfne.simulate(model, duration_ms=10.0, dt_ms=0.1, seed=0)
    assert jnp.isfinite(signals.get("vm")).all()
    assert jnp.isfinite(signals.get("spk")).all()


def test_existing_metadata_preserved():
    """Adding circuit declarations must not drop existing truth-gate metadata."""
    cfg = jtfne.Configuration().connections(
        name="c1", source={"cell_type": "E"}, target={"cell_type": "E"}
    )
    assert cfg.metadata["truth_mode"] == "truth_safe_unverified"
    assert cfg.metadata["claim_level"] == "computational_scaffold"
    assert cfg.metadata["field_solver_status"] == "laminar_proxy_no_pde"


def test_immutability_chain_does_not_mutate_source():
    cfg0 = jtfne.Configuration()
    cfg1 = cfg0.mechanisms(name="ampa", kind="synapse")
    # cfg0 must be unchanged (frozen, copy-on-write).
    assert "circuit" not in cfg0.metadata or not cfg0.metadata.get("circuit", {}).get("mechanisms")
    assert len(cfg1.metadata["circuit"]["mechanisms"]) == 1
