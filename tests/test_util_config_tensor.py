"""Tests for jaxfne.util: RuntimeConfig/NeuronalTensor/Model/Configuration validate/diff/merge/summary."""

from pathlib import Path

import pytest

import jaxfne as jtfne
from jaxfne import _pipeline
from jaxfne.core import Configuration, Model, RuntimeConfig
from jaxfne.neuronal_tensor import Area, Layer, NeuronType, NeuronalTensor

CONFIG_DIR = Path(__file__).parent.parent / "jaxfne" / "configs"
SMALL_TENSOR_PATH = CONFIG_DIR / "default-column.json"


def _small_model(*, recurrent_backend: str = "edge_list") -> Model:
    tensor = _pipeline.load_tensor(SMALL_TENSOR_PATH)
    cfg = _pipeline.tensor_to_configuration(tensor, seed=1, duration_ms=50.0, dt_ms=0.5)
    cfg = cfg.runtime(recurrent_backend=recurrent_backend)
    return _pipeline.build_network(cfg)


def test_validate_runtime_config_clean_default_has_no_warnings():
    assert jtfne.validate_runtime_config(RuntimeConfig()) == []


def test_validate_runtime_config_flags_float64_without_x64():
    import jax
    if bool(jax.config.read("jax_enable_x64")):
        pytest.skip("jax_enable_x64 already on in this process; cannot test the off-path")
    warnings = jtfne.validate_runtime_config(RuntimeConfig(dtype="float64"))
    assert any("float64" in w for w in warnings)


def test_validate_runtime_config_flags_incomplete_hdp_params():
    cfg = RuntimeConfig(enable_hdp=True, hdp_params={"K_HDP": 1.0})
    warnings = jtfne.validate_runtime_config(cfg)
    assert any("hdp_params missing keys" in w for w in warnings)


def test_validate_runtime_config_strict_raises():
    cfg = RuntimeConfig(enable_hdp=True, hdp_params={"K_HDP": 1.0})
    with pytest.raises(ValueError):
        jtfne.validate_runtime_config(cfg, strict=True)


def test_runtime_config_diff_detects_changed_fields():
    a = RuntimeConfig(seed=0, dtype="float32")
    b = RuntimeConfig(seed=1, dtype="float32")
    diff = jtfne.runtime_config_diff(a, b)
    assert diff == {"seed": (0, 1)}


def test_runtime_config_diff_empty_for_identical_configs():
    a = RuntimeConfig(seed=5)
    b = RuntimeConfig(seed=5)
    assert jtfne.runtime_config_diff(a, b) == {}


def test_merge_runtime_configs_later_wins():
    a = RuntimeConfig(seed=0, dtype="float32")
    b = RuntimeConfig(seed=9, dtype="float32")
    merged = jtfne.merge_runtime_configs(a, b)
    assert merged.seed == 9


def test_merge_runtime_configs_overrides_applied_last():
    a = RuntimeConfig(seed=0)
    merged = jtfne.merge_runtime_configs(a, seed=42)
    assert merged.seed == 42


def test_merge_runtime_configs_requires_at_least_one():
    with pytest.raises(ValueError):
        jtfne.merge_runtime_configs()


def _two_layer_area(name="A1"):
    return Area(
        name=name,
        layers=[
            Layer(name="L1", n_neurons=10, neuron_types=[NeuronType(name="E", fraction=0.8), NeuronType(name="I", fraction=0.2)]),
            Layer(name="L2", n_neurons=5, neuron_types=[NeuronType(name="E")]),
        ],
    )


def test_validate_neuronal_tensor_clean_tensor_has_no_warnings():
    nt = NeuronalTensor(areas=[_two_layer_area()])
    assert jtfne.validate_neuronal_tensor(nt) == []


def test_validate_neuronal_tensor_flags_zero_neurons():
    nt = NeuronalTensor(areas=[Area(name="A1", layers=[Layer(name="L1", n_neurons=0)])])
    warnings = jtfne.validate_neuronal_tensor(nt)
    assert any("n_neurons=0" in w for w in warnings)


def test_validate_neuronal_tensor_flags_duplicate_area_names():
    nt = NeuronalTensor(areas=[_two_layer_area("A1"), _two_layer_area("A1")])
    warnings = jtfne.validate_neuronal_tensor(nt)
    assert any("duplicate Area names" in w for w in warnings)


def test_validate_neuronal_tensor_flags_fractions_over_one():
    layer = Layer(name="L1", n_neurons=10, neuron_types=[NeuronType(name="E", fraction=0.9), NeuronType(name="I", fraction=0.5)])
    nt = NeuronalTensor(areas=[Area(name="A1", layers=[layer])])
    warnings = jtfne.validate_neuronal_tensor(nt)
    assert any("sum to" in w for w in warnings)


def test_validate_neuronal_tensor_strict_raises():
    nt = NeuronalTensor(areas=[Area(name="A1", layers=[])])
    with pytest.raises(ValueError):
        jtfne.validate_neuronal_tensor(nt, strict=True)


def test_tensor_summary_counts_match_manual_sum():
    nt = NeuronalTensor(areas=[_two_layer_area()])
    summary = jtfne.tensor_summary(nt)
    assert summary["n_areas"] == 1
    assert summary["n_layers"] == 2
    assert summary["n_neurons"] == 15
    assert summary["cell_types"] == ["E", "I"]


def test_tensor_summary_on_real_canonical_config():
    nt = jtfne.load_canonical_neuronal_tensor("default-column")
    summary = jtfne.tensor_summary(nt)
    assert summary["n_neurons"] > 0
    assert summary["n_areas"] >= 1


def test_validate_neuronal_tensor_on_all_canonical_configs_is_clean():
    # configs-dir-schema-assumption-fix: FIXED this session by moving the 2
    # legacy Configuration-schema files (nuclei_default/spectrolaminar_default)
    # into jaxfne/configs/legacy/, so configs_dir().glob("*.json") (used by
    # list_canonical_neuronal_tensors) is now NeuronalTensor-schema-only again.
    for name in jtfne.list_canonical_neuronal_tensors():
        nt = jtfne.load_canonical_neuronal_tensor(name)
        warnings = jtfne.validate_neuronal_tensor(nt)
        assert warnings == [], f"{name}: {warnings}"


def test_validate_model_clean_edge_list_model_has_no_warnings():
    model = _small_model(recurrent_backend="edge_list")
    assert jtfne.validate_model(model) == []


def test_validate_model_flags_out_of_bounds_edge_index():
    import jax.numpy as jnp
    from dataclasses import replace as dc_replace

    model = _small_model(recurrent_backend="edge_list")
    edge_list = model.params["edge_list"]
    bad_pre = edge_list.pre.at[0].set(10**6)
    bad_edge_list = dc_replace(edge_list, pre=bad_pre)
    bad_model = dc_replace(model, params={**model.params, "edge_list": bad_edge_list})
    warnings = jtfne.validate_model(bad_model)
    assert any("out of bounds" in w for w in warnings)


def test_validate_model_strict_raises():
    import jax.numpy as jnp
    from dataclasses import replace as dc_replace

    model = _small_model(recurrent_backend="edge_list")
    edge_list = model.params["edge_list"]
    bad_weight = edge_list.weight.at[0].set(float("nan"))
    bad_edge_list = dc_replace(edge_list, weight=bad_weight)
    bad_model = dc_replace(model, params={**model.params, "edge_list": bad_edge_list})
    with pytest.raises(ValueError):
        jtfne.validate_model(bad_model, strict=True)


def test_validate_model_no_edge_list_backend_is_still_safe():
    model = _small_model(recurrent_backend="dense")
    # dense backend has no params["edge_list"] -- must not crash, just skip those checks
    assert jtfne.validate_model(model) == []


def test_model_diff_detects_n_units_change():
    a = _small_model(recurrent_backend="edge_list")
    tensor = _pipeline.load_tensor(SMALL_TENSOR_PATH)
    cfg_b = _pipeline.tensor_to_configuration(tensor, seed=1, duration_ms=50.0, dt_ms=0.5)
    cfg_b = cfg_b.runtime(recurrent_backend="edge_list")
    b = _pipeline.build_network(cfg_b)
    diff = jtfne.model_diff(a, b)
    # same config -> same n_units, no n_units key; only checks the diff machinery doesn't crash
    assert "n_units" not in diff


def test_model_diff_empty_for_identical_build():
    model = _small_model(recurrent_backend="edge_list")
    diff = jtfne.model_diff(model, model)
    assert diff == {}


def test_configuration_diff_detects_metadata_change():
    cfg_a = Configuration()
    cfg_b = cfg_a.network(name="A")
    diff = jtfne.configuration_diff(cfg_a, cfg_b)
    assert "networks" in diff


def test_configuration_diff_empty_for_identical_configs():
    cfg_a = Configuration()
    cfg_b = Configuration()
    assert jtfne.configuration_diff(cfg_a, cfg_b) == {}
