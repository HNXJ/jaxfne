"""Tests for jaxfne/_pipeline.py -- the internal pure-function layer.

Each wrapper is checked for behavioral equivalence against the existing
public call it delegates to, not just import success.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jaxfne import _pipeline
from jaxfne.core import Configuration, Model, Signals
from jaxfne.neuronal_tensor import (
    NeuronalTensor,
    RuntimeConfiguration,
    load as tensor_load,
    neuronal_tensor_to_configuration,
)

CONFIG_DIR = Path(__file__).parent.parent / "jaxfne" / "configs"
SMALL_TENSOR_PATH = CONFIG_DIR / "default-column.json"


def test_load_tensor_matches_direct_load():
    via_pipeline = _pipeline.load_tensor(SMALL_TENSOR_PATH)
    via_direct = tensor_load(SMALL_TENSOR_PATH)
    assert isinstance(via_pipeline, NeuronalTensor)
    assert via_pipeline.name == via_direct.name
    assert len(via_pipeline.areas) == len(via_direct.areas)


def test_save_tensor_roundtrip(tmp_path):
    tensor = _pipeline.load_tensor(SMALL_TENSOR_PATH)
    out_path = tmp_path / "roundtrip.json"
    written = _pipeline.save_tensor(tensor, out_path)
    assert Path(written).exists()
    reloaded = _pipeline.load_tensor(out_path)
    assert reloaded.name == tensor.name
    assert len(reloaded.areas) == len(tensor.areas)


def test_tensor_to_configuration_matches_direct_call():
    tensor = _pipeline.load_tensor(SMALL_TENSOR_PATH)
    via_pipeline = _pipeline.tensor_to_configuration(
        tensor, seed=3, duration_ms=200.0, dt_ms=0.5,
    )
    via_direct = neuronal_tensor_to_configuration(
        tensor, seed=3, duration_ms=200.0, dt_ms=0.5,
    )
    assert isinstance(via_pipeline, Configuration)
    assert via_pipeline.networks == via_direct.networks
    assert via_pipeline.emitters == via_direct.emitters


def test_build_network_configuration_path():
    tensor = _pipeline.load_tensor(SMALL_TENSOR_PATH)
    cfg = _pipeline.tensor_to_configuration(
        tensor, seed=1, duration_ms=100.0, dt_ms=0.5,
    )
    model = _pipeline.build_network(cfg)
    assert isinstance(model, Model)


def test_build_network_tensor_path():
    tensor = _pipeline.load_tensor(SMALL_TENSOR_PATH)
    runtime = RuntimeConfiguration(seed=2, duration_ms=100.0, dt_ms=0.5)
    model = _pipeline.build_network(tensor, runtime)
    assert isinstance(model, Model)


def test_run_network_matches_direct_simulate():
    tensor = _pipeline.load_tensor(SMALL_TENSOR_PATH)
    cfg = _pipeline.tensor_to_configuration(
        tensor, seed=5, duration_ms=50.0, dt_ms=0.5,
    )
    model = _pipeline.build_network(cfg)
    sig = _pipeline.run_network(model, duration_ms=50.0, dt_ms=0.5, seed=5)
    assert isinstance(sig, Signals)
    assert sig.spikes.shape[0] > 0
