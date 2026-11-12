"""`build_config` dropped the layer structure it declares (release plan 10B B4).

``scripts/cortical_column_localized_workflow.py::build_config`` declares the six
canonical layers (``jtfne.CANONICAL_LAYERS_6L``) and the canonical per-layer E:I
profile, then asked ``build_laminar_column`` for ``geometry="uniform3d"``. That
mode builds one ``uniform_3d`` layer and writes no ``layer_fractions`` /
``layer_cell_types``, so the declared laminar structure never reaches the
Configuration; at HEAD the request is refused outright, and before the 0.5.5
guard it silently collapsed the column to a single ``uniform_3d`` layer.

The declared ``local`` connection rule is not part of the defect -- it compiles
with its ``max_in_degree`` cap on either geometry -- so the connectivity
assertions below are a regression guard, not the failure being fixed.
"""
import numpy as np
import pytest

import jaxfne as jtfne
from scripts.cortical_column_localized_workflow import build_config

N = 400
DECLARED_LAYERS = list(jtfne.CANONICAL_LAYERS_6L)


@pytest.fixture(scope="module")
def config():
    """One cheap (construct-free) build_config call, shared by every test."""
    return build_config(N)


@pytest.fixture(scope="module")
def model(config):
    """construct() once (~3 s at N=400); no simulate(), so no 60 s simulation."""
    return jtfne.construct(config)


def test_build_config_keeps_the_declared_layers(config):
    assert config.metadata["columns"][0]["layers"] == DECLARED_LAYERS
    assert list(config.metadata["layer_fractions"]) == DECLARED_LAYERS
    assert not config.metadata.get("uniform_3d")
    assert config.metadata.get("column_radius_mm") == 1.0
    assert config.metadata.get("column_height_mm") == 2.0


def test_build_config_keeps_the_canonical_per_layer_composition(config):
    per_layer = config.metadata["layer_cell_types"]
    assert list(per_layer) == DECLARED_LAYERS
    # The canonical E:I gradient peaks E deep (L6 0.95) and is inhibitory-heavy
    # superficial (L1 0.50); a flat composition would read 0.75 in every layer.
    assert per_layer["L6"]["E"] == pytest.approx(0.95)
    assert per_layer["L1"]["E"] == pytest.approx(0.50)


def test_constructed_model_labels_every_neuron_with_a_declared_layer(model):
    labels = tuple(model.params["emitter"].layer_labels)
    assert len(labels) == N
    assert set(labels) == set(DECLARED_LAYERS)


def test_declared_local_connection_still_compiles_under_its_cap(model):
    rule = model.cfg.metadata["circuit"]["connections"][0]
    assert rule["status"] == "compiled"
    assert rule["max_in_degree"] == 100
    assert rule["spatial_sigma"] == pytest.approx(0.15)
    post = np.asarray(model.params["edge_list"].post)
    assert post.size > 0
    assert int(np.bincount(post, minlength=N).max()) <= 100
