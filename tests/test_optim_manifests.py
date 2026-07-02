"""Tests for jaxfne/optim/manifests.py::serialize_optimization_manifest.

Confirmed 2026-07-01 to have zero prior test coverage (grep -rln
'serialize_optimization_manifest' tests/ -> no hits) despite being exported
from jaxfne.optim.__init__.
"""
from __future__ import annotations

import dataclasses
import json

import jax.numpy as jnp
import pytest

from jaxfne.optim.gsgd import GSGDState
from jaxfne.optim.manifests import serialize_optimization_manifest


def test_none_state_produces_empty_state_dict():
    manifest = serialize_optimization_manifest(None, {"lr": 0.1})
    assert manifest["state"] == {}
    assert manifest["hyperparams"] == {"lr": 0.1}


def test_namedtuple_state_uses_asdict_path():
    state = GSGDState(count=jnp.asarray(3), step_size=jnp.asarray(0.5))
    manifest = serialize_optimization_manifest(state, {})
    assert manifest["state"]["count"] == 3
    assert manifest["state"]["step_size"] == pytest.approx(0.5)


def test_dataclass_state_uses_dict_path():
    @dataclasses.dataclass
    class _PlainState:
        step: int
        scale: float

    state = _PlainState(step=5, scale=1.5)
    manifest = serialize_optimization_manifest(state, {})
    assert manifest["state"]["step"] == 5
    assert manifest["state"]["scale"] == 1.5


def test_frozen_dataclass_state_uses_dict_path_not_asdict():
    # Frozen only blocks __setattr__, not __dict__ access -- confirmed this
    # falls through the hasattr(state, "_asdict") check correctly since a
    # plain dataclass has no _asdict method (that's NamedTuple-only).
    @dataclasses.dataclass(frozen=True)
    class _FrozenState:
        step: int

    state = _FrozenState(step=7)
    assert not hasattr(state, "_asdict")
    manifest = serialize_optimization_manifest(state, {})
    assert manifest["state"]["step"] == 7


def test_state_with_no_dict_or_asdict_falls_back_to_empty():
    manifest = serialize_optimization_manifest(object(), {})
    assert manifest["state"] == {}


def test_jax_array_state_field_converted_to_list():
    @dataclasses.dataclass
    class _ArrayState:
        weights: jnp.ndarray

    state = _ArrayState(weights=jnp.asarray([1.0, 2.0, 3.0]))
    manifest = serialize_optimization_manifest(state, {})
    assert manifest["state"]["weights"] == [1.0, 2.0, 3.0]


def test_hyperparams_jax_array_and_callable_handling():
    manifest = serialize_optimization_manifest(
        None,
        {
            "lr": jnp.asarray(0.01),
            "steps": 10,
            "skip_me": lambda x: x,
        },
    )
    assert manifest["hyperparams"]["lr"] == pytest.approx(0.01)
    assert manifest["hyperparams"]["steps"] == 10
    assert "skip_me" not in manifest["hyperparams"]


def test_manifest_is_json_serializable():
    @dataclasses.dataclass
    class _State:
        step: int
        weights: jnp.ndarray

    state = _State(step=2, weights=jnp.asarray([0.1, 0.2]))
    manifest = serialize_optimization_manifest(state, {"lr": jnp.asarray(0.05)})
    json_str = json.dumps(manifest, allow_nan=False)
    reloaded = json.loads(json_str)
    assert reloaded["state"]["step"] == 2
    assert reloaded["hyperparams"]["lr"] == pytest.approx(0.05)


def test_unserializable_field_falls_back_to_str_not_raise():
    class _Unrepresentable:
        def __repr__(self):
            return "<unrepresentable>"

    @dataclasses.dataclass
    class _State:
        obj: _Unrepresentable

    manifest = serialize_optimization_manifest(_State(obj=_Unrepresentable()), {})
    assert manifest["state"]["obj"] == "<unrepresentable>"
