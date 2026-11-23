"""Shared test fixtures deduplicated from repeated local copies.

Each helper here replaces AST-identical local copies only; a copy that
differs (extra fields, different signature) stays in its own test module.
"""

from __future__ import annotations

import importlib.util
import json
import types

import jax.numpy as jnp

import jaxfne
from jaxfne.emitters import EdgeList, IzhikevichParams


class MockNeuronsDataFrame:
    """Mock neuron dataframe for testing."""

    def __init__(self, n=20, areas=None, layers=None, cell_types=None):
        self.n = n
        self.data = {
            "area": areas or (["V1"] * (n // 2) + ["PFC"] * (n - n // 2)),
            "layer": layers or (["L4"] * (n // 2) + ["L5"] * (n - n // 2)),
            "cell_type": cell_types or (["E"] * int(0.75 * n) + ["PV"] * (n - int(0.75 * n))),
        }

    def __len__(self):
        return self.n

    def get(self, key, default=None):
        return self.data.get(key, default)


def _cfg(n=8):
    """Minimal configuration."""
    return (
        jaxfne.configuration()
        .network(n=n)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="p", n_contacts=4)
    )


def _empty_edges() -> EdgeList:
    jdtype = jnp.float32
    return EdgeList(
        pre=jnp.zeros((0,), dtype=jnp.int32),
        post=jnp.zeros((0,), dtype=jnp.int32),
        weight=jnp.zeros((0,), dtype=jdtype),
        receptor_index=jnp.zeros((0,), dtype=jnp.int32),
        tau_ms=jnp.zeros((0,), dtype=jdtype),
    )


def _isolated_neuron_params() -> IzhikevichParams:
    jdtype = jnp.float32
    return IzhikevichParams(
        v0=jnp.asarray([-65.0], dtype=jdtype),
        u0=jnp.zeros((1,), dtype=jdtype),
        a=jnp.asarray([0.02], dtype=jdtype),
        b=jnp.asarray([0.2], dtype=jdtype),
        c=jnp.asarray([-65.0], dtype=jdtype),
        d=jnp.asarray([8.0], dtype=jdtype),
        drive=jnp.zeros((1,), dtype=jdtype),
        sign=jnp.ones((1,), dtype=jdtype),
        W=jnp.zeros((1, 1), dtype=jdtype),
        source_scale=jnp.ones((1,), dtype=jdtype),
        labels=("E",),
        layer_labels=("L4",),
        source_calibration_status="uncalibrated_izhikevich_native_current",
    )


def load_score(path):
    """The stored score document at path (one scorer's score.json)."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(name, path) -> types.ModuleType:
    """Execute the file at path as a module named name (not registered in sys.modules)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
