"""Tests for v0.3.18 sharding mesh stubs.

Verifies that sharding_utils functions are importable, return correct types,
degrade cleanly on single-device environments, and expose the expected
public API through the jaxfne top-level namespace.
"""
from __future__ import annotations

import jax
import pytest


# ---------------------------------------------------------------------------
# 1. Import smoke — module must load without errors
# ---------------------------------------------------------------------------

class TestImportSmoke:
    """sharding_utils must import cleanly with no side-effects."""

    def test_module_importable(self):
        import jaxfne.sharding_utils as su  # noqa: F401

    def test_public_api_importable_from_jaxfne(self):
        from jaxfne import (  # noqa: F401
            get_sharding_context,
            make_candidate_sharding,
            make_population_mesh,
            make_replicated_sharding,
        )

    def test_symbols_in_all(self):
        import jaxfne
        for sym in ("get_sharding_context", "make_population_mesh",
                    "make_candidate_sharding", "make_replicated_sharding"):
            assert sym in jaxfne.__all__, f"{sym!r} missing from jaxfne.__all__"


# ---------------------------------------------------------------------------
# 2. make_population_mesh — single-device fallback
# ---------------------------------------------------------------------------

class TestMakePopulationMesh:
    """Single-device CI must return None; multi-device must return a Mesh."""

    def test_returns_none_or_mesh(self):
        from jaxfne.sharding_utils import make_population_mesh
        from jax.sharding import Mesh

        result = make_population_mesh()
        n_devices = len(jax.devices())
        if n_devices <= 1:
            assert result is None, (
                f"Expected None on single-device env but got {result!r}"
            )
        else:
            assert isinstance(result, Mesh), (
                f"Expected Mesh on multi-device env but got {type(result)}"
            )

    def test_single_device_returns_none(self, monkeypatch):
        """Force single-device view: must return None."""
        import jax
        import jaxfne.sharding_utils as su

        real_device = jax.devices()[0]  # capture before patching
        monkeypatch.setattr(jax, "devices", lambda: [real_device])
        result = su.make_population_mesh()
        assert result is None

    def test_mesh_axis_name(self, monkeypatch):
        """Mesh must carry axis name 'population_sweep'."""
        import jax
        import numpy as np
        import jaxfne.sharding_utils as su
        from jax.sharding import Mesh

        # Simulate 2 devices by patching jax.devices to return same device twice
        real_device = jax.devices()[0]
        monkeypatch.setattr(jax, "devices", lambda: [real_device, real_device])

        result = su.make_population_mesh()
        # With 2 devices it should return a Mesh
        assert isinstance(result, Mesh)
        assert "population_sweep" in result.axis_names


# ---------------------------------------------------------------------------
# 3. make_candidate_sharding / make_replicated_sharding
# ---------------------------------------------------------------------------

class TestShardings:
    """NamedSharding factories must produce correct PartitionSpec."""

    @pytest.fixture
    def two_device_mesh(self, monkeypatch):
        """Return a Mesh patched to appear as 2 devices."""
        import jax
        import jaxfne.sharding_utils as su

        real_device = jax.devices()[0]
        monkeypatch.setattr(jax, "devices", lambda: [real_device, real_device])
        mesh = su.make_population_mesh()
        assert mesh is not None, "fixture requires make_population_mesh to succeed"
        return mesh

    def test_candidate_sharding_type(self, two_device_mesh):
        from jax.sharding import NamedSharding
        from jaxfne.sharding_utils import make_candidate_sharding

        sharding = make_candidate_sharding(two_device_mesh)
        assert isinstance(sharding, NamedSharding)

    def test_candidate_sharding_partitions_batch_axis(self, two_device_mesh):
        from jaxfne.sharding_utils import make_candidate_sharding

        sharding = make_candidate_sharding(two_device_mesh)
        spec = sharding.spec
        # First axis partitioned by population_sweep
        assert spec[0] == "population_sweep"
        # Second axis (n_params) replicated
        assert spec[1] is None

    def test_replicated_sharding_type(self, two_device_mesh):
        from jax.sharding import NamedSharding
        from jaxfne.sharding_utils import make_replicated_sharding

        sharding = make_replicated_sharding(two_device_mesh)
        assert isinstance(sharding, NamedSharding)

    def test_replicated_sharding_no_partitioning(self, two_device_mesh):
        from jaxfne.sharding_utils import make_replicated_sharding

        sharding = make_replicated_sharding(two_device_mesh)
        # All axes replicated (None)
        assert all(s is None for s in sharding.spec)


# ---------------------------------------------------------------------------
# 4. get_sharding_context — bundle
# ---------------------------------------------------------------------------

class TestGetShardingContext:
    """get_sharding_context must return None on single-device or a valid dict."""

    def test_returns_none_or_dict(self):
        from jaxfne.sharding_utils import get_sharding_context

        ctx = get_sharding_context()
        if len(jax.devices()) <= 1:
            assert ctx is None
        else:
            assert isinstance(ctx, dict)

    def test_single_device_returns_none(self, monkeypatch):
        import jax
        import jaxfne.sharding_utils as su

        real_device = jax.devices()[0]  # capture before patching
        monkeypatch.setattr(jax, "devices", lambda: [real_device])
        assert su.get_sharding_context() is None

    def test_multi_device_dict_keys(self, monkeypatch):
        import jax
        import jaxfne.sharding_utils as su

        real_device = jax.devices()[0]
        monkeypatch.setattr(jax, "devices", lambda: [real_device, real_device])
        ctx = su.get_sharding_context()
        assert ctx is not None
        assert set(ctx.keys()) == {"mesh", "candidate", "replicated"}

    def test_multi_device_context_types(self, monkeypatch):
        import jax
        import jaxfne.sharding_utils as su
        from jax.sharding import Mesh, NamedSharding

        real_device = jax.devices()[0]
        monkeypatch.setattr(jax, "devices", lambda: [real_device, real_device])
        ctx = su.get_sharding_context()
        assert ctx is not None
        assert isinstance(ctx["mesh"], Mesh)
        assert isinstance(ctx["candidate"], NamedSharding)
        assert isinstance(ctx["replicated"], NamedSharding)
