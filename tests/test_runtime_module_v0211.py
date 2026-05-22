"""Tests for jaxfne.runtime compatibility module (v0.2.11).

Verifies that the public runtime module provides stable import surface
and correct re-exports.

NOTE: All imports of jaxfne.runtime are isolated to function-local scope
to prevent module-level caching from affecting other tests' access to the
jaxfne.runtime() function. See GitHub issue: test isolation regression.
"""

import jaxfne
from jaxfne.core import RuntimeConfig as CoreRuntimeConfig


class TestRuntimeModule:
    """Test suite for runtime module imports and contracts."""

    def test_runtime_module_import(self):
        """Test that jaxfne.runtime module can be imported."""
        # Import locally to avoid module-level namespace pollution
        import jaxfne.runtime
        assert jaxfne.runtime is not None

    def test_runtime_config_import(self):
        """Test that RuntimeConfig can be imported from jaxfne.runtime."""
        # Import locally to avoid module-level namespace pollution
        from jaxfne.runtime import RuntimeConfig
        assert RuntimeConfig is not None
        assert isinstance(RuntimeConfig, type)

    def test_runtime_config_same_as_core(self):
        """Test that runtime.RuntimeConfig is the same as core.RuntimeConfig."""
        from jaxfne.runtime import RuntimeConfig
        assert RuntimeConfig is CoreRuntimeConfig

    def test_runtime_config_same_as_package_export(self):
        """Test that runtime.RuntimeConfig is the same as jaxfne.RuntimeConfig."""
        from jaxfne.runtime import RuntimeConfig
        assert RuntimeConfig is jaxfne.RuntimeConfig

    def test_runtime_config_default_construct(self):
        """Test that default RuntimeConfig() can be constructed."""
        from jaxfne.runtime import RuntimeConfig
        cfg = RuntimeConfig()
        assert cfg is not None
        assert isinstance(cfg, RuntimeConfig)

    def test_runtime_module_all(self):
        """Test that module __all__ includes RuntimeConfig."""
        import sys
        import jaxfne.runtime  # noqa: F401 - trigger module import
        # Note: jaxfne.runtime attribute returns the function (due to __getattr__),
        # so we access the module directly from sys.modules instead
        runtime_module = sys.modules['jaxfne.runtime']
        assert hasattr(runtime_module, '__all__')
        assert 'RuntimeConfig' in runtime_module.__all__

    def test_version_unchanged(self):
        """Test that jaxfne version remains 0.2.10."""
        assert jaxfne.__version__ == "0.2.25"
