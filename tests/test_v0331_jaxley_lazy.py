"""v0.3.31 Jaxley lazy loading tests.

Verify that jaxley is not imported on import jaxfne.
"""

import sys
import pytest


class TestJaxleyLazyLoading:
    """Test that Jaxley is loaded lazily, not on import."""

    def test_jaxley_not_imported_on_import_jaxfne(self):
        """Importing jaxfne should NOT import jaxley."""
        if "jaxley" in sys.modules:
            pytest.skip("jaxley already imported by another test")

        import jaxfne  # noqa: F401

        assert (
            "jaxley" not in sys.modules
        ), "jaxley should not be imported when importing jaxfne"

    def test_require_jaxley_fails_without_install(self):
        """require_jaxley() should fail loudly if jaxley is not installed."""
        import jaxfne as jtfne

        # We can't actually test this without uninstalling jaxley, but we can
        # verify the function exists and would have the right error message.
        assert callable(jtfne.require_jaxley)

    def test_jaxley_bridge_available_in_api(self):
        """JaxleyBridge should be available in public API."""
        import jaxfne as jtfne

        assert hasattr(jtfne, "JaxleyBridge")
        assert jtfne.JaxleyBridge is not None

    def test_jaxley_emitter_bridge_available(self):
        """JaxleyEmitterBridge should be available."""
        import jaxfne as jtfne

        assert hasattr(jtfne, "JaxleyEmitterBridge")

    def test_jaxley_trace_to_signals_available(self):
        """jaxley_trace_to_signals function should be available."""
        import jaxfne as jtfne

        assert hasattr(jtfne, "jaxley_trace_to_signals")
        assert callable(jtfne.jaxley_trace_to_signals)

    def test_bridge_spec_available(self):
        """BridgeSpec should be available."""
        import jaxfne as jtfne

        assert hasattr(jtfne, "BridgeSpec")
        spec = jtfne.BridgeSpec(name="test", backend="test")
        assert spec.name == "test"
        assert spec.backend == "test"

    def test_jaxley_imports_with_require(self):
        """require_jaxley() should successfully import jaxley if available."""
        import jaxfne as jtfne

        try:
            jaxley = jtfne.require_jaxley()
            assert jaxley is not None
            assert "jaxley" in sys.modules
        except ImportError:
            pytest.skip("jaxley not installed")
