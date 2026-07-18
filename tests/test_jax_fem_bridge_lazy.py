"""jax-fem bridge lazy-loading tests.

Mirrors tests/test_v0331_jaxley_lazy.py's pattern. jax-fem is GPLv3
(commercial licensing available separately from the author) -- deliberately
NOT a core jaxfne dependency, so this bridge must never import it at
`import jaxfne` time, and must fail with a clear, actionable error if a
caller tries to use it without installing the optional extra.
"""

import sys
import pytest


class TestJaxFemLazyLoading:
    def test_jax_fem_not_imported_on_import_jaxfne(self):
        """Importing jaxfne should NOT import jax_fem."""
        if "jax_fem" in sys.modules:
            pytest.skip("jax_fem already imported by another test")

        import jaxfne  # noqa: F401

        assert "jax_fem" not in sys.modules, (
            "jax_fem should not be imported when importing jaxfne"
        )

    def test_require_jax_fem_available_in_api(self):
        import jaxfne as jtfne

        assert callable(jtfne.require_jax_fem)

    def test_jax_fem_field_bridge_available_in_api(self):
        import jaxfne as jtfne

        assert hasattr(jtfne, "JaxFemFieldBridge")
        assert jtfne.JaxFemFieldBridge is not None

    def test_bridge_spec_schema_only_no_import_required(self):
        """to_spec() must work without jax-fem installed -- it's pure metadata,
        matching JaxleyEmitterBridge's own schema-first contract."""
        import jaxfne as jtfne

        bridge = jtfne.JaxFemFieldBridge(geometry="laminar_column", n_layers=6)
        spec = bridge.to_spec().to_dict()
        assert spec["name"] == "jax_fem_field_bridge"
        assert spec["backend"] == "jax_fem"
        assert spec["status"] == "schema_only_no_field_solve"
        assert spec["physical_amplitude_calibrated"] is False
        assert "GPLv3" in spec["metadata"]["license"]

    def test_construct_raises_clear_error_without_install(self):
        """construct() must fail loudly (not silently no-op) if jax-fem isn't
        installed, with an actionable install hint -- matching
        require_jaxley()'s contract."""
        import jaxfne as jtfne

        try:
            import jax_fem  # noqa: F401
            pytest.skip("jax_fem is installed -- this test only checks the absent-dependency path")
        except ImportError:
            pass

        bridge = jtfne.JaxFemFieldBridge()
        with pytest.raises(ImportError, match="jax-fem"):
            bridge.construct()
