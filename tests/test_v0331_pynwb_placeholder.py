"""v0.3.31 PyNWB placeholder tests.

Verify that PyNWB operations fail loudly with clear error messages.
"""

import pytest

import jaxfne as jtfne
from jaxfne import pynwb_compat


class TestPyNWBPlaceholder:
    """Test PyNWB placeholder functions."""

    def test_write_nwb_not_implemented(self):
        """write_nwb should raise NotImplementedError."""
        with pytest.raises(
            NotImplementedError,
            match="NWB write is not yet implemented",
        ):
            pynwb_compat.write_nwb()

    def test_read_nwb_not_implemented(self):
        """read_nwb should raise NotImplementedError."""
        with pytest.raises(
            NotImplementedError,
            match="NWB read is not yet implemented",
        ):
            pynwb_compat.read_nwb()

    def test_pynwb_functions_namespaced_not_root_exports(self):
        """PyNWB placeholders live in jaxfne.pynwb_compat, not the root namespace."""
        assert "write_nwb" not in jtfne.__all__
        assert "read_nwb" not in jtfne.__all__
        assert not hasattr(jtfne, "write_nwb")
        assert not hasattr(jtfne, "read_nwb")
        assert callable(pynwb_compat.write_nwb)
        assert callable(pynwb_compat.read_nwb)

    def test_pynwb_error_message_helpful(self):
        """Error message should suggest alternatives."""
        with pytest.raises(NotImplementedError) as exc_info:
            pynwb_compat.write_nwb()
        error_msg = str(exc_info.value)
        assert "manifest.json" in error_msg
        assert "Signals" in error_msg

    def test_write_nwb_with_args_fails(self):
        """write_nwb should fail even with arguments."""
        with pytest.raises(NotImplementedError):
            pynwb_compat.write_nwb("fake_file.nwb", {})

    def test_read_nwb_with_args_fails(self):
        """read_nwb should fail even with arguments."""
        with pytest.raises(NotImplementedError):
            pynwb_compat.read_nwb("fake_file.nwb")
