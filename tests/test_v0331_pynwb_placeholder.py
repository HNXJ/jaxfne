"""v0.3.31 PyNWB placeholder tests.

Verify that PyNWB operations fail loudly with clear error messages.
"""

import pytest

import jaxfne as jtfne


class TestPyNWBPlaceholder:
    """Test PyNWB placeholder functions."""

    def test_write_nwb_not_implemented(self):
        """write_nwb should raise NotImplementedError."""
        with pytest.raises(
            NotImplementedError,
            match="NWB write is not yet implemented",
        ):
            jtfne.write_nwb()

    def test_read_nwb_not_implemented(self):
        """read_nwb should raise NotImplementedError."""
        with pytest.raises(
            NotImplementedError,
            match="NWB read is not yet implemented",
        ):
            jtfne.read_nwb()

    def test_pynwb_functions_in_api(self):
        """write_nwb and read_nwb should be in public API."""
        assert hasattr(jtfne, "write_nwb")
        assert hasattr(jtfne, "read_nwb")
        assert callable(jtfne.write_nwb)
        assert callable(jtfne.read_nwb)

    def test_pynwb_error_message_helpful(self):
        """Error message should suggest alternatives."""
        with pytest.raises(NotImplementedError) as exc_info:
            jtfne.write_nwb()
        error_msg = str(exc_info.value)
        assert "manifest.json" in error_msg
        assert "Signals" in error_msg

    def test_write_nwb_with_args_fails(self):
        """write_nwb should fail even with arguments."""
        with pytest.raises(NotImplementedError):
            jtfne.write_nwb("fake_file.nwb", {})

    def test_read_nwb_with_args_fails(self):
        """read_nwb should fail even with arguments."""
        with pytest.raises(NotImplementedError):
            jtfne.read_nwb("fake_file.nwb")
