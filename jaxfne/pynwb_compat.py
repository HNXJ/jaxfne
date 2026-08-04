"""PyNWB compatibility placeholder.

This module is a placeholder for future NWB file I/O support.
Currently, no round-trip NWB write/read is implemented.

Do not use production NWB files with jaxfne output yet.
"""


def write_nwb(*args, **kwargs):
    """NWB write — NOT implemented (intentional placeholder).

    Exported for API-surface stability only. No round-trip NWB write exists;
    calling this always raises ``NotImplementedError``.

    Raises:
        NotImplementedError: Always. NWB I/O is not yet implemented.
    """
    raise NotImplementedError(
        "NWB write is not yet implemented in jaxfne. "
        "This is a placeholder for future compatibility. "
        "Use manifest.json + Signals objects for now."
    )


def read_nwb(*args, **kwargs):
    """NWB read — NOT implemented (intentional placeholder).

    Exported for API-surface stability only. No round-trip NWB read exists;
    calling this always raises ``NotImplementedError``.

    Raises:
        NotImplementedError: Always. NWB I/O is not yet implemented.
    """
    raise NotImplementedError(
        "NWB read is not yet implemented in jaxfne. "
        "This is a placeholder for future compatibility."
    )


__all__ = ["write_nwb", "read_nwb"]
