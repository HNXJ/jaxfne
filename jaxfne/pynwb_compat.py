"""NWB files from simulation output, written through ``jaxfne.jnwb_view``.

``write_nwb`` stores one ``Signals`` (or a ``JnwbView``) as an NWB file:
spikes in the ``Units`` table (spike times in seconds), ``V_m``, ``sources``
and the field proxies as ``TimeSeries`` in ``/acquisition`` at the view's
sampling rate and start time. Proxies keep their proxy level as the NWB
``unit`` string (``RELATIVE_PROXY`` until calibrated); they are never written
as an ``ElectricalSeries``, whose unit is volts. jnwb therefore reads the
spikes (``jnwb.unit_spike_times``) from the file; analyse the proxies from the
in-memory view (``jaxfne.jnwb_view.to_jnwb``).

``read_nwb`` opens a file with ``jnwb.read_nwb``. Both need the ``jnwb``
extra (``pynwb`` for writing; jnwb needs Python >= 3.12 for reading).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .jnwb_view import JnwbView, to_jnwb

# No real session exists for a simulation; NWB requires a start time, so the
# default is a fixed, obviously synthetic one and files are reproducible.
_SIMULATION_START = datetime(1970, 1, 1, tzinfo=timezone.utc)


def write_nwb(
    signals: Any,
    path: str | Path,
    *,
    session_description: str = "jaxfne simulation",
    identifier: Optional[str] = None,
    session_start_time: Optional[datetime] = None,
) -> Path:
    """Write ``signals`` (``Signals`` or ``JnwbView``) to ``path``; return the path.

    Raises:
        ImportError: ``pynwb`` is not installed (``pip install jaxfne[jnwb]``).
        ValueError: whatever ``to_jnwb`` refuses.
    """
    try:
        from pynwb import NWBHDF5IO, NWBFile, TimeSeries
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("write_nwb needs pynwb: pip install 'jaxfne[jnwb]'") from exc

    view = signals if isinstance(signals, JnwbView) else to_jnwb(signals)
    notes = {
        "writer": "jaxfne.pynwb_compat.write_nwb",
        "dt_ms": view.dt_ms,
        "representation": view.representation,
        "field_level": view.field_level,
        "spikes": view.metadata.get("jnwb_view_spikes", "counts_per_step"),
    }
    nwb = NWBFile(
        session_description=session_description,
        identifier=identifier or str(uuid.uuid4()),
        session_start_time=session_start_time or _SIMULATION_START,
        notes=json.dumps(notes, sort_keys=True),
    )
    for times in view.spike_times_s:
        nwb.add_unit(spike_times=times)

    series = {
        "V_m": (view.V_m, view.representation or "unspecified", "membrane state per unit"),
        "sources": (view.sources, view.representation or "unspecified", "source proxy per unit"),
        "lfp_proxy": (view.lfp_proxy, view.field_level, "LFP proxy per contact, not volts"),
        "csd_proxy": (view.csd_proxy, view.field_level, "CSD proxy per contact"),
        "phi_e_proxy": (view.phi_e_proxy, view.field_level, "extracellular potential proxy per contact"),
    }
    for name, (data, unit, description) in series.items():
        if data is None:
            continue
        nwb.add_acquisition(TimeSeries(
            name=name, data=data, unit=str(unit), rate=view.fs_hz,
            starting_time=view.t0_s, description=f"jaxfne simulation: {description}",
        ))

    path = Path(path)
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    return path


def read_nwb(path: str | Path, **kwargs: Any) -> Any:
    """Read an NWB file with ``jnwb.read_nwb`` (needs the ``jnwb`` extra)."""
    try:
        import jnwb
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("read_nwb needs jnwb: pip install 'jaxfne[jnwb]' (Python >= 3.12)") from exc
    return jnwb.read_nwb(str(path), **kwargs)


__all__ = ["write_nwb", "read_nwb"]
