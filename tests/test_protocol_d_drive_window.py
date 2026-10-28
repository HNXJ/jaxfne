"""Protocol D drive window: a negative onset or a nonpositive duration is refused.

A negative onset made ``sched[onset:end]`` index from the end of the array (T-41).
"""

from __future__ import annotations

import pytest

from jaxfne.protocol_d_biological_rbs.d1_execution import _drive_schedule
from jaxfne.protocol_d_biological_rbs.d1_protocol import load_d1_spec
from jaxfne.protocol_d_biological_rbs.d2b_execution import _drive_off_index


def _with_drive(**drive):
    spec = load_d1_spec()
    sim = dict(spec["simulation_policy"])
    sim["drive"] = {**sim["drive"], **drive}
    return {**spec, "simulation_policy": sim}


def test_frozen_drive_window_is_unchanged():
    spec = load_d1_spec()
    sim = spec["simulation_policy"]
    sched = _drive_schedule(spec)
    on = sched[:, 0] != 0.0
    assert on.any(), "fixture must drive cell 0"
    first = int(on.argmax())
    assert first == int(round(float(sim["drive"]["onset_ms"]) / float(sim["dt_ms"])))
    assert _drive_off_index(spec, sim) == first + int(on.sum())


@pytest.mark.parametrize("drive", [{"onset_ms": -5.0}, {"duration_ms": 0.0}, {"duration_ms": -1.0},
                                   {"onset_ms": float("nan")}])
def test_drive_window_refuses_bad_values(drive):
    spec = _with_drive(**drive)
    with pytest.raises(ValueError, match="drive"):
        _drive_schedule(spec)
    with pytest.raises(ValueError, match="drive"):
        _drive_off_index(spec, spec["simulation_policy"])
