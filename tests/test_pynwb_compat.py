"""NWB files from simulation output (todo 0d): write_nwb round trip, read via jnwb."""

import json

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne import pynwb_compat
from jaxfne.jnwb_view import to_jnwb

pynwb = pytest.importorskip("pynwb")


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    model = jtfne.construct(jtfne.suite2_four_celltype_config(seed=0))
    sig = jtfne.simulate(model, duration_ms=50, dt_ms=0.1, seed=0)
    path = pynwb_compat.write_nwb(sig, tmp_path_factory.mktemp("nwb") / "sim.nwb",
                                  identifier="jaxfne-test")
    return to_jnwb(sig), path


def test_round_trip_keeps_spikes_rate_and_proxy_status(written):
    view, path = written
    assert sum(t.size for t in view.spike_times_s) > 0, "fixture must spike"
    with pynwb.NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        units = nwb.units
        assert len(units) == len(view.spike_times_s)
        for u, times in enumerate(view.spike_times_s):
            np.testing.assert_array_equal(units["spike_times"][u], times)
        lfp = nwb.acquisition["lfp_proxy"]
        np.testing.assert_array_equal(lfp.data[:], view.lfp_proxy)
        assert lfp.rate == view.fs_hz and lfp.starting_time == view.t0_s
        assert lfp.unit == "RELATIVE_PROXY"
        assert not isinstance(lfp, pynwb.ecephys.ElectricalSeries)
        np.testing.assert_array_equal(nwb.acquisition["V_m"].data[:], view.V_m)
        assert json.loads(nwb.notes)["dt_ms"] == view.dt_ms


def test_jnwb_reads_the_written_spikes(written):
    jnwb = pytest.importorskip("jnwb")
    view, path = written
    unit = int(np.argmax([t.size for t in view.spike_times_s]))
    np.testing.assert_array_equal(jnwb.unit_spike_times(str(path), unit), view.spike_times_s[unit])
    assert pynwb_compat.read_nwb(path).identifier == "jaxfne-test"


def test_nwb_functions_stay_off_the_root_namespace():
    assert "write_nwb" not in jtfne.__all__ and not hasattr(jtfne, "write_nwb")
    assert "read_nwb" not in jtfne.__all__ and not hasattr(jtfne, "read_nwb")
