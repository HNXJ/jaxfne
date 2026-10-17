"""Fig. 6 data labels from oxm0818 MT/MST units (recipe per GSDR ``misc/main.py``).

Fail-first: on the pre-label tree ``data_labels.py`` does not exist, so the
import below fails. The recipe mirrors ``misc/main.py:109-141`` (read as
data): trial slice ``5:6``, baseline window ``:200``, stim window
``600:1200``, mean over the MT/MST units, rFFT power ``/(N*fs)`` at
``fs=1000``, interp onto ``linspace(15, 60, 40)``, ``uniform_filter1d``
smoothing (size 5), min-subtracted, max-normalized. The real warehouse file
stays out of the repo; the real-data test below skips when it is absent.
"""

from __future__ import annotations

import importlib.util
import io
import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage


def _load_data_labels():
    path = (Path(__file__).resolve().parents[1] / "artifacts" / "etudes"
            / "papers" / "gsdr2026" / "run" / "data_labels.py")
    spec = importlib.util.spec_from_file_location("gsdr2026_data_labels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dl = _load_data_labels()

FS_HZ = 1000.0
GRID = np.linspace(15.0, 60.0, 8)


def _expected_label_psd(mean_trace, fs=FS_HZ, grid=GRID):
    """Independent plain-numpy copy of the ``misc/main.py`` label recipe."""
    signal = np.asarray(mean_trace, dtype=np.float64)
    n = signal.shape[0]
    power = np.abs(np.fft.rfft(signal)) ** 2 / (n * fs)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    interp = np.interp(np.asarray(grid, dtype=np.float64), freqs, power)
    smooth = ndimage.uniform_filter1d(interp, size=5)
    smooth = smooth - smooth.min()
    return smooth / (smooth.max() + 1e-6)


def _synthetic_units():
    rng = np.random.default_rng(7)
    # Sparse exponentially-decaying spike kernels like the warehouse file:
    # isolated onsets with peak ~200, tau 1 sample, plus exact zeros.
    u = np.zeros((2, 5, 300), dtype=np.float64)
    for t in range(2):
        for k in range(5):
            for onset in (10, 60, 120, 200, 260):
                amp = 150.0 + 50.0 * rng.random()
                decay = amp * np.exp(-np.arange(40) / 1.0)
                u[t, k, onset:onset + 40] += decay[: min(40, 300 - onset)]
    return u


def _synthetic_info():
    return [
        {"name": "unknown", "ids": np.arange(2, 2 + 2)},
        {"name": "unknown", "ids": np.arange(207, 207 + 1)},
        {"name": "unknown", "ids": np.arange(362, 362 + 2)},
    ]


def test_label_psds_match_recipe_on_synthetic():
    """Label PSDs equal an independent reimplementation on a tiny array."""
    u = _synthetic_units()
    info = _synthetic_info()  # sizes (2, 1, 2): MT/MST slice is units 3:5
    out = dl.fig6_label_psds(u, info, trial=0, base_window=(0, 50),
                             stim_window=(50, 150), fs_hz=FS_HZ, grid=GRID)
    assert out["mtmst_slice"] == (3, 5)
    for key, (w0, w1) in (("baseline", (0, 50)), ("stim", (50, 150))):
        mean_trace = u[0, 3:5, w0:w1].mean(axis=0)
        np.testing.assert_allclose(out[key], _expected_label_psd(mean_trace),
                                   rtol=1e-10, err_msg=key)
        assert out[key].shape == (8,)
        assert np.all(np.isfinite(out[key]))
    assert out["baseline"].min() == pytest.approx(0.0, abs=1e-12)
    # max-normalized as max/(max+1e-6): within 1e-3 of 1 from below.
    assert out["baseline"].max() == pytest.approx(1.0, abs=1e-3)
    assert out["baseline"].max() <= 1.0


def test_label_psd_without_smoothing_would_differ():
    """The smoothing step is load-bearing: raw interp is not the label."""
    u = _synthetic_units()
    out = dl.fig6_label_psds(u, _synthetic_info(), trial=0,
                             base_window=(0, 50), stim_window=(50, 150),
                             fs_hz=FS_HZ, grid=GRID)
    mean_trace = u[0, 3:5, 0:50].mean(axis=0)
    n = mean_trace.shape[0]
    power = np.abs(np.fft.rfft(mean_trace)) ** 2 / (n * FS_HZ)
    freqs = np.fft.rfftfreq(n, d=1.0 / FS_HZ)
    raw = np.interp(GRID, freqs, power)
    raw = raw - raw.min()
    raw = raw / (raw.max() + 1e-6)
    assert not np.allclose(out["baseline"], raw, rtol=1e-6), \
        "smoothing must change the label PSD on this fixture"


def test_group_slice_covers_units_axis():
    """Group sizes cumulate to the units axis; MT/MST is the third group."""
    info = _synthetic_info()
    assert dl.group_sizes(info) == (2, 1, 2)
    assert dl.mtmst_unit_slice(info) == (3, 5)
    assert sum(dl.group_sizes(info)) == 5


def test_restricted_unpickler_refuses_foreign_global():
    """A pickle naming anything outside the three allowed globals raises."""
    import os

    payload = pickle.dumps(os.system)
    with pytest.raises(pickle.UnpicklingError):
        dl.loads_info_payload(payload)


def test_restricted_unpickler_roundtrips_object_array():
    """A real ``np.save`` object array loads through the restricted path."""
    buf = io.BytesIO()
    np.save(buf, np.array(_synthetic_info(), dtype=object), allow_pickle=True)
    info = dl.loads_info_payload(buf.getvalue())
    assert [d["name"] for d in info] == ["unknown"] * 3
    assert [len(np.asarray(d["ids"])) for d in info] == [2, 1, 2]


def test_spike_onsets_count_kernel_onsets():
    """Onset detection counts one crossing per kernel onset in the fixture."""
    u = _synthetic_units()
    onsets = dl.spike_onsets(u[0, 0, :], threshold=20.0)
    assert int(onsets.sum()) == 5, "five kernel onsets, got %d" % int(onsets.sum())


def test_window_kappa_is_paper_pairwise_fleiss():
    """``window_kappa`` follows ``gsdr/analysis.py:109`` (pairwise Fleiss).

    The jaxfne unanimous-agreement variant saturates at its -1 clamp on
    sparse data (raw -9.1 baseline / -21.7 stim on oxm0818 trial 5 MT/MST);
    the paper statistic stays near chance there and is exactly 1.0 under
    perfect synchrony.
    """
    # Perfect synchrony (mixed active/silent bins): kappa is exactly 1.
    train = np.zeros(200, dtype=np.float64)
    train[[10, 50, 90, 150]] = 250.0
    sync = np.repeat(train[None, :], 6, axis=0)
    assert dl.window_kappa(sync, 0, 200) == pytest.approx(1.0)
    # Sparse independent onsets: finite, near chance, never the -1 clamp.
    rng = np.random.default_rng(3)
    flat = (rng.random((20, 1000)) < 0.02)
    sparse = np.zeros((20, 1000), dtype=np.float64)
    sparse[flat] = 250.0
    k = dl.window_kappa(sparse, 0, 1000)
    assert np.isfinite(k)
    assert k > -1.0, "sparse independent spikes must not hit the -1 clamp"
    assert abs(k) < 0.2, "independent spikes stay near chance, got %r" % k


_REAL = [dl.DATA_UNITS_PATH, dl.DATA_INFO_PATH]
_NEEDS_REAL = pytest.mark.skipif(
    not all(Path(p).exists() for p in _REAL),
    reason="warehouse oxm0818 files absent (they stay out of the repo)")


@_NEEDS_REAL
def test_real_file_axes_groups_and_hashes():
    """OBSERVED on the warehouse file: axes, groups, hashes, label sanity."""
    import hashlib

    for path, want in dl.DATA_SHA256.items():
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        assert h.hexdigest() == want, "hash drift on %s" % path
    units = dl.load_units_mmap()
    assert units.shape == (190, 257, 6000)
    assert str(units.dtype) == "float64"
    info = dl.load_units_info()
    assert dl.group_sizes(info) == (122, 40, 95)
    assert dl.mtmst_unit_slice(info) == (162, 257)
    out = dl.fig6_label_psds(units, info)
    assert out["grid"].shape == (40,)
    for key in ("baseline", "stim"):
        assert out[key].shape == (40,)
        assert np.all(np.isfinite(out[key]))
        assert out[key].min() == pytest.approx(0.0, abs=1e-12)
        assert out[key].max() == pytest.approx(1.0, abs=1e-3)
        assert out[key].max() <= 1.0
