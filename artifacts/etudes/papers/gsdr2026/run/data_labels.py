"""Fig. 6 data labels from the warehouse oxm0818 MT/MST units.

Recipe (read as data, never run): ``HNXJ/GSDR`` ``misc/main.py:109-141``.
Labels are the PSD of the mean-over-units trace on trial slice ``5:6``,
baseline window ``:200``, stim window ``600:1200``; rFFT power ``/(N*fs)``
at ``fs=1000`` Hz, interp onto ``linspace(15, 60, 40)``, ``uniform_filter1d``
smoothing (size 5), min-subtracted, max-normalized. Unit selection
(``kappa_synch.ipynb:109-121,190``): the ``(190, 257, 6000)`` array carries
axes trial/unit/sample; the three info groups cumulate along the unit axis
(0818: PFC; TEO, FST; MT, MST), so MT/MST is the third group.

The warehouse files stay out of the repo (paths + hashes below); tests that
need them skip when absent. Sampling: 1 sample = 1 ms (``fs=1000`` in both
repo PSD functions); 6000 samples = 6 s trials.

Data spike admission (lane-assumed): the traces are spike trains convolved
with an exponential kernel (peak ~10^2, tau 1 sample; zeros elsewhere), so
``DATA_SPIKE_THRESHOLD = 20.0`` admits kernel onsets while the decay tails
fall below 1.0 within ~7 samples.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[5]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
from scipy import ndimage

DATA_UNITS_PATH = "/mnt/f/warehouse/gsdr/data/oxm0818_units.npy"
DATA_INFO_PATH = "/mnt/f/warehouse/gsdr/data/oxm0818_units_info.npy"
DATA_SHA256 = {
    DATA_UNITS_PATH: "794b5ee6459b7ea38aefde15456a6d62ca3c017761091b7652c0980b2b0defb5",
    DATA_INFO_PATH: "759349b22e0552cc27a3032e27fedce5bfdbc085dcf36e494a042b3381ece6ed",
}

FS_HZ = 1000.0
LABEL_TRIAL = 5
LABEL_BASE_WINDOW = (0, 200)
LABEL_STIM_WINDOW = (600, 1200)
DATA_SPIKE_THRESHOLD = 20.0

ALLOWED_PICKLE_GLOBALS = frozenset({
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
})


def target_grid(n_bins: int = 40):
    """Label grid ``linspace(15, 60, n_bins)`` (``misc/main.py:129``)."""
    return np.linspace(15.0, 60.0, n_bins)


class _RestrictedUnpickler(pickle.Unpickler):
    """``pickle.Unpickler`` allowing exactly ``ALLOWED_PICKLE_GLOBALS``."""

    def find_class(self, module, name):
        if (module, name) in ALLOWED_PICKLE_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            "blocked pickle global %s.%s (info files may name only %s)"
            % (module, name, sorted(ALLOWED_PICKLE_GLOBALS)))


def loads_info_payload(payload: bytes):
    """Unpickle one ``np.save``-framed object array through the restricted path."""
    import io

    buf = io.BytesIO(payload)
    magic = buf.read(8)
    if magic[:6] != b"\x93NUMPY":
        # Bare pickle (tests); the warehouse file carries the .npy frame.
        buf.seek(0)
        return _RestrictedUnpickler(buf).load()
    import struct

    hlen = struct.unpack("<H" if magic[6] == 1 else "<I",
                         buf.read(2 if magic[6] == 1 else 4))[0]
    buf.read(hlen)
    return _RestrictedUnpickler(buf).load()


def load_units_info(path: str = DATA_INFO_PATH):
    """Load the ``(3,)`` object array of ``{'name', 'ids'}`` dicts (restricted)."""
    with open(path, "rb") as f:
        payload = f.read()
    info = loads_info_payload(payload)
    info = list(np.asarray(info).reshape(-1))
    if len(info) != 3 or any(set(d) != {"name", "ids"} for d in info):
        raise ValueError("units info must hold 3 {'name', 'ids'} dicts")
    return info


def load_units_mmap(path: str = DATA_UNITS_PATH):
    """Memory-map the ``(190, 257, 6000)`` float64 trial/unit/sample array."""
    units = np.load(path, mmap_mode="r", allow_pickle=False)
    if units.shape[1] != sum(len(np.asarray(d["ids"])) for d in load_units_info()):
        raise ValueError("info group sizes do not cumulate to the unit axis %s"
                         % (units.shape,))
    return units


def group_sizes(info) -> tuple:
    """Per-group unit counts, in probe order (PFC; TEO/FST; MT/MST for 0818)."""
    return tuple(int(np.asarray(d["ids"]).shape[0]) for d in info)


def mtmst_unit_slice(info) -> tuple:
    """``(start, stop)`` of the third (MT/MST) group along the unit axis."""
    sizes = group_sizes(info)
    start = sizes[0] + sizes[1]
    return (start, start + sizes[2])


def psd_of_mean_trace(signal_1d, fs_hz: float = FS_HZ, grid=None):
    """Label PSD of one mean-over-units trace (``misc/main.py:109-127``)."""
    signal = np.asarray(signal_1d, dtype=np.float64)
    n = signal.shape[0]
    power = np.abs(np.fft.rfft(signal)) ** 2 / (n * fs_hz)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    grid = target_grid() if grid is None else np.asarray(grid, dtype=np.float64)
    interp = np.interp(grid, freqs, power)
    size = int(np.ceil(5.0 * np.sqrt(fs_hz / 1000.0)))
    smooth = ndimage.uniform_filter1d(interp, size=size)
    smooth = smooth - smooth.min()
    return smooth / (smooth.max() + 1e-6)


def fig6_label_psds(units, info, *, trial: int = LABEL_TRIAL,
                    base_window=LABEL_BASE_WINDOW,
                    stim_window=LABEL_STIM_WINDOW,
                    fs_hz: float = FS_HZ, grid=None):
    """Baseline + stim label PSDs over the MT/MST units (``misc/main.py:135-141``)."""
    grid = target_grid() if grid is None else np.asarray(grid, dtype=np.float64)
    u0, u1 = mtmst_unit_slice(info)
    out = {"grid": grid, "trial": int(trial), "mtmst_slice": (u0, u1),
           "fs_hz": float(fs_hz)}
    for key, (w0, w1) in (("baseline", tuple(base_window)),
                          ("stim", tuple(stim_window))):
        mean_trace = np.asarray(units[trial, u0:u1, w0:w1], dtype=np.float64)
        out[key] = psd_of_mean_trace(mean_trace.mean(axis=0), fs_hz, grid)
    return out


def band_powers(psd, freqs, bands=None):
    """Mean power in each band (bands per ``misc/main.py`` ``band_definitions``)."""
    if bands is None:
        bands = {"gamma": (35.0, 60.0), "beta": (13.0, 30.0),
                 "alpha": (8.0, 12.0), "theta": (4.0, 8.0)}
    psd = np.asarray(psd, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    out = {}
    for name, (f_min, f_max) in bands.items():
        sel = (freqs >= f_min) & (freqs < f_max)
        out[name] = float(psd[sel].mean()) if sel.any() else 0.0
    return out


def spike_onsets(trace, threshold: float = DATA_SPIKE_THRESHOLD):
    """Upward threshold crossings (bool): one per convolved-spike onset."""
    v = np.asarray(trace, dtype=np.float64)
    return (v[:-1] < threshold) & (v[1:] >= threshold)


def window_rates_hz(units_2d, w0: int, w1: int, fs_hz: float = FS_HZ,
                    threshold: float = DATA_SPIKE_THRESHOLD):
    """Per-unit onset rates (Hz) in samples ``[w0, w1)``; ``units_2d`` is (unit, sample)."""
    mat = np.asarray(units_2d, dtype=np.float64)[:, w0:w1]
    onsets = (mat[:, :-1] < threshold) & (mat[:, 1:] >= threshold)
    return onsets.sum(axis=1) / ((w1 - w0) / fs_hz)


def window_kappa(units_2d, w0: int, w1: int, bin_ms: float = 5.0,
                 fs_hz: float = FS_HZ,
                 threshold: float = DATA_SPIKE_THRESHOLD):
    """Pairwise Fleiss-kappa synchrony of the onset matrix.

    NumPy port of ``gsdr/analysis.py:109`` ``compute_kappa`` (read as data):
    binary onset matrix ``(units, time)``, ``samples_per_bin =
    int(fs * bin_ms / 1000)``, bins by max, pairwise observed agreement
    ``P_i`` over neuron pairs, chance agreement from the pooled spike rate,
    ``(P_o - P_e) / max(1e-6, 1 - P_e)`` with no clamp. This replaces the
    earlier jaxfne unanimous-agreement variant, whose ``(p_o - p_e)/(1 -
    p_e)`` saturates at its -1 clip on sparse data (raw -9.1 baseline /
    -21.7 stim on oxm0818 trial 5 MT/MST) while the paper statistic stays
    near chance there.
    """
    mat = np.asarray(units_2d, dtype=np.float64)[:, w0:w1]
    spikes = ((mat[:, :-1] < threshold) & (mat[:, 1:] >= threshold)).astype(
        np.float64)
    n_units, n_time = spikes.shape
    samples_per_bin = int(fs_hz * (bin_ms / 1000.0)) or 1
    n_bins = n_time // samples_per_bin
    if n_units < 2 or n_bins == 0:
        return float("nan")
    binned = spikes[:, : n_bins * samples_per_bin].reshape(
        n_units, n_bins, samples_per_bin).max(axis=2)
    spikes_per_bin = binned.sum(axis=0)
    silences_per_bin = n_units - spikes_per_bin
    den_pi = max(1.0, float(n_units * (n_units - 1)))
    p_i = (spikes_per_bin * (spikes_per_bin - 1)
           + silences_per_bin * (silences_per_bin - 1)) / den_pi
    p_o = float(p_i.mean())
    p_spike = float(spikes_per_bin.sum() / max(1.0, n_units * n_bins))
    p_e = p_spike ** 2 + (1.0 - p_spike) ** 2
    return float((p_o - p_e) / max(1e-6, 1.0 - p_e))
