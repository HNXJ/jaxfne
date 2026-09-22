"""PSD, spectrogram, band-power, and laminar depth-profile visualization (Plotly)."""

from __future__ import annotations

import numpy as np

from ._common import require_plotly, field_proxy, contact_depths_mm, dt_ms


def _welch_psd(x: np.ndarray, fs_hz: float, freq_max_hz: float = 150.0):
    from ..core import welch_psd as _contract

    return _contract(x, fs_hz, nperseg=256, freq_max_hz=freq_max_hz)


def plot_psd(
    signals,
    *,
    signal_key: str = "lfp",
    contact: int | None = None,
    freq_max_hz: float = 150.0,
    title: str = "Power spectral density",
):
    """PSD of a laminar proxy signal, per contact (or one contact if given)."""
    require_plotly()
    import plotly.graph_objects as go

    sig = field_proxy(signals, signal_key)  # (n_steps, n_contacts)
    fs_hz = 1000.0 / dt_ms(signals)
    depths = contact_depths_mm(signals)

    contacts = [contact] if contact is not None else range(sig.shape[1])
    fig = go.Figure()
    for c in contacts:
        f, pxx = _welch_psd(sig[:, c], fs_hz, freq_max_hz)
        fig.add_trace(go.Scatter(x=f, y=pxx, mode="lines", name=f"depth={depths[c]:.3f}mm"))

    fig.update_layout(
        title=title, xaxis_title="frequency (Hz)", yaxis_title="power", yaxis_type="log"
    )
    return fig


def plot_spectrogram(
    signals,
    *,
    signal_key: str = "lfp",
    contact: int = 0,
    freq_max_hz: float = 150.0,
    title: str = "Spectrogram",
):
    """Time x frequency spectrogram of one contact's proxy signal."""
    require_plotly()
    import plotly.graph_objects as go
    from scipy.signal import spectrogram as sp_spectrogram

    sig = field_proxy(signals, signal_key)[:, contact]
    fs_hz = 1000.0 / dt_ms(signals)
    nperseg = min(128, sig.shape[0])
    f, t, sxx = sp_spectrogram(sig, fs=fs_hz, nperseg=nperseg, noverlap=nperseg // 2)
    keep = f <= freq_max_hz

    fig = go.Figure(
        data=go.Heatmap(
            z=np.log10(sxx[keep] + 1e-12),
            x=t * 1000.0,
            y=f[keep],
            colorscale="Viridis",
            colorbar=dict(title="log10 power"),
        )
    )
    fig.update_layout(title=title, xaxis_title="time (ms)", yaxis_title="frequency (Hz)")
    return fig


def plot_band_power(
    signals,
    *,
    signal_key: str = "lfp",
    bands: dict | None = None,
    title: str = "Band power vs depth",
):
    """Absolute mean in-band power per contact, vs depth (P5 canonical contract).

    Same numbers as :func:`jaxfne.vis.fields.bandpower`: Welch PSD
    (``nperseg=256``) averaged over in-band freqs, absolute proxy units —
    NOT normalized across depths, so a >2x vs shuffled comparison (tutorial
    07) remains meaningful. For the peak-normalized single-band view use
    :func:`plot_depth_profile`.
    """
    require_plotly()
    import plotly.graph_objects as go

    from ..core import inband_power_mean, welch_psd

    bands = bands or {"alpha_beta": (8.0, 25.0), "gamma": (40.0, 150.0)}
    sig = field_proxy(signals, signal_key)
    fs_hz = 1000.0 / dt_ms(signals)
    depths = contact_depths_mm(signals)
    order = np.argsort(depths)

    fig = go.Figure()
    for band_name, (lo, hi) in bands.items():
        powers = []
        for c in order:
            f, pxx = welch_psd(sig[:, c], fs_hz, nperseg=256)
            powers.append(float(inband_power_mean(pxx, f, lo, hi)))
        powers = np.asarray(powers)
        fig.add_trace(go.Scatter(x=depths[order], y=powers, mode="lines+markers", name=band_name))

    fig.update_layout(
        title=title, xaxis_title="depth (mm)", yaxis_title="mean in-band power (proxy a.u.)"
    )
    return fig


def plot_depth_profile(
    signals,
    *,
    signal_key: str = "lfp",
    freq_band: tuple = (8.0, 25.0),
    title: str = "Laminar depth profile",
):
    """Relative power within one frequency band, plotted vs laminar depth (the
    spectrolaminar-style single-band readout)."""
    require_plotly()
    import plotly.graph_objects as go

    sig = field_proxy(signals, signal_key)
    fs_hz = 1000.0 / dt_ms(signals)
    depths = contact_depths_mm(signals)
    order = np.argsort(depths)
    lo, hi = freq_band

    powers = []
    for c in order:
        f, pxx = _welch_psd(sig[:, c], fs_hz, freq_max_hz=max(hi, 150.0))
        sel = (f >= lo) & (f <= hi)
        powers.append(float(pxx[sel].sum()) if np.any(sel) else 0.0)
    powers = np.asarray(powers)
    peak = float(powers.max()) or 1.0

    fig = go.Figure(go.Scatter(x=powers / peak, y=depths[order], mode="lines+markers"))
    fig.add_hline(y=0.0, line_color="gray", line_width=1)
    fig.update_layout(
        title=title,
        xaxis_title=f"relative power ({lo:.0f}-{hi:.0f} Hz)",
        yaxis_title="depth (mm)",
        yaxis_autorange="reversed",
    )
    return fig
