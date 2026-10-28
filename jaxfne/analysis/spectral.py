"""JAX-native spectral analysis and objectives for JAXFNE.

Docs: ``docs/reference/api/index.md`` spectral-analysis section
(https://jaxfne.readthedocs.io/en/latest/reference/api/): update when this module changes.

Provides JIT-compilable, batchable functions for calculating PSDs, bandpower
profiles, and similarity objectives.
"""

from functools import partial
from typing import Dict
import numpy as _np
from scipy.optimize import curve_fit as _curve_fit

import jax
import jax.numpy as jnp

# Zero-guard for band/peak ratios on empty bands; similarity score scale
# and decay for the spectrolaminar motif objective (percent scale).
_ZERO_GUARD = 1e-12
_SIMILARITY_SCALE = 100.0
_SIMILARITY_DECAY = 3.0


@partial(jax.jit, static_argnames=("fs", "window"))
def spectrolaminar_psd_jax(
    signal: jax.Array,          # (n_trials, n_steps, n_contacts)
    fs: float = 1000.0,
    freqs: jax.Array | None = None,   # (n_freqs,)
    window: str = "rectangular",
) -> jax.Array:
    """Compute spectrolaminar PSD averaged across trials in a JAX-native way.

    Projects each contact's timeseries onto a bank of complex exponentials and
    averages the magnitude spectrum across trials; a JIT-compilable, batchable
    DFT at the requested target frequencies.

    Window/normalization convention: ``window`` selects the taper applied to
    each timeseries before projection: ``"rectangular"`` (default, no taper;
    the historical behavior, bit-identical) or ``"hann"``. The projection uses
    an unnormalized rectangular DFT basis and divides by ``n_steps``; band
    profiles derived downstream are additionally peak-normalized per contact
    (see :func:`bandpower_jax`), so absolute amplitude scale is not preserved
    by ``bandpower_jax`` by construction.

    Parameters
    ----------
    signal : jax.Array
        Timeseries of shape ``(n_trials, n_steps, n_contacts)``.
    fs : float, default 1000.0
        Sampling frequency in Hz (static under JIT). For a ``dt_ms`` timestep,
        ``fs = 1000.0 / dt_ms``.
    freqs : jax.Array, optional
        Target frequencies of shape ``(n_freqs,)``. Default: 96 points linearly
        spaced over 1–150 Hz, the canonical spectrolaminar band.
    window : str, default "rectangular"
        Taper convention (static under JIT): ``"rectangular"`` or ``"hann"``.

    Returns
    -------
    jax.Array
        PSD heatmap of shape ``(n_freqs, n_contacts)``.
    """
    if window == "rectangular":
        taper = jnp.ones_like(signal)
    elif window == "hann":
        n_steps = jnp.shape(signal)[1]
        taper = jnp.hanning(n_steps).astype(signal.dtype)[None, :, None]
    else:
        raise ValueError(
            f"unsupported spectral window {window!r}; "
            "expected 'rectangular' or 'hann'"
        )
    tapered = signal * taper
    if freqs is None:
        freqs = jnp.linspace(1.0, 150.0, 96, dtype=jnp.float32)
    n_trials, n_steps, n_contacts = tapered.shape
    # Phase of sample n in cycles is (f * n mod fs) / fs. Formed as (f / fs) * n in float32 it
    # loses ~log2(n) bits. f * 2**k and its remainder mod fs are exact, so sum the remainders
    # over the set bits k of n; only that short sum and the final division round.
    n_bits = max(1, (n_steps - 1).bit_length())
    pow2 = (2.0 ** jnp.arange(n_bits)).astype(freqs.dtype)
    fs_t = jnp.asarray(fs, dtype=freqs.dtype)
    rem_k = jax.lax.rem(freqs[:, None] * pow2[None, :], fs_t)
    bits = ((jnp.arange(n_steps)[:, None] >> jnp.arange(n_bits)[None, :]) & 1).astype(freqs.dtype)
    rem_n = jnp.einsum("fk,sk->fs", rem_k, bits, precision=jax.lax.Precision.HIGHEST)
    cycles = jax.lax.rem(rem_n, fs_t) / fs_t
    # basis: (n_freqs, n_steps)
    basis = jnp.exp(-1j * 2.0 * jnp.pi * cycles)
    # DFT projection: (n_freqs, n_steps) @ (n_trials, n_steps, n_contacts) -> (n_trials, n_freqs, n_contacts)
    spec = jnp.einsum("fs,tsc->tfc", basis, tapered)
    # Average magnitude across trials, normalized by timesteps
    psd = jnp.mean(jnp.abs(spec), axis=0) / float(n_steps)
    return psd.astype(jnp.float32)


@jax.jit
def bandpower_jax(
    psd: jax.Array,            # (n_freqs, n_contacts)
    freqs: jax.Array,          # (n_freqs,)
    band_range: jax.Array,     # (2,) e.g. jnp.array([10.0, 25.0])
) -> jax.Array:
    """Compute average power within a frequency band normalized by channel max.

    Parameters
    ----------
    psd : jax.Array
        PSD heatmap of shape (n_freqs, n_contacts).
    freqs : jax.Array
        Frequencies of shape (n_freqs,).
    band_range : jax.Array
        Band frequency range [min_hz, max_hz] of shape (2,).

    Returns
    -------
    jax.Array
        Normalized band power profile of shape (n_contacts,).
    """
    mask = (freqs >= band_range[0]) & (freqs <= band_range[1])
    mask_float = mask.astype(psd.dtype)
    sum_mask = jnp.sum(mask_float)
    # Average power across the band frequencies
    band_power = jnp.sum(psd * mask_float[:, None], axis=0) / (sum_mask + _ZERO_GUARD)
    # Normalize by the peak channel value
    profile = band_power / (jnp.max(band_power) + _ZERO_GUARD)
    return profile


@jax.jit
def spectrolaminar_readout_kernel_jax(
    psd: jax.Array,                # (n_freqs, n_contacts)
    freqs: jax.Array,              # (n_freqs,)
    alpha_beta_range: jax.Array,   # (2,)
    gamma_range: jax.Array,        # (2,)
) -> Dict[str, jax.Array]:
    """Batchable readout kernel computing relative power and normalized band profiles.

    Parameters
    ----------
    psd : jax.Array
        PSD heatmap of shape (n_freqs, n_contacts).
    freqs : jax.Array
        Frequencies of shape (n_freqs,).
    alpha_beta_range : jax.Array
        Alpha/Beta frequency range [min_hz, max_hz] of shape (2,).
    gamma_range : jax.Array
        Gamma frequency range [min_hz, max_hz] of shape (2,).

    Returns
    -------
    dict
        Dictionary containing relative power, alpha_beta profile, and gamma profile.
    """
    depth_sum = jnp.sum(psd, axis=1, keepdims=True)
    relative_power = psd / (depth_sum + 1e-12)
    
    ab_profile = bandpower_jax(psd, freqs, alpha_beta_range)
    gm_profile = bandpower_jax(psd, freqs, gamma_range)
    
    return {
        "relative_power": relative_power,
        "alpha_beta": ab_profile,
        "gamma": gm_profile,
    }


@jax.jit
def spectrolaminar_similarity_kernel_jax(
    alpha_beta: jax.Array,        # (n_contacts,)
    gamma: jax.Array,             # (n_contacts,)
    target_alpha_beta: jax.Array, # (n_contacts,)
    target_gamma: jax.Array,      # (n_contacts,)
) -> jax.Array:
    """Compute the profile similarity score in JAX.

    Parameters
    ----------
    alpha_beta : jax.Array
        Alpha/Beta profile of shape (n_contacts,).
    gamma : jax.Array
        Gamma profile of shape (n_contacts,).
    target_alpha_beta : jax.Array
        Target Alpha/Beta profile of shape (n_contacts,).
    target_gamma : jax.Array
        Target Gamma profile of shape (n_contacts,).

    Returns
    -------
    jax.Array
        Similarity score scalar.
    """
    mse_ab = jnp.mean((alpha_beta - target_alpha_beta) ** 2)
    mse_gamma = jnp.mean((gamma - target_gamma) ** 2)
    mse_total = mse_ab + mse_gamma
    similarity = _SIMILARITY_SCALE * jnp.exp(-_SIMILARITY_DECAY * mse_total)
    return jnp.clip(similarity, 0.0, _SIMILARITY_SCALE)


# Expose batched vectorization paths
spectrolaminar_similarity_candidates_jax = jax.vmap(
    spectrolaminar_similarity_kernel_jax,
    in_axes=(0, 0, None, None)
)

spectrolaminar_similarity_candidates_seeds_jax = jax.vmap(
    jax.vmap(spectrolaminar_similarity_kernel_jax, in_axes=(0, 0, None, None)),
    in_axes=(0, 0, None, None)
)


# --- CAMINOS lane estimators (T-32d stage 2: generic core homes) ----------------


def welch_numpy(x, fs, nperseg, noverlap):
    """Welch power density with Hann windows (fallback, no scipy needed).

    Same convention as ``scipy.signal.welch`` with ``detrend=False`` and
    ``scaling="density"``: segments demeaned here, scale
    ``1 / (fs * sum(win**2))``, one-sided doubling off DC/Nyquist.
    """
    x = _np.asarray(x, dtype=float)
    step = nperseg - noverlap
    win = _np.hanning(nperseg)
    scale = 1.0 / (fs * float(_np.sum(win**2)))
    acc = None
    n = 0
    for start in range(0, len(x) - nperseg + 1, step):
        seg = x[start:start + nperseg]
        seg = seg - seg.mean()
        spec = _np.fft.rfft(win * seg)
        pxx = (spec.real**2 + spec.imag**2) * scale
        if nperseg % 2 == 0:
            pxx[1:-1] *= 2.0
        else:
            pxx[1:] *= 2.0
        acc = pxx if acc is None else acc + pxx
        n += 1
    if n == 0:
        raise ValueError("signal shorter than one window: %d < %d" % (len(x), nperseg))
    return _np.fft.rfftfreq(nperseg, 1.0 / fs), acc / n


def welch_spectrum(x, fs, nperseg, noverlap):
    """Welch density via scipy when present, else :func:`welch_numpy`."""
    try:
        from scipy.signal import welch as _welch
    except ImportError:
        return welch_numpy(x, fs, nperseg, noverlap)
    return _welch(x, fs=fs, window="hann", nperseg=nperseg,
                  noverlap=noverlap, detrend=False, scaling="density")


def aperiodic_log10(f, a, k, chi):
    """Aperiodic model in log10 power: log10 P = a - log10(k + f**chi)."""
    return a - _np.log10(k + _np.power(f, chi))


def fit_aperiodic(freqs, pxx, keep, *, k_min=0.0, chi_bounds=(0.0, 5.0),
                  init_k=1.0, init_chi=1.0):
    """Bounded least-squares fit of the aperiodic model on the bins in ``keep``.

    Bounds: k >= k_min, chi in chi_bounds, a free. Start: k0 = init_k,
    chi0 = init_chi, and a0 set so the model equals the median log10 power
    at the median fitted frequency. Returns ``(a, k, chi)``.
    """
    f = freqs[keep]
    y = _np.log10(pxx[keep])
    a0 = float(_np.median(y) + _np.log10(init_k + _np.median(f) ** init_chi))
    popt, _ = _curve_fit(aperiodic_log10, f, y, p0=[a0, init_k, init_chi],
                         bounds=([-_np.inf, k_min, chi_bounds[0]],
                                 [_np.inf, _np.inf, chi_bounds[1]]),
                         maxfev=20000)
    return tuple(float(v) for v in popt)


def rate_welch_psd(rate, dt, *, win_s=1.0, overlap_frac=0.5):
    """Welch PSD of a rate trace: mean removed, Hann windows, fractional overlap."""
    x = _np.asarray(rate, dtype=float)
    x = x - x.mean()
    fs = 1.0 / float(dt)
    nperseg = int(round(float(win_s) * fs))
    return welch_spectrum(x, fs, nperseg, int(nperseg * float(overlap_frac)))


def aperiodic_spectral_peak(freqs, pxx, *, fit_lo_hz=10.0, fit_hi_hz=150.0,
                            band_lo_hz=20.0, band_hi_hz=80.0, excl_half_hz=8.0,
                            edge_hz=2.0):
    """Peak over the fitted aperiodic background, without the oscillation call.

    Returns a dict: ``peak_freq_Hz`` (f_pk), ``peak_excess`` (PE),
    ``background_Hz``, ``local_max``, ``edge_ok``, ``fit`` (refit) and
    ``fit_pass1`` (first fit).
    """
    fit_band = (freqs >= fit_lo_hz) & (freqs <= fit_hi_hz)
    pk_idx = _np.flatnonzero((freqs >= band_lo_hz) & (freqs <= band_hi_hz))
    if pk_idx.size == 0:
        raise ValueError("no frequency bins in %g-%g Hz" % (band_lo_hz, band_hi_hz))
    pars1 = fit_aperiodic(freqs, pxx, fit_band)
    resid = _np.log10(pxx) - aperiodic_log10(freqs, *pars1)
    i = int(pk_idx[_np.argmax(resid[pk_idx])])
    f_pk = float(freqs[i])
    pars2 = fit_aperiodic(freqs, pxx,
                          fit_band & (_np.abs(freqs - f_pk) > excl_half_hz))
    background = float(10.0 ** aperiodic_log10(f_pk, *pars2))
    pe = float(pxx[i]) / background
    local_max = bool(pxx[i] > pxx[i - 1] and pxx[i] > pxx[i + 1])
    edge_ok = bool(abs(f_pk - band_lo_hz) > edge_hz
                   and abs(f_pk - band_hi_hz) > edge_hz)
    return {
        "peak_freq_Hz": f_pk,
        "peak_excess": pe,
        "background_Hz": background,
        "local_max": local_max,
        "edge_ok": edge_ok,
        "fit": pars2,
        "fit_pass1": pars1,
    }


def oscillation_call(peak_excess, edge_ok, pe_threshold=4.0):
    """Oscillation rule: PE >= pe_threshold with f_pk off the band edges."""
    return bool(peak_excess >= pe_threshold and edge_ok)


def _ref42_window(n_steps, dt_s, drop_s=2.0, keep_s=10.0):
    """Row bounds (first, end) of the analysed window."""
    fs = 1.0 / float(dt_s)
    i0 = int(round(float(drop_s) * fs))
    i1 = i0 + int(round(float(keep_s) * fs))
    if n_steps < i1:
        raise ValueError("run has %d steps; the ref. 42 window needs %d" % (n_steps, i1))
    return i0, i1


def ref42_lfp(counts, dt_s, *, drop_s=2.0, keep_s=10.0, lfp_sigma_ms=1.0):
    """LFP proxy of the ref. 42 method: E spikes per bin, Gaussian-smoothed.

    The smoothing runs on the whole run, so the window edges are not
    truncated kernels.
    """
    c = _np.asarray(counts)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError("counts must be [steps, 3], got shape %r" % (c.shape,))
    fs = 1.0 / float(dt_s)
    sigma = float(lfp_sigma_ms) * 1e-3 * fs
    half = int(_np.ceil(4.0 * sigma))
    t = _np.arange(-half, half + 1, dtype=float)
    kernel = _np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()
    lfp = _np.convolve(c[:, 0].astype(float), kernel, mode="same")
    i0, i1 = _ref42_window(c.shape[0], dt_s, drop_s, keep_s)
    return lfp[i0:i1]


def ref42_phase_delay(counts, dt_s, f_peak_hz, half_band_hz=5.0, *, drop_s=2.0,
                      keep_s=10.0, lfp_sigma_ms=1.0, butter_order=4):
    """PV->SOM delay by the ref. 42 phase method, from counts in [E | PV | SOM] order.

    Returns a dict: delay_ms, phase_PV_rad, phase_SOM_rad, n_spikes_PV,
    n_spikes_SOM, f_peak_Hz, half_band_Hz. Positive when SOM fires after PV;
    PV or SOM with no spikes in the window gives ``delay_ms`` None.
    """
    from scipy import signal as _signal

    c = _np.asarray(counts)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError("counts must be [steps, 3], got shape %r" % (c.shape,))
    f_peak = float(f_peak_hz)
    half = float(half_band_hz)
    fs = 1.0 / float(dt_s)
    if not (0.0 < half < f_peak and f_peak + half < fs / 2.0):
        raise ValueError("band must satisfy 0 < half_band_hz < f_peak_hz and "
                         "f_peak_hz + half_band_hz < Nyquist (got half=%r, f=%r)" % (half, f_peak))
    x = ref42_lfp(c, dt_s, drop_s=drop_s, keep_s=keep_s, lfp_sigma_ms=lfp_sigma_ms)
    sos = _signal.butter(int(butter_order),
                         [(f_peak - half) / (fs / 2.0), (f_peak + half) / (fs / 2.0)],
                         btype="bandpass", output="sos")
    phase = _np.angle(_signal.hilbert(_signal.sosfiltfilt(sos, x)))
    i0, i1 = _ref42_window(c.shape[0], dt_s, drop_s, keep_s)
    w = c[i0:i1].astype(float)

    def group_phase(col):
        n = w[:, col].sum()
        if n <= 0:
            return None, 0
        return float(_np.angle(_np.sum(w[:, col] * _np.exp(1j * phase)))), int(round(n))

    pv_phi, n_pv = group_phase(1)
    som_phi, n_som = group_phase(2)
    delay_ms = None
    if pv_phi is not None and som_phi is not None:
        d = float(_np.angle(_np.exp(1j * (som_phi - pv_phi))))
        delay_ms = d / (2.0 * _np.pi * f_peak) * 1e3
    return {"delay_ms": delay_ms, "phase_PV_rad": pv_phi, "phase_SOM_rad": som_phi,
            "n_spikes_PV": n_pv, "n_spikes_SOM": n_som,
            "f_peak_Hz": f_peak, "half_band_Hz": half}
