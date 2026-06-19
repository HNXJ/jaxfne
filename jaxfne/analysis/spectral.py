"""JAX-native spectral analysis and objectives for JAXFNE.

Docs: ``docs/api/index.md`` spectral-analysis section
(https://jaxfne.readthedocs.io/en/latest/api/) — update when this module changes.

Provides JIT-compilable, batchable functions for calculating PSDs, bandpower
profiles, and similarity objectives.
"""

from functools import partial
from typing import Dict
import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("fs",))
def spectrolaminar_psd_jax(
    signal: jax.Array,          # (n_trials, n_steps, n_contacts)
    fs: float = 1000.0,
    freqs: jax.Array | None = None,   # (n_freqs,)
) -> jax.Array:
    """Compute spectrolaminar PSD averaged across trials in a JAX-native way.

    Projects each contact's timeseries onto a bank of complex exponentials and
    averages the magnitude spectrum across trials — a JIT-compilable, batchable
    DFT at the requested target frequencies.

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

    Returns
    -------
    jax.Array
        PSD heatmap of shape ``(n_freqs, n_contacts)``.
    """
    if freqs is None:
        freqs = jnp.linspace(1.0, 150.0, 96, dtype=jnp.float32)
    n_trials, n_steps, n_contacts = signal.shape
    sample_idx = jnp.arange(n_steps, dtype=freqs.dtype)
    # basis: (n_freqs, n_steps)
    basis = jnp.exp(-1j * 2.0 * jnp.pi * (freqs[:, None] / fs) * sample_idx[None, :])
    # DFT projection: (n_freqs, n_steps) @ (n_trials, n_steps, n_contacts) -> (n_trials, n_freqs, n_contacts)
    spec = jnp.einsum("fs,tsc->tfc", basis, signal)
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
    band_power = jnp.sum(psd * mask_float[:, None], axis=0) / (sum_mask + 1e-12)
    # Normalize by the peak channel value
    profile = band_power / (jnp.max(band_power) + 1e-12)
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
    similarity = 100.0 * jnp.exp(-3.0 * mse_total)
    return jnp.clip(similarity, 0.0, 100.0)


# Expose batched vectorization paths
spectrolaminar_similarity_candidates_jax = jax.vmap(
    spectrolaminar_similarity_kernel_jax,
    in_axes=(0, 0, None, None)
)

spectrolaminar_similarity_candidates_seeds_jax = jax.vmap(
    jax.vmap(spectrolaminar_similarity_kernel_jax, in_axes=(0, 0, None, None)),
    in_axes=(0, 0, None, None)
)
