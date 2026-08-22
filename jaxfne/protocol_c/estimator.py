"""Traveling-wave estimator W_hat for Protocol C (synthetic and field readouts)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import signal

from jaxfne.protocol_c.protocol import load_protocol_spec


@dataclass(frozen=True)
class WaveEstimate:
    """Structured estimator output (downstream receipt contract)."""

    classification: str
    frequency_hz: float
    wave_vector: np.ndarray
    direction: np.ndarray
    phase_velocity: float
    phase_fit_r2: float
    spatial_coherence: float
    null_score: float
    quality_reasons: tuple[str, ...]
    finite_status: bool
    errors: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        k = np.asarray(self.wave_vector, dtype=np.float64)
        d = np.asarray(self.direction, dtype=np.float64)
        return {
            "classification": self.classification,
            "frequency_hz": float(self.frequency_hz),
            "wave_vector": k.tolist(),
            "direction": d.tolist(),
            "phase_velocity": float(self.phase_velocity),
            "phase_fit_r2": float(self.phase_fit_r2),
            "spatial_coherence": float(self.spatial_coherence),
            "null_score": float(self.null_score),
            "quality_reasons": list(self.quality_reasons),
            "finite_status": bool(self.finite_status),
            "errors": dict(self.errors),
        }


def _bandpass_rows(phi: np.ndarray, fs_hz: float, band: tuple[float, float]) -> np.ndarray:
    lo, hi = band
    nyq = 0.5 * fs_hz
    low = max(lo / nyq, 1e-6)
    high = min(hi / nyq, 0.999)
    if low >= high:
        raise ValueError(f"invalid band {band} for fs={fs_hz}")
    b, a = signal.butter(4, [low, high], btype="band")
    out = np.zeros_like(phi, dtype=np.float64)
    for i in range(phi.shape[1]):
        out[:, i] = signal.filtfilt(b, a, phi[:, i])
    return out


def _estimate_frequency_hz(phi_bp: np.ndarray, fs_hz: float, band: tuple[float, float]) -> float:
    """Dominant in-band frequency from summed site spectral power.

    P2V repair (A-1a, 2026-08-15): the prior implementation argmaxed the
    power spectrum of the spatial mean trace. On cyclic geometries (the C3
    ring) integer spatial modes have vanishing spatial mean, so the mean
    trace is ~0 and the argmax returns a degenerate bin, corrupting f_hat
    and every downstream quantity. Summed site power (total spectral power)
    is identical to the mean-trace spectrum for synchronous fields (all
    sites in phase) and remains well-defined for cancelling ring modes.
    """
    freqs = np.fft.rfftfreq(phi_bp.shape[0], d=1.0 / fs_hz)
    spec = np.sum(np.abs(np.fft.rfft(phi_bp, axis=0)) ** 2, axis=1)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return float("nan")
    idx = np.argmax(spec[mask])
    return float(freqs[mask][idx])


def _phasor_at_frequency(phi_bp: np.ndarray, fs_hz: float, f_hz: float) -> np.ndarray:
    """Per-site complex phasor at f_hz via narrow DFT projection."""
    t = np.arange(phi_bp.shape[0], dtype=np.float64) / fs_hz
    carrier = np.exp(-1j * 2.0 * np.pi * f_hz * t)
    return np.sum(phi_bp * carrier[:, None], axis=0)


def _neighbor_phase_coherence(phi_bp: np.ndarray, positions: np.ndarray) -> float:
    """Mean resultant length of wrapped neighbor phase differences (time × pairs)."""
    from scipy.spatial import cKDTree

    analytic = signal.hilbert(phi_bp, axis=0)
    phases = np.angle(analytic)
    r = np.asarray(positions, dtype=np.float64)
    n = r.shape[0]
    if n < 2:
        return 0.0
    pair_phases: list[np.ndarray] = []
    if r.shape[1] == 1:
        order = np.argsort(r[:, 0])
        ph = phases[:, order]
        dph = (np.diff(ph, axis=1) + np.pi) % (2.0 * np.pi) - np.pi
        pair_phases.append(dph.ravel())
    else:
        tree = cKDTree(r)
        for i in range(n):
            _, idx = tree.query(r[i], k=2)
            j = int(idx[1])
            if j <= i:
                continue
            dph = (phases[:, j] - phases[:, i] + np.pi) % (2.0 * np.pi) - np.pi
            pair_phases.append(dph)
    if not pair_phases:
        return 0.0
    z = np.exp(1j * np.concatenate(pair_phases))
    return float(np.abs(np.mean(z)))


def _fit_wave_vector(phasors: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate k from nearest-neighbor wrapped phase differences (translation invariant)."""
    from scipy.spatial import cKDTree

    r = np.asarray(positions, dtype=np.float64)
    r = r - r.mean(axis=0, keepdims=True)
    z = np.asarray(phasors, dtype=np.complex128)
    n = r.shape[0]
    if n < 2:
        return np.zeros(r.shape[1], dtype=np.float64), 0.0
    rows: list[np.ndarray] = []
    targets: list[float] = []
    tree = cKDTree(r)
    for i in range(n):
        _, idx = tree.query(r[i], k=min(4, n))
        idx_arr = np.atleast_1d(idx)
        for j in idx_arr:
            j = int(j)
            if j == i:
                continue
            dr = r[j] - r[i]
            if float(np.linalg.norm(dr)) < 1e-12:
                continue
            dphi = float(np.angle(z[j] * np.conj(z[i])))
            dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi
            rows.append(dr)
            targets.append(dphi)
    if not rows:
        return np.zeros(r.shape[1], dtype=np.float64), 0.0
    design = np.asarray(rows, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    k_fit, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ k_fit
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2) + 1e-12)
    r2 = 1.0 - ss_res / ss_tot
    return np.asarray(k_fit, dtype=np.float64), float(r2)


def _k_stability_score(phi_bp: np.ndarray, positions: np.ndarray, fs_hz: float, f_hz: float) -> float:
    """Cosine similarity of k estimates across time halves (standing waves flip)."""
    mid = phi_bp.shape[0] // 2
    if mid < 4:
        return 1.0
    ks = []
    for sl in (slice(0, mid), slice(mid, None)):
        z = _phasor_at_frequency(phi_bp[sl], fs_hz, f_hz)
        k, _ = _fit_wave_vector(z, positions)
        kn = np.linalg.norm(k)
        if kn > 1e-9:
            ks.append(k / kn)
    if len(ks) < 2:
        return 1.0
    return float(np.dot(ks[0], ks[1]))


def estimate_traveling_wave(
    phi: np.ndarray,
    positions: np.ndarray,
    *,
    dt_ms: float,
    spec: dict[str, Any] | None = None,
    ground_truth: dict[str, Any] | None = None,
) -> WaveEstimate:
    """Estimate W_hat from a spatiotemporal field Phi (T, N)."""
    spec = spec or load_protocol_spec()
    params = spec["preregistered_estimator_parameters"]
    band = tuple(params["frequency_band_hz"])
    min_coh = float(params["minimum_spatial_coherence"])
    r2_thr = float(params["R2_phase_traveling_threshold"])
    coh_thr = float(params["coherence_traveling_threshold"])

    phi = np.asarray(phi, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    reasons: list[str] = []

    if phi.ndim != 2:
        raise ValueError(f"phi must be (T, N); got {phi.shape}")
    if positions.shape[0] != phi.shape[1]:
        raise ValueError("positions rows must match phi sites")
    if positions.shape[0] < 4:
        return WaveEstimate(
            classification="UNRESOLVED",
            frequency_hz=float("nan"),
            wave_vector=np.full(positions.shape[1], np.nan),
            direction=np.full(positions.shape[1], np.nan),
            phase_velocity=float("nan"),
            phase_fit_r2=0.0,
            spatial_coherence=0.0,
            null_score=1.0,
            quality_reasons=("fewer_than_4_spatial_samples",),
            finite_status=False,
        )

    fs_hz = 1000.0 / float(dt_ms)
    phi_bp = _bandpass_rows(phi, fs_hz, band)
    band_power = float(np.mean(phi_bp**2))
    noise_floor = float(params.get("noise_band_power_floor", 1e-4))

    f_hat = _estimate_frequency_hz(phi_bp, fs_hz, band)
    phasors = _phasor_at_frequency(phi_bp, fs_hz, f_hat)
    coherence = _neighbor_phase_coherence(phi_bp, positions)
    k_hat, r2 = _fit_wave_vector(phasors, positions)
    k_norm = float(np.linalg.norm(k_hat))
    omega_hat = 2.0 * np.pi * f_hat
    v_hat = float(omega_hat / k_norm) if k_norm > 1e-9 else float("nan")
    direction = k_hat / k_norm if k_norm > 1e-9 else np.zeros_like(k_hat)

    extent = float(np.max(np.linalg.norm(positions - positions.mean(axis=0), axis=1)) + 1e-9)
    # Declared heuristic (preregistered c0 spec scale): k_min is one-eighth of
    # the smallest resolvable half-wave across the array extent; not fitted.
    k_min = np.pi / extent / 8.0
    k_stability = _k_stability_score(phi_bp, positions, fs_hz, f_hat)

    null_score = float(max(0.0, 1.0 - coherence))

    errors: dict[str, float] = {}
    if ground_truth is not None:
        f_true = float(ground_truth.get("frequency_hz", f_hat))
        k_true = np.asarray(ground_truth.get("k_vector", k_hat), dtype=np.float64).reshape(-1)
        v_true = float(ground_truth.get("phase_velocity", omega_hat / (np.linalg.norm(k_true) + 1e-12)))
        k_cmp = np.asarray(k_hat, dtype=np.float64)
        if np.dot(k_cmp, k_true) < 0.0 and np.linalg.norm(k_true) > 1e-12:
            k_cmp = -k_cmp
        errors["epsilon_f"] = float(abs(f_hat - f_true))
        kn_true = float(np.linalg.norm(k_true))
        kn_cmp = float(np.linalg.norm(k_cmp))
        errors["epsilon_k"] = float(abs(kn_cmp - kn_true) / (kn_true + 1e-12)) if kn_true > 1e-12 else float(kn_cmp)
        if kn_true > 1e-12 and kn_cmp > 1e-9:
            cosang = float(np.clip(np.dot(k_cmp, k_true) / (kn_cmp * kn_true), -1.0, 1.0))
            errors["epsilon_theta_rad"] = float(np.arccos(cosang))
            errors["epsilon_theta_deg"] = float(np.degrees(errors["epsilon_theta_rad"]))
        else:
            errors["epsilon_theta_rad"] = 0.0 if k_norm < k_min else float("inf")
            errors["epsilon_theta_deg"] = float(np.degrees(errors["epsilon_theta_rad"]))
        errors["epsilon_v"] = float(abs(v_hat - v_true) / (abs(v_true) + 1e-12))

    finite = bool(
        np.isfinite(f_hat)
        and np.all(np.isfinite(k_hat))
        and np.isfinite(coherence)
        and np.isfinite(r2)
    )

    if band_power < noise_floor:
        reasons.append("noise_only_power_below_floor")
        cls = "NO_WAVE"
    elif not finite:
        reasons.append("non_finite_k_or_f")
        cls = "UNRESOLVED"
    elif k_norm < k_min:
        reasons.append("synchronous_oscillation_k_near_zero")
        cls = "NO_WAVE"
    elif k_stability < 0.0:
        reasons.append("standing_or_flipping_spatial_gradient")
        cls = "NO_WAVE"
    elif coherence < coh_thr or r2 < r2_thr:
        reasons.append("structured_but_fails_traveling_gates")
        cls = "NO_WAVE"
    else:
        reasons.append("planar_phase_gradient_gates_passed")
        cls = "TRAVELING_WAVE"

    return WaveEstimate(
        classification=cls,
        frequency_hz=float(f_hat),
        wave_vector=k_hat,
        direction=direction,
        phase_velocity=v_hat,
        phase_fit_r2=float(r2),
        spatial_coherence=float(coherence),
        null_score=null_score,
        quality_reasons=tuple(reasons),
        finite_status=finite,
        errors=errors,
    )
