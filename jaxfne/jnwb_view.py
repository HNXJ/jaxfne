"""Hand simulation output to jnwb in jnwb's conventions.

jnwb: data/signal -> processing -> analysis -> results.
jaxfne: model/signal -> implementation -> simulation -> results.

``to_jnwb`` turns one ``Signals`` into plain numpy arrays and metadata that
jnwb functions take directly: sampling rate in Hz, times in seconds, spikes
as per-unit spike-time arrays, time on axis 0. It reads only ``Signals`` and
its ``FieldOutput`` and does not import jnwb, so it works without jnwb
installed.

Field readouts stay proxies. ``lfp_proxy``, ``csd_proxy`` and ``phi_e_proxy``
carry ``field_level`` (``RELATIVE_PROXY`` until an explicit calibration) and
``field_calibration`` and are never relabelled as volts; jnwb operations
that require volts, such as ``current_source_density_1d``, are not valid on
a ``RELATIVE_PROXY`` view. ``contact_depths`` are passed as recorded, with no
unit attached.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np

# Bridges whose ``spikes`` mark every sample at or above a voltage threshold,
# not spike events. The trace bridges also record ``metadata['spike_threshold']``.
_LEVEL_SPIKE_BRIDGES = frozenset({"jaxley_hh_laminar_field"})


@dataclass(frozen=True)
class JnwbView:
    """One simulation run as jnwb inputs. Arrays are read-only numpy, time on axis 0."""

    fs_hz: float
    dt_ms: float
    t0_s: float
    n_steps: int
    spike_times_s: tuple[np.ndarray, ...]
    V_m: np.ndarray
    sources: Optional[np.ndarray]
    lfp_proxy: Optional[np.ndarray]
    csd_proxy: Optional[np.ndarray]
    phi_e_proxy: Optional[np.ndarray]
    contact_depths: Optional[np.ndarray]
    field_level: Optional[str]
    field_calibration: Any
    representation: Optional[str]
    neuron_metadata: Any
    metadata: dict[str, Any]

    @property
    def time_s(self) -> np.ndarray:
        """Sample times in seconds, float64, rebuilt from ``t0_s`` and ``dt_ms``."""
        return self.t0_s + np.arange(self.n_steps, dtype=np.float64) * (self.dt_ms / 1000.0)


def _array(x: Any) -> Optional[np.ndarray]:
    """Read-only numpy view; sub-32-bit floats (bfloat16, float16) widen exactly to float32."""
    if x is None:
        return None
    arr = np.asarray(x)
    if jnp.issubdtype(arr.dtype, jnp.floating) and arr.dtype.itemsize < 4:
        arr = arr.astype(np.float32)
    else:
        arr = arr.view()
    arr.flags.writeable = False
    return arr


def _resolve_dt_ms(time_ms: np.ndarray, metadata: dict[str, Any]) -> float:
    n = time_ms.shape[0]
    meta_dt = metadata.get("dt_ms")
    if meta_dt is not None:
        dt = float(meta_dt)
        if not (np.isfinite(dt) and dt > 0):
            raise ValueError(f"to_jnwb: metadata dt_ms must be positive and finite, got {meta_dt!r}")
    elif n < 2:
        raise ValueError("to_jnwb: one sample and no metadata dt_ms; the sampling rate is unknown")
    else:
        dt = (float(time_ms[-1]) - float(time_ms[0])) / (n - 1)
    if n < 2:
        return dt
    t = time_ms.astype(np.float64)
    if not np.all(np.isfinite(t)) or t[-1] <= t[0]:
        raise ValueError("to_jnwb: time_ms is not finite and increasing")
    # Each sample must sit within a quarter step of t0 + k*dt, widened to the
    # rounding the time axis dtype allows (a bfloat16 axis rounds 0.1 to 0.10009766).
    eps = float(jnp.finfo(time_ms.dtype).eps) if jnp.issubdtype(time_ms.dtype, jnp.floating) else 0.0
    tol = np.maximum(0.25 * dt, 4.0 * eps * np.abs(t))
    dev = np.abs(t - (t[0] + np.arange(n, dtype=np.float64) * dt))
    if np.any(dev > tol):
        k = int(np.argmax(dev - tol))
        raise ValueError(
            f"to_jnwb: time_ms is not uniform at dt_ms={dt}: sample {k} is {t[k]} ms, "
            f"expected {t[0] + k * dt} ms; the sampling rate is ambiguous"
        )
    return dt


def to_jnwb(signals: Any) -> JnwbView:
    """Return ``signals`` as a ``JnwbView``.

    The sampling rate comes from ``metadata['dt_ms']`` when present, else from
    the span of ``time_ms``, and every sample of ``time_ms`` must agree with
    it. ``t0`` is ``time_ms[0]``; a spike at step ``k`` has time
    ``t0 + k * dt`` (float64, seconds), so the rounding of later ``time_ms``
    samples does not reach spike times. A count of ``c`` at one step gives
    ``c`` equal times. The view copies ``metadata`` and exposes arrays
    read-only, so changing the view cannot change ``signals``.

    Raises:
        ValueError: ``V_m`` or ``spikes`` is not ``(n_steps, n_units)``; the
            time axis is not finite, increasing and uniform at ``dt_ms``;
            ``spikes`` holds values other than non-negative integers; or the
            spikes are threshold levels from a bridge, not events.
    """
    time_ms = np.asarray(signals.time_ms)
    spikes = np.asarray(signals.spikes)
    metadata = copy.deepcopy(dict(signals.metadata or {}))
    if time_ms.ndim != 1:
        raise ValueError(f"to_jnwb: time_ms must be 1D, got shape {time_ms.shape}")
    n = time_ms.shape[0]
    V_m = _array(signals.V_m)
    for name, arr in (("V_m", V_m), ("spikes", spikes)):
        if arr.ndim != 2 or arr.shape[0] != n:
            raise ValueError(
                f"to_jnwb: {name} must be (n_steps={n}, n_units), got shape {arr.shape}; "
                "trial batches convert one run at a time"
            )
    if spikes.shape != V_m.shape:
        raise ValueError(f"to_jnwb: spikes {spikes.shape} and V_m {V_m.shape} differ in shape")
    if metadata.get("spike_threshold") is not None or metadata.get("bridge") in _LEVEL_SPIKE_BRIDGES:
        raise ValueError(
            "to_jnwb: these spikes mark every sample at or above a voltage threshold, "
            "not spike events, so they have no spike times; derive events from V_m first"
        )

    dt_ms = _resolve_dt_ms(time_ms, metadata)
    t0_s = float(time_ms[0]) / 1000.0 if n else 0.0
    dt_s = dt_ms / 1000.0
    counts = spikes.astype(np.float64)
    if not np.all(np.isfinite(counts)) or np.any(counts < 0) or np.any(counts != np.rint(counts)):
        raise ValueError(
            "to_jnwb: spikes must be non-negative integer counts per step; "
            "a rate or probability readout has no spike times"
        )
    counts = counts.astype(np.int64)
    # A count c at step k becomes c spike times at t0 + k*dt.
    spike_times_s = tuple(
        _array(t0_s + np.repeat(np.arange(n, dtype=np.float64), counts[:, u]) * dt_s)
        for u in range(counts.shape[1])
    )

    field = signals.field
    return JnwbView(
        fs_hz=1000.0 / dt_ms,
        dt_ms=dt_ms,
        t0_s=t0_s,
        n_steps=n,
        spike_times_s=spike_times_s,
        V_m=V_m,
        sources=_array(signals.sources),
        lfp_proxy=None if field is None else _array(field.lfp_proxy),
        csd_proxy=None if field is None else _array(field.csd_proxy),
        phi_e_proxy=None if field is None else _array(field.phi_e_proxy),
        contact_depths=None if field is None else _array(field.contact_depths),
        field_level=None if field is None else str(field.epistemic_level),
        field_calibration=None if field is None else copy.deepcopy(field.calibration),
        representation=metadata.get("representation"),
        neuron_metadata=metadata.get("neuron_metadata"),
        metadata=metadata,
    )


@dataclass(frozen=True)
class JnwbTrials:
    """A trial batch as jnwb inputs: one ``JnwbView`` per trial on a shared time grid."""

    fs_hz: float
    dt_ms: float
    t0_s: float
    n_steps: int
    trial_ids: tuple[str, ...]
    condition_labels: tuple[Optional[str], ...]
    views: tuple[JnwbView, ...]

    def stack(self, name: str) -> np.ndarray:
        """``(n_trials, n_steps, ...)`` read-only array of view field ``name`` (trials on axis 0)."""
        parts = [getattr(v, name) for v in self.views]
        if any(p is None for p in parts):
            raise ValueError(f"to_jnwb_trials: {name!r} was not recorded in every trial")
        if not all(isinstance(p, np.ndarray) and p.shape[:1] == (self.n_steps,) for p in parts):
            raise ValueError(f"to_jnwb_trials: {name!r} is not a per-step array")
        out = np.stack(parts)
        out.flags.writeable = False
        return out


def to_jnwb_trials(batch: Any) -> JnwbTrials:
    """Return a ``TrialBatchResult`` as ``JnwbTrials``, trial order kept.

    Every trial is converted with ``to_jnwb`` and must share ``dt_ms``, ``t0``,
    ``n_steps`` and the unit count, so trial ``i`` of ``stack(...)`` is
    ``batch.results[i]``.

    Raises:
        ValueError: the batch is empty, a trial failed or has no signals (a
            dropped trial would shift every later index; filter the batch
            first), or the trials do not share one time grid and unit count.
    """
    results = tuple(batch.results)
    if not results:
        raise ValueError("to_jnwb_trials: empty batch")
    failed = [r.trial_id for r in results if not r.success or r.signals is None]
    if failed:
        raise ValueError(f"to_jnwb_trials: trials without signals {failed}; filter the batch first")
    views = tuple(to_jnwb(r.signals) for r in results)
    first = views[0]
    for r, v in zip(results, views):
        if (v.dt_ms, v.t0_s, v.n_steps, v.V_m.shape) != (first.dt_ms, first.t0_s, first.n_steps, first.V_m.shape):
            raise ValueError(
                f"to_jnwb_trials: trial {r.trial_id!r} has dt_ms={v.dt_ms}, t0_s={v.t0_s}, "
                f"V_m {v.V_m.shape}; trial {results[0].trial_id!r} has dt_ms={first.dt_ms}, "
                f"t0_s={first.t0_s}, V_m {first.V_m.shape}"
            )
    return JnwbTrials(
        fs_hz=first.fs_hz,
        dt_ms=first.dt_ms,
        t0_s=first.t0_s,
        n_steps=first.n_steps,
        trial_ids=tuple(str(r.trial_id) for r in results),
        condition_labels=tuple(r.condition_label for r in results),
        views=views,
    )


__all__ = ["JnwbTrials", "JnwbView", "to_jnwb", "to_jnwb_trials"]
