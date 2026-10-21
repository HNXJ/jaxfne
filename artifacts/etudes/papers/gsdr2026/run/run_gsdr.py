"""GSDR Fig. 6 baseline on the jaxfne core HH route (P3 runner, PORT_SPEC.md).

Builds the 36 E / 8 Ig / 6 Il circuit only from the jaxfne public API,
constructs it with ``jaxfne._construct_hh.construct_hh_model`` (standalone;
NOT wired into ``construct()`` dispatch), and runs it through
``jaxfne.emitters_hh.simulate_hh``. No scored runs: outputs land in the
caller's directory (``scratch/gsdr2026/``), never under a ``scored`` path.

Every biophysical value comes from ``PORT_SPEC.md``; assumed values carry
their A-label. Units: mV, ms, mS/cm2. Lane runs are float32 on CUDA
(deviation D7); stimulus adds as depolarizing inward-positive (D9).

Baseline runs the Fig. 6 batch amps ``(0.0, 0.6, 0.6)`` and reports model
PSDs / rates / kappa. The Fig. 6 loss needs label spectra from data (none
in the lane), so no loss is scored; ``fig6_loss``/``mcdp_factors`` below are
the GSDR-training skeleton for P4+, kept exact to PORT_SPEC.md section 9-10.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent

_repo_root = Path(__file__).resolve().parents[5]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# PORT_SPEC.md sections 1, 7-9 (code-observed Fig. 6 run values).
N_E, N_IG, N_IL = 36, 8, 6
N_CELLS = N_E + N_IG + N_IL
N_AMPA_EDGES = N_E * N_CELLS  # 1800
N_GABAA_GLOBAL_EDGES = N_IG * N_CELLS  # 400
N_GABAA_LOCAL_EDGES = N_IL * 5  # 30
LOCAL_POST_COUNT = 5
E_STIM_CELLS = list(range(0, N_E, 2))  # 18 every-other-E cells (D3)

DT_MS = 0.1
T_MAX_MS = 1500.0
N_STEPS = int(round(T_MAX_MS / DT_MS)) + 1  # 15001
STIM_ON_MS, STIM_OFF_MS = 500.0, 1000.0
STIM_FREQ_HZ = 120.0
STIM_AMP = 0.6  # code run value (paper text: 1 nA, D1)
BATCH_AMPS = (0.0, STIM_AMP, STIM_AMP)

PSD_WEIGHT = 990.0
RATE_WEIGHT = 9.0
RATE_LO, RATE_HI = 5.0, 50.0
PSD_EPS = 1e-6

# Reference-unit conversions (PORT_SPEC.md fidelity record; sources read as data).
#
# Each reference cell is 2 branches x ncomp=2 compartments of radius 1 um and
# length 10 um: the ``cell.length = 1.0`` assignments are plain-attribute
# no-ops (jaxley defines no ``__setattr__``), so the 10 um default stands and
# each compartment covers 2*pi*1*10 = 20pi um2, the cell 80pi um2.
# The lane is one compartment per cell, so point-process inputs convert with
# the whole-cell area (``convert_point_process_to_distributed``):
# density = value / AREA_CELL_UM2 * 1e5.
AREA_CELL_UM2 = 80.0 * math.pi
# Stimulus nA -> uA/cm2; synapse conductance uS -> mS/cm2 (same factor).
NA_TO_UACM2 = 1e5 / AREA_CELL_UM2
US_TO_MSCM2 = 1e5 / AREA_CELL_UM2
# OU channel current mA/cm2 -> uA/cm2 (the reference ``*1000``).
MA_TO_UA = 1000.0


def target_grid():
    import numpy as np

    return np.linspace(15.0, 60.0, 40)


def build_spec(local_seed=1, edge_seed=0, weight_unit="density"):
    """The 50-cell Fig. 6 circuit built only from the public API (PORT_SPEC.md).

    ``weight_unit`` is ``"density"`` (the default: connection weights in
    mS/cm2, converted from the reference uS with ``US_TO_MSCM2``) or
    ``"uS"`` (reference point-process conductances, for the
    multi-compartment reference morphology).
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    import jaxfne as jtfne

    if weight_unit not in ("density", "uS"):
        raise ValueError(
            f"weight_unit must be 'density' or 'uS'; got {weight_unit!r}")
    scale = US_TO_MSCM2 if weight_unit == "density" else 1.0

    posts = np.asarray(
        jax.random.choice(jax.random.PRNGKey(int(local_seed)), jnp.arange(N_CELLS),
                          shape=(LOCAL_POST_COUNT,), replace=False)
    )
    cfg = (
        jtfne.Configuration()
        .network(name="gsdr50", n=N_CELLS,
                 cell_types={"E": N_E / N_CELLS, "Ig": N_IG / N_CELLS,
                             "Il": N_IL / N_CELLS})
        .mechanisms(name="ampa", kind="receptor",
                    params={"tau_ms": 2.0, "reversal_mV": 0.0, "tau_r_ms": 0.2,
                            "slope_mV": 5.0, "v_th_mV": -20.0, "s0": 0.1})
        .mechanisms(name="gaba_global", kind="receptor",
                    params={"tau_ms": 5.0, "reversal_mV": -80.0, "tau_r_ms": 0.5,
                            "slope_mV": 5.0, "v_th_mV": -20.0, "s0": 0.1})
        .mechanisms(name="gaba_local", kind="receptor",
                    params={"tau_ms": 2.0, "reversal_mV": -80.0, "tau_r_ms": 0.5,
                            "slope_mV": 5.0, "v_th_mV": -20.0, "s0": 0.1})
        # Reference conductances are uS (point processes); in the density
        # default the kernel takes densities, converted with US_TO_MSCM2
        # (2.5/5.0 uS reference scale).
        .connections(name="e_all", source={"cell_type": "E"}, target={},
                     probability=None, weight=2.5 * scale, mechanism="ampa",
                     sign="excitatory")
        .connections(name="ig_all", source={"cell_type": "Ig"}, target={},
                     probability=None, weight=5.0 * scale, mechanism="gaba_global",
                     sign="inhibitory")
        .connections(name="il_local", source={"cell_type": "Il"},
                     target={"ids": [int(i) for i in posts.tolist()]},
                     probability=None, weight=5.0 * scale, mechanism="gaba_local",
                     sign="inhibitory")
    )
    return cfg.connectivity(edge_seed=int(edge_seed)), [int(i) for i in posts.tolist()]


def init_conductances(key, weight_unit="density"):
    """PORT_SPEC.md section 6: gAMPA ~ U(-4,10)->[0,8], gGABAa ~ U(2,10)->[0,10].

    Drawn in the reference uS scale; in the density default converted to
    mS/cm2 (US_TO_MSCM2), in ``"uS"`` returned as drawn.
    """
    import jax
    import jax.numpy as jnp

    if weight_unit not in ("density", "uS"):
        raise ValueError(
            f"weight_unit must be 'density' or 'uS'; got {weight_unit!r}")
    scale = US_TO_MSCM2 if weight_unit == "density" else 1.0
    k1, k2 = jax.random.split(key)
    g_ampa = jnp.clip(jax.random.uniform(k1, (N_AMPA_EDGES,), minval=-4.0, maxval=10.0),
                      0.0, 8.0)
    g_gaba = jnp.clip(jax.random.uniform(k2, (N_GABAA_GLOBAL_EDGES + N_GABAA_LOCAL_EDGES,),
                                         minval=2.0, maxval=10.0),
                      0.0, 10.0)
    return (jnp.concatenate([g_ampa, g_gaba]) * scale).astype(jnp.float32)


def clip_conductances(g, lo=0.0, hi=10.0):
    """Optimizer clamp transform (both families to [0, 10]).

    Bounds are in the reference uS scale; the P4+ trainer scales them by
    US_TO_MSCM2 when it converts to densities.
    """
    import jax.numpy as jnp

    return jnp.clip(g, lo, hi)


def ou_hyperparams(key):
    """Per-cell OU tau ~ U(10, 50), sigma = clip(U(-0.1, 0.1), 0, 0.05) (D5 run values).

    The reference amplitudes are channel-current mA/cm2; converted to the
    lane uA/cm2 (MA_TO_UA).
    """
    import jax
    import jax.numpy as jnp

    k1, k2 = jax.random.split(key)
    tau = jax.random.uniform(k1, (N_CELLS,), minval=10.0, maxval=50.0)
    sigma = jnp.clip(jax.random.uniform(k2, (N_CELLS,), minval=-0.1, maxval=0.1),
                     0.0, 0.05) * MA_TO_UA
    return tau.astype(jnp.float32), sigma.astype(jnp.float32)


def stimulus_matrix(stim_amp, n_steps=N_STEPS, dt_ms=DT_MS, unit="density"):
    """Gated 120 Hz sine on [500, 1000] ms over the 18 every-other-E cells (D8).

    ``stim_amp`` is the reference nA amplitude; in the density default it is
    converted to uA/cm2 (NA_TO_UACM2) for the density kernel, in ``"nA"``
    returned as drawn for the point-process reference morphology.
    """
    import numpy as np

    if unit not in ("density", "nA"):
        raise ValueError(f"unit must be 'density' or 'nA'; got {unit!r}")
    scale = NA_TO_UACM2 if unit == "density" else 1.0
    t = np.arange(n_steps, dtype=np.float32) * np.float32(dt_ms)
    mask = (t >= np.float32(STIM_ON_MS)) & (t <= np.float32(STIM_OFF_MS))
    wave = np.asarray(float(stim_amp) * scale, dtype=np.float32) * np.sin(
        2.0 * np.pi * np.float32(STIM_FREQ_HZ) * (t / 1000.0))
    col = np.where(mask, wave, 0.0).astype(np.float32)
    mat = np.zeros((n_steps, N_CELLS), dtype=np.float32)
    mat[:, np.asarray(E_STIM_CELLS)] = col[:, None]
    return mat


def detect_spikes(v_trace, threshold=-20.0):
    """Upward threshold crossings; (time, cells) -> bool (time-1, cells)."""
    import numpy as np

    v = np.asarray(v_trace)
    return (v[:-1] < threshold) & (v[1:] >= threshold)


def firing_rates_hz(v_trace, t_max_ms=T_MAX_MS, threshold=-20.0):
    """Per-cell rates from upward -20 mV crossings (A5) over t_max seconds."""
    import numpy as np

    spikes = detect_spikes(np.asarray(v_trace), threshold)
    return spikes.sum(axis=0) / (t_max_ms / 1000.0)


def rate_penalty(firing_rates, lo=RATE_LO, hi=RATE_HI):
    """One-sided penalty: mean(exp(lo-F) + exp(F-hi))."""
    import numpy as np

    f = np.asarray(firing_rates, dtype=np.float64)
    return float(np.mean(np.exp(lo - f) + np.exp(f - hi)))


def psd_mean_voltage(v_trace, dt_ms=DT_MS, t_l_ms=100.0, t_r_ms=1400.0):
    """Max-normalized PSD of the mean cell voltage in [t_l, t_r] (A6 window)."""
    import numpy as np

    v = np.asarray(v_trace, dtype=np.float64)
    i_l, i_r = int(t_l_ms / dt_ms), int(t_r_ms / dt_ms)
    signal = v[i_l:i_r, :].mean(axis=1)
    n = signal.shape[0]
    fs = 1000.0 / dt_ms
    power = np.abs(np.fft.rfft(signal)) ** 2 / (n * fs)
    freqs = np.fft.rfftfreq(n, d=dt_ms / 1000.0)
    interp = np.interp(np.asarray(target_grid(), dtype=np.float64), freqs, power)
    return interp / (interp.max() + 1e-6)


def psd_loss(pred, label, eps=PSD_EPS):
    """Label-weighted squared error (training skeleton; needs data labels)."""
    import numpy as np

    pred = np.asarray(pred, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    return float(np.sum((label * ((pred + eps) - (label + eps))) ** 2))


def fig6_loss(v_batch, labels):
    """main.py loss over a batch (training skeleton; P4+ with data labels)."""
    import numpy as np

    psds = [psd_loss(psd_mean_voltage(v), lab) for v, lab in zip(v_batch, labels)]
    pen = np.mean([rate_penalty(firing_rates_hz(v)) for v in v_batch])
    return float(np.mean(psds) * PSD_WEIGHT + RATE_WEIGHT * pen)


def mcdp_factors(v_traces, ampa_pre, ampa_post, gaba_pre, gaba_post):
    """z-scored squared Pearson pre/post (post +10 steps) per group (training skeleton).

    Simplification S1: no Gaussian edge-smoothing here; the trainer restores it.
    v_traces: (batch, time, cells) array; pre/post are int edge-index arrays.
    """
    import numpy as np

    v = np.asarray(v_traces)
    out = {}
    for group, (pre, post) in {"gAMPA": (np.asarray(ampa_pre), np.asarray(ampa_post)),
                               "gGABAa": (np.asarray(gaba_pre), np.asarray(gaba_post))}.items():
        pv = v[:, :-10][:, :, pre] if v.ndim == 3 else v[:-10, pre]
        qv = v[:, 10:][:, :, post] if v.ndim == 3 else v[10:, post]
        pv = pv - pv.mean(axis=-2, keepdims=True)
        qv = qv - qv.mean(axis=-2, keepdims=True)
        num = (pv * qv).sum(axis=-2)
        den = np.sqrt((pv ** 2).sum(axis=-2) * (qv ** 2).sum(axis=-2)) + 1e-6
        corr = (num / den) ** 2
        if corr.ndim == 2:
            corr = corr.mean(axis=0)
        out[group] = (corr - corr.mean()) / (corr.std() + 1e-6)
    return out


def population_kappa(v_trace, dt_ms=DT_MS, bin_ms=10.0, threshold=-20.0):
    """Fleiss-kappa synchrony through the jaxfne public API."""
    import jaxfne.analysis as _ana

    spikes = detect_spikes(v_trace, threshold).astype(float).T  # (neurons, time)
    return float(_ana.fleiss_kappa_binary(spikes, bin_ms=bin_ms, dt_ms=dt_ms))


def kappa_of_onsets(onsets, dt_ms=DT_MS, bin_ms=5.0):
    """Pairwise Fleiss kappa on a binary onset matrix (time, cells).

    Same statistic as ``data_labels.window_kappa`` (the paper's
    ``compute_kappa``, 5 ms bins, no clamp), but starting from model spike
    onsets instead of convolved-trace threshold crossings, so model kappa is
    comparable to the y kappa labels. ``onsets`` is a (time, cells) bool
    array at ``dt_ms``; bins take the max over ``round(bin_ms / dt_ms)``
    steps.
    """
    import numpy as np

    b = np.asarray(onsets, dtype=bool)
    if b.ndim != 2:
        raise ValueError("onsets must be (time, cells), got %s" % (b.shape,))
    n_time, n_units = b.shape
    steps_per_bin = int(round(bin_ms / dt_ms)) or 1
    n_bins = n_time // steps_per_bin
    if n_units < 2 or n_bins == 0:
        return float("nan")
    binned = b[: n_bins * steps_per_bin, :].reshape(
        n_bins, steps_per_bin, n_units).max(axis=1)
    spikes_per_bin = binned.sum(axis=1)
    silences_per_bin = n_units - spikes_per_bin
    den_pi = max(1.0, float(n_units * (n_units - 1)))
    p_i = (spikes_per_bin * (spikes_per_bin - 1)
           + silences_per_bin * (silences_per_bin - 1)) / den_pi
    p_o = float(p_i.mean())
    p_spike = float(spikes_per_bin.sum() / max(1.0, n_units * n_bins))
    p_e = p_spike ** 2 + (1.0 - p_spike) ** 2
    return float((p_o - p_e) / max(1e-6, 1.0 - p_e))


def window_rates_hz(v_trace, t_l_ms, t_r_ms, dt_ms=DT_MS, threshold=-20.0):
    """Per-cell onset rates (Hz) in ``[t_l, t_r)`` ms (model-time analogue of
    ``data_labels.window_rates_hz``).

    y stim rates count convolved-trace threshold crossings in samples
    ``[600, 1200)`` (``data_labels.py:39-40,165-170``); the functionally
    matched model window is the stim window ``[500, 1000]`` ms
    (``run_gsdr.STIM_ON_MS/OFF_MS``). Whole-trial ``firing_rates_hz`` stays
    the paper-loss rate (``main_fig6.py:306-344`` divides by the full 1.5 s).
    """
    import numpy as np

    v = np.asarray(v_trace)
    n_steps = v.shape[0]
    i0 = max(0, min(int(t_l_ms / dt_ms), n_steps - 2))
    i1 = max(i0 + 2, min(int(t_r_ms / dt_ms), n_steps - 1))
    spikes = detect_spikes(v[i0:i1, :], threshold)
    return spikes.sum(axis=0) / ((i1 - i0) * dt_ms / 1000.0)


def spike_trace_psd(v_trace, dt_ms=DT_MS, t_l_ms=STIM_ON_MS,
                    t_r_ms=STIM_OFF_MS, threshold=-20.0):
    """Diagnostic model PSD through the label recipe (``main_fig6.py:109-126``).

    The paper loss compares a mean-voltage model PSD (max-only normalization,
    ``main_fig6.py:407-421``) against spike-trace labels (smoothed,
    min-subtracted, max-normalized, ``main_fig6.py:109-126``). This function
    puts the model through the label side instead: -20 mV onsets
    (``main_fig6.py:306-330``) binned by max to 1 ms (``fs=1000``, the data
    sampling), mean over cells, rFFT power ``/(N*fs)``, interp onto
    ``target_grid``, ``uniform_filter1d`` size 5, min-subtracted,
    max-normalized (+1e-6). No spike-shape kernel is applied: the warehouse
    kernel (peak ~1e2, tau ~1 sample) is data-side only and unrecorded, so a
    binary-binned comparison is the closest determinable like-for-like.
    """
    import numpy as np

    from scipy import ndimage

    v = np.asarray(v_trace)
    n_steps = v.shape[0]
    i0 = max(0, min(int(t_l_ms / dt_ms), n_steps - 2))
    i1 = max(i0 + 2, min(int(t_r_ms / dt_ms), n_steps - 1))
    onsets = detect_spikes(v[i0:i1, :], threshold)
    steps_per_ms = int(round(1.0 / dt_ms)) or 1
    n_avail = onsets.shape[0]
    n_bins = n_avail // steps_per_ms
    if n_bins == 0:
        raise ValueError("window holds no 1 ms bin at dt %s" % (dt_ms,))
    binned = onsets[: n_bins * steps_per_ms, :].reshape(
        n_bins, steps_per_ms, onsets.shape[1]).max(axis=1)
    signal = binned.mean(axis=1).astype(np.float64)
    fs_hz = 1000.0
    n = signal.shape[0]
    power = np.abs(np.fft.rfft(signal)) ** 2 / (n * fs_hz)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    interp = np.interp(np.asarray(target_grid(), dtype=np.float64), freqs,
                       power)
    smooth = ndimage.uniform_filter1d(interp, size=5)
    smooth = smooth - smooth.min()
    return smooth / (smooth.max() + 1e-6)


def _refuse_scored_dir(out_dir):
    parts = Path(out_dir).resolve().parts
    if any("scored" in part.lower() for part in parts):
        raise SystemExit("error: refusing to write under a directory containing "
                         "'scored': %s" % out_dir)


def git_sha():
    """Git SHA via ``git rev-parse HEAD`` at the repository root."""
    out = subprocess.run(["git", "-C", str(_repo_root), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return "unknown"
    return out.stdout.strip()


def run_trial(stim_amp, *, g_seed=0, ou_seed=0, local_seed=1, edge_seed=0,
              t_max_ms=T_MAX_MS, g_override=None, noise_gain=1.0,
              morphology="single"):
    """One baseline trial: build, init, simulate, metrics. Returns (result, blobs).

    ``g_override`` is an optional conductance vector in the reference uS
    scale (the optimizer's trainable space, bounds [0, 10]); the runner
    converts it to densities. When None, ``init_conductances(PRNGKey(g_seed))``
    supplies the densities. ``noise_gain`` is the lane-4e residual X
    (``jaxfne.emitters_hh.simulate_hh`` noise level; 1.0 = reference).
    ``morphology`` is ``"single"`` (the default: one compartment per cell,
    the density kernel, bit-identical to before) or ``"reference"`` (the
    Fig. 6 4-compartment cell: synapses onto branch 1 loc 1.0, presynaptic
    V and recording at branch 0 loc 0.0, stimulus onto branch 1 loc 0.0,
    gates at the reference default 0.2, point-process uS/nA inputs
    converted by the target compartment area).
    """
    import jax
    import numpy as np

    from jaxfne._construct_hh import construct_hh_model, hh_kernel_args
    from jaxfne.emitters_hh import simulate_hh

    if morphology not in ("single", "reference"):
        raise ValueError(
            f"morphology must be 'single' or 'reference'; got {morphology!r}")

    n_steps = int(round(t_max_ms / DT_MS)) + 1
    if morphology == "reference":
        from jaxfne.emitters_hh import reference_4comp_morphology

        cfg, local_posts = build_spec(local_seed=local_seed,
                                      edge_seed=edge_seed, weight_unit="uS")
        model = construct_hh_model(
            cfg, morphology=reference_4comp_morphology(),
            edge_pre_comp=0, edge_post_comp=3, stim_comp=2, record_comp=0,
            weight_unit="uS", gate_init=0.2)
        st = model.static["hh"]
        assert int(st["pre"].shape[0]) == N_AMPA_EDGES + N_GABAA_GLOBAL_EDGES + N_GABAA_LOCAL_EDGES
        rec = np.asarray(st["receptor"])
        assert int((rec == 0).sum()) == N_AMPA_EDGES
        assert int((rec == 1).sum()) == N_GABAA_GLOBAL_EDGES
        assert int((rec == 2).sum()) == N_GABAA_LOCAL_EDGES
        if str(model.params["emitter"].gNa_mS_cm2.dtype) != "float32":
            raise SystemExit("error: GSDR runs are float32, params resolved %s"
                             % model.params["emitter"].gNa_mS_cm2.dtype)
        kwargs = hh_kernel_args(model, DT_MS)
        if g_override is None:
            g = init_conductances(jax.random.PRNGKey(int(g_seed)),
                                  weight_unit="uS")
        else:
            g = np.asarray(g_override, dtype=np.float32)
            if g.shape != (N_AMPA_EDGES + N_GABAA_GLOBAL_EDGES + N_GABAA_LOCAL_EDGES,):
                raise ValueError("g_override must hold 2230 reference-uS values, got %s"
                                 % (g.shape,))
        kwargs["edge_weight_uS"] = np.asarray(g)
        ou_key = jax.random.fold_in(jax.random.PRNGKey(int(ou_seed)), 0)
        ou_tau, ou_sigma = ou_hyperparams(ou_key)
        i_nA = stimulus_matrix(stim_amp, n_steps=n_steps, unit="nA")
        t0 = time.time()
        out = simulate_hh(**kwargs, n_steps=n_steps, dt_ms=DT_MS, I_ext_nA=i_nA,
                          ou_tau=np.asarray(ou_tau), ou_sigma=np.asarray(ou_sigma),
                          ou_key=ou_key, noise_gain=noise_gain, record_v=True)
        jax.block_until_ready(out["V_mV"])
        wall_s = time.time() - t0
        # Metrics read the recorded compartment (branch 0 loc 0.0).
        v = np.ascontiguousarray(
            np.asarray(out["V_mV"], dtype=np.float32)[:, :, 0])
        rates = firing_rates_hz(v, t_max_ms=t_max_ms)
        result = {
            "stim_amp": float(stim_amp),
            "morphology": "reference",
            "noise_gain": float(noise_gain),
            "n_steps": int(n_steps),
            "rates_hz": [float(r) for r in rates],
            "mean_rate_hz": float(np.mean(rates)),
            "rate_penalty": rate_penalty(rates),
            "psd": [float(x) for x in psd_mean_voltage(v)],
            "kappa": population_kappa(v),
            "wall_s": float(wall_s),
        }
        return result, {"v": v, "g": np.asarray(g, dtype=np.float32),
                        "local_posts": local_posts}
    cfg, local_posts = build_spec(local_seed=local_seed, edge_seed=edge_seed)
    model = construct_hh_model(cfg)
    st = model.static["hh"]
    assert int(st["pre"].shape[0]) == N_AMPA_EDGES + N_GABAA_GLOBAL_EDGES + N_GABAA_LOCAL_EDGES
    rec = np.asarray(st["receptor"])
    assert int((rec == 0).sum()) == N_AMPA_EDGES
    assert int((rec == 1).sum()) == N_GABAA_GLOBAL_EDGES
    assert int((rec == 2).sum()) == N_GABAA_LOCAL_EDGES
    if str(model.params["emitter"].gNa_mS_cm2.dtype) != "float32":
        raise SystemExit("error: GSDR runs are float32, params resolved %s"
                         % model.params["emitter"].gNa_mS_cm2.dtype)
    kwargs = hh_kernel_args(model, DT_MS)
    if g_override is None:
        g = init_conductances(jax.random.PRNGKey(int(g_seed)))
    else:
        g = (np.asarray(g_override, dtype=np.float32)
             * np.float32(US_TO_MSCM2)).astype(np.float32)
        if g.shape != (N_AMPA_EDGES + N_GABAA_GLOBAL_EDGES + N_GABAA_LOCAL_EDGES,):
            raise ValueError("g_override must hold 2230 reference-uS values, got %s"
                             % (g.shape,))
    kwargs["edge_weight"] = np.asarray(g)
    ou_key = jax.random.fold_in(jax.random.PRNGKey(int(ou_seed)), 0)
    ou_tau, ou_sigma = ou_hyperparams(ou_key)
    i_ext = stimulus_matrix(stim_amp, n_steps=n_steps)
    t0 = time.time()
    out = simulate_hh(**kwargs, n_steps=n_steps, dt_ms=DT_MS, I_ext=i_ext,
                      ou_tau=np.asarray(ou_tau), ou_sigma=np.asarray(ou_sigma),
                      ou_key=ou_key, noise_gain=noise_gain, record_v=True)
    jax.block_until_ready(out["V_mV"])
    wall_s = time.time() - t0
    v = np.asarray(out["V_mV"], dtype=np.float32)
    rates = firing_rates_hz(v, t_max_ms=t_max_ms)
    result = {
        "stim_amp": float(stim_amp),
        "morphology": "single",
        "noise_gain": float(noise_gain),
        "n_steps": int(n_steps),
        "rates_hz": [float(r) for r in rates],
        "mean_rate_hz": float(np.mean(rates)),
        "rate_penalty": rate_penalty(rates),
        "psd": [float(x) for x in psd_mean_voltage(v)],
        "kappa": population_kappa(v),
        "wall_s": float(wall_s),
    }
    return result, {"v": v, "g": np.asarray(g, dtype=np.float32),
                    "local_posts": local_posts}


def write_trial(out_dir, trial, result, blobs, *, wall_s, start_time, sha, seeds):
    """Write V trace, conductances and the manifest entry. Refuses overwrites."""
    import numpy as np

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prior = None
    if (out / "manifest.json").exists():
        prior = json.loads((out / "manifest.json").read_text())
        if prior.get("git_sha") != sha:
            raise SystemExit("error: %s holds runs from code %s; this run is %s (one code SHA "
                             "per directory)" % (out, prior.get("git_sha"), sha))
    entries = {e["file"]: e for e in prior["trials"]} if prior else {}
    fname = "v_trial%d_amp%g_float32.npy" % (int(trial), float(result["stim_amp"]))
    if (out / fname).exists() or fname in entries:
        raise SystemExit("error: refusing to overwrite %s" % (out / fname))
    tmp = out / (fname + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, blobs["v"])
    os.replace(tmp, out / fname)
    gname = "g_seed%d_float32.npy" % int(seeds["g_seed"])
    if not (out / gname).exists():
        gtmp = out / (gname + ".tmp")
        with open(gtmp, "wb") as f:
            np.save(f, blobs["g"])
        os.replace(gtmp, out / gname)
    entry = {"trial": int(trial), "file": fname, "g_file": gname,
             "dtype": "float32", "dt_ms": DT_MS, "t_max_ms": T_MAX_MS,
             "local_posts": blobs["local_posts"], **result, **seeds}
    entries[fname] = entry
    manifest = {"git_sha": sha, "start_time": prior["start_time"] if prior else start_time,
                "trials": list(entries.values())}
    try:
        import jax

        manifest["jax_version"] = jax.__version__
        manifest["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception:
        pass
    mtmp = out / "manifest.json.tmp"
    mtmp.write_text(json.dumps(manifest, indent=2))
    os.replace(mtmp, out / "manifest.json")
    return out / fname


def run(out_dir, *, amps=BATCH_AMPS, g_seed=0, ou_seed=0, local_seed=1,
        edge_seed=0, t_max_ms=T_MAX_MS, morphology="single"):
    """Baseline: one trial per amp. Writes traces plus ``manifest.json``."""
    _refuse_scored_dir(out_dir)
    sha = git_sha()
    if (Path(out_dir) / "manifest.json").exists():
        prior = json.loads((Path(out_dir) / "manifest.json").read_text())
        if prior.get("git_sha") != sha:
            raise SystemExit("error: %s holds runs from code %s; this run is %s" %
                             (out_dir, prior.get("git_sha"), sha))
    start_time = datetime.now(timezone.utc).isoformat()
    seeds = {"g_seed": int(g_seed), "ou_seed": int(ou_seed),
             "local_seed": int(local_seed), "edge_seed": int(edge_seed)}
    results = []
    for trial, amp in enumerate(amps):
        tseeds = {"g_seed": int(g_seed + trial), "ou_seed": int(ou_seed + trial),
                  "local_seed": int(local_seed), "edge_seed": int(edge_seed)}
        result, blobs = run_trial(amp, g_seed=g_seed + trial, ou_seed=ou_seed + trial,
                                  local_seed=local_seed, edge_seed=edge_seed,
                                  t_max_ms=t_max_ms, morphology=morphology)
        write_trial(out_dir, trial, result, blobs, wall_s=result["wall_s"],
                    start_time=start_time, sha=sha, seeds=tseeds)
        results.append(result)
        print("trial %d amp %g: mean_rate %.3f Hz penalty %.4f kappa %.4f wall %.1fs"
              % (trial, amp, result["mean_rate_hz"], result["rate_penalty"],
                 result["kappa"], result["wall_s"]), flush=True)
    return results


def build_parser():
    ap = argparse.ArgumentParser(description="GSDR Fig. 6 baseline on the jaxfne core HH route.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--amps", type=float, nargs="*", default=None,
                    help="stimulus amps, one trial each (default: 0 0.6 0.6)")
    ap.add_argument("--g-seed", type=int, default=0)
    ap.add_argument("--ou-seed", type=int, default=0)
    ap.add_argument("--local-seed", type=int, default=1)
    ap.add_argument("--edge-seed", type=int, default=0)
    ap.add_argument("--t-max-ms", type=float, default=T_MAX_MS)
    ap.add_argument("--morphology", choices=["single", "reference"],
                    default="single",
                    help="compartment layout (default: single)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args.out, amps=tuple(args.amps) if args.amps is not None else BATCH_AMPS,
        g_seed=args.g_seed, ou_seed=args.ou_seed, local_seed=args.local_seed,
        edge_seed=args.edge_seed, t_max_ms=args.t_max_ms,
        morphology=args.morphology)


if __name__ == "__main__":
    main()
