"""Protocol H3 — localized RBS perturbation and nested linear decodability.

Scientific question: does a localized RBS perturbation remain decodable after the
perturbation offset is applied (no ongoing forcing)? See
``docs/doctrine/protocol_h_rbd_memory.md`` §5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jaxfne.emitters import EdgeList, IzhikevichParams, simulate_edge_recurrent_izhikevich_rbd

DecoderObservationKind = Literal["H", "X", "XH"]
PRIMARY_DECODER_KINDS: tuple[DecoderObservationKind, ...] = ("H", "X", "XH")


@dataclass(frozen=True)
class H3ProtocolConfig:
    """Frozen H3 measurement contract (not H4 topology matrix)."""

    delta_h: float = 0.2
    perturbation_step: int = 0
    lag_steps: tuple[int, ...] = (1, 3, 5, 10, 20, 40)
    rbd_family: str = "f1"
    beta_h: float = 0.5
    kappa_h: float = 0.0
    tau_h_ms: float = 80.0
    dt_ms: float = 1.0
    ridge_lambda: float = 1e-2
    n_shuffle: int = 8
    train_seeds: tuple[int, ...] = field(default_factory=tuple)
    test_seeds: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.train_seeds and self.test_seeds:
            _validate_disjoint_seeds(self.train_seeds, self.test_seeds)


def _validate_disjoint_seeds(train: Sequence[int], test: Sequence[int]) -> None:
    overlap = set(train) & set(test)
    if overlap:
        raise ValueError(f"train and test seeds must be disjoint; overlap={sorted(overlap)}")


def trial_drive_schedule(
    n_steps: int,
    n_neurons: int,
    seed: int,
) -> jax.Array:
    """Deterministic per-trial background drive (independent seeds, noise_scale=0)."""
    rng = np.random.default_rng(int(seed) + 7919)
    drive = np.zeros((n_steps, n_neurons), dtype=np.float32)
    n_pulses = 2 + (int(seed) % 3)
    for _ in range(n_pulses):
        step = int(rng.integers(3, max(4, n_steps - 3)))
        neu = int(rng.integers(0, n_neurons))
        drive[step, neu] = float(rng.uniform(25.0, 55.0))
    return jnp.asarray(drive)


def localized_h_state(
    n_neurons: int,
    perturbed_index: int,
    *,
    delta_h: float,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    h = jnp.ones((n_neurons,), dtype=dtype)
    if delta_h == 0.0:
        return h
    return h.at[int(perturbed_index)].set(1.0 + float(delta_h))


def run_localized_rbs_trial(
    params: IzhikevichParams,
    edges: EdgeList,
    *,
    n_steps: int,
    perturbed_index: int,
    seed: int,
    cfg: H3ProtocolConfig,
    drive_schedule: jax.Array | None = None,
) -> dict[str, Any]:
    """Single independent trial with localized RBS offset at ``perturbation_step``."""
    n_neurons = int(params.v0.shape[0])
    if not (0 <= int(perturbed_index) < n_neurons):
        raise ValueError(f"perturbed_index must be in [0, {n_neurons}); got {perturbed_index}")
    t0 = int(cfg.perturbation_step)
    if t0 < 0 or t0 >= n_steps:
        raise ValueError(f"perturbation_step must be in [0, {n_steps}); got {t0}")

    if drive_schedule is None:
        drive_schedule = trial_drive_schedule(n_steps, n_neurons, seed)

    key = jax.random.PRNGKey(int(seed))
    rbd_kw = dict(
        rbd_family=cfg.rbd_family,
        beta_h=cfg.beta_h,
        kappa_h=cfg.kappa_h,
        tau_h_ms=cfg.tau_h_ms,
        dtype="float32",
        noise_scale=0.0,
    )

    h_pert = localized_h_state(
        n_neurons, perturbed_index, delta_h=cfg.delta_h
    )

    if t0 == 0:
        init_state = {"H": h_pert}
        v, s, q, st = simulate_edge_recurrent_izhikevich_rbd(
            params,
            edges,
            n_steps,
            cfg.dt_ms,
            key,
            drive_schedule=drive_schedule,
            init_state=init_state,
            **rbd_kw,
        )
    else:
        v0, s0, q0, st0 = simulate_edge_recurrent_izhikevich_rbd(
            params,
            edges,
            t0,
            cfg.dt_ms,
            key,
            drive_schedule=None if drive_schedule is None else drive_schedule[:t0],
            init_state={"H": jnp.ones((n_neurons,), dtype=jnp.float32)},
            **rbd_kw,
        )
        cont = dict(st0)
        cont["H"] = h_pert
        cont["H_final"] = h_pert
        key, key2 = jax.random.split(key)
        v1, s1, q1, st1 = simulate_edge_recurrent_izhikevich_rbd(
            params,
            edges,
            n_steps - t0,
            cfg.dt_ms,
            key2,
            drive_schedule=None if drive_schedule is None else drive_schedule[t0:],
            init_state=cont,
            **rbd_kw,
        )
        v = jnp.concatenate([v0, v1], axis=0)
        s = jnp.concatenate([s0, s1], axis=0)
        q = jnp.concatenate([q0, q1], axis=0)
        st = dict(st1)
        st["H_trace"] = jnp.concatenate([st0["H_trace"], st1["H_trace"]], axis=0)

    return {
        "voltages": v,
        "spikes": s,
        "sources": q,
        "state": st,
        "perturbed_index": int(perturbed_index),
        "seed": int(seed),
        "label": int(perturbed_index),
    }


def _feature_vector(
    trial: Mapping[str, Any],
    *,
    lag: int,
    kind: DecoderObservationKind,
) -> np.ndarray:
    """Label-independent decoder features at ``lag`` (global step index)."""
    step = int(lag)
    v = np.asarray(trial["voltages"][step], dtype=np.float64)
    spikes = np.asarray(trial["spikes"][step], dtype=np.float64)
    h = np.asarray(trial["state"]["H_trace"][step], dtype=np.float64)
    x_activity = np.concatenate([v, spikes], axis=0)

    if kind == "H":
        feat = h
    elif kind == "X":
        feat = x_activity
    elif kind == "XH":
        feat = np.concatenate([h, x_activity], axis=0)
    else:
        raise ValueError(f"unknown decoder observation kind {kind!r}")
    return feat


def local_h_perturbation_magnitude(
    trial: Mapping[str, Any],
    *,
    lag: int,
    perturbation_step: int,
) -> float:
    """Local persistence diagnostic: ``|H_k(t_0+Δ)-1|`` at the intervened coordinate."""
    k = int(trial["perturbed_index"])
    step = int(perturbation_step) + int(lag)
    h_k = float(np.asarray(trial["state"]["H_trace"][step, k], dtype=np.float64))
    return abs(h_k - 1.0)


def distributed_rbs_dispersion(
    trial: Mapping[str, Any],
    *,
    lag: int,
    perturbation_step: int,
) -> float:
    r"""Label-safe propagation diagnostic: ``D_H(Δ)=\sum_{j\neq k}|H_j(t_0+Δ)-1|``."""
    k = int(trial["perturbed_index"])
    step = int(perturbation_step) + int(lag)
    h = np.asarray(trial["state"]["H_trace"][step], dtype=np.float64)
    off_diag = np.delete(h, k)
    return float(np.sum(np.abs(off_diag - 1.0)))


def diagnostic_curves(
    trials: Sequence[Mapping[str, Any]],
    *,
    cfg: H3ProtocolConfig,
) -> dict[str, dict[int, float]]:
    """Non-decoder RBS diagnostics (may use intervention index ``k``)."""
    local_h: dict[int, list[float]] = {int(lag): [] for lag in cfg.lag_steps}
    d_h: dict[int, list[float]] = {int(lag): [] for lag in cfg.lag_steps}
    for tr in trials:
        for lag in cfg.lag_steps:
            local_h[int(lag)].append(
                local_h_perturbation_magnitude(
                    tr, lag=int(lag), perturbation_step=cfg.perturbation_step
                )
            )
            d_h[int(lag)].append(
                distributed_rbs_dispersion(
                    tr, lag=int(lag), perturbation_step=cfg.perturbation_step
                )
            )
    return {
        "local_h_k": {lag: float(np.mean(vals)) for lag, vals in local_h.items()},
        "D_H": {lag: float(np.mean(vals)) for lag, vals in d_h.items()},
    }


def _fit_ridge_ovr(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    *,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One-vs-rest ridge classifier weights ``(n_classes, n_features)``."""
    n_features = X.shape[1]
    W = np.zeros((n_classes, n_features), dtype=np.float64)
    b = np.zeros((n_classes,), dtype=np.float64)
    for c in range(n_classes):
        y_bin = (y == c).astype(np.float64)
        XtX = X.T @ X + ridge_lambda * np.eye(n_features)
        W[c] = np.linalg.solve(XtX, X.T @ y_bin)
        b[c] = 0.0
    return W, b


def _predict_scores(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return X @ W.T + b[None, :]


def linear_decode_accuracy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_classes: int,
    ridge_lambda: float,
) -> float:
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    Xtr = (X_train - mu) / sigma
    Xte = (X_test - mu) / sigma
    W, b = _fit_ridge_ovr(Xtr, y_train, n_classes, ridge_lambda=ridge_lambda)
    scores = _predict_scores(Xte, W, b)
    pred = np.argmax(scores, axis=1)
    return float(np.mean(pred == y_test))


def chance_accuracy(n_classes: int) -> float:
    return 1.0 / float(n_classes)


def build_trial_dataset(
    params: IzhikevichParams,
    edges: EdgeList,
    *,
    n_steps: int,
    seeds: Sequence[int],
    cfg: H3ProtocolConfig,
    drive_schedule: jax.Array | None = None,
) -> list[dict[str, Any]]:
    n_neurons = int(params.v0.shape[0])
    trials: list[dict[str, Any]] = []
    for seed in seeds:
        drive = (
            drive_schedule
            if drive_schedule is not None
            else trial_drive_schedule(n_steps, n_neurons, int(seed))
        )
        for k in range(n_neurons):
            trials.append(
                run_localized_rbs_trial(
                    params,
                    edges,
                    n_steps=n_steps,
                    perturbed_index=k,
                    seed=int(seed) * 1000 + int(k) + 1,
                    cfg=cfg,
                    drive_schedule=drive,
                )
            )
    return trials


def features_labels_for_lag(
    trials: Sequence[Mapping[str, Any]],
    *,
    lag: int,
    kind: DecoderObservationKind,
    perturbation_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    feats = []
    labels = []
    step = int(perturbation_step) + int(lag)
    for tr in trials:
        feats.append(_feature_vector(tr, lag=step, kind=kind))
        labels.append(int(tr["label"]))
    return np.stack(feats, axis=0), np.asarray(labels, dtype=np.int64)


def decodability_curve(
    train_trials: Sequence[Mapping[str, Any]],
    test_trials: Sequence[Mapping[str, Any]],
    *,
    cfg: H3ProtocolConfig,
    kind: DecoderObservationKind,
    n_classes: int,
) -> dict[int, float]:
    curve: dict[int, float] = {}
    for lag in cfg.lag_steps:
        Xtr, ytr = features_labels_for_lag(
            train_trials, lag=lag, kind=kind, perturbation_step=cfg.perturbation_step
        )
        Xte, yte = features_labels_for_lag(
            test_trials, lag=lag, kind=kind, perturbation_step=cfg.perturbation_step
        )
        curve[int(lag)] = linear_decode_accuracy(
            Xtr,
            ytr,
            Xte,
            yte,
            n_classes=n_classes,
            ridge_lambda=cfg.ridge_lambda,
        )
    return curve


def shuffle_baseline_curve(
    train_trials: Sequence[Mapping[str, Any]],
    test_trials: Sequence[Mapping[str, Any]],
    *,
    cfg: H3ProtocolConfig,
    kind: DecoderObservationKind,
    n_classes: int,
    rng: np.random.Generator,
) -> dict[int, float]:
    lags = cfg.lag_steps
    accs = np.zeros((cfg.n_shuffle, len(lags)), dtype=np.float64)
    for si in range(cfg.n_shuffle):
        perm_trials = list(train_trials)
        labels = [int(t["label"]) for t in perm_trials]
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        permuted = []
        for tr, lab in zip(perm_trials, shuffled):
            t2 = dict(tr)
            t2["label"] = lab
            permuted.append(t2)
        curve = decodability_curve(
            permuted, test_trials, cfg=cfg, kind=kind, n_classes=n_classes
        )
        for j, lag in enumerate(lags):
            accs[si, j] = curve[int(lag)]
    mean_curve = {int(lag): float(accs[:, j].mean()) for j, lag in enumerate(lags)}
    return mean_curve


def area_above_shuffle(
    m_curve: Mapping[int, float],
    m_shuffle: Mapping[int, float],
) -> float:
    lags = sorted(m_curve.keys())
    vals = [
        max(float(m_curve[l]) - float(m_shuffle.get(l, 0.0)), 0.0) for l in lags
    ]
    if len(lags) < 2:
        return float(vals[0]) if vals else 0.0
    x = np.asarray(lags, dtype=np.float64)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(vals, x=x))
    return float(np.trapz(vals, x=x))


def run_h3_decodability_study(
    params: IzhikevichParams,
    edges: EdgeList,
    *,
    n_steps: int,
    cfg: H3ProtocolConfig,
    drive_schedule: jax.Array | None = None,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Run label-independent linear decodability plus RBS propagation diagnostics."""
    _validate_disjoint_seeds(cfg.train_seeds, cfg.test_seeds)
    n_classes = int(params.v0.shape[0])
    train_trials = build_trial_dataset(
        params, edges, n_steps=n_steps, seeds=cfg.train_seeds, cfg=cfg, drive_schedule=drive_schedule
    )
    test_trials = build_trial_dataset(
        params, edges, n_steps=n_steps, seeds=cfg.test_seeds, cfg=cfg, drive_schedule=drive_schedule
    )
    rng = np.random.default_rng(int(rng_seed))
    curves: dict[str, dict[int, float]] = {}
    shuffle_curves: dict[str, dict[int, float]] = {}
    areas: dict[str, float] = {}
    for kind in PRIMARY_DECODER_KINDS:
        curves[kind] = decodability_curve(
            train_trials, test_trials, cfg=cfg, kind=kind, n_classes=n_classes
        )
        shuffle_curves[kind] = shuffle_baseline_curve(
            train_trials,
            test_trials,
            cfg=cfg,
            kind=kind,
            n_classes=n_classes,
            rng=rng,
        )
        areas[kind] = area_above_shuffle(curves[kind], shuffle_curves[kind])
    all_trials = list(train_trials) + list(test_trials)
    diagnostics = diagnostic_curves(all_trials, cfg=cfg)
    return {
        "config": cfg,
        "n_classes": n_classes,
        "chance": chance_accuracy(n_classes),
        "curves": curves,
        "shuffle_curves": shuffle_curves,
        "area_above_shuffle": areas,
        "decoder_observation_kinds": PRIMARY_DECODER_KINDS,
        "diagnostics": diagnostics,
    }
