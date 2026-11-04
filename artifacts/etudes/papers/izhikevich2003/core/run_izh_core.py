"""Izhikevich 2003 Fig. 3 network on the jaxfne library path (packet 9C-E1, pilot 2).

Unscored pilot. ``build_core_model`` declares the network through ``Configuration``
and ``construct``; ``run_core_seed`` simulates it through ``Model.simulate``. The
kernel route (``run_izh.run_fig3_seed``) is the comparison arm. An ``--out`` containing
"scored" is refused before any compute.

Settings (packet 9C-E1 pilot 2):
- 800 E and 200 I cells; all-to-all rules that include self-edges
  (``with_rule_allow_self_connections``); ``connectivity_mode="explicit"``.
- Per-cell a, b, c, d, v0, u0 and drive through ``cell_params``; the per-cell values
  are the kernel route's own draw (``run_izh.fig3_keys``, ``jax.random.uniform``).
- W = 2S through ``weight_range``: E |w| ~ U[0, 1.0), I |w| ~ U[0, 2.0); the
  inhibitory sign comes from the presynaptic I cells.
- Noise through ``Simulation.gaussian_drive`` at dt 0.5 ms: std E 5 sqrt(2), I 2 sqrt(2).
- Recurrent backend ``edge_list``: the dense backend runs on ``emitter.W`` only and
  refuses compiled rule edges, so the declared rules run on the edge list.
- Mechanism ``instantaneous`` (``tau_ms`` 0.0, the tau -> 0 limit, P-051): the current
  at step t is the weighted sum of presynaptic spikes at t-1 with no carry-over, as
  ``W @ prev_spikes`` is in the kernel route (``simulate_eig_izhikevich``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jaxfne as jtfne  # noqa: E402

assert "site-packages" not in jtfne.__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

# The lane's own runner: kernel route, Fig. 3 builder, score call and calibration.
_LANE = HERE.parent / "run_izh.py"
_lane_spec = importlib.util.spec_from_file_location("run_izh_lane", _LANE)
lane = importlib.util.module_from_spec(_lane_spec)
_lane_spec.loader.exec_module(lane)

N_E = 800
N_I = 200
N = N_E + N_I
DT_MS = float(lane.FIG3_DT)
DISCARD_MS = float(lane.DISCARD_MS)
NOISE_STD = {"E": 5.0 * np.sqrt(2.0), "I": 2.0 * np.sqrt(2.0)}
W_RANGE = {"E": (0.0, 1.0), "I": (0.0, 2.0)}
CELL_KEYS = ("a", "b", "c", "d", "v0", "u0", "drive")
PE_CHANGE_STOP = 0.25  # packet stop rule: library vs kernel E or I rate differs by more than 25%
INSTANTANEOUS_TAU_MS = 0.0  # the tau -> 0 limit (P-051); the only zero tau the edge route accepts

# P6 scored run (PREREG_P6_v2_signed.md): seeds 2-6 at 10.2 s, option B for Q5.
# Reference (mean, margin): N1 rate 8.677 Hz (0.29); N2 median CV 0.4028 (0.0085);
# N3 f_pk 9.10 Hz (1.0), PE 8.01 (5.8); N4 f_pk 38.1 Hz (5.2), PE 5.83 (2.5).
PREREG_SIGNED = HERE / "PREREG_P6_v2_signed.md"
P6_SEEDS = (2, 3, 4, 5, 6)
P6_T_S = float(lane.FIG3_T)
P5_N1_RATE = (8.677, 0.29)
P5_N2_CV = (0.4028, 0.0085)
P5_N3_FPK = (9.10, 1.0)
P5_N4_FPK = (38.1, 5.2)
P5_N3_PE = (8.01, 5.8)
P5_N4_PE = (5.83, 2.5)


def check_out(out):
    """Refuse an --out containing "scored" (no scored artifact is written here)."""
    if "scored" in str(out).lower():
        raise SystemExit("refusing --out containing 'scored'")


def intended_cell_arrays(seed):
    """Per-cell parameters from the kernel route's draw, E first (indices 0-799)."""
    import jax

    k_r, _, _ = lane.fig3_keys(seed)
    r = np.asarray(jax.random.uniform(k_r, (N,)), dtype=np.float64)
    r_e, r_i = r[:N_E], r[N_E:]
    e = {"a": np.full(N_E, 0.02), "b": np.full(N_E, 0.2),
         "c": -65.0 + 15.0 * r_e ** 2, "d": 8.0 - 6.0 * r_e ** 2}
    i = {"a": 0.02 + 0.08 * r_i, "b": 0.25 - 0.05 * r_i,
         "c": np.full(N_I, -65.0), "d": np.full(N_I, 2.0)}
    out = {}
    for name, block, n in (("E", e, N_E), ("I", i, N_I)):
        block["v0"] = np.full(n, -65.0)
        block["u0"] = block["b"] * -65.0
        block["drive"] = np.zeros(n)
        out[name] = block
    return out


def _full_intended(seed):
    arrs = intended_cell_arrays(seed)
    return {k: np.concatenate([arrs["E"][k], arrs["I"][k]]) for k in CELL_KEYS}


def core_simulation(seed, t_s):
    return jtfne.Simulation(
        duration_ms=float(t_s) * 1000.0,
        dt_ms=DT_MS,
        seed=int(seed),
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"),
        gaussian_drive={"std_by_cell_type": dict(NOISE_STD)},
    )


def build_core_model(seed):
    """Construct the Fig. 3 network (explicit mode, edge_list, ampa mechanism)."""
    from jaxfne.connectivity import with_rule_allow_self_connections

    arrs = intended_cell_arrays(seed)
    payload = {
        name: {k: block[k].tolist() for k in CELL_KEYS}
        for name, block in arrs.items()
    }
    cfg = (
        jtfne.configuration()
        .network(name="izh2003_fig3", kind="balanced_ei_population", n=N,
                 cell_types={"E": 0.8, "I": 0.2})
        .mechanisms(name="instantaneous", kind="receptor", params={"tau_ms": INSTANTANEOUS_TAU_MS})
        .connections(name="e_all", source={"cell_type": "E"}, target={}, probability=1.0,
                     mechanism="instantaneous", weight_range=W_RANGE["E"])
        .connections(name="i_all", source={"cell_type": "I"}, target={}, probability=1.0,
                     mechanism="instantaneous", weight_range=W_RANGE["I"])
        .connectivity(edge_seed=int(seed))
        .cell_params({"cell_type": "E"}, payload["E"])
        .cell_params({"cell_type": "I"}, payload["I"])
        .update_metadata(connectivity_mode="explicit")
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="declared_proxy",
               gauge="mean_zero")
        .probe(name="spike_probe", modes=["spikes"])
        .runtime(random_v0=False)
    )
    cfg = with_rule_allow_self_connections(cfg, "e_all", True)
    cfg = with_rule_allow_self_connections(cfg, "i_all", True)
    return jtfne.construct(cfg)


def simulate_core(model, seed, t_s):
    """Run ``Model.simulate``; returns spike rows, cols (kernel-style indexing) and n_steps."""
    sim = core_simulation(seed, t_s)
    n_steps = int(round(float(t_s) * 1000.0 / DT_MS))
    res = model.simulate(sim)
    spikes = np.asarray(res.spikes)
    if spikes.shape != (n_steps, N):
        raise RuntimeError(f"spikes shape {spikes.shape}, expected {(n_steps, N)}")
    idx = np.flatnonzero(spikes)
    return (idx // N).astype(np.int64), (idx % N).astype(np.int64), n_steps


def run_core_seed(seed, t_s, model=None):
    """Library route, one seed. Returns (rows, cols, n_steps, timing) like run_fig3_seed."""
    t0 = time.perf_counter()
    if model is None:
        model = build_core_model(seed)
    construct_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    rows, cols, n_steps = simulate_core(model, seed, t_s)
    return rows, cols, n_steps, {"construct_s": construct_s,
                                 "simulate_s": time.perf_counter() - t1}


def run_kernel_seed(seed, t_s):
    """Kernel route: ``run_izh.run_fig3_seed``. construct_s is one separate builder call."""
    t0 = time.perf_counter()
    lane.build_fig3_network(seed)
    construct_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    rows, cols, n_steps, _ = lane.run_fig3_seed(seed, t_s)
    total = time.perf_counter() - t1
    return rows, cols, n_steps, {"construct_s": construct_s,
                                 "simulate_s": total - construct_s}


def score_seed(rows, cols, n_steps, theta, alpha_band):
    """N1-N4 for one seed from a spike raster, through ``run_izh.score_network_seed``.

    ``rows``/``cols`` are spike time steps and cell indices (E 0-799, I 800-999), kernel or
    library. The raster is grouped per cell exactly as ``run_izh.cmd_fig3`` groups it. Returns the
    N1-N4 entries as the lane scorer computes them. Writes no file and adds no CLI command.
    """
    cells = [[] for _ in range(N)]
    for r, c in zip(np.asarray(rows).tolist(), np.asarray(cols).tolist()):
        cells[int(c)].append(int(r))
    res = lane.score_network_seed(cells, int(n_steps), DT_MS, theta, alpha_band,
                                  populations={"E": np.arange(N_E), "I": np.arange(N_E, N)})
    return {nid: res[nid] for nid in ("N1", "N2", "N3", "N4")}


def prereg_sha256():
    """sha256 of the signed P6 prereg, recorded in the scored receipt."""
    import hashlib

    return hashlib.sha256(PREREG_SIGNED.read_bytes()).hexdigest()


def score_one_seed(rows, cols, n_steps):
    """Gate-2 scorer: N1-N4 for one raster through ``run_izh.score_network_seed``.

    Theta and the alpha band come from the calibration file the prereg names
    (``lane.load_calibration``); the estimators are the lane's, not reimplemented here.
    """
    theta, alpha_band = lane.load_calibration()
    return score_seed(rows, cols, n_steps, theta, alpha_band)


def realization_ok(rep):
    """Gate-1 verdict for one seed's ``realization_checks`` report.

    Mirrors the assertions in ``tests/test_izh2003_core_path.py``: per-cell
    parameters within 1e-4 of the kernel draw, noise coefficients within 1e-6,
    1e6 edges with 1000 self edges, signed weight blocks, receipt counts and
    zero decay. A missing field reads as not ok.
    """
    try:
        if rep["n_E_cells"] != N_E or rep["n_I_cells"] != N_I:
            return False
        if not rep["labels_E_first_then_I"]:
            return False
        cell = rep["cell_params_max_abs_diff"]
        for k in CELL_KEYS:
            if cell[k]["realized_vs_kernel"] > 1e-4 or cell[k]["intended_vs_kernel"] > 1e-4:
                return False
        if cell["sign"]["realized_vs_kernel"] != 0.0:
            return False
        e_std, i_std = rep["noise_std_E_I"]
        for got, want in ((e_std, 5.0 * np.sqrt(2.0)), (i_std, 2.0 * np.sqrt(2.0))):
            if abs(got - want) > max(1e-12 * abs(want), 1e-12):
                return False
        if rep["noise_coef_max_abs_diff_vs_kernel"] > 1e-6:
            return False
        if rep["n_edges"] != N * N or rep["self_edges"] != N:
            return False
        if rep["rule_compiled_n_edges"] != {"e_all": N_E * N, "i_all": N_I * N}:
            return False
        if [float(x) for x in rep["tau_ms_unique"]] != [0.0]:
            return False
        if rep["decay_float32_max"] != 0.0:
            return False
        mechs = rep["mechanisms"]
        if len(mechs) != 1 or mechs[0].get("tau_ms") != 0.0:
            return False
        if set(rep["connection_mechanisms"].values()) != {mechs[0].get("name")}:
            return False
        if rep["connectivity_compilation"] != {
                "connectivity_mode": "explicit", "default_edge_count": 0,
                "declared_rule_edge_count": N * N, "total_compiled_edge_count": N * N}:
            return False
        if rep["emitter_W_has_coupling"] is not False:
            return False
        e = rep["weight_blocks_by_pre_type"]["E"]
        i = rep["weight_blocks_by_pre_type"]["I"]
        if e["n_edges"] != N_E * N or e["n_negative"] != 0 or e["abs_max"] >= 1.0:
            return False
        if abs(e["abs_mean"] - 0.5) >= 0.01:
            return False
        if i["n_edges"] != N_I * N or i["n_positive"] != 0 or i["n_negative"] != i["n_edges"]:
            return False
        if i["abs_max"] >= 2.0 or abs(i["abs_mean"] - 1.0) >= 0.02:
            return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def evaluate_p6_verdicts(per_seed):
    """Q1-Q5 from per-seed N1-N4, per the prereg table (option B for Q5).

    ``per_seed`` maps a seed to its ``score_one_seed`` dict. Returns the
    five seed means, each Q verdict with its numbers, and the overall verdict
    ("pass" only when Q1-Q5 all pass). Completeness (seeds, T, signature) is
    decided by the caller, not here.
    """
    seeds = sorted(per_seed)
    n1 = [per_seed[s]["N1"]["numbers"]["mean_rate_hz"] for s in seeds]
    n2 = [per_seed[s]["N2"]["numbers"]["median_cv"] for s in seeds]
    n3f = [per_seed[s]["N3"]["numbers"]["f_pk"] for s in seeds]
    n4f = [per_seed[s]["N4"]["numbers"]["f_pk"] for s in seeds]
    n3pe = [per_seed[s]["N3"]["numbers"]["pe"] for s in seeds]
    n4pe = [per_seed[s]["N4"]["numbers"]["pe"] for s in seeds]
    n1_v = [per_seed[s]["N1"]["verdict"] for s in seeds]
    n2_v = [per_seed[s]["N2"]["verdict"] for s in seeds]

    def _mean(xs):
        if not xs or any(x is None for x in xs):
            return None
        return float(sum(xs) / len(xs))

    means = {"n1_rate_hz": _mean(n1), "n2_median_cv": _mean(n2),
             "n3_f_pk": _mean(n3f), "n4_f_pk": _mean(n4f),
             "n3_pe": _mean(n3pe), "n4_pe": _mean(n4pe)}

    def _within(value, ref_margin):
        ref, margin = ref_margin
        return value is not None and abs(value - ref) <= margin

    legs = {
        "Q1": _within(means["n1_rate_hz"], P5_N1_RATE),
        "Q2": _within(means["n2_median_cv"], P5_N2_CV),
        "Q3": _within(means["n3_f_pk"], P5_N3_FPK),
        "Q4": _within(means["n4_f_pk"], P5_N4_FPK),
    }
    # Arm verdicts as P5 forms them (PREREG_P3.md: pass "in every seed"): N1 passes
    # when every seed passes; N2 fails when any seed fails.
    legs["Q5"] = (n1_v == ["pass"] * len(seeds) and any(v != "pass" for v in n2_v)
                  and _within(means["n3_pe"], P5_N3_PE)
                  and _within(means["n4_pe"], P5_N4_PE))
    criteria = {
        "Q1": {"verdict": "pass" if legs["Q1"] else "fail",
               "numbers": {"mean_rate_hz": means["n1_rate_hz"],
                           "p5_mean_hz": P5_N1_RATE[0], "margin_hz": P5_N1_RATE[1]}},
        "Q2": {"verdict": "pass" if legs["Q2"] else "fail",
               "numbers": {"mean_median_cv": means["n2_median_cv"],
                           "p5_mean": P5_N2_CV[0], "margin": P5_N2_CV[1]}},
        "Q3": {"verdict": "pass" if legs["Q3"] else "fail",
               "numbers": {"mean_f_pk": means["n3_f_pk"],
                           "p5_mean_hz": P5_N3_FPK[0], "margin_hz": P5_N3_FPK[1]}},
        "Q4": {"verdict": "pass" if legs["Q4"] else "fail",
               "numbers": {"mean_f_pk": means["n4_f_pk"],
                           "p5_mean_hz": P5_N4_FPK[0], "margin_hz": P5_N4_FPK[1]}},
        "Q5": {"verdict": "pass" if legs["Q5"] else "fail",
               "numbers": {"n1_seed_verdicts": n1_v, "n2_seed_verdicts": n2_v,
                           "mean_n3_pe": means["n3_pe"], "p5_n3_pe": P5_N3_PE[0],
                           "n3_margin": P5_N3_PE[1], "mean_n4_pe": means["n4_pe"],
                           "p5_n4_pe": P5_N4_PE[0], "n4_margin": P5_N4_PE[1]}},
    }
    overall = "pass" if all(legs[q] for q in ("Q1", "Q2", "Q3", "Q4", "Q5")) else "fail"
    return {"means": means, "criteria": criteria, "overall": overall}


def _mad(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def edge_arrays(model):
    import jax.numpy as jnp

    from jaxfne.emitters import _resolved_edge_weight

    edges = model.params["edge_list"]
    pre = np.asarray(edges.pre).astype(np.int64)
    post = np.asarray(edges.post).astype(np.int64)
    w = np.asarray(_resolved_edge_weight(edges, jnp.float32, model.params["emitter"]),
                   dtype=np.float64)
    return pre, post, w


def realization_checks(model, seed):
    """Realized values read through the model's accessors, against the intended arrays."""
    import jax.numpy as jnp

    from jaxfne._model_simulate import _gaussian_noise_for_route
    from jaxfne.emitters import _noise_coef, resolve_edge_tau_ms, resolve_receptor_index

    em = model.params["emitter"]
    kp, kern_noise, _ = lane.build_fig3_network(seed)
    full = _full_intended(seed)
    labels = np.asarray(em.labels)
    rep = {"seed": int(seed)}
    rep["n_E_cells"] = int(np.sum(labels == "E"))
    rep["n_I_cells"] = int(np.sum(labels == "I"))
    rep["labels_E_first_then_I"] = bool(np.all(labels[:N_E] == "E") and np.all(labels[N_E:] == "I"))
    cell = {}
    for k in CELL_KEYS:
        realized = np.asarray(getattr(em, k), dtype=np.float64)
        kern = np.asarray(getattr(kp, k), dtype=np.float64)
        cell[k] = {"realized_vs_kernel": _mad(realized, kern),
                   "intended_vs_kernel": _mad(full[k], kern)}
    cell["sign"] = {"realized_vs_kernel": _mad(np.asarray(em.sign, dtype=np.float64),
                                               np.asarray(kp.sign, dtype=np.float64))}
    rep["cell_params_max_abs_diff"] = cell
    sim = core_simulation(seed, 1.0)
    std = np.asarray(_gaussian_noise_for_route(model, sim, sim.runtime), dtype=np.float64)
    coef = np.asarray(_noise_coef(std, DT_MS, jnp.float32), dtype=np.float64)
    rep["noise_std_E_I"] = [float(std[0]), float(std[-1])]
    rep["noise_coef_max_abs_diff_vs_kernel"] = _mad(coef, np.asarray(kern_noise))
    edges = model.params["edge_list"]
    pre, post, w = edge_arrays(model)
    rep["n_edges"] = int(edges.n_edges)
    rep["self_edges"] = int(np.sum(pre == post))
    tau = np.asarray(resolve_edge_tau_ms(edges, jnp.float32), dtype=np.float64)
    rep["tau_ms_unique"] = [float(x) for x in np.unique(tau)]
    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.where(tau > 0, -np.float32(DT_MS) / tau.astype(np.float32),
                        np.float32("-inf")).astype(np.float32)
        decay = np.exp(step)
    rep["decay_float32_max"] = float(decay.max())
    rep["mechanisms"] = [
        {"name": m.get("name"), "tau_ms": (m.get("params") or {}).get("tau_ms")}
        for m in model.cfg.metadata["circuit"]["mechanisms"]
    ]
    rep["connection_mechanisms"] = {
        r["name"]: r.get("mechanism") for r in model.cfg.metadata["circuit"]["connections"]
    }
    rep["receptor_index_unique"] = [int(x) for x in np.unique(np.asarray(resolve_receptor_index(edges)))]
    blocks = {}
    for t in ("E", "I"):
        wb = w[labels[pre] == t]
        blocks[t] = {"n_edges": int(wb.size),
                     "abs_min": float(np.abs(wb).min()), "abs_max": float(np.abs(wb).max()),
                     "abs_mean": float(np.abs(wb).mean()),
                     "n_positive": int(np.sum(wb > 0)), "n_negative": int(np.sum(wb < 0))}
    rep["weight_blocks_by_pre_type"] = blocks
    rep["connectivity_compilation"] = model.cfg.metadata.get("connectivity_compilation")
    rep["rule_compiled_n_edges"] = {r["name"]: int(r["compiled_n_edges"])
                                    for r in model.cfg.metadata["circuit"]["connections"]}
    W = np.asarray(em.W)
    rep["emitter_W_has_coupling"] = bool(W.size and np.any(W))
    return rep


def _summarize(rows, cols, n_steps, theta):
    """Population rates over the kept window and the lane's gamma call.

    The gamma call is ``run_izh._safe_peak_excess(signal, 1000.0, GAMMA_BAND)``, which wraps
    ``izh_analysis.peak_excess``, then ``izh_analysis.holds``. The signal is the 1 ms bin count
    of all cells over the kept window, built as ``run_izh.score_network_seed`` builds it. It is
    built with numpy because that function's per-spike Python lists would hold about 2e7
    spikes in the saturated library run.
    """
    discard = int(round(DISCARD_MS / DT_MS))
    steps_per_bin = int(round(1.0 / DT_MS))
    kept_s = (n_steps - discard) * DT_MS / 1000.0
    keep = rows >= discard
    e_hz = float(np.sum(keep & (cols < N_E)) / (N_E * kept_s))
    i_hz = float(np.sum(keep & (cols >= N_E)) / (N_I * kept_s))
    n_bins = (n_steps - discard) // steps_per_bin
    bins = (rows[keep] - discard) // steps_per_bin
    signal = np.bincount(bins, minlength=n_bins)[:n_bins].astype(float)
    gamma = lane._safe_peak_excess(signal, 1000.0, lane.GAMMA_BAND)
    return {"E_hz": e_hz, "I_hz": i_hz, "n_spikes": int(rows.size),
            "gamma_f_pk_hz": gamma["f_pk"], "gamma_pe": gamma["pe"],
            "gamma_local_max": gamma["local_max"], "gamma_edge_ok": gamma["edge_ok"],
            "gamma_band_call": bool(lane.izh.holds(gamma, theta))}


def _py(x):
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _git_sha():
    try:
        return subprocess.run(["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _write(out, doc):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(_py(doc), indent=1), encoding="utf-8")


def cmd_pilot(out, t_s):
    check_out(out)
    theta, alpha_band = lane.load_calibration()
    doc = {"packet": "9C-E1 pilot 2 (unscored)", "code_sha": _git_sha(), "t_s": t_s,
           "dt_ms": DT_MS, "discard_ms": DISCARD_MS, "theta": theta,
           "alpha_band": list(alpha_band),
           "synaptic": {"receptor": "instantaneous", "tau_ms_declared": INSTANTANEOUS_TAU_MS,
                        "dt_ms": DT_MS},
           "runs": [], "realization": {}, "stop": None}
    _write(out, doc)
    models = {}
    for seed in (0, 1):
        rows, cols, n_steps, tk = run_kernel_seed(seed, t_s)
        ker = dict(path="kernel", seed=seed, **tk, **_summarize(rows, cols, n_steps, theta))
        doc["runs"].append(ker)
        _write(out, doc)

        t0 = time.perf_counter()
        models[seed] = build_core_model(seed)
        construct = time.perf_counter() - t0
        doc["realization"][f"seed{seed}"] = realization_checks(models[seed], seed)
        _write(out, doc)
        rows, cols, n_steps, tl = run_core_seed(seed, t_s, model=models[seed])
        lib = _summarize(rows, cols, n_steps, theta)
        doc["runs"].append(dict(path="library", seed=seed, **dict(tl, construct_s=construct), **lib))
        rel_e = abs(lib["E_hz"] - ker["E_hz"]) / ker["E_hz"] if ker["E_hz"] else float("inf")
        rel_i = abs(lib["I_hz"] - ker["I_hz"]) / ker["I_hz"] if ker["I_hz"] else float("inf")
        doc[f"rate_rel_diff_seed{seed}"] = {"E": rel_e, "I": rel_i}
        if max(rel_e, rel_i) > PE_CHANGE_STOP:
            doc["stop"] = (f"library vs kernel rate differs by more than {PE_CHANGE_STOP:.0%} "
                           f"at seed {seed} (E {rel_e:.3f}, I {rel_i:.3f}); later runs not run")
            _write(out, doc)
            print(doc["stop"])
            return 3
        _write(out, doc)

    lib0, lib1 = doc["runs"][1], doc["runs"][3]
    pre0, post0, w0 = edge_arrays(models[0])
    pre1, post1, w1 = edge_arrays(models[1])
    doc["seed_change_library"] = {
        "topology_identical": bool(np.array_equal(pre0, pre1) and np.array_equal(post0, post1)),
        "weights_fraction_differ": float(np.mean(w0 != w1)),
        "cell_params_max_abs_diff": {
            k: _mad(np.asarray(getattr(models[0].params["emitter"], k)),
                    np.asarray(getattr(models[1].params["emitter"], k)))
            for k in ("a", "b", "c", "d", "u0")},
        "noise_std_unchanged": doc["realization"]["seed0"]["noise_std_E_I"]
        == doc["realization"]["seed1"]["noise_std_E_I"],
        "delta_E_hz": lib1["E_hz"] - lib0["E_hz"],
        "delta_I_hz": lib1["I_hz"] - lib0["I_hz"],
        "delta_gamma_f_pk_hz": (None if lib1["gamma_f_pk_hz"] is None or lib0["gamma_f_pk_hz"] is None
                                else lib1["gamma_f_pk_hz"] - lib0["gamma_f_pk_hz"]),
        "delta_n_spikes": lib1["n_spikes"] - lib0["n_spikes"],
    }
    _write(out, doc)
    return 0


def cmd_score(seed, t_s, out, route="kernel"):
    """Score one raster through the gate-2 scorer: simulate seed, N1-N4, write JSON."""
    if route == "core":
        rows, cols, n_steps, timing = run_core_seed(seed, t_s)
    else:
        rows, cols, n_steps, timing = run_kernel_seed(seed, t_s)
    theta, alpha_band = lane.load_calibration()
    doc = {"seed": int(seed), "route": route, "t_s": float(t_s), "n_steps": int(n_steps),
           "code_sha": _git_sha(), "theta": theta, "alpha_band": list(alpha_band),
           "timing": timing, "scores": score_one_seed(rows, cols, n_steps)}
    _write(out, doc)
    print(f"score seed={seed} route={route} -> {out}")
    return doc


def cmd_scored(seeds, t_s, out, signed=False):
    """P6 scored run: gate 1 per seed, one code SHA (gate 3), core route, Q1-Q5 receipt.

    Without ``signed`` the receipt is still written but marked unscored with an
    "incomplete" overall verdict (smoke path). With ``signed`` a dirty tree
    refuses (exit 2, the lane helper); anything but seeds 2-6 at 10.2 s, a gate-1
    stop, a mixed SHA or a short run scores "incomplete".
    """
    out = Path(out)
    receipt = out if out.suffix == ".json" else out / "p6_score.json"
    sha0 = _git_sha()
    if signed and lane._v8_tree_dirty():
        raise SystemExit(2)
    theta, alpha_band = lane.load_calibration()
    seeds = [int(s) for s in seeds]
    realization, runs, scores, stop = {}, {}, {}, None
    for seed in seeds:
        model = build_core_model(seed)
        rep = realization_checks(model, seed)
        rep["ok"] = realization_ok(rep)
        realization[str(seed)] = rep
        if not rep["ok"]:
            stop = f"gate 1 (realization) fails at seed {seed}; later seeds not run"
            break
        rows, cols, n_steps, timing = run_core_seed(seed, float(t_s), model=model)
        runs[str(seed)] = {"n_steps": int(n_steps), **timing}
        res = score_seed(rows, cols, n_steps, theta, alpha_band)
        scores[str(seed)] = {nid: res[nid] for nid in ("N1", "N2", "N3", "N4")}
    sha1 = _git_sha()
    shas = sorted({sha0, sha1})
    one_sha = len(shas) == 1 and all(s not in ("", "unknown") for s in shas)
    want_steps = int(round(P6_T_S * 1000.0 / DT_MS))
    short = sorted(s for s, r in runs.items() if r["n_steps"] != want_steps)
    asked = sorted(str(s) for s in seeds)
    scored_all_asked = stop is None and sorted(scores) == asked
    complete = (bool(signed) and scored_all_asked and one_sha and seeds == list(P6_SEEDS)
                and abs(float(t_s) - P6_T_S) < 1e-12 and not short
                and asked == [str(s) for s in P6_SEEDS])
    if complete:
        verdict = evaluate_p6_verdicts(scores)
    else:
        verdict = {"means": {},
                   "criteria": {q: {"verdict": "incomplete", "numbers": {}}
                                for q in ("Q1", "Q2", "Q3", "Q4", "Q5")},
                   "overall": "incomplete"}
    overall = verdict["overall"]
    doc = {"prereg": "PREREG_P6_v2_signed.md", "prereg_sha256": prereg_sha256(),
           "signed": bool(signed), "scored": bool(signed) and complete,
           "complete": complete, "stop": stop, "short_seeds": short,
           "seeds": seeds, "t_s": float(t_s), "dt_ms": DT_MS,
           "discard_ms": DISCARD_MS, "theta": theta, "alpha_band": list(alpha_band),
           "gamma_band": list(lane.GAMMA_BAND),
           "code_sha": sha1, "run_code_shas": shas,
           "realization": realization, "runs": runs, "scores": scores,
           "means": verdict["means"], "criteria": verdict["criteria"], "overall": overall}
    _write(receipt, doc)
    print(f"scored seeds={seeds} t_s={t_s} signed={bool(signed)} overall={overall} -> {receipt}")
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pilot", help="kernel vs library pilot, seeds 0 and 1 (unscored)")
    p.add_argument("--out", required=True)
    p.add_argument("--t-s", type=float, default=float(lane.FIG3_T))
    s = sub.add_parser("score", help="N1-N4 for one raster through the gate-2 scorer")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--t-s", type=float, default=float(lane.FIG3_T))
    s.add_argument("--route", choices=("kernel", "core"), default="kernel")
    s.add_argument("--out", required=True)
    s.add_argument("--signed", action="store_true")
    c = sub.add_parser("scored", help="P6 scored run: gates 1-3, core route, Q1-Q5 receipt")
    c.add_argument("--seeds", type=int, nargs="+", required=True)
    c.add_argument("--t-s", type=float, default=P6_T_S)
    c.add_argument("--out", required=True)
    c.add_argument("--signed", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "score":
        lane.check_out(args.out, args.signed)
        cmd_score(args.seed, args.t_s, args.out, args.route)
        return 0
    if args.cmd == "scored":
        lane.check_out(args.out, args.signed)
        doc = cmd_scored(args.seeds, args.t_s, args.out, args.signed)
        return 0 if doc["overall"] == "pass" else 1
    check_out(args.out)
    return cmd_pilot(args.out, args.t_s)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
