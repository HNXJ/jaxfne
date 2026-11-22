"""O-NUM evidence: measure the float32 differences the exactness tests compare, and their mutants.

For each comparison in the five O-NUM tests (``mac_gate_attribution.md`` #1-5) this records the
maximum absolute V_m difference (mV), the count of differing elements, whether spikes are equal,
and the same for mutants that scale one side's recurrent weights (or the Gaussian drive std) by
1 + eps. Run from the repo root: ``python scripts/internal/onum_measure.py OUT.json``.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import jaxfne as jtfne  # noqa: E402
from jaxfne._model import Model  # noqa: E402

assert "site-packages" not in jtfne.__file__, "P-025: jaxfne must resolve to the repo tree"

ROOT = Path(__file__).resolve().parents[2]
EPS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _diff(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = np.abs(a - b)
    return {"max_abs_mV": float(d.max()), "n_diff": int((d > 0).sum()), "n": int(d.size)}


def _pair(v_a, v_b, s_a=None, s_b=None):
    out = _diff(v_a, v_b)
    if s_a is not None:
        out["spikes_equal"] = bool(np.array_equal(np.asarray(s_a), np.asarray(s_b)))
    return out


def _scaled(model, eps):
    edges = model.params["edge_list"]
    w = np.asarray(edges.weight, dtype=np.float64) * (1.0 + eps)
    params = dict(model.params)
    params["edge_list"] = replace(edges, weight=jnp.asarray(w, dtype=edges.weight.dtype),
                                  weight_storage="per_edge", weight_magnitude=None,
                                  mechanism_weight_magnitude_table=None)
    return Model(cfg=model.cfg, params=params, static=dict(model.static))


def gaussian_drive():
    g = _load("test_construct_weight_range_gaussian_drive")
    model = g._ei_model()

    def run(gauss=None, hdp=None):
        rt = jtfne.RuntimeConfig(recurrent_backend="edge_list",
                                 synaptic_kernel="receptor_exponential", hdp_params=hdp)
        sim = jtfne.Simulation(duration_ms=50.0, dt_ms=0.5, seed=0, runtime=rt, gaussian_drive=gauss)
        return np.asarray(model.simulate(sim).V_m)

    ref = run(hdp={"noise_scale": 3.0})
    out = {"compare": _pair(run(gauss=g._gauss(3.0, 3.0)), ref), "mutants": {}}
    for e in EPS:
        out["mutants"][str(e)] = _pair(run(gauss=g._gauss(3.0 * (1 + e), 3.0 * (1 + e))), ref)
    return out


def composition():
    a = _load("test_composition_aspects_054")
    i = _load("test_composition_identity_054")
    res = {}

    # continuation (aspects): full vs concatenated segments, and member 0 vs solo.
    m1, m2 = a._col("V1", a.N1, 0), a._col("V2", a.N2, 1)
    ens = jtfne.connect(m1, m2, namespace=("A", "B"))
    full = a._sim(ens, a.SEED)
    seg1, st = jtfne.simulate(ens, duration_ms=20.0, dt_ms=a.DT_MS, seed=a.SEED, return_state=True)
    seg2 = jtfne.simulate(ens, duration_ms=20.0, dt_ms=a.DT_MS, seed=a.SEED, continuation=st)
    cat = np.concatenate([np.asarray(seg1.V_m), np.asarray(seg2.V_m)], axis=0)
    s0 = jtfne.ensemble_member_seed(a.SEED, 0, 2)
    solo0 = a._sim(m1, s0)
    res["continuation_segments"] = {"compare": _pair(full.V_m, cat)}
    res["continuation_member0_solo"] = {
        "compare": _pair(np.asarray(full.V_m)[:, :a.N1], solo0.V_m,
                         np.asarray(full.spikes)[:, :a.N1], solo0.spikes),
        "mutants": {str(e): _pair(np.asarray(full.V_m)[:, :a.N1], a._sim(_scaled(m1, e), s0).V_m,
                                  np.asarray(full.spikes)[:, :a.N1], a._sim(_scaled(m1, e), s0).spikes)
                    for e in EPS},
    }

    # three-area middle member vs solo.
    m = [a._col(f"V{k}", n, k) for k, n in enumerate((10, 8, 6))]
    ens3 = jtfne.connect(*m, namespace=("A", "B", "C"))
    sig = a._sim(ens3, a.SEED)
    s1 = jtfne.ensemble_member_seed(a.SEED, 1, 3)
    solo = a._sim(m[1], s1)
    mid_v, mid_s = np.asarray(sig.V_m)[:, 10:18], np.asarray(sig.spikes)[:, 10:18]
    res["three_area_middle"] = {
        "compare": _pair(mid_v, solo.V_m, mid_s, solo.spikes),
        "mutants": {str(e): (lambda r: _pair(mid_v, r.V_m, mid_s, r.spikes))(a._sim(_scaled(m[1], e), s1))
                    for e in EPS},
    }

    # ensemble slice vs solo, eager and jit.
    for tag, kw in (("slice_eager", {}), ("slice_jit", {"jit": True})):
        c1, c2 = i._col("V1", 12, 0, **kw), i._col("V2", 8, 1, **kw)
        e2 = jtfne.connect(c1, c2, namespace=("A", "B"))
        sg = i._sim(e2, i.SEED)
        t0, t1 = jtfne.ensemble_member_seed(i.SEED, 0, 2), jtfne.ensemble_member_seed(i.SEED, 1, 2)
        so0, so1 = i._sim(c1, t0), i._sim(c2, t1)
        v, s = np.asarray(sg.V_m), np.asarray(sg.spikes)
        res[tag] = {
            "compare_member0": _pair(v[:, :12], so0.V_m, s[:, :12], so0.spikes),
            "compare_member1": _pair(v[:, 12:], so1.V_m, s[:, 12:], so1.spikes),
            "mutants_member0": {str(e): (lambda r: _pair(v[:, :12], r.V_m, s[:, :12], r.spikes))(
                i._sim(_scaled(c1, e), t0)) for e in EPS},
        }
    return res


def defects():
    """Defect-class mutants: the faults each identity test exists to catch."""
    g = _load("test_construct_weight_range_gaussian_drive")
    a = _load("test_composition_aspects_054")
    out = {}

    # Gaussian drive (P-045: per-cell std equals the scalar noise_scale).
    model = g._ei_model()

    def run(gauss=None, hdp=None, seed=0):
        rt = jtfne.RuntimeConfig(recurrent_backend="edge_list",
                                 synaptic_kernel="receptor_exponential", hdp_params=hdp)
        sim = jtfne.Simulation(duration_ms=50.0, dt_ms=0.5, seed=seed, runtime=rt, gaussian_drive=gauss)
        return np.asarray(model.simulate(sim).V_m)

    ref = run(hdp={"noise_scale": 3.0})
    out["gaussian_std_as_variance"] = _pair(run(gauss=g._gauss(9.0, 9.0)), ref)
    out["gaussian_one_type_dropped"] = _pair(run(gauss=g._gauss(3.0, 0.0)), ref)
    out["gaussian_wrong_seed"] = _pair(run(gauss=g._gauss(3.0, 3.0), seed=1), ref)

    # Composition (member RNG domains, zero cross coupling, block placement, continuation state).
    m1, m2 = a._col("V1", a.N1, 0), a._col("V2", a.N2, 1)
    ens = jtfne.connect(m1, m2, namespace=("A", "B"))
    full = a._sim(ens, a.SEED)
    v, s = np.asarray(full.V_m), np.asarray(full.spikes)
    s0, s1 = (jtfne.ensemble_member_seed(a.SEED, k, 2) for k in (0, 1))
    solo0, solo1 = a._sim(m1, s0), a._sim(m2, s1)
    wrong = a._sim(m1, s1)
    out["member_seed_swapped"] = _pair(v[:, :a.N1], wrong.V_m, s[:, :a.N1], wrong.spikes)
    out["block_offset_by_one"] = _pair(v[:, 1:a.N1 + 1], solo0.V_m, s[:, 1:a.N1 + 1], solo0.spikes)
    for w in (0.05, 0.5):
        leak = jtfne.connect(m1, m2, namespace=("A", "B"), edges=[dict(
            source={"model": 0, "area": "V1", "cell_type": "E"}, target={"model": 1, "area": "V2"},
            probability=0.3, weight=w, sign="excitatory")])
        lk = a._sim(leak, a.SEED)
        out[f"cross_leak_w{w}"] = _pair(np.asarray(lk.V_m)[:, a.N1:], solo1.V_m,
                                        np.asarray(lk.spikes)[:, a.N1:], solo1.spikes)
    seg1 = jtfne.simulate(ens, duration_ms=20.0, dt_ms=a.DT_MS, seed=a.SEED)
    fresh = jtfne.simulate(ens, duration_ms=20.0, dt_ms=a.DT_MS, seed=a.SEED)
    cat = np.concatenate([np.asarray(seg1.V_m), np.asarray(fresh.V_m)], axis=0)
    out["continuation_state_lost"] = _pair(v, cat)
    return out


def main(out_path):
    rec = {
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "python": sys.version.split()[0], "jax": jax.__version__,
                 "jaxlib": jaxlib.__version__, "backend": jax.default_backend()},
        "gaussian_drive_receptor": gaussian_drive(),
        "composition": composition(),
        "defects": defects(),
    }
    Path(out_path).write_text(json.dumps(rec, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps(rec["host"]))


if __name__ == "__main__":
    main(sys.argv[1])
