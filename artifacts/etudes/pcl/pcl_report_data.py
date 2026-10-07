"""Traces for the PCL report: Fig. 2 network and K1 column dynamics, laminar LFP proxy.

Fig. 2 network: K0b seed-10 trained weights (time averaged, ``fig2_hdp_avg.json``) on one
fresh test sample, with and without the learned inhibition; membrane potential and spikes.
K1 column: trained weights (``--k1`` npz written by ``pcl_column_hdp.py``), one sequence per
orientation, PCL vs distant + top-down inhibition removed; spikes, source proxy, and the
laminar LFP/CSD proxy from ``jaxfne.fields.proxy.project_laminar_sources``. Laminar depths
are assigned, not modelled: input relays 0.55 (layer-4-like), simple cells 0.35, complex
cells 0.2; the field is a RELATIVE_PROXY.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcl_column as C  # noqa: E402
import pcl_column_hdp as H  # noqa: E402
import pcl_fig2 as F  # noqa: E402
import pcl_fig2_hdp as K  # noqa: E402
import pcl_hdp_rule as R  # noqa: E402
from jaxfne._hdp_registrable_kernel import simulate_edge_recurrent_izhikevich_hdp_registered as sim  # noqa: E402
from jaxfne.fields.proxy import project_laminar_sources  # noqa: E402

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)
HERE = Path(__file__).resolve().parent


def fig2_traces():
    res = json.loads((HERE / "fig2_hdp_avg.json").read_text())
    p, run = res["params"], res["runs"][0]  # seed 10
    K.DT = p["dt_ms"]
    w = np.asarray(run["w_avg_last_epoch_mV"]) * p["scale"]
    sample = F.make_sample(np.random.default_rng(1234))
    sched = K.schedule(sample, p["pulse_amp"], F.T_MS)
    out = {}
    for name, wk in (("pcl", w), ("no_inh", np.zeros(3))):
        st = {"v": jnp.full(4, -65.0), "u": jnp.full(4, -13.0), "prev_spikes": jnp.zeros(4),
              "syn_state": jnp.zeros(3), "w_final": -jnp.asarray(wk, jnp.float32),
              "aux_final": jnp.zeros((3, 3), jnp.float32)}
        v, s, _, _ = sim(K.izh(4), K.edges(wk, 3), int(sched.shape[0]), K.DT, jax.random.PRNGKey(0),
                         drive_schedule=sched, noise_scale=0.0, init_state=st, hdp_rule=R.NAME,
                         hdp_rule_params=K.null_params(3), record_weight_trace=False,
                         plasticity_mask=jnp.zeros(3))
        out[f"f2_v_{name}"], out[f"f2_s_{name}"] = np.asarray(v), np.asarray(s)
    out["f2_dt"] = K.DT
    out["f2_w_mV"] = np.asarray(run["w_avg_last_epoch_mV"])
    return out


def positions():
    pos = np.zeros((H.N, 3), np.float32)
    pix = np.arange(C.G * C.G)
    pos[:H.N_IN, 0] = np.repeat(pix % C.G, 2) / (C.G - 1)
    pos[:H.N_IN, 1] = np.repeat(pix // C.G, 2) / (C.G - 1)
    pos[:H.N_IN, 2] = 0.55
    py, px = np.divmod(np.arange(C.NS) // C.FS, C.NP_S)
    pos[H.S0:H.C0, 0] = (C.STRIDE * px + C.RF / 2) / C.G
    pos[H.S0:H.C0, 1] = (C.STRIDE * py + C.RF / 2) / C.G
    pos[H.S0:H.C0, 2] = 0.35
    pos[H.C0:, 2] = 0.2
    return jnp.asarray(pos)


def column_traces(k1_npz, k1_json):
    p = json.loads(Path(k1_json).read_text())["params"]
    w = np.load(k1_npz)["w"]
    K.DT = H.DT
    runner = H.make_runner(p["scale"], p["pulse_amp"], (), full=True)
    pos = positions()
    out = {"col_dt": H.DT}
    for name, wk in (("pcl", w), ("no_inh", np.where(np.isin(H.TYP, (2, 3)), 0.0, w))):
        lfp, csd, counts, pop = [], [], [], []
        for ori in range(C.N_ORI):
            v, s, src = runner(jnp.asarray(wk, jnp.float32), jax.random.PRNGKey(100 + ori), ori, 1.0)
            fld = project_laminar_sources(src, pos, n_contacts=16)
            lfp.append(np.asarray(fld.lfp_proxy)), csd.append(np.asarray(fld.csd_proxy))
            counts.append(np.asarray(s).sum(0))
            sa = np.asarray(src)
            pop.append(np.stack([sa[:, sl].mean(1) for sl in (slice(0, H.S0), slice(H.S0, H.C0), slice(H.C0, H.N))], -1))
            if ori == 2:
                out[f"col_s_{name}"], out[f"col_v_{name}"] = np.asarray(s), np.asarray(v)[:, H.S0:H.S0 + 32]
        out[f"col_lfp_{name}"], out[f"col_csd_{name}"] = np.array(lfp), np.array(csd)
        out[f"col_counts_{name}"] = np.array(counts)
        out[f"col_popsrc_{name}"] = np.array(pop)  # [ori, T, relay/simple/complex] mean source proxy
    out["contact_depths"] = np.linspace(0.0, 1.0, 16)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k1", type=Path, required=True, help="K1 weights npz from pcl_column_hdp.py")
    ap.add_argument("--k1-json", type=Path, default=HERE / "confirm" / "k1_seed10.json")
    ap.add_argument("--out", type=Path, default=HERE / "report" / "report_data.npz")
    a = ap.parse_args()
    R.register()
    data = fig2_traces()
    data.update(column_traces(a.k1, a.k1_json))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **data)
    print({k: np.shape(v) for k, v in data.items()})


if __name__ == "__main__":
    main()
