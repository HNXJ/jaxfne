"""Spectrolaminar-suite replication: targeted drive boost to E@L2/3, E+PV@L4,
E@L5/L6, swept over three magnitudes, 500ms trials.

Builds the 1000-neuron canonical column ONCE (DEFAULT_LAYERS, 5-layer:
L1, L2/3, L4, L5, L6) and reuses it across drive values (construct() is the
slow step). For each drive boost value, adds a uniform additive offset to the
native per-neuron Izhikevich drive for exactly three groups:
  - E in L5 and L6
  - E and PV in L4
  - E in L2/3
via Model.with_emitter_parameters(drive_per_neuron=...), then runs the
standard-path spectrolaminar_suite_3panel (n_trials trials @ duration_ms=500).

Writes scripts/out/spectrolaminar_drive_sweep_boost_<value>.png per value.
"""
import numpy as np

import jaxfne as jtfne

OUT_DIR = "scripts/out"
DRIVE_BOOSTS = (1.0, 3.0, 5.0)  # three values, additive on top of native drive
N_TRIALS = 8
DURATION_MS = 500.0
DT_MS = 0.5


def build_column(n: int = 1000):
    cfg = (
        jtfne.build_laminar_column("V1", n=n, ei_profile="canonical")
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m", "LFP", "CSD"], n_contacts=24)
        .field(domain="laminar_column", conductivity="proxy")
    )
    return jtfne.construct(cfg)


def boosted_drive_array(model, boost: float) -> np.ndarray:
    nt = model.neuron_table()
    base = np.asarray(model.params["emitter"].drive, dtype=float).copy()
    for i, r in enumerate(nt):
        layer, ct = r["layer"], r["cell_type"]
        targeted = (
            (ct == "E" and layer in ("L5", "L6"))
            or (layer == "L4" and ct in ("E", "PV"))
            or (ct == "E" and layer == "L2/3")
        )
        if targeted:
            base[i] += boost
    return base


def main() -> None:
    jtfne.enable_x64()
    model = build_column()
    nt = model.neuron_table()
    n_targeted = sum(
        1
        for r in nt
        if (r["cell_type"] == "E" and r["layer"] in ("L5", "L6"))
        or (r["layer"] == "L4" and r["cell_type"] in ("E", "PV"))
        or (r["cell_type"] == "E" and r["layer"] == "L2/3")
    )
    print(f"targeted neurons: {n_targeted} / {len(nt)}")

    for boost in DRIVE_BOOSTS:
        drive = boosted_drive_array(model, boost)
        model_v = model.with_emitter_parameters(drive_per_neuron=drive)
        stage = f"drive_boost_{boost:.1f}".replace(".", "p")
        figs = jtfne.vis.tutorial_panels.spectrolaminar_suite_3panel(
            model_v,
            stage=stage,
            areas=["V1"],
            output_dir=None,
            n_trials=N_TRIALS,
            duration_ms=DURATION_MS,
            dt_ms=DT_MS,
            seed=0,
            signal="csd",
            recurrent_backend="edge_list",
        )
        out_path = f"{OUT_DIR}/spectrolaminar_drive_sweep_boost_{boost:.1f}.png".replace(
            "boost_1.0", "boost_1p0"
        ).replace("boost_3.0", "boost_3p0").replace("boost_5.0", "boost_5p0")
        figs["V1"].savefig(out_path, dpi=150)
        print(f"boost={boost}: wrote {out_path}")


if __name__ == "__main__":
    main()
