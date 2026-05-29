"""Suite No. 2 upgrade: package-native grammar, line limits, and smoke path."""

import json
from pathlib import Path

import jax.numpy as jnp
import pytest

import jaxfne as jtfne


NOTEBOOK = Path("tutorials/jaxfne_suite_no_2_spectrolaminar_motif.ipynb")


def test_suite2_notebook_uses_short_code_cells_and_no_local_functions():
    nb = json.loads(NOTEBOOK.read_text())
    code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for cell in code_cells:
        lines = [line for line in cell["source"].splitlines() if line.strip()]
        assert len(lines) <= 10
        assert not any(line.lstrip().startswith("def ") for line in lines)
        assert not any("class " in line for line in lines)


def test_suite2_builders_emit_constructable_configs():
    cfg_net1 = jtfne.suite2_net1_config(seed=3, n=12, duration_ms=20.0, dt_ms=0.5)
    model_net1 = jtfne.construct(cfg_net1)
    assert model_net1.summary()["n_units"] == 12
    assert len(model_net1.neuron_table()) == 12
    assert cfg_net1.metadata["uniform_3d"] is True

    cfg_v1v4 = jtfne.suite2_v1_v4_config(seed=3, n_per_area=6, duration_ms=20.0, dt_ms=0.5)
    model_v1v4 = jtfne.construct(cfg_v1v4)
    rows = model_v1v4.neuron_table()
    assert model_v1v4.summary()["n_units"] == 12
    assert {row["area"] for row in rows} == {"V1", "V4"}
    assert "feedforward" in cfg_v1v4.metadata["connectivity"]
    assert "feedback" in cfg_v1v4.metadata["connectivity"]


def test_suite2_smoke_simulation_and_tuning_are_finite():
    cfg = jtfne.suite2_net1_config(seed=4, n=10, duration_ms=20.0, dt_ms=0.5)
    model = jtfne.construct(cfg)
    sim = jtfne.suite2_simulation(seed=4, duration_ms=20.0, dt_ms=0.5)
    signals = jtfne.simulate(model, sim)
    assert signals.field is not None
    assert bool(jnp.all(jnp.isfinite(signals.V_m)))
    assert bool(jnp.all(jnp.isfinite(signals.spikes)))

    result = jtfne.suite2_tune_noise_agsdr_adam(
        model,
        simulation=sim,
        amplitudes=(0.0, 0.1),
        adam_steps=1,
        seed=4,
    )
    assert "noise_amplitude" in result.best_parameters
    assert result.summary["optimizer"] == "AGSDR_outer_finite_difference_Adam_inner"
    json.dumps(result.to_dict(), allow_nan=False)


def test_suite2_visualization_facade_smoke():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    cfg = jtfne.suite2_net1_config(seed=5, n=8, duration_ms=20.0, dt_ms=0.5)
    model = jtfne.construct(cfg)
    signals = jtfne.simulate(model, jtfne.suite2_simulation(seed=5, duration_ms=20.0, dt_ms=0.5))
    figs = [
        jtfne.vis.raster(signals),
        jtfne.vis.lfp_traces(signals),
        jtfne.vis.csd_traces(signals),
        jtfne.vis.eeg(signals),
        jtfne.vis.meg(signals),
        jtfne.vis.emm(signals),
        jtfne.vis.spectrolaminar_suite(signals),
        jtfne.vis.circuit3d(signals),
    ]
    assert all(fig is not None for fig in figs)
    for fig in figs:
        plt.close(fig)
