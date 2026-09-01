"""0.4.17 closure: regression tests for the deep-reconciliation HP items.

HP-01 tensor RuntimeConfiguration wiring (dtype/device/jit/vmap)
HP-05 HDP + nonzero delay loud-fail
HP-07 population-restoring H domain (h_state_dim == 2)
HP-09 tensor identity / provenance through run_receipt
HP-04 sdist hygiene (forbidden path prefixes absent)
HP-03 conceptual CircuitSpec hygiene (construct rejects experimental class)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.neuronal_tensor import (
    RuntimeConfiguration,
    NeuronalTensor,
    Area,
    Layer,
    NeuronType,
    Geometry3D,
)

ROOT = Path(__file__).resolve().parent.parent


def _mk_tensor(depth: float) -> NeuronalTensor:
    area = Area(name="A", layers=[
        Layer(name="L2/3", n_neurons=20,
              neuron_types=(NeuronType(name="E", fraction=0.8),
                            NeuronType(name="I", fraction=0.2)),
              geometry=Geometry3D(x_range=(0.0, 1.0), y_range=(0.0, 0.5),
                                  z_range=(0.0, depth))),
    ])
    return NeuronalTensor(areas=(area,), name=f"t-{depth}")


# --------------------------------------------------------------------------- #
# HP-01 — tensor-path RuntimeConfiguration wiring
# --------------------------------------------------------------------------- #

class TestTensorRuntimeConfiguration:
    def test_dtype_float64_honored_under_x64(self):
        import jax
        if not bool(jax.config.read("jax_enable_x64")):
            pytest.skip("requires jax_enable_x64")
        t = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
        m = jtfne.construct(t, RuntimeConfiguration(seed=1, duration_ms=20.0,
                                                    dt_ms=1.0, dtype="float64"))
        s = jtfne.simulate(m)
        assert str(s.V_m.dtype).startswith("float64")
        assert m.cfg.metadata.get("dtype") == "float64"

    def test_default_dtype_stays_float32(self):
        t = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
        m = jtfne.construct(t, RuntimeConfiguration(seed=1, duration_ms=20.0, dt_ms=1.0))
        assert m.cfg.metadata.get("dtype") == "float32"

    def test_device_jit_vmap_forwarded_into_metadata(self):
        t = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
        m = jtfne.construct(t, RuntimeConfiguration(seed=1, duration_ms=20.0,
                                                    dt_ms=1.0, jit=True, vmap=True,
                                                    device="cpu"))
        assert m.cfg.metadata.get("jit") is True
        assert m.cfg.metadata.get("vmap") is True
        assert m.cfg.metadata.get("backend") == "cpu"

    def test_noarg_simulate_inherits_tensor_runtime(self):
        t = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
        m = jtfne.construct(t, RuntimeConfiguration(seed=3, duration_ms=40.0, dt_ms=1.0))
        s = jtfne.simulate(m)
        assert s.time_ms.shape[0] == 40  # duration_ms inherited


# --------------------------------------------------------------------------- #
# HP-05 — HDP + nonzero delay must be rejected loudly
# --------------------------------------------------------------------------- #

class TestHdpDelaySupported:
    def test_hdp_nonzero_delay_supported(self):
        from dataclasses import replace
        import jax.numpy as jnp
        cfg = jtfne.suite2_net1_config(seed=7, n=3, duration_ms=50.0, dt_ms=1.0)
        cfg = cfg.hdp(enable_hdp=True, hdp_params={"noise_scale": 0.0})
        model = jtfne.construct(cfg)
        edges = model.params["edge_list"]
        ds = jnp.full((edges.n_edges,), 10, dtype=jnp.int32)
        new_edges = replace(edges, delay_steps=ds)
        object.__setattr__(model, "params", {**model.params, "edge_list": new_edges})
        sim = jtfne.simulation(
            duration_ms=50.0, dt_ms=1.0, seed=7,
            runtime=jtfne.RuntimeConfig(enable_hdp=True, recurrent_backend="edge_list",
                                        hdp_params={"noise_scale": 0.0}),
        )
        s = jtfne.simulate(model, sim)
        assert s.V_m.shape == (50, 3)
        # simulate_batch must also succeed
        batch_out = model.simulate_batch(sim, n_seeds=2)
        assert batch_out["V_m"].shape == (2, 50, 3)

    def test_hdp_zero_delay_still_works(self):
        cfg = jtfne.suite2_net1_config(seed=7, n=3, duration_ms=50.0, dt_ms=1.0)
        cfg = cfg.hdp(enable_hdp=True, hdp_params={"noise_scale": 0.0})
        model = jtfne.construct(cfg)
        s = jtfne.simulate(model)
        assert s.V_m.shape[1] == 3

    def test_hdp_nonzero_delay_supported_on_continuation_path(self):
        """Finite delays must also work on full-state continuation path."""
        from dataclasses import replace
        import jax.numpy as jnp
        cfg = jtfne.suite2_net1_config(seed=7, n=3, duration_ms=80.0, dt_ms=1.0)
        model = jtfne.construct(cfg)
        edges = model.params["edge_list"]
        ds = jnp.full((edges.n_edges,), 8, dtype=jnp.int32)
        new_edges = replace(edges, delay_steps=ds)
        object.__setattr__(model, "params", {**model.params, "edge_list": new_edges})
        rt = jtfne.RuntimeConfig(recurrent_backend="edge_list", enable_hdp=True,
                                 hdp_params={"noise_scale": 0.0})
        s_cont, cont_st = model.simulate(
            jtfne.simulation(duration_ms=40.0, dt_ms=1.0, seed=7, runtime=rt),
            return_state=True,
        )
        assert s_cont.V_m.shape == (40, 3)
        assert cont_st.delay_state is not None


# --------------------------------------------------------------------------- #
# HP-07 — population-restoring H domain
# --------------------------------------------------------------------------- #

class TestPopulationHDomain:
    def test_population_h_dim_must_be_two(self):
        import jax.numpy as jnp
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "tests"))
        from test_hdp_population_restoring import _mcc3_model
        model, mei_mask, e_mask = _mcc3_model()
        for dim in (1, 3):
            hp = {
                "K_HDP": 0.0, "h_state_locality": "population", "h_state_dim": dim,
                "controller_B": jnp.eye(dim), "controller_lambda": 0.5,
                "controller_tau_H_s": 1.0, "controller_tau_theta_s": 1.0,
                "controller_rate_setpoint_E_hz": 10.9,
                "controller_rate_setpoint_I_hz": 9.14,
                "controller_theta_S_init": (1.0, 1.0),
                "m_ei_edge_mask": mei_mask.astype(bool),
                "e_neuron_mask": e_mask.astype(bool),
                "theta_m_EI_bounds": (0.1, 5.0), "theta_eta_a_bounds": (0.25, 4.0),
            }
            runtime = jtfne.RuntimeConfig(enable_hdp=True, recurrent_backend="edge_list",
                                          jit=False, hdp_params=hp)
            with pytest.raises(ValueError, match="two-coordinate controller"):
                model.simulate(jtfne.simulation(duration_ms=50.0, dt_ms=0.1,
                                                seed=17, runtime=runtime))


# --------------------------------------------------------------------------- #
# HP-09 — tensor identity / provenance through run_receipt
# --------------------------------------------------------------------------- #

class TestProvenanceIdentity:
    def test_distinct_geometry_distinct_receipt(self):
        t1, t2 = _mk_tensor(0.5), _mk_tensor(2.0)
        assert t1 != t2
        m1 = jtfne.construct(t1, RuntimeConfiguration(seed=1, duration_ms=50.0, dt_ms=1.0))
        m2 = jtfne.construct(t2, RuntimeConfiguration(seed=1, duration_ms=50.0, dt_ms=1.0))
        r1 = m1.run_receipt(jtfne.simulate(m1))
        r2 = m2.run_receipt(jtfne.simulate(m2))
        assert r1.config_hash != r2.config_hash
        assert r1.receipt_id != r2.receipt_id

    def test_same_tensor_stable_receipt(self):
        t1 = _mk_tensor(0.5)
        m1 = jtfne.construct(t1, RuntimeConfiguration(seed=1, duration_ms=50.0, dt_ms=1.0))
        m1b = jtfne.construct(t1, RuntimeConfiguration(seed=1, duration_ms=50.0, dt_ms=1.0))
        r1 = m1.run_receipt(jtfne.simulate(m1))
        r1b = m1b.run_receipt(jtfne.simulate(m1b))
        assert r1.receipt_id == r1b.receipt_id

    def test_distinct_development_seed_distinct_receipt(self):
        g = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")
        d0 = jtfne.develop(g, seed=0)
        d7 = jtfne.develop(g, seed=7)
        assert d0.provenance["genome_sha256"] == d7.provenance["genome_sha256"]
        m0 = jtfne.construct(d0, RuntimeConfiguration(seed=1, duration_ms=20.0, dt_ms=1.0))
        m7 = jtfne.construct(d7, RuntimeConfiguration(seed=1, duration_ms=20.0, dt_ms=1.0))
        r0 = m0.run_receipt(jtfne.simulate(m0))
        r7 = m7.run_receipt(jtfne.simulate(m7))
        assert r0.receipt_id != r7.receipt_id

    def test_tensor_identity_in_metadata(self):
        g = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")
        dt = jtfne.develop(g, seed=0)
        m = jtfne.construct(dt, RuntimeConfiguration(seed=1, duration_ms=20.0, dt_ms=1.0))
        assert "tensor_identity" in m.cfg.metadata
        assert len(m.cfg.metadata["tensor_identity"]) == 64

    def test_receipt_stable_across_save_load_roundtrip(self, tmp_path):
        """F5: receipt identity is stable across the documented persistence
        path (save_neuronal_tensor strips in-memory provenance; the structural
        digest must still match so the same phenotype gets the same receipt)."""
        import os
        from jaxfne.neuronal_tensor import save_neuronal_tensor, load_neuronal_tensor
        g = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")
        dt = jtfne.develop(g, seed=0)
        m = jtfne.construct(dt, RuntimeConfiguration(seed=1, duration_ms=20.0, dt_ms=1.0))
        r = m.run_receipt(jtfne.simulate(m))
        p = os.path.join(str(tmp_path), "t.json")
        save_neuronal_tensor(dt, p)
        dt2 = load_neuronal_tensor(p)
        m2 = jtfne.construct(dt2, RuntimeConfiguration(seed=1, duration_ms=20.0, dt_ms=1.0))
        r2 = m2.run_receipt(jtfne.simulate(m2))
        assert r.receipt_id == r2.receipt_id


# --------------------------------------------------------------------------- #
# Fresh-review residuals — F7 backend validation, F3 null boundary
# --------------------------------------------------------------------------- #

class TestBackendValidation:
    def test_bogus_backend_rejected(self):
        with pytest.raises(ValueError, match="backend must be one of"):
            jtfne.RuntimeConfig(backend="bogus")

    def test_valid_backends_accepted(self):
        for b in ("auto", "cpu", "gpu", "tpu"):
            jtfne.RuntimeConfig(backend=b)

    def test_tensor_device_forwarded_and_validated(self):
        # tensor-path device maps to RuntimeConfig.backend; bogus rejected
        t = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
        with pytest.raises(ValueError, match="backend must be one of"):
            jtfne.simulate(
                jtfne.construct(t, RuntimeConfiguration(seed=1, duration_ms=20.0,
                                                        dt_ms=1.0, device="bogus"))
            )


class TestDisconnectedNullHdpBoundary:
    def test_default_gains_floor_scale(self):
        """Shipped/default HDP gains: disconnected_null keeps |w| at floor."""
        cfg = jtfne.suite2_net1_config(seed=7, n=3, duration_ms=100.0, dt_ms=1.0)
        cfg = cfg.hdp(enable_hdp=True, hdp_params={"noise_scale": 0.0})
        model = jtfne.construct(cfg)
        jtfne.simulate(model, ablation="disconnected_null")
        diag = model.last_hdp_diagnostics()
        wf = np.asarray(diag["w_final"]).reshape(-1)
        assert np.abs(wf).max() <= 2.0 * 1.0e-3

    def test_elevated_gains_can_grow_above_floor(self):
        """Boundary documented in guides/hdp.md: explicit elevated HDP gains
        (C_spike>0) can drive |w| above the floor even with zeroed initial
        weights — disconnected_null zeroes initial weights, not the dynamics."""
        cfg = jtfne.suite2_net1_config(seed=7, n=3, duration_ms=100.0, dt_ms=1.0)
        model = jtfne.construct(cfg)
        sim = jtfne.simulation(
            duration_ms=100.0, dt_ms=1.0, seed=7, ablation="disconnected_null",
            runtime=jtfne.RuntimeConfig(
                enable_hdp=True, recurrent_backend="edge_list",
                hdp_params={"K_HDP": 2.0, "C_spike": 0.5},
            ),
        )
        model.simulate(sim)
        diag = model.last_hdp_diagnostics()
        wf = np.asarray(diag["w_final"]).reshape(-1)
        assert np.abs(wf).max() > 2.0 * 1.0e-3


# --------------------------------------------------------------------------- #
# HP-04 — sdist hygiene
# --------------------------------------------------------------------------- #

class TestSdistHygiene:
    @pytest.mark.slow
    def test_sdist_excludes_harness_trees(self, tmp_path):
        import tarfile
        build = tmp_path / "build"
        build.mkdir()
        subprocess.run(
            [sys.executable, "-m", "build", "--sdist", "--outdir", str(build), str(ROOT)],
            check=True, capture_output=True, cwd=ROOT,
        )
        sdist = next(build.glob("*.tar.gz"))
        names = tarfile.open(sdist).getnames()
        joined = "\n".join(names)
        for bad in ("/.opencode/", "/.cursor/", "/node_modules/", "/receipts/"):
            assert bad not in joined, f"sdist leaks {bad!r}"
        assert any("canonical-v1-column-1000n.json" in n for n in names)


# --------------------------------------------------------------------------- #
# HP-03 — conceptual CircuitSpec hygiene
# --------------------------------------------------------------------------- #

class TestCircuitSpecHygiene:
    def test_experimental_circuitspec_not_accepted_by_construct(self):
        from jaxfne.experimental_hpc.contracts import CircuitSpec
        from jaxfne._construct_core import construct
        with pytest.raises((NotImplementedError, TypeError, ValueError)):
            construct(CircuitSpec())


# --------------------------------------------------------------------------- #
# HP-02 — objective fluent grammar
# --------------------------------------------------------------------------- #

class TestObjectiveGrammar:
    def test_objective_fluent_loss_builder(self):
        o = jtfne.objective().loss(name="rate", metric="spike_rate_hz", target=10.0)
        assert o.losses[0]["metric"] == "spike_rate_hz"
        assert o.losses[0]["target"] == 10.0

    def test_objective_factory_takes_no_args(self):
        with pytest.raises(TypeError):
            jtfne.objective(name="rate")  # type: ignore[misc]
