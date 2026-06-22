"""End-to-end proof that the Tensor-operator pipeline composes on a custom,
non-canonical Configuration -- not just the one canonical column used to
validate cable_filter_sources/synaptic_current_tensor/csd_tensor individually.

Pipeline: custom cfg -> construct -> simulate -> raw source ->
synaptic_current_tensor (optional) -> cable_filter_sources ->
project_laminar_sources (LFP + CSD via csd_tensor) ->
eeg_proxy_transform / meg_proxy_transform.
"""

import numpy as np
import jax.numpy as jnp

import jaxfne as jtfne

MECHANISM_MAP = {"E": "AMPA", "PV": "GABA_A", "SST": "GABA_B", "VIP": "GABA_B"}


def _build_custom_model():
    cfg = jtfne.laminar_cortex_config(
        seed=7,
        duration_ms=100.0,
        dt_ms=0.5,
        areas=("V1",),
        layers=("L1", "L4", "L6"),                       # non-canonical: 3 layers
        cell_types={"E": 0.6, "PV": 0.25, "SST": 0.15},   # non-canonical mix
        n=40,
        emitter="izhikevich",
    )
    model = jtfne.construct(cfg)
    return model, cfg


DURATION_MS = 100.0
DT_MS = 0.5


def test_full_tensor_pipeline_composes_on_custom_config():
    model, cfg = _build_custom_model()
    sig = jtfne.simulate(model, duration_ms=DURATION_MS, dt_ms=DT_MS, seed=3)

    nt = model.neuron_table()
    cell_type = np.array([row["cell_type"] for row in nt])
    depth_z = np.array([row["z"] for row in nt])
    n = len(nt)

    source_native = np.asarray(jtfne.get_signal(sig, "source"))
    assert source_native.shape[1] == n
    assert np.all(np.isfinite(source_native))

    # stage 1: Synaptic Tensor (optional pre-filter)
    mechanisms = [MECHANISM_MAP[c] for c in cell_type]
    tau_syn_ms = jtfne.synaptic_tau_from_mechanism(mechanisms)
    syn_filtered = jtfne.synaptic_current_tensor(jnp.asarray(source_native), tau_syn_ms, dt_ms=DT_MS)
    assert syn_filtered.shape == source_native.shape
    assert jnp.all(jnp.isfinite(syn_filtered))

    # stage 2: Cable-filter Tensor (LFP frequency-domain stage)
    tau_cable_s = jtfne.cable_filter_tau(cell_type, depth_z)
    cable_filtered = jtfne.cable_filter_sources(jnp.asarray(source_native), tau_cable_s, dt_ms=DT_MS, order=2)
    assert cable_filtered.shape == source_native.shape
    assert jnp.all(jnp.isfinite(cable_filtered))

    # stage 3: readout tensors (LFP, CSD via csd_tensor, EEG, MEG)
    positions = np.stack([np.zeros(n), np.zeros(n), depth_z], axis=1)
    fo = jtfne.project_laminar_sources(cable_filtered, positions, n_contacts=12)
    assert jnp.all(jnp.isfinite(fo.lfp_proxy))
    assert jnp.all(jnp.isfinite(fo.csd_proxy))

    dz = fo.contact_depths[1] - fo.contact_depths[0]
    csd_standalone = jtfne.csd_tensor(fo.phi_e_proxy, dz)
    np.testing.assert_allclose(np.asarray(csd_standalone), np.asarray(fo.csd_proxy))

    n_contacts = fo.lfp_proxy.shape[1]
    rng = np.random.default_rng(0)
    eeg_leadfield = jnp.asarray(rng.normal(size=(8, n_contacts)).astype(np.float32))
    meg_leadfield = jnp.asarray(rng.normal(size=(6, n_contacts)).astype(np.float32))

    eeg = jtfne.eeg_proxy_transform(fo.lfp_proxy, eeg_leadfield)
    meg = jtfne.meg_proxy_transform(fo.lfp_proxy, meg_leadfield)
    assert eeg.shape == (source_native.shape[0], 8)
    assert meg.shape == (source_native.shape[0], 6)
    assert jnp.all(jnp.isfinite(eeg))
    assert jnp.all(jnp.isfinite(meg))
