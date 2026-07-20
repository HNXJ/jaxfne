"""1000-neuron fast float32 laminar default cortex LFP+CSD+HDP smoke test
(plans.json:test-1000n-fast-laminar-lfp-csd-hdp).

N=1000, float32, dt_ms=0.1, duration_ms=1000.0, DEFAULT_HDP preset
(K_ctrl=5.0/tau_0_ms=200.0/rho_passive=0.0), LFP-proxy + CSD-proxy readout.
Marked slow -- a real ~70s run at this scale, not a fast-lane test.
"""
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.hdp_network import DEFAULT_HDP
from jaxfne._runtime_config import RuntimeConfig

pytestmark = pytest.mark.slow


def test_1000n_laminar_hdp_lfp_csd_smoke():
    tensor = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
    model = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=0, duration_ms=1000.0, dt_ms=0.1))

    signals = jtfne.simulate(
        model, duration_ms=1000.0, dt_ms=0.1, seed=0, record_fields=True,
        runtime=RuntimeConfig(enable_hdp=True, hdp_params=dict(DEFAULT_HDP)),
    )

    V_m = np.asarray(signals.V_m)
    spikes = np.asarray(signals.spikes)
    assert V_m.shape == (10_000, 1000)  # 1000ms / 0.1ms, 1000 neurons
    assert np.all(np.isfinite(V_m))
    assert np.all(np.abs(V_m) < 150.0), f"|V_m| > 150mV indicates a blowup, got max {np.abs(V_m).max()}"
    assert np.all(np.isfinite(spikes))

    assert signals.field is not None, "record_fields=True must produce a FieldOutput"
    lfp_proxy = np.asarray(signals.field.lfp_proxy)
    csd_proxy = np.asarray(signals.field.csd_proxy)
    assert lfp_proxy.shape == (10_000, 16)
    assert csd_proxy.shape == (10_000, 16)
    assert np.all(np.isfinite(lfp_proxy))
    assert np.all(np.isfinite(csd_proxy))

    diag = model.last_hdp_diagnostics()
    assert diag is not None, "HDP must be active with explicit RuntimeConfig(enable_hdp=True)"
    H_final = np.asarray(diag["H_final"])
    assert H_final.shape == (1000,)
    assert np.all(np.isfinite(H_final))
    # DEFAULT_HDP's K_ctrl=5.0 pins H tightly near equilibrium 1.0 -- verified
    # empirically at this scale: mean 1.0006, std 0.001.
    assert abs(float(H_final.mean()) - 1.0) < 0.05
    assert float(H_final.std()) < 0.1

    kappa = jtfne.kappa_synchrony(spikes, dt_ms=0.1)
    assert np.isfinite(kappa)
    # Asynchronous-irregular regime (per project memory: low-synchrony-kappa
    # for spectrolaminar suites) -- verified empirically ~0.018 at this scale.
    assert kappa < 0.2, f"kappa_synchrony={kappa} is too high for an async-irregular regime"
