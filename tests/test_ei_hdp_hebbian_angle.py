"""Config #3 of the small-network smart-test matrix (see plans.json:
smart-test-matrix-configs-2-5) -- HDP Hebbian/activity-tracking angle.

Uses DEFAULT_HDP_DESYNC (alpha=0.05, gamma=0.5 nonzero -- "responsive H",
vs DEFAULT_HDP's near-static alpha=gamma=0), rho_passive=0.0 (canonical,
per F-017 -- rho_passive alone is not a working restoring mechanism).
Asserts H_i genuinely tracks per-neuron spike-rate-dependent drift: silencing
half the population (silence_mask) creates a real, large activity contrast
(rate=0 vs rate>0), and DEFAULT_HDP_DESYNC's rate-drain term (gamma>0) should
produce a real, measurable H difference between the two groups -- silent
neurons retain higher H (an income/homeostatic-resource term that active
neurons spend by spiking), confirming H is activity-dependent, not a fixed
constant like DEFAULT_HDP's near-static H.
"""
import numpy as np
import jax
import jax.numpy as jnp

import jaxfne as jtfne
from jaxfne.hdp_network import (
    BASE_DRIVE_BY_CELL_TYPE_DEFAULT,
    BASE_HDP_KWARGS_DEFAULT,
    DEFAULT_HDP,
    DEFAULT_HDP_DESYNC,
    HDPColumnConfig,
    build_model,
)

N_NEURONS = 20
DURATION_MS = 4000.0
DT_MS = 0.5


def _run(hdp_preset: dict, silence_mask=None):
    cfg = HDPColumnConfig(
        n_neurons=N_NEURONS, duration_ms=DURATION_MS, dt_ms=DT_MS, seed=0,
        base_drive_by_cell_type=dict(BASE_DRIVE_BY_CELL_TYPE_DEFAULT),
    )
    model = build_model(cfg)
    emitter = model.params["emitter"]
    edges = model.params["edge_list"]
    combined = {**hdp_preset, **BASE_HDP_KWARGS_DEFAULT}
    n_steps = int(DURATION_MS / DT_MS)
    kwargs = dict(
        params=emitter, edges=edges, n_steps=n_steps, dt_ms=DT_MS,
        key=jax.random.PRNGKey(0), **combined,
    )
    if silence_mask is not None:
        kwargs["silence_mask"] = silence_mask
    _, sig, _, diag = jtfne.emitters.simulate_edge_recurrent_izhikevich_hdp(**kwargs)
    return np.asarray(sig), np.asarray(diag["H_trace"])


def test_hdp_desync_h_tracks_activity_contrast_between_silenced_and_active_groups():
    # First 10 neurons silenced (rate must be 0), last 10 active.
    silence_mask = jnp.array([False] * 10 + [True] * 10)
    spikes, H_trace = _run(DEFAULT_HDP_DESYNC, silence_mask=silence_mask)

    rate_silenced = spikes[:, :10].mean()
    rate_active = spikes[:, 10:].mean()
    assert rate_silenced == 0.0, "silence_mask must genuinely zero the silenced group's rate"
    assert rate_active > 0.0, "the unsilenced group must actually spike"

    H_final_silenced = H_trace[-1, :10].mean()
    H_final_active = H_trace[-1, 10:].mean()
    assert np.all(np.isfinite(H_trace))
    # A real, non-trivial H difference between the two activity groups --
    # confirms H is genuinely activity-dependent under gamma>0, not a fixed
    # constant. Direction: active (spiking) neurons drain H via the rate-drain
    # term, so silenced neurons retain higher H.
    assert abs(H_final_silenced - H_final_active) > 0.002, (
        f"expected a real H gap between silenced ({H_final_silenced}) and active "
        f"({H_final_active}) groups under DEFAULT_HDP_DESYNC's gamma=0.5 rate-drain term"
    )
    assert H_final_silenced > H_final_active


def test_default_hdp_h_stays_near_static_regardless_of_activity_contrast():
    """Contrast case: DEFAULT_HDP (alpha=gamma=0, near-static H by design) does
    NOT show this activity-tracking behavior -- confirms the Hebbian angle is
    specific to DEFAULT_HDP_DESYNC's nonzero alpha/gamma, not a universal HDP
    property."""
    silence_mask = jnp.array([False] * 10 + [True] * 10)
    spikes, H_trace = _run(DEFAULT_HDP, silence_mask=silence_mask)
    H_final_silenced = H_trace[-1, :10].mean()
    H_final_active = H_trace[-1, 10:].mean()
    assert np.all(np.isfinite(H_trace))
    assert abs(H_final_silenced - H_final_active) < 0.01, (
        f"DEFAULT_HDP's near-static H (alpha=gamma=0) should show little activity-tracking, "
        f"got a gap of {abs(H_final_silenced - H_final_active)}"
    )
