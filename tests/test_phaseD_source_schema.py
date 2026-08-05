"""Phase D (D-04) cross-path source schema regression tests.

Phase D discovery (commit 07f99ab, docs/fullroadmap.md D-01/D-02/D-03) found
no ``Source``/``SourceTensor`` class anywhere in ``jaxfne/``. The supported
contract is:

1. Every emitter's third return value is a raw source *proxy-current trace*
   array, ordinarily shape ``(T, N)`` (time-by-neuron), with no schema object.
2. Model-level source metadata is the ``source_bookkeeping`` dict.
3. ``source_calibration_status`` is a non-empty source-derived string.
4. ``physical_amplitude_calibrated`` must remain ``False`` on every path.
5. Generic Izhikevich paths use ``uncalibrated_izhikevich_native_current``;
   the homeostatic E/I path has its own explicit native-current proxy status
   ``uncalibrated_homeostatic_ei_native_current`` (HomeostaticEIParams,
   ``jaxfne/emitters_homeostatic_ei.py:337``).

Five emitter runtime paths are covered:

1. ``simulate_eig_izhikevich`` (dense backend)
2. ``simulate_edge_recurrent_izhikevich`` (edge_list backend)
3. ``simulate_edge_recurrent_izhikevich_homeostatic`` (``enable_homeostasis``)
4. ``simulate_edge_recurrent_izhikevich_hdp`` (``enable_hdp``)
5. ``simulate_homeostatic_ei`` (homeostatic_ei emitter family)

``test_source_bookkeeping_v020.py`` already asserts the required fields and
canonical string of ``source_bookkeeping`` on a single izhikevich config; this
module adds the cross-path dimension and does not duplicate it. The
homeostatic_ei path does NOT produce ``source_bookkeeping`` — its model-level
metadata (``jaxfne/_model_simulate.py`` ``_simulate_homeostatic_ei``) omits the
dict by design. Its nearest supported surface is asserted instead:
``signals.metadata["source_calibration_status"]``, the params default, the
``cfg.metadata`` truth gate, and a direct kernel-level call to
``simulate_homeostatic_ei`` for the raw trace contract.
"""

import jax.numpy as jnp

import jaxfne as jtfne

CANONICAL_IZH_STATUS = "uncalibrated_izhikevich_native_current"
CANONICAL_HOMEOSTATIC_EI_STATUS = "uncalibrated_homeostatic_ei_native_current"


def _izhikevich_config(n=8):
    return (
        jtfne.configuration()
        .network(
            name="V1",
            kind="cortical_column",
            n=n,
            cell_types={"E": 0.7, "PV": 0.2, "SST": 0.1},
        )
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="mean_zero_neumann",
            gauge="mean_zero",
        )
        .probe(name="p", modes=["spikes", "V_m"])
    )


def _run_izhikevich(runtime_kwargs, n=8, duration_ms=10.0, dt_ms=0.5, seed=0):
    model = jtfne.construct(_izhikevich_config(n=n))
    sim = jtfne.simulation(
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        seed=seed,
        runtime=jtfne.RuntimeConfig(**runtime_kwargs),
    )
    return model.simulate(sim)


def _assert_raw_source_trace(sig):
    """Raw source trace contract: plain array, (T, N), finite, no schema object."""
    assert sig.sources is not None, "record_sources must be True for a source trace"
    assert hasattr(sig.sources, "shape"), "sources output must be an array, not a schema object"
    assert sig.sources.ndim == 2, f"sources must be time-by-neuron; got ndim={sig.sources.ndim}"
    n_steps = int(sig.metadata["n_steps"])
    n_neurons = sig.V_m.shape[1]
    assert sig.sources.shape == (n_steps, n_neurons), (
        f"sources (T,N) must match the recorded run dims; got {sig.sources.shape}, "
        f"expected ({n_steps}, {n_neurons})"
    )
    assert bool(jnp.all(jnp.isfinite(sig.sources))), "source trace must be finite"


def _assert_bookkeeping_metadata(sig, expected_status):
    """Model-level source_bookkeeping metadata contract (paths that produce it)."""
    sb = sig.metadata.get("source_bookkeeping")
    assert sb is not None, "source_bookkeeping must exist on the signals metadata surface"
    status = sb.get("source_calibration_status")
    assert isinstance(status, str) and status, "source_calibration_status must be a non-empty string"
    assert status == expected_status, (
        f"source_calibration_status must equal the source-derived value; "
        f"got {status!r}, expected {expected_status!r}"
    )
    assert sb.get("physical_amplitude_calibrated") is False, (
        "physical_amplitude_calibrated must be explicitly False"
    )


def test_dense_eig_path_source_schema():
    sig = _run_izhikevich({"recurrent_backend": "dense", "jit": False})
    _assert_raw_source_trace(sig)
    _assert_bookkeeping_metadata(sig, CANONICAL_IZH_STATUS)


def test_edge_recurrent_path_source_schema():
    sig = _run_izhikevich({"recurrent_backend": "edge_list", "jit": False})
    _assert_raw_source_trace(sig)
    _assert_bookkeeping_metadata(sig, CANONICAL_IZH_STATUS)


def test_homeostatic_path_source_schema():
    sig = _run_izhikevich({"enable_homeostasis": True, "jit": False})
    _assert_raw_source_trace(sig)
    _assert_bookkeeping_metadata(sig, CANONICAL_IZH_STATUS)


def test_hdp_path_source_schema():
    sig = _run_izhikevich({"enable_hdp": True, "jit": False})
    _assert_raw_source_trace(sig)
    _assert_bookkeeping_metadata(sig, CANONICAL_IZH_STATUS)


def test_status_not_inferred_from_trace_shape():
    """The status string is metadata, never derived from V_m/source shape."""
    small = _run_izhikevich(
        {"recurrent_backend": "dense", "jit": False}, n=4, duration_ms=5.0, dt_ms=0.5
    )
    large = _run_izhikevich(
        {"recurrent_backend": "dense", "jit": False}, n=16, duration_ms=20.0, dt_ms=0.5
    )
    assert small.sources.shape != large.sources.shape
    small_status = small.metadata["source_bookkeeping"]["source_calibration_status"]
    large_status = large.metadata["source_bookkeeping"]["source_calibration_status"]
    assert small_status == large_status == CANONICAL_IZH_STATUS
    assert small.metadata["source_bookkeeping"]["physical_amplitude_calibrated"] is False
    assert large.metadata["source_bookkeeping"]["physical_amplitude_calibrated"] is False


def test_truth_gate_false_across_all_izhikevich_paths():
    """Null-control: physical_amplitude_calibrated stays False on every path."""
    for runtime_kwargs in (
        {"recurrent_backend": "dense", "jit": False},
        {"recurrent_backend": "edge_list", "jit": False},
        {"enable_homeostasis": True, "jit": False},
        {"enable_hdp": True, "jit": False},
    ):
        sig = _run_izhikevich(runtime_kwargs)
        sb = sig.metadata["source_bookkeeping"]["physical_amplitude_calibrated"]
        assert sb is False, f"path {runtime_kwargs}: physical_amplitude_calibrated must be False"


def _build_homeostatic_ei_model(n=8):
    cfg = (
        jtfne.Configuration()
        .runtime(seed=0)
        .network(name="ei", n=n)
        .set_emitter("homeostatic_ei")
        .field(domain="none")
        .probe(modes=["vm"])
    )
    return jtfne.construct(cfg)


def test_homeostatic_ei_model_level_surface():
    """homeostatic_ei surfaces status at the top-level metadata dict (no source_bookkeeping)."""
    model = _build_homeostatic_ei_model()
    sim = jtfne.simulation(duration_ms=20.0, dt_ms=0.5, seed=0)
    sig = model.simulate(sim)
    # documented surface: top-level status, no source_bookkeeping dict
    assert "source_bookkeeping" not in sig.metadata
    assert sig.metadata["source_calibration_status"] == CANONICAL_HOMEOSTATIC_EI_STATUS
    # nearest supported metadata surface for the truth gate is the config metadata
    assert model.cfg.metadata["physical_amplitude_calibrated"] is False
    _assert_raw_source_trace(sig)


def test_homeostatic_ei_direct_kernel_contract():
    """Direct simulate_homeostatic_ei call with params — raw source trace array."""
    import jax

    from jaxfne.emitters_homeostatic_ei import make_minimal_ei_params, simulate_homeostatic_ei

    params = make_minimal_ei_params(n=8)
    assert params.source_calibration_status == CANONICAL_HOMEOSTATIC_EI_STATUS
    voltages, spikes, sources, G_hist, H_hist, diag = simulate_homeostatic_ei(
        params,
        n_steps=20,
        dt_ms=0.5,
        key=jax.random.PRNGKey(0),
        noise_scale=0.0,
        dtype="float32",
    )
    assert voltages.shape == (20, 8)
    assert spikes.shape == (20, 8)
    assert sources.shape == (20, 8)
    assert sources.ndim == 2
    assert bool(jnp.all(jnp.isfinite(sources)))
    assert not bool(diag["error"])