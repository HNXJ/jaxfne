"""T-25: frozen W0 provenance on NeuronalTensor + with_trained_weights.

A trained run's final weights (W_T) become a new tensor A' = with_trained_weights(A, ...)
carrying per-edge trained W0 plus W0Provenance(origin="trained", parent_digest=digest(A)).
Constructing A' realizes the trained weights exactly; mismatches are refused, never coerced.
"""

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.neuronal_tensor import (
    W0Provenance,
    make_minimal_ei_tensor,
    with_trained_weights,
    _tensor_identity_digest,
)

# Pinned on the baseline tree (pre-T-25 digests); T-25 keeps default-declared
# tensors byte-identical by omitting default fields from to_dict.
PIN_MINIMAL8 = "9ea62ed326374f6cf0543c4504f62ff1a5c61303232765b1380994dd60257eb6"
PIN_MINIMAL16 = "c2905d714068d789d5c3cb8d088122bed134364cc9e787f9e1468cb7c64025d8"

SEED = 0
HDP_PARAMS = dict(
    K_HDP=0.2,
    K_ctrl=0.3,
    K_w_ctrl=0.002,
    alpha=0.05,
    tau_0_ms=5.0,
    noise_scale=0.0,
)


def _model(tensor=None, seed=SEED):
    tensor = make_minimal_ei_tensor() if tensor is None else tensor
    runtime = jtfne.neuronal_tensor.RuntimeConfiguration(
        seed=seed, duration_ms=20.0, dt_ms=0.5)
    return jtfne.construct(tensor, runtime)


def _hdp_run(model, seed=7):
    sigs = jtfne.simulate(
        model,
        runtime=jtfne.RuntimeConfig(enable_hdp=True, hdp_params=dict(HDP_PARAMS)),
        duration_ms=20.0,
        dt_ms=0.5,
        seed=seed,
    )
    return sigs


def _edges_of(model):
    """Compiled (pre, post) plus native per-edge weights.

    Native weights resolve through the kernel's own
    ``_resolved_edge_weight`` (the compact default storage keeps
    ``edge_list.weight`` a size-0 placeholder); the independent oracle for
    sign is the source cell type in ``test_model_form_end_to_end`` below.
    """
    from jaxfne.emitters import _resolved_edge_weight

    el = model.params["edge_list"]
    native = np.asarray(
        _resolved_edge_weight(el, np.float32, model.params["emitter"]),
        dtype=np.float64,
    ).reshape(-1)
    return np.asarray(el.pre).reshape(-1), np.asarray(el.post).reshape(-1), native


# ----- provenance type ------------------------------------------------------

def test_w0_provenance_defaults_and_validation():
    assert W0Provenance().origin == "declared"
    with pytest.raises(ValueError):
        W0Provenance(origin="learned")
    with pytest.raises(ValueError):
        W0Provenance(origin="trained")  # trained requires a parent digest
    with pytest.raises(ValueError):
        W0Provenance(origin="declared", parent_digest="abc")  # declared carries nothing
    prov = W0Provenance(origin="trained", parent_digest="abc", n_steps=40)
    assert prov.parent_digest == "abc" and prov.n_steps == 40


def test_declared_tensors_omit_trained_fields_from_to_dict():
    d = make_minimal_ei_tensor().to_dict()
    assert "w0_provenance" not in d and "trained_w0" not in d


def test_pinned_identity_digests_unchanged():
    assert _tensor_identity_digest(make_minimal_ei_tensor()) == PIN_MINIMAL8
    assert (
        _tensor_identity_digest(make_minimal_ei_tensor(n=16, e_fraction=0.5))
        == PIN_MINIMAL16
    )


# ----- refusals ---------------------------------------------------------------

def test_bare_signals_refused_with_directive():
    tensor = make_minimal_ei_tensor()
    sigs = jtfne.simulate(_model(tensor), duration_ms=20.0, dt_ms=0.5, seed=1)
    with pytest.raises(ValueError, match="summaries only"):
        with_trained_weights(tensor, sigs)


def test_model_without_hdp_run_refused():
    tensor = make_minimal_ei_tensor()
    with pytest.raises(ValueError, match="enable_hdp"):
        with_trained_weights(tensor, _model(tensor))


def test_foreign_model_refused_by_ownership():
    a = make_minimal_ei_tensor()
    b = make_minimal_ei_tensor(n=16, e_fraction=0.5)
    model_b = _model(b)
    _hdp_run(model_b)
    with pytest.raises(ValueError, match="not built from this tensor"):
        with_trained_weights(a, model_b)


def test_shape_mismatch_refused():
    tensor = make_minimal_ei_tensor()
    model = _model(tensor)
    pre, post, _ = _edges_of(model)
    bad_w = np.ones((pre.shape[0] + 1,), dtype=np.float32)
    with pytest.raises(ValueError, match="same length"):
        with_trained_weights(tensor, {"pre": pre, "post": post, "w": bad_w}, seed=SEED)


def test_non_float_weights_refused():
    tensor = make_minimal_ei_tensor()
    model = _model(tensor)
    pre, post, _ = _edges_of(model)
    with pytest.raises(ValueError, match="float32 or float64"):
        with_trained_weights(
            tensor, {"pre": pre, "post": post, "w": np.ones_like(pre)}, seed=SEED)


def test_nonfinite_weights_refused():
    tensor = make_minimal_ei_tensor()
    model = _model(tensor)
    pre, post, w = _edges_of(model)
    w = w.astype(np.float32).copy()
    w[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        with_trained_weights(tensor, {"pre": pre, "post": post, "w": w}, seed=SEED)


def test_sign_flip_refused():
    tensor = make_minimal_ei_tensor()
    model = _model(tensor)
    pre, post, w = _edges_of(model)
    with pytest.raises(ValueError, match="sign"):
        with_trained_weights(tensor, {"pre": pre, "post": post, "w": -w}, seed=SEED)


def test_reordered_edges_refused():
    tensor = make_minimal_ei_tensor()
    model = _model(tensor)
    pre, post, w = _edges_of(model)
    assert not np.array_equal(pre, post), "fixture: the swap must reorder"
    with pytest.raises(ValueError, match="same edges"):
        with_trained_weights(
            tensor, {"pre": post.copy(), "post": pre.copy(), "w": w}, seed=SEED)


def test_checkpoint_leaves_refused_with_directive():
    tensor = make_minimal_ei_tensor()
    with pytest.raises(ValueError, match="explicit pre/post"):
        with_trained_weights(tensor, [np.zeros(3), np.zeros(2)])


# ----- positive paths -----------------------------------------------------------

def test_model_form_end_to_end_and_provenance():
    tensor = make_minimal_ei_tensor()
    model = _model(tensor)
    _hdp_run(model)
    w_final = np.asarray(model.last_hdp_diagnostics()["w_final"], dtype=np.float64)
    assert np.all(np.isfinite(w_final)), "fixture: HDP run must stay finite"
    # Independent sign oracle (not the resolver): an edge is excitatory iff
    # its source neuron is E. HDP updates are sign-preserving, so the trained
    # signs must match the declared biology exactly, with no exact zeros.
    pre, post, native = _edges_of(model)
    src_is_e = np.asarray(
        [row["cell_type"] == "E" for row in model.neuron_table()], dtype=bool)
    expected = np.where(src_is_e[pre], 1.0, -1.0)
    assert np.array_equal(np.sign(native), expected), "fixture: native signs"
    assert np.array_equal(np.sign(w_final), expected), "HDP must preserve signs"
    trained = with_trained_weights(tensor, model)
    assert trained.w0_provenance.origin == "trained"
    assert trained.w0_provenance.parent_digest == _tensor_identity_digest(tensor)
    # A' is a distinct circuit from A: the digest covers the trained vector
    # and its provenance, so retraining from A' records A' as parent (lineage
    # holds the full chain, not just the declared structure).
    assert _tensor_identity_digest(trained) != _tensor_identity_digest(tensor)
    assert np.sign(
        np.asarray(trained.trained_w0.w)).tolist() == np.sign(w_final).tolist()
    # Realized W0 of A' is the trained vector, exactly, deterministically.
    w0_a = np.asarray(_model(trained).params["edge_list"].weight)
    w0_b = np.asarray(_model(trained).params["edge_list"].weight)
    assert np.array_equal(w0_a, w_final.astype(np.float32))
    assert np.array_equal(w0_a, w0_b)


def test_dict_form_binds_identically_to_model_form():
    tensor = make_minimal_ei_tensor()
    model = _model(tensor)
    _hdp_run(model)
    pre, post, _ = _edges_of(model)
    w_final = np.asarray(model.last_hdp_diagnostics()["w_final"])
    via_model = with_trained_weights(tensor, model)
    via_dict = with_trained_weights(
        tensor, {"pre": pre, "post": post, "w": w_final}, seed=SEED)
    assert np.array_equal(
        np.asarray(via_dict.trained_w0.w), np.asarray(via_model.trained_w0.w))
    assert via_dict.w0_provenance.parent_digest == via_model.w0_provenance.parent_digest


def test_tuple_and_state_forms_bind():
    tensor = make_minimal_ei_tensor()
    model = _model(tensor)
    sigs, state = jtfne.simulate(
        model,
        runtime=jtfne.RuntimeConfig(
            recurrent_backend="edge_list",
            enable_hdp=True,
            hdp_params=dict(HDP_PARAMS),
        ),
        duration_ms=20.0,
        dt_ms=0.5,
        seed=7,
        return_state=True,
    )
    pre, post, _ = _edges_of(model)
    w_final = np.asarray(model.last_hdp_diagnostics()["w_final"])
    via_tuple = with_trained_weights(tensor, (pre, post, w_final), seed=SEED)
    via_state = with_trained_weights(
        tensor, state, pre=pre, post=post, n_steps=int(sigs.time_ms.shape[0]))
    assert np.array_equal(
        np.asarray(via_tuple.trained_w0.w), np.asarray(via_state.trained_w0.w))
    assert via_state.w0_provenance.n_steps == int(sigs.time_ms.shape[0])


def test_trained_tensor_round_trips_through_save_load(tmp_path):
    tensor = make_minimal_ei_tensor()
    model = _model(tensor)
    _hdp_run(model)
    trained = with_trained_weights(tensor, model)
    path = tmp_path / "trained.jcfg.json"
    jtfne.neuronal_tensor.save_neuronal_tensor(trained, path)
    reloaded = jtfne.neuronal_tensor.load(path)
    assert reloaded.w0_provenance.origin == "trained"
    assert reloaded.w0_provenance.parent_digest == _tensor_identity_digest(tensor)
    assert np.array_equal(
        np.asarray(reloaded.trained_w0.w), np.asarray(trained.trained_w0.w))
    assert np.array_equal(
        np.asarray(reloaded.trained_w0.pre), np.asarray(trained.trained_w0.pre))
    assert _tensor_identity_digest(reloaded) == _tensor_identity_digest(trained)


def test_lif_tensor_refuses_trained_w0_at_construct():
    from jaxfne.neuronal_tensor import Area, InterConnection, Layer, NeuronType, StaticParams
    from jaxfne.presets import RECEPTOR_KINETICS

    def _lif_nt(name):
        return NeuronType.make(name, fraction=0.5, emitter="lif")

    layer = Layer(name="L1", n_neurons=8,
                  neuron_types=[_lif_nt("E"), _lif_nt("PV")])
    area = Area(
        name="lif",
        layers=[layer],
        inter_connections=[
            InterConnection(
                source_layer="L1", source_neuron_type="E",
                target_layer="L1", target_neuron_type="E",
                mechanism="AMPA",
                static=StaticParams(
                    dT_ms=float(RECEPTOR_KINETICS["AMPA"]["tau_ms"])),
            ),
        ],
    )
    from jaxfne.neuronal_tensor import NeuronalTensor, TrainedW0
    from dataclasses import replace

    tensor = NeuronalTensor(areas=[area])
    model = _model(make_minimal_ei_tensor())
    pre, post, w = _edges_of(model)
    trained_lif = replace(
        tensor,
        trained_w0=TrainedW0(pre=pre, post=post, w=np.asarray(w, dtype=np.float64)),
        w0_provenance=W0Provenance(
            origin="trained", parent_digest=_tensor_identity_digest(tensor)),
    )
    with pytest.raises(ValueError, match="trained"):
        jtfne.neuronal_tensor._construct_neuronal_tensor_impl(
            trained_lif, seed=SEED, duration_ms=20.0, dt_ms=0.5, emitter="lif")


# ----- review fixes: distinct identity, bridge refusal, dtype exactness -----

def _declared_emitter_tensor(emitter):
    """Single-area tensor with every NeuronType declaring `emitter`."""
    from jaxfne.neuronal_tensor import (
        Area,
        InterConnection,
        Layer,
        NeuronalTensor,
        NeuronType,
        StaticParams,
    )
    from jaxfne.presets import RECEPTOR_KINETICS

    layer = Layer(
        name="L1",
        n_neurons=8,
        neuron_types=[
            NeuronType.make("E", fraction=0.5, emitter=emitter),
            NeuronType.make("PV", fraction=0.5, emitter=emitter),
        ],
    )
    area = Area(
        name=emitter,
        layers=[layer],
        inter_connections=[
            InterConnection(
                source_layer="L1", source_neuron_type="E",
                target_layer="L1", target_neuron_type="E",
                mechanism="AMPA",
                static=StaticParams(
                    dT_ms=float(RECEPTOR_KINETICS["AMPA"]["tau_ms"])),
            ),
        ],
    )
    return NeuronalTensor(areas=[area])


def _sparse_tensor():
    """Minimal tensor with seed-dependent compiled edges (in-degree sampling)."""
    from jaxfne.neuronal_tensor import (
        Area,
        InterConnection,
        Layer,
        NeuronalTensor,
        NeuronType,
        StaticParams,
    )
    from jaxfne.presets import RECEPTOR_KINETICS

    layer = Layer(
        name="L1",
        n_neurons=16,
        neuron_types=[
            NeuronType.make("E", fraction=0.75),
            NeuronType.make("PV", fraction=0.25),
        ],
    )
    conns = [
        InterConnection(
            source_layer="L1", source_neuron_type=src,
            target_layer="L1", target_neuron_type=tgt,
            mechanism=("AMPA" if src == "E" else "GABA_A"),
            static=StaticParams(
                dT_ms=float(
                    RECEPTOR_KINETICS["AMPA" if src == "E" else "GABA_A"]["tau_ms"])),
            max_in_degree=4,
        )
        for src in ("E", "PV")
        for tgt in ("E", "PV")
    ]
    return NeuronalTensor(
        areas=[Area(name="m", layers=[layer], inter_connections=conns)])


def test_second_generation_records_trained_parent_as_lineage():
    tensor = make_minimal_ei_tensor()
    model = _model(tensor)
    _hdp_run(model)
    trained = with_trained_weights(tensor, model)  # A'
    d_trained = _tensor_identity_digest(trained)
    assert d_trained != _tensor_identity_digest(tensor)
    model2 = _model(trained)
    _hdp_run(model2)
    trained2 = with_trained_weights(trained, model2)  # A''
    assert trained2.w0_provenance.parent_digest == d_trained
    assert _tensor_identity_digest(trained2) != d_trained


def test_to_configuration_refuses_trained_tensor():
    tensor = make_minimal_ei_tensor()
    model = _model(tensor)
    _hdp_run(model)
    trained = with_trained_weights(tensor, model)
    with pytest.raises(ValueError, match=r"construct\(tensor\)"):
        jtfne.neuronal_tensor.neuronal_tensor_to_configuration(
            trained, seed=SEED)


def test_to_adex_configuration_refuses_trained_tensor():
    tensor = make_minimal_ei_tensor()
    model = _model(tensor)
    _hdp_run(model)
    trained = with_trained_weights(tensor, model)
    with pytest.raises(ValueError, match="trained W0"):
        jtfne.neuronal_tensor.neuronal_tensor_to_adex_configuration(trained, seed=SEED)


def test_zero_trained_weight_is_not_a_sign_flip():
    # A weight clamped to exactly 0 keeps no sign; it binds and realizes as 0.
    tensor = make_minimal_ei_tensor()
    model = _model(tensor)
    pre, post, w = _edges_of(model)
    assert np.any(w != 0), "fixture: native weights must be nonzero"
    w0 = w.copy()
    w0[0] = 0.0
    trained = with_trained_weights(tensor, {"pre": pre, "post": post, "w": w0}, seed=SEED)
    _, _, realized = _edges_of(_model(trained))
    assert realized[0] == 0.0


def test_float64_detail_refused_on_float32_apply():
    tensor = make_minimal_ei_tensor()
    model = _model(tensor)
    _hdp_run(model)
    pre, post, _ = _edges_of(model)
    w_final = np.asarray(model.last_hdp_diagnostics()["w_final"])
    # Control: float64 weights exactly representable in float32 apply exactly.
    w_exact = w_final.astype(np.float64)
    assert np.array_equal(
        w_exact.astype(np.float32).astype(np.float64), w_exact)
    bound_exact = with_trained_weights(
        tensor, {"pre": pre, "post": post, "w": w_exact}, seed=SEED)
    assert bound_exact.trained_w0.dtype == "float64"
    realized = np.asarray(_model(bound_exact).params["edge_list"].weight)
    assert np.array_equal(realized, w_final.astype(np.float32))
    # Genuine float64 detail (sub-float32, sign-preserving) binds but is
    # refused when the float32 model would silently round it off.
    w_detail = w_exact * (1.0 + 1e-9)
    assert np.all(np.isfinite(w_detail))
    assert np.array_equal(np.sign(w_detail), np.sign(w_exact))
    assert not np.array_equal(
        w_detail.astype(np.float32).astype(np.float64), w_detail
    ), "fixture: perturbation must be real float64 detail"
    bound_detail = with_trained_weights(
        tensor, {"pre": pre, "post": post, "w": w_detail}, seed=SEED)
    with pytest.raises(ValueError, match="detail"):
        _model(bound_detail)


@pytest.mark.parametrize("emitter", ["lif", "adex"])
def test_bind_refuses_non_izhikevich_tensors_on_both_forms(emitter):
    tensor = _declared_emitter_tensor(emitter)
    model = _model(make_minimal_ei_tensor())
    _hdp_run(model)
    pre, post, w = _edges_of(model)
    with pytest.raises(ValueError, match="izhikevich route"):
        with_trained_weights(tensor, model)
    with pytest.raises(ValueError, match="izhikevich route"):
        with_trained_weights(
            tensor, {"pre": pre, "post": post, "w": w}, seed=SEED)


def test_merge_refuses_trained_tensor():
    tensor = make_minimal_ei_tensor()
    model = _model(tensor)
    _hdp_run(model)
    trained = with_trained_weights(tensor, model)
    with pytest.raises(ValueError, match="trained W0"):
        jtfne.merge_neuronal_tensors([trained, make_minimal_ei_tensor()])


def test_wrong_seed_refused():
    tensor = _sparse_tensor()
    model = _model(tensor)
    pre, post, native = _edges_of(model)
    ref = jtfne.neuronal_tensor._construct_neuronal_tensor_impl(
        tensor, seed=999, duration_ms=10.0, dt_ms=0.5)
    ref_pre = np.asarray(ref.params["edge_list"].pre).reshape(-1)
    assert not np.array_equal(pre, ref_pre), "fixture: seeds must disagree"
    with pytest.raises(ValueError, match="same edges"):
        with_trained_weights(
            tensor, {"pre": pre, "post": post, "w": native}, seed=999)
