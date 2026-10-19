"""Tests for the Wilson-Cowan-type rate-population emitter (Mejias G1).

Covers: transfer functions (phi(0) = 1, refusals), per-triple param
validation and overrides, the tensor route (construct -> simulate through
the public API), drive/input/delay/noise realization, and refusal of every
input the route does not consume.
"""

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt
from jaxfne.emitters_rate import (
    RATE_DEFAULTS,
    RATE_TRANSFER_FUNCTIONS,
    _RATE_PARAM_FIELDS,
    check_rate_params,
    make_rate_params,
    rate_transfer,
    simulate_rate_edge,
)


def _tensor(n=4, params_e=None, params_i=None, delay_ms=1.0):
    pairs = [("E", 0.5, params_e), ("I", 0.5, params_i)]
    types = []
    for name, frac, params in pairs:
        kw = {} if params is None else {"emitter_params": dict(params)}
        types.append(nt.NeuronType.make(
            name, fraction=frac, emitter="rate", **kw))
    conn = nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="I",
        mechanism="AMPA", static=nt.StaticParams(dT_ms=2.0),
        delay_ms=delay_ms,
    )
    return nt.NeuronalTensor(
        name="rate-test",
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(name="L4", n_neurons=n, neuron_types=types)],
            inter_connections=[conn],
        )],
    )


def _model(tensor, seed=0, duration_ms=100.0, dt_ms=0.5):
    return jtfne.construct(
        tensor, nt.RuntimeConfiguration(
            seed=seed, duration_ms=duration_ms, dt_ms=dt_ms, emitter="rate"))


def _sim(duration_ms=100.0, dt_ms=0.5, seed=0, poisson=None, **kw):
    return jtfne.Simulation(
        duration_ms=duration_ms, dt_ms=dt_ms, seed=seed,
        poisson_drive=poisson,
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"), **kw)


class TestTransferFunctions:
    @pytest.mark.parametrize("transfer", RATE_TRANSFER_FUNCTIONS)
    def test_phi_zero_is_one(self, transfer):
        """Every transfer function is normalized to phi(0) = 1."""
        got = rate_transfer(np.zeros(5), transfer, 2.0)
        assert np.all(got == 1.0), f"{transfer} must read 1 at 0"

    def test_threshold_linear_floor_and_slope(self):
        got = rate_transfer(np.array([-2.0, -1.0, 0.0, 1.0]), "threshold_linear", 1.0)
        assert list(got) == [0.0, 0.0, 1.0, 2.0]

    def test_sigmoid_bounded(self):
        got = rate_transfer(np.array([-100.0, 100.0]), "sigmoid", 1.0)
        assert got[0] == pytest.approx(0.0, abs=1e-9)
        assert got[1] == pytest.approx(2.0, abs=1e-9)

    def test_x_over_one_minus_exp_positive(self):
        xs = np.array([-3.0, 3.0])
        got = rate_transfer(xs, "x_over_one_minus_exp", 1.0)
        assert bool((got > 0.0).all()), "the Wilson form stays positive"
        assert float(rate_transfer(np.array([1e-12]), "x_over_one_minus_exp", 1.0)[0]) == \
            pytest.approx(1.0)

    def test_gain_scales_the_curve(self):
        x = np.array([1.0])
        assert float(rate_transfer(x, "threshold_linear", 2.0)[0]) == pytest.approx(3.0)
        assert float(rate_transfer(x, "threshold_linear", 0.0)[0]) == pytest.approx(1.0)

    def test_unknown_transfer_refused(self):
        with pytest.raises(ValueError, match="not realized"):
            rate_transfer(np.zeros(2), "tanh", 1.0)

    def test_negative_gain_refused(self):
        with pytest.raises(ValueError, match="gain"):
            rate_transfer(np.zeros(2), "sigmoid", -1.0)

    def test_non_finite_gain_refused(self):
        with pytest.raises(ValueError, match="gain"):
            rate_transfer(np.zeros(2), "sigmoid", np.inf)


class TestParamValidation:
    def test_none_reads_as_empty(self):
        assert check_rate_params(None) == {}

    def test_subset_accepted(self):
        assert check_rate_params({"tau_ms": 20.0}) == {"tau_ms": 20.0}

    def test_unknown_key_refused(self):
        with pytest.raises(ValueError, match="does not consume"):
            check_rate_params({"tau_m_ms": 20.0})

    def test_non_positive_tau_refused(self):
        for bad in (0.0, -5.0):
            with pytest.raises(ValueError, match="tau_ms"):
                check_rate_params({"tau_ms": bad})

    def test_bad_transfer_refused(self):
        with pytest.raises(ValueError, match="transfer"):
            check_rate_params({"transfer": "tanh"})

    def test_negative_sigma_and_r0_refused(self):
        with pytest.raises(ValueError, match="sigma"):
            check_rate_params({"sigma": -0.1})
        with pytest.raises(ValueError, match="r0"):
            check_rate_params({"r0": -0.1})

    def test_non_finite_refused(self):
        with pytest.raises(ValueError, match="gain"):
            check_rate_params({"gain": np.nan})

    def test_bool_refused_never_coerced(self):
        with pytest.raises(ValueError, match="not realized"):
            check_rate_params({"sigma": True})


class TestTripleOverrides:
    def _rows(self):
        return [{"area": "V1", "layer": "L4", "cell_type": "E"},
                {"area": "V1", "layer": "L4", "cell_type": "I"}]

    def test_defaults_without_overrides(self):
        params = make_rate_params(["E", "I"])
        assert tuple(params.transfer) == (
            RATE_DEFAULTS["transfer"], RATE_DEFAULTS["transfer"])
        assert bool((np.asarray(params.tau_ms) == RATE_DEFAULTS["tau_ms"]).all())

    def test_override_applies_to_exactly_its_triple(self):
        from jaxfne._construct_rate import apply_rate_triple_param_overrides

        params = make_rate_params(["E", "I"])
        out = apply_rate_triple_param_overrides(
            params, self._rows(), {("V1", "L4", "E"): {"tau_ms": 25.0}})
        assert float(np.asarray(out.tau_ms)[0]) == 25.0
        assert float(np.asarray(out.tau_ms)[1]) == RATE_DEFAULTS["tau_ms"]

    def test_transfer_override_applies(self):
        from jaxfne._construct_rate import apply_rate_triple_param_overrides

        params = make_rate_params(["E", "I"])
        out = apply_rate_triple_param_overrides(
            params, self._rows(), {("V1", "L4", "I"): {"transfer": "sigmoid"}})
        assert tuple(out.transfer) == (RATE_DEFAULTS["transfer"], "sigmoid")

    def test_unmatched_triple_refused(self):
        from jaxfne._construct_rate import apply_rate_triple_param_overrides

        with pytest.raises(ValueError, match="matches no neuron"):
            apply_rate_triple_param_overrides(
                make_rate_params(["E"]), self._rows()[:1],
                {("V1", "L4", "PV"): {"tau_ms": 5.0}})

    def test_unknown_field_refused(self):
        from jaxfne._construct_rate import apply_rate_triple_param_overrides

        with pytest.raises(ValueError, match="does not consume"):
            apply_rate_triple_param_overrides(
                make_rate_params(["E"]), self._rows()[:1],
                {("V1", "L4", "E"): {"tau_m_ms": 5.0}})


class TestRateRoute:
    def test_registry_consumes_the_rate_fields(self):
        from jaxfne import emitter_registry as reg

        assert tuple(reg.get_emitter("rate").params) == tuple(_RATE_PARAM_FIELDS)

    def test_end_to_end_rates_finite_and_shaped(self):
        """The lane gap case builds and runs through the public API."""
        model = _model(_tensor())
        signals = model.simulate(_sim())
        rates = np.asarray(signals.V_m, dtype=float)
        assert rates.shape == (200, 4), "rate trace must cover every neuron"
        assert bool(np.isfinite(rates).all()), "rate state must stay finite"
        assert bool((rates >= 0.0).all()), "rates floor at zero"
        assert int(np.sum(np.asarray(signals.spikes))) == 0, "no spikes emitted"
        assert signals.metadata["emitter_family"] == "rate"

    def test_same_seed_same_rates(self):
        """One declaration plus one seed gives one trajectory."""
        first = np.asarray(_model(_tensor()).simulate(_sim()).V_m, dtype=float)
        second = np.asarray(_model(_tensor()).simulate(_sim()).V_m, dtype=float)
        assert np.array_equal(first, second)

    def test_sigma_zero_vs_positive_differs(self):
        """The noise stream is live: sigma > 0 moves the trajectory."""
        still = np.asarray(_model(_tensor(params_e={"sigma": 0.0})).simulate(_sim()).V_m)
        noisy = np.asarray(
            _model(_tensor(params_e={"sigma": 0.5})).simulate(_sim()).V_m)
        assert float(np.max(np.abs(still - noisy))) > 0.0

    def test_drive_moves_the_trajectory(self):
        """NeuronType.drive is the executed external current."""
        tensor = _tensor()
        plain = np.asarray(_model(tensor).simulate(_sim()).V_m, dtype=float)
        driven_types = [
            nt.NeuronType.make("E", fraction=0.5, emitter="rate", drive=9.0),
            nt.NeuronType.make("I", fraction=0.5, emitter="rate"),
        ]
        tensor2 = nt.NeuronalTensor(
            name="rate-test-driven",
            areas=[nt.Area(name="V1", layers=[
                nt.Layer(name="L4", n_neurons=4, neuron_types=driven_types)])],
        )
        driven = np.asarray(_model(tensor2).simulate(_sim()).V_m, dtype=float)
        assert float(np.max(np.abs(plain - driven))) > 0.0

    def test_poisson_drive_moves_the_trajectory(self):
        """The declared input composes with the drive channel."""
        model = _model(_tensor())
        quiet = np.asarray(model.simulate(_sim()).V_m, dtype=float)
        loud = np.asarray(model.simulate(
            _sim(poisson={"rate_hz": 400.0, "amplitude": 2.0, "seed": 0})).V_m,
            dtype=float)
        assert float(np.max(np.abs(quiet - loud))) > 0.0

    def test_delay_moves_the_trajectory(self):
        """A -> A' (delay 1 vs 5 steps) changes when recurrence arrives."""
        near = np.asarray(
            _model(_tensor(delay_ms=0.5)).simulate(_sim()).V_m, dtype=float)
        far = np.asarray(
            _model(_tensor(delay_ms=2.5)).simulate(_sim()).V_m, dtype=float)
        assert float(np.max(np.abs(near - far))) > 0.0

    def test_zero_step_edges_floor_with_receipt(self):
        """A rule without declared delay compiles 0 and floors to 1 step."""
        bare_types = [
            nt.NeuronType.make("E", fraction=0.5, emitter="rate"),
            nt.NeuronType.make("I", fraction=0.5, emitter="rate"),
        ]
        bare_conn = nt.InterConnection(
            source_layer="L4", source_neuron_type="E",
            target_layer="L4", target_neuron_type="I",
            mechanism="AMPA", static=nt.StaticParams(dT_ms=2.0),
        )
        bare = nt.NeuronalTensor(
            name="rate-test-bare",
            areas=[nt.Area(
                name="V1",
                layers=[nt.Layer(name="L4", n_neurons=4, neuron_types=bare_types)],
                inter_connections=[bare_conn],
            )],
        )
        model = _model(bare)
        signals = model.simulate(_sim())
        assert int(signals.metadata["n_floored_to_one_step"]) == int(
            signals.metadata["n_edges"]) > 0

    def test_declared_delay_needs_no_floor(self):
        """A declared 1 ms delay at dt 0.5 realizes 2 steps, floored 0."""
        model = _model(_tensor())
        signals = model.simulate(_sim())
        assert int(signals.metadata["n_floored_to_one_step"]) == 0
        assert int(signals.metadata["n_edges"]) > 0

    def test_kernel_refuses_sub_one_delays(self):
        params = make_rate_params(["E"])
        with pytest.raises(ValueError, match=">= 1"):
            simulate_rate_edge(
                params, n_steps=4, dt_ms=0.5,
                edge_pre=np.array([0]), edge_post=np.array([0]),
                edge_weight=np.array([1.0]), edge_delay_steps=np.array([0]),
                drive=np.zeros(1), seed=0)


@pytest.mark.parametrize("dt_ms", [0.1, 1.0])
def test_stationary_variance_matches_euler_maruyama(dt_ms):
    """tau dr = (mu - r) dt + sigma sqrt(tau) dW: stationary variance sigma**2 / (2 - dt/tau).

    The value tends to sigma**2 / 2 as dt -> 0, so the noise intensity does not
    depend on the step (a sqrt(tau/dt) increment would scale it by tau/dt).
    """
    from dataclasses import replace

    tau, sigma, n = 10.0, 0.2, 64
    base = make_rate_params(["E"] * n)
    params = replace(base, tau_ms=np.full(n, tau), sigma=np.full(n, sigma),
                     gain=np.zeros(n), r0=np.ones(n))  # phi = 1: an OU unit at mu = 1
    n_steps = int(round(20_000.0 / dt_ms))
    empty = np.zeros(0, dtype=np.int64)
    rates = simulate_rate_edge(
        params, n_steps=n_steps, dt_ms=dt_ms, edge_pre=empty, edge_post=empty,
        edge_weight=np.zeros(0), edge_delay_steps=empty, drive=np.zeros(n), seed=3,
    )["rates"]
    tail = rates[n_steps // 10:]
    assert tail.min() > 0.0, "fixture must stay above the rate floor"
    want = sigma ** 2 / (2.0 - dt_ms / tau)
    assert float(tail.var()) == pytest.approx(want, rel=0.05)
    assert float(tail.mean()) == pytest.approx(1.0, abs=0.01)


class TestRateRefusals:
    def test_rate_params_on_lif_route_refused(self):
        """Rate-only keys are unconsumed on the spiking routes."""
        types = [nt.NeuronType.make(
            "E", fraction=1.0, emitter="lif", emitter_params={"tau_ms": 20.0})]
        tensor = nt.NeuronalTensor(
            name="rate-on-lif",
            areas=[nt.Area(name="V1", layers=[
                nt.Layer(name="L4", n_neurons=2, neuron_types=types)])],
        )
        with pytest.raises(ValueError, match="does not consume"):
            jtfne.construct(tensor, nt.RuntimeConfiguration(
                seed=0, duration_ms=20.0, dt_ms=0.5, emitter="lif"))

    def test_paradigm_refused(self):
        model = _model(_tensor())
        with pytest.raises(ValueError, match="paradigm"):
            model.simulate(_sim(), paradigm={"kind": "step"})

    def test_ablation_refused(self):
        model = _model(_tensor())
        with pytest.raises(ValueError, match="ablation"):
            model.simulate(_sim(ablation="E_silence"))

    def test_homeostasis_refused(self):
        model = _model(_tensor())
        with pytest.raises(ValueError, match="homeostasis"):
            model.simulate(jtfne.Simulation(
                duration_ms=100.0, dt_ms=0.5, seed=0,
                runtime=jtfne.RuntimeConfig(
                    recurrent_backend="edge_list", enable_homeostasis=True)))

    def test_enable_hdp_refused(self):
        model = _model(_tensor())
        with pytest.raises(ValueError, match="enable_hdp"):
            model.simulate(jtfne.Simulation(
                duration_ms=100.0, dt_ms=0.5, seed=0,
                runtime=jtfne.RuntimeConfig(
                    recurrent_backend="edge_list", enable_hdp=True,
                    hdp_params={"hdp_rule": "pcl_stdp"})))

    def test_dense_backend_refused(self):
        model = _model(_tensor())
        with pytest.raises(ValueError, match="recurrent_backend"):
            model.simulate(jtfne.Simulation(
                duration_ms=100.0, dt_ms=0.5, seed=0,
                runtime=jtfne.RuntimeConfig(recurrent_backend="dense")))

    def test_n_contacts_refused(self):
        with pytest.raises(ValueError, match="n_contacts"):
            jtfne.construct(
                _tensor(), nt.RuntimeConfiguration(
                    seed=0, duration_ms=20.0, dt_ms=0.5, emitter="rate",
                    n_contacts=8))

    def test_weight_sharing_refused(self):
        conn = nt.InterConnection(
            source_layer="L4", source_neuron_type="E",
            target_layer="L4", target_neuron_type="E",
            mechanism="AMPA", static=nt.StaticParams(dT_ms=2.0),
            delay_ms=1.0,
            footprint={"size": [10.0, 10.0], "offset": [-5.0, -5.0]},
            weight_sharing="patch",
        )
        tensor = nt.NeuronalTensor(
            name="rate-sharing",
            areas=[nt.Area(name="V1", layers=[nt.Layer(
                name="L4", n_neurons=4, neuron_types=[
                    nt.NeuronType.make("E", fraction=1.0, emitter="rate")])],
                inter_connections=[conn])],
        )
        with pytest.raises(ValueError, match="weight_sharing"):
            _model(tensor)

    def test_trained_w0_refused(self):
        bare_types = [
            nt.NeuronType.make("E", fraction=0.5),
            nt.NeuronType.make("I", fraction=0.5),
        ]
        bare_conn = nt.InterConnection(
            source_layer="L4", source_neuron_type="E",
            target_layer="L4", target_neuron_type="I",
            mechanism="AMPA", static=nt.StaticParams(dT_ms=2.0),
            delay_ms=1.0,
        )
        bare = nt.NeuronalTensor(
            name="rate-w0-bare",
            areas=[nt.Area(
                name="V1",
                layers=[nt.Layer(name="L4", n_neurons=4, neuron_types=bare_types)],
                inter_connections=[bare_conn],
            )],
        )
        ref = jtfne.construct(
            bare, nt.RuntimeConfiguration(
                seed=0, duration_ms=20.0, dt_ms=0.5, emitter="izhikevich"))
        el = ref.params["edge_list"]
        from jaxfne.emitters import resolve_edge_weight

        w = np.asarray(
            resolve_edge_weight(
                el, np.float64,
                presynaptic_sign=np.asarray(ref.params["emitter"].sign)),
            dtype=np.float64)
        w0tensor = jtfne.with_trained_weights(
            bare,
            {"pre": np.asarray(el.pre), "post": np.asarray(el.post), "w": w},
            seed=0)
        with pytest.raises(ValueError, match="trained W0"):
            _model(w0tensor)
