"""Tests for spectral Granger causality and directionality (Mejias G5).

Covers: Geweke VAR(order) directed spectra on synthetic drives (x -> y
reads positive DAI, y -> x negative, uncoupled reads ~0), the DAI Eq. 1
form (0/0 reads 0.0, never NaN), and refusal of every input the
estimators do not consume.
"""

import numpy as np
import pytest

from jaxfne.analysis import directionality_index, granger_spectral


def _drive(n=2000, seed=0, coupling=0.6, direction="x_to_y"):
    """Two coupled AR(1) traces with a known direction.

    ``x_to_y``: y[t] carries lagged x; ``y_to_x``: the reverse;
    ``uncoupled``: two independent AR(1) traces.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    ex = rng.standard_normal(n)
    ey = rng.standard_normal(n)
    for t in range(1, n):
        if direction == "x_to_y":
            x[t] = 0.5 * x[t - 1] + ex[t]
            y[t] = 0.5 * y[t - 1] + coupling * x[t - 1] + ey[t]
        elif direction == "y_to_x":
            y[t] = 0.5 * y[t - 1] + ey[t]
            x[t] = 0.5 * x[t - 1] + coupling * y[t - 1] + ex[t]
        else:
            x[t] = 0.5 * x[t - 1] + ex[t]
            y[t] = 0.5 * y[t - 1] + ey[t]
    return x, y


FREQS = np.array([5.0, 10.0, 40.0])
DT_MS = 1.0


def test_matches_the_analytic_spectral_gc_of_the_var():
    """x -> y VAR(1) with unit diagonal noise: GC_{x->y}(f) = ln(S_yy / (S_yy - |H_yx|^2)).

    H(f) = (I - A1 z)^-1 with z = exp(-2 pi i f dt) and S = H H* (Geweke 1982 with
    Sigma = I); the reverse direction is zero.
    """
    c = 0.6
    x, y = _drive(n=200_000, seed=11, coupling=c, direction="x_to_y")
    got = granger_spectral(x, y, DT_MS, 1, FREQS)
    a1 = np.array([[0.5, 0.0], [c, 0.5]])
    want = []
    for f in FREQS:
        z = np.exp(-2j * np.pi * f * DT_MS / 1000.0)
        h = np.linalg.inv(np.eye(2) - a1 * z)
        s = h @ h.conj().T
        syy = s[1, 1].real
        want.append(np.log(syy / (syy - abs(h[1, 0]) ** 2)))
    want = np.asarray(want)
    assert np.all(want > 0.05), "fixture must carry a clear x -> y spectrum"
    np.testing.assert_allclose(got["gc_xy"], want, rtol=0.05)
    assert np.all(np.asarray(got["gc_yx"]) < 0.01), "reverse direction must read near zero"


class TestDirectedSpectra:
    def test_x_to_y_reads_positive_dai(self):
        """The x -> y drive must dominate the reverse at every probe freq."""
        x, y = _drive(direction="x_to_y")
        out = granger_spectral(x, y, DT_MS, 2, FREQS)
        assert out["gc_xy"].shape == FREQS.shape
        assert out["gc_yx"].shape == FREQS.shape
        assert bool((out["gc_xy"] > out["gc_yx"]).all()), \
            "x -> y must dominate on an x-driving fixture"
        dai = directionality_index(out["gc_xy"], out["gc_yx"])
        assert bool((dai > 0.0).all()), "DAI must read x -> y as positive"

    def test_y_to_x_reads_negative_dai(self):
        """The reverse drive mirrors: y -> x dominates, DAI negative."""
        x, y = _drive(direction="y_to_x")
        out = granger_spectral(x, y, DT_MS, 2, FREQS)
        assert bool((out["gc_yx"] > out["gc_xy"]).all()), \
            "y -> x must dominate on a y-driving fixture"
        dai = directionality_index(out["gc_xy"], out["gc_yx"])
        assert bool((dai < 0.0).all()), "DAI must read y -> x as negative"

    def test_uncoupled_reads_near_zero_and_dai_zero(self):
        """Independent traces carry no directed influence either way."""
        x, y = _drive(direction="uncoupled")
        out = granger_spectral(x, y, DT_MS, 2, FREQS)
        assert bool((out["gc_xy"] < 0.05).all()), "uncoupled gc_xy must read ~0"
        assert bool((out["gc_yx"] < 0.05).all()), "uncoupled gc_yx must read ~0"

    def test_every_spectrum_finite_and_non_negative(self):
        """Spectra are non-negative and finite on a coupled fixture."""
        x, y = _drive(direction="x_to_y")
        out = granger_spectral(x, y, DT_MS, 2, FREQS)
        for key in ("gc_xy", "gc_yx"):
            assert bool(np.isfinite(out[key]).all()), f"{key} must stay finite"
            assert bool((out[key] >= 0.0).all()), f"{key} must stay >= 0"

    def test_same_traces_same_spectra(self):
        """One trace pair gives one spectrum (no hidden randomness)."""
        x, y = _drive(direction="x_to_y")
        first = granger_spectral(x, y, DT_MS, 2, FREQS)
        second = granger_spectral(x, y, DT_MS, 2, FREQS)
        assert np.array_equal(first["gc_xy"], second["gc_xy"])
        assert np.array_equal(first["gc_yx"], second["gc_yx"])

    def test_rate_traces_flow_through_the_public_estimator(self):
        """Rate-route traces are valid estimator inputs (lane G5 path)."""
        import jaxfne as jtfne
        from jaxfne import neuronal_tensor as nt

        types = [nt.NeuronType.make("E", fraction=0.5, emitter="rate"),
                 nt.NeuronType.make("I", fraction=0.5, emitter="rate")]
        tensor = nt.NeuronalTensor(
            name="granger-rate",
            areas=[nt.Area(name="V1", layers=[
                nt.Layer(name="L4", n_neurons=4, neuron_types=types)])],
        )
        model = jtfne.construct(tensor, nt.RuntimeConfiguration(
            seed=0, duration_ms=200.0, dt_ms=1.0, emitter="rate"))
        signals = model.simulate(jtfne.Simulation(
            duration_ms=200.0, dt_ms=1.0, seed=0,
            runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list")))
        rates = np.asarray(signals.V_m, dtype=float)
        out = granger_spectral(rates[:, 0], rates[:, 1], 1.0, 2, FREQS)
        assert bool(np.isfinite(out["gc_xy"]).all())
        assert bool(np.isfinite(out["gc_yx"]).all())


class TestDirectionalityIndex:
    def test_eq1_form(self):
        """DAI is (xy - yx) / (xy + yx) elementwise."""
        got = directionality_index(np.array([3.0, 1.0]), np.array([1.0, 1.0]))
        assert list(got) == pytest.approx([0.5, 0.0])

    def test_zero_over_zero_reads_zero_never_nan(self):
        """No coupling either way reads 0.0, never NaN."""
        got = directionality_index(np.array([0.0]), np.array([0.0]))
        assert float(got[0]) == 0.0
        assert bool(np.isfinite(got).all())

    def test_scalar_and_broadcast(self):
        """Scalars and broadcastable pairs both work elementwise."""
        assert float(directionality_index(3.0, 1.0)) == pytest.approx(0.5)
        got = directionality_index(np.array([2.0, 0.0]), 1.0)
        assert list(got) == pytest.approx([1.0 / 3.0, -1.0])

    def test_non_finite_refused(self):
        with pytest.raises(ValueError, match="finite"):
            directionality_index(np.array([np.nan]), np.array([1.0]))

    def test_unbroadcastable_refused(self):
        with pytest.raises(ValueError, match="broadcast"):
            directionality_index(np.zeros(3), np.zeros(4))


class TestGrangerRefusals:
    def test_order_below_one_refused(self):
        x, y = _drive(n=500, direction="uncoupled")
        for bad in (0, -1):
            with pytest.raises(ValueError, match="order"):
                granger_spectral(x, y, DT_MS, bad, FREQS)

    def test_non_integer_order_refused(self):
        x, y = _drive(n=500, direction="uncoupled")
        with pytest.raises(ValueError, match="order"):
            granger_spectral(x, y, DT_MS, 2.5, FREQS)

    def test_short_series_refused(self):
        with pytest.raises(ValueError, match="short"):
            granger_spectral(np.ones(5), np.ones(5), DT_MS, 2, FREQS)

    def test_mismatched_traces_refused(self):
        with pytest.raises(ValueError, match="disagree"):
            granger_spectral(np.ones(500), np.ones(400), DT_MS, 2, FREQS)

    def test_non_finite_trace_refused(self):
        x, y = _drive(n=500, direction="uncoupled")
        x[10] = np.nan
        with pytest.raises(ValueError, match="finite"):
            granger_spectral(x, y, DT_MS, 2, FREQS)

    def test_bad_dt_refused(self):
        x, y = _drive(n=500, direction="uncoupled")
        for bad in (0.0, -1.0, np.nan):
            with pytest.raises(ValueError, match="dt_ms"):
                granger_spectral(x, y, bad, 2, FREQS)

    def test_freqs_outside_nyquist_refused(self):
        x, y = _drive(n=500, direction="uncoupled")
        with pytest.raises(ValueError, match="Nyquist"):
            granger_spectral(x, y, DT_MS, 2, np.array([0.0]))
        with pytest.raises(ValueError, match="Nyquist"):
            granger_spectral(x, y, DT_MS, 2, np.array([1000.0]))


class TestZeroResidualVariance:
    def test_silent_series_is_refused_not_read_as_zero_coupling(self):
        """A series with zero variance gave 0.0 spectra silently; it is refused."""
        x, _ = _drive(n=2000, direction="uncoupled")
        with pytest.raises(ValueError, match="zero variance"):
            granger_spectral(x, np.zeros(len(x)), DT_MS, 2, FREQS)

    def test_constant_series_is_refused_not_read_as_zero_coupling(self):
        """A constant series leaves a rounding-sized residual (about 1e-32), so variance must catch it."""
        x, y = _drive(n=2000, direction="uncoupled")
        with pytest.raises(ValueError, match="zero variance"):
            granger_spectral(np.ones(len(x)), y, DT_MS, 2, FREQS)
        with pytest.raises(ValueError, match="zero variance"):
            granger_spectral(x, np.full(len(y), 3.5), DT_MS, 2, FREQS)
