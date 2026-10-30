"""Hamm's 2026-10-28 rulings on the analysis metrics (packet W1A).

Covers textbook Fleiss kappa, refusal of off-grid bin widths, the ``dt_ms`` and
``discard_ms`` keywords of ``population_rates``, and the periodic Hann window of
``welch_numpy``.
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from jaxfne.analysis import fano_factor, fleiss_kappa_binary, population_rates
from jaxfne.analysis.metrics import pairwise_cc_binned_disjoint
from jaxfne.analysis.spectral import welch_numpy

# --- item 1: textbook Fleiss kappa (bins are subjects, neurons are raters) ----


def test_fleiss_identical_trains_give_one():
    rng = np.random.default_rng(0)
    row = rng.random(2000) < 0.3
    assert 0.0 < row.mean() < 1.0, "fixture must mix active and silent bins"
    spikes = np.tile(row, (20, 1))
    assert fleiss_kappa_binary(spikes, bin_ms=1.0, dt_ms=1.0) == pytest.approx(1.0)


def test_fleiss_independent_trains_give_about_zero():
    rng = np.random.default_rng(0)
    spikes = rng.random((20, 20000)) < 0.2
    assert abs(spikes.mean() - 0.2) < 0.01, "fixture must fire near 0.2"
    kappa = fleiss_kappa_binary(spikes, bin_ms=1.0, dt_ms=1.0)
    assert abs(kappa) < 0.02, kappa


def test_fleiss_matches_hand_computed_textbook_value():
    # 4 raters (neurons), 2 subjects (bins): bin 0 has 2 active, bin 1 has 1.
    # P_0 = (2*1 + 2*1)/12 = 1/3; P_1 = (1*0 + 3*2)/12 = 1/2; P_bar = 5/12.
    # p_1 = 3/8, P_e = 9/64 + 25/64 = 17/32; kappa = (5/12 - 17/32)/(15/32) = -11/45.
    spikes = np.array([[1, 1], [1, 0], [0, 0], [0, 0]], dtype=float)
    assert fleiss_kappa_binary(spikes, bin_ms=1.0, dt_ms=1.0) == pytest.approx(
        -11 / 45, rel=1e-12
    )


# --- item 2: off-grid bin widths are refused ----------------------------------


def test_fano_refuses_off_grid_bin():
    rng = np.random.default_rng(1)
    spikes = rng.random((4, 100)) < 0.3
    with pytest.raises(ValueError, match=r"bin_size_ms=2\.3.*nearest valid bin width is 2 ms"):
        fano_factor(spikes, bin_size_ms=2.3, dt_ms=1.0)


def test_fleiss_refuses_off_grid_bin():
    rng = np.random.default_rng(2)
    spikes = rng.random((4, 100)) < 0.3
    with pytest.raises(ValueError, match=r"bin_ms=2\.3.*nearest valid bin width is 2 ms"):
        fleiss_kappa_binary(spikes, bin_ms=2.3, dt_ms=1.0)


def test_pairwise_cc_refuses_off_grid_bin():
    rng = np.random.default_rng(3)
    spikes = rng.random((200, 6)) < 0.1
    with pytest.raises(ValueError, match=r"bin_ms=1\.3.*nearest valid bin width is 1 ms"):
        pairwise_cc_binned_disjoint(spikes, 1.0, bin_ms=1.3)


@pytest.mark.parametrize(
    "fn", ["mean_pairwise_spike_correlation", "burst_index", "wang_buzsaki_kappa"]
)
def test_shared_binning_refuses_off_grid_bin(fn):
    import jaxfne.analysis.metrics as m

    rng = np.random.default_rng(4)
    spikes = (rng.random((4, 100)) < 0.3).astype(float)
    with pytest.raises(ValueError, match=r"bin_ms=2\.3.*nearest valid bin width is 2 ms"):
        getattr(m, fn)(spikes, bin_ms=2.3, dt_ms=1.0)


# --- item 3: population_rates ms keywords and deprecated seconds forms --------

_SIZES = {"E": 1000, "PV": 50, "SOM": 50}


def _rates_counts(seed=3, steps=120000):
    """Poisson [steps, 3] counts at dt = 0.1 ms; every population spikes."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson([0.2, 0.06, 0.035], size=(steps, 3))
    assert all(counts[:, i].sum() > 0 for i in range(3)), "fixture must spike in every pop"
    return counts


def test_population_rates_ms_keywords_match_seconds_form():
    counts = _rates_counts()
    with pytest.warns(DeprecationWarning):
        old = population_rates(counts, 1e-4, _SIZES, discard_s=2.0)
    new = population_rates(counts, dt_ms=0.1, sizes=_SIZES, discard_ms=2000.0)
    assert list(new) == list(old) == ["E", "PV", "SOM"]
    for pop in old:
        assert_allclose(new[pop], old[pop], rtol=1e-12)


def test_population_rates_seconds_forms_warn_and_ms_forms_do_not():
    counts = _rates_counts()
    with pytest.warns(DeprecationWarning, match="dt"):
        population_rates(counts, 1e-4, _SIZES)
    with pytest.warns(DeprecationWarning, match="discard_s"):
        population_rates(counts, 1e-4, _SIZES, discard_s=2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        population_rates(counts, dt_ms=0.1, sizes=_SIZES, discard_ms=2000.0)


def test_population_rates_both_forms_of_one_quantity_refused():
    counts = _rates_counts()
    with pytest.raises(ValueError, match="dt"):
        population_rates(counts, 1e-4, _SIZES, dt_ms=0.1)
    with pytest.raises(ValueError, match="discard"):
        population_rates(
            counts, dt_ms=0.1, sizes=_SIZES, discard_s=2.0, discard_ms=2000.0
        )


# --- item 4: welch_numpy uses the periodic Hann window ------------------------


def test_welch_numpy_matches_scipy_periodic_hann():
    """The fallback equals the scipy branch of ``welch_spectrum`` (detrend=False)."""
    signal = pytest.importorskip("scipy.signal")
    rng = np.random.default_rng(9)
    x = 3.0 + rng.standard_normal(4000)
    assert abs(x.mean()) > 2.0, "fixture must carry a mean offset"
    fs, nperseg, noverlap = 1000.0, 256, 128
    f_np, p_np = welch_numpy(x, fs, nperseg, noverlap)
    f_sp, p_sp = signal.welch(
        x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend=False
    )
    assert_allclose(f_np, f_sp, rtol=1e-12)
    assert_allclose(p_np, p_sp, rtol=1e-10)


# --- item 5: pairwise phase consistency (PPC; paper ref. 36) -------------------


def test_ppc_uniform_phases_exact():
    from jaxfne.analysis.spectral import pairwise_phase_consistency
    n = 50
    phases = 2.0 * np.pi * np.arange(n) / n
    per, _ = pairwise_phase_consistency([phases])
    assert_allclose(per[0], -1.0 / (n - 1), rtol=1e-12)


def test_ppc_identical_phases_is_one():
    from jaxfne.analysis.spectral import pairwise_phase_consistency
    phases = np.full(7, 0.3)
    per, _ = pairwise_phase_consistency([phases])
    assert_allclose(per[0], 1.0, rtol=1e-12)


def test_ppc_matches_pair_mean():
    from jaxfne.analysis.spectral import pairwise_phase_consistency
    rng = np.random.default_rng(57)
    groups = [rng.uniform(-np.pi, np.pi, 5),
              rng.uniform(-np.pi, np.pi, 9),
              rng.uniform(-np.pi, np.pi, 20)]
    expected = []
    for ph in groups:
        s = 0.0
        m = len(ph)
        for i in range(m):
            for j in range(i + 1, m):
                s += np.cos(ph[i] - ph[j])
        expected.append(s / (m * (m - 1) / 2.0))
    per, _ = pairwise_phase_consistency(groups)
    assert_allclose(per, expected, rtol=1e-12)


def test_ppc_fewer_than_two_spikes_is_nan():
    from jaxfne.analysis.spectral import pairwise_phase_consistency
    groups = [np.array([]), np.array([0.5]), np.zeros(4)]
    assert max(len(g) for g in groups) >= 2, "at least one neuron needs >= 2 spikes"
    per, mean = pairwise_phase_consistency(groups)
    assert np.isnan(per[0])
    assert np.isnan(per[1])
    assert_allclose(per[2], 1.0, rtol=1e-12)
    assert_allclose(mean, 1.0, rtol=1e-12)


def test_ppc_population_mean_all_nan():
    from jaxfne.analysis.spectral import pairwise_phase_consistency
    groups = [np.array([]), np.array([1.2])]
    per, mean = pairwise_phase_consistency(groups)
    assert np.isnan(per).all()
    assert np.isnan(mean)
