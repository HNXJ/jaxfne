# Metric definitions declared 2026-10-13 (T-13)

Owner's call 2026-10-13 (todo T-13): the three misnamed metrics in
`jaxfne/analysis/metrics.py` are replaced by the definitions below. Their
`DeprecationWarning`s are removed.

Shared binning (all three): `b = max(1, int(np.round(bin_ms / dt_ms)))`
steps; non-overlapping bins over axis 1 of `spikes` (n_neurons, n_time);
`n_bins = n_time // b`; the trailing partial bin is dropped; if
`n_bins == 0` the metric returns 0.0. Implemented once as the private helper
`_bin_counts(spikes, bin_ms, dt_ms) -> (n_neurons, n_bins)` float array.

Worked fixture: `spikes = [[1,0,1,0],[0,0,1,1]]`, `dt_ms = 1.0`.
Bin 1 step means `bin_ms = 1.0`; bin 2 steps means `bin_ms = 2.0`.
Old values were produced by `git show 65c5c9ef:jaxfne/analysis/metrics.py`;
new values by the current `jaxfne/analysis/metrics.py`. Both runs used the
fixture above.

| metric | old definition | new definition | bin 1 step: old vs new | bin 2 steps: old vs new |
|---|---|---|---|---|
| `fano_factor` | variance/mean of the population-summed binned count (one value) | per-neuron Fano factor averaged over neurons: `F_i = var(c_i)/mean(c_i)` (ddof 0) for neurons with `mean(c_i) > 0`; 0.0 when none or not finite | 0.5 vs 0.5 | 0.5 vs 0.5 |
| `burst_index` | fraction of raw time steps whose cross-neuron mean reaches threshold (`bin_ms` accepted but unused) | per bin, the fraction of neurons with count > 0; the fraction of bins where that reaches `active_fraction_threshold` | 0.75 vs 0.75 | 0.75 vs 1.0 |
| `mean_pairwise_spike_correlation` | mean upper-triangle Pearson r of raw per-step spike rows (no binning; no `bin_ms`/`dt_ms` arguments) | same algorithm (nonzero-variance rows, `>= min_neurons`; `>= min_events` active bins; mean finite upper-triangle r) applied to the binned counts; `bin_ms` and `dt_ms` required keywords | 0.0 vs 0.0 | n/a (no binning before) vs 0.0 |

Notes on the worked values: on this fixture the per-neuron Fano means
coincide with the old population-summed ratios (0.5 at both widths), and the
old and new correlations coincide (0.0: the two rows are uncorrelated
per-step, and the 2-step binned rows `[1,1]` and `[0,2]` leave only one row
with variance, so 0.0). The burst index moves 0.75 -> 1.0 when binning at 2
steps, since both binned windows then reach the 0.5 active-fraction
threshold.

## Undefined values (2026-10-13)

Owner's call 2026-10-13: an undefined metric returns NaN, never a
placeholder 0.0. A real computed 0.0 stays 0.0.

- `mean_pairwise_spike_correlation`: NaN when fewer than 2 bins result,
  fewer than `min_neurons` binned-count rows have variance, fewer than
  `min_events` bins carry any activity, or no finite pair remains.
- `burst_index`: NaN when there are zero neurons or zero full bins. A
  defined fraction 0.0 (e.g. no spikes) stays 0.0.
- `fano_factor`: NaN when zero full bins result, no neuron has a positive
  mean count, or the mean ratio is not finite.
- `fleiss_kappa_binary`: NaN exactly where kappa is mathematically
  undefined: fewer than 2 neurons, zero time steps, zero full bins
  (`n_bins == 0`, i.e. the bin window covers the whole trace), expected
  agreement `p_e == 1`, or a non-finite kappa.
