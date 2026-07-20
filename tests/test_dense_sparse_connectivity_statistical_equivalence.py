"""Statistical equivalence check between jaxfne's dense-masked and sparse-
direct within-area connectivity constructions, at a small N where both
paths can be directly invoked and compared (jax-fem-harden skill rule #3:
"verify dense vs. sparse give identical connectivity on a small case before
trusting the sparse path" -- this was a documented rule without a
dedicated regression test before this pass; see
plans.json:novelty::tfne-cross-backend-reframing).

_make_sparse_within_area_edges's own docstring already states it is
"statistically equivalent to the dense path; NOT bit-identical" -- so this
test checks aggregate statistics across many seeds, not exact per-edge
values. The desired, correct outcome is FAILING to reject the null
hypothesis of no difference (a large p-value) -- unlike every other
significance test in this repo (ed9/ed10), where a small p-value was the
sought-after result.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jaxfne.emitters import izhikevich_params_from_labels
from jaxfne._construct import _suite2_apply_connectivity, _make_sparse_within_area_edges

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str((_Path(__file__).resolve().parent.parent / "scripts")))
from _ed9_common import _significance_test


def _build_params(n=200, e_fraction=0.75):
    n_e = int(n * e_fraction)
    labels = tuple(["E"] * n_e + ["PV"] * (n - n_e))
    layer_labels = tuple(["L4"] * n)
    area_labels = tuple(["V1"] * n)
    params = izhikevich_params_from_labels(
        labels=labels, layer_labels=layer_labels, dtype="float32",
        build_dense_connectivity=False)
    return params, labels, layer_labels, area_labels


def _mean_incoming_weight(n, seed, p_connect=0.3, within_gain=0.45):
    params, labels, layer_labels, area_labels = _build_params(n)
    metadata = {"connectivity": {"p_connect": p_connect, "within_gain": within_gain}}

    params_dense, edges_dense = _suite2_apply_connectivity(
        params, area_labels, layer_labels, labels, metadata, seed=seed, dtype="float32")
    assert edges_dense is None, "expected the dense-masked path at this N"
    W = np.asarray(params_dense.W)
    dense_mean = float(np.abs(W).sum(axis=1).mean())

    edges_sparse = _make_sparse_within_area_edges(
        area_labels, params.sign, n, within_gain=within_gain, p_connect=p_connect,
        seed=seed, jdtype=jnp.float32)
    post = np.asarray(edges_sparse.post)
    w = np.asarray(edges_sparse.weight)
    incoming = np.zeros(n)
    np.add.at(incoming, post, np.abs(w))
    sparse_mean = float(incoming.mean())

    return dense_mean, sparse_mean


def test_dense_and_sparse_paths_produce_statistically_equivalent_incoming_weight():
    """Across 20 seeds, the dense-masked and sparse-direct constructions'
    per-network mean incoming weight must be statistically indistinguishable
    -- NOT bit-identical (the sparse path's own docstring says so), but
    equivalent in aggregate. Correct outcome: FAIL to reject the null
    (large p-value), unlike ed9/ed10's tests where a small p-value was
    sought."""
    dense_means, sparse_means = [], []
    for seed in range(20):
        d, s = _mean_incoming_weight(n=200, seed=seed)
        dense_means.append(d)
        sparse_means.append(s)

    result = _significance_test(dense_means, sparse_means)
    assert result["p_value"] is not None
    print(f"dense mean incoming weight: {np.mean(dense_means):.4f} +/- {np.std(dense_means):.4f}")
    print(f"sparse mean incoming weight: {np.mean(sparse_means):.4f} +/- {np.std(sparse_means):.4f}")
    print(f"Mann-Whitney U p={result['p_value']:.4g}, Cohen's d={result['cohens_d']:.3f}")

    # The correct outcome here is a LARGE p-value (fail to reject the null of
    # no difference) -- a small p-value would mean the two construction paths
    # are NOT statistically equivalent, contradicting the documented design intent.
    assert result["p_value"] > 0.05, (
        f"dense and sparse paths differ significantly (p={result['p_value']:.4g}) -- "
        f"this would mean the sparse-direct path is NOT a valid statistical substitute "
        f"for the dense path, a real correctness concern, not just a style difference"
    )
    # Effect size should also be small in absolute terms, not just non-significant
    # by p-value alone (a real check against p-hacking-style false reassurance).
    assert abs(result["cohens_d"]) < 0.5, (
        f"Cohen's d={result['cohens_d']:.3f} suggests a non-trivial effect size even "
        f"if not statistically significant at n=20 -- worth more seeds before trusting this"
    )


def test_dense_and_sparse_edge_counts_match_expected_degree():
    """Both paths should produce roughly the same expected total edge count
    for the same n/p_connect (a coarser, simpler sanity check independent of
    the weight-magnitude statistics test above)."""
    n, p_connect = 200, 0.3
    params, labels, layer_labels, area_labels = _build_params(n)
    metadata = {"connectivity": {"p_connect": p_connect, "within_gain": 0.45}}

    _, edges_dense_none = _suite2_apply_connectivity(
        params, area_labels, layer_labels, labels, metadata, seed=1, dtype="float32")
    params_dense, _ = _suite2_apply_connectivity(
        params, area_labels, layer_labels, labels, metadata, seed=1, dtype="float32")
    n_edges_dense = int(np.sum(np.asarray(params_dense.W) != 0.0))

    edges_sparse = _make_sparse_within_area_edges(
        area_labels, params.sign, n, within_gain=0.45, p_connect=p_connect, seed=1, jdtype=jnp.float32)
    n_edges_sparse = int(edges_sparse.n_edges)

    expected = n * (n - 1) * p_connect
    assert abs(n_edges_dense - expected) / expected < 0.15
    assert abs(n_edges_sparse - expected) / expected < 0.15
