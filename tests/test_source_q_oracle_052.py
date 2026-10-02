"""0.5.2 item 3 oracle: one source representation Q, bit-identical proxies.

Frozen literals below are pre-change outputs (seed 20260924): every probe /
transform / projection run on the frozen arrays must reproduce them exactly,
whether fed a bare array or the single ``CanonicalSource`` Q. Q-path reports
add provenance keys; array-path reports are unchanged.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import jax.numpy as jnp

from jaxfne.fields import (
    CanonicalSource,
    canonical_source,
    CANONICAL_SOURCE_REPRESENTATION,
    spk_probe,
    vm_probe,
    source_probe,
    lfp_proxy_probe,
    csd_proxy_probe,
    eeg_proxy_probe,
    meg_proxy_probe,
    emm_proxy_probe,
    eeg_proxy_transform,
    meg_proxy_transform,
    emm_proxy_transform,
    sample_phi_at_probe_depths,
    project_laminar_sources,
    construct_source_tensor,
    filtered_spike_source,
    csd_tensor,
    LinearReadout,
)

SEED = 20260924
T, N, C, S = 24, 6, 8, 5

# Pre-change frozen output hashes (oracle_dump.py on the pristine tree).
ORACLE = {
    "spk": "3f8f6ac99ab0b999",
    "vm": "377e6ba866fa8d72",
    "source": "7686733b61977385",
    "lfp_pass": "1ad9e6d6033f527a",
    "lfp_interp": "95a4ebfea8d62b81",
    "csd": "4efc4677c7ed4724",
    "eeg": "6e8f2f85b1c28232",
    "meg": "b51122c7f6f80e73",
    "emm": "59307b82eaf3a352",
    "sample": "95a4ebfea8d62b81",
    "proj_csd": "40f3eab9675d96ec",
    "proj_kernel": "d3b7938fabe1228f",
    "q_total": "b40fa73903159f8b",
    "q_spike": "3f8f6ac99ab0b999",
    "csd_t": "34f77db0e3577da0",
}


def _h(a) -> str:
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()[:16]


@pytest.fixture(scope="module")
def frozen():
    rng = np.random.default_rng(SEED)
    spikes = (rng.random((T, N)) < 0.2).astype(np.float32)
    vm = rng.normal(size=(T, N)).astype(np.float32)
    src = rng.normal(size=(T, N)).astype(np.float32)
    phi = rng.normal(size=(T, C)).astype(np.float32)
    csd = rng.normal(size=(T, C)).astype(np.float32)
    eeg = rng.normal(size=(T, S)).astype(np.float32)
    meg = rng.normal(size=(T, S)).astype(np.float32)
    emm = rng.normal(size=(T, 1)).astype(np.float32)
    lead = rng.normal(size=(S, N)).astype(np.float32)
    pos = np.zeros((N, 3), dtype=np.float32)
    pos[:, 2] = np.linspace(0, 1, N).astype(np.float32)
    return {
        "spikes": spikes,
        "vm": vm,
        "src": src,
        "phi": phi,
        "csd": csd,
        "eeg": eeg,
        "meg": meg,
        "emm": emm,
        "lead": lead,
        "pos": pos,
        "field_z": np.linspace(0, 1, C).astype(np.float32),
        "probe_z": np.array([0.2, 0.8], dtype=np.float32),
    }


def test_oracle_array_path_matches_prechange(frozen):
    f = frozen
    assert _h(spk_probe(jnp.asarray(f["spikes"])).data) == ORACLE["spk"]
    assert _h(vm_probe(jnp.asarray(f["vm"])).data) == ORACLE["vm"]
    assert _h(source_probe(jnp.asarray(f["src"])).data) == ORACLE["source"]
    assert _h(lfp_proxy_probe(jnp.asarray(f["phi"])).data) == ORACLE["lfp_pass"]
    r = lfp_proxy_probe(
        jnp.asarray(f["phi"]),
        contact_depths=jnp.asarray(f["probe_z"]),
        field_contact_depths=jnp.asarray(f["field_z"]),
    )
    assert _h(r.data) == ORACLE["lfp_interp"]
    assert r.report["method"] == "depth_interpolation_on_phi_e_proxy"
    assert _h(csd_proxy_probe(jnp.asarray(f["csd"])).data) == ORACLE["csd"]
    assert _h(eeg_proxy_probe(jnp.asarray(f["eeg"])).data) == ORACLE["eeg"]
    assert _h(meg_proxy_probe(jnp.asarray(f["meg"])).data) == ORACLE["meg"]
    assert _h(emm_proxy_probe(jnp.asarray(f["emm"])).data) == ORACLE["emm"]
    # eeg_t/meg_t are one float32 matmul (source @ leadfield.T): bit-hashes
    # flip on 1-ulp BLAS reorder (the 3.11 CI leg), while values agree to
    # ~2e-6 relative against float64 truth (measured locally). Pinned instead
    # by twin equality (same op and inputs: identical bytes on any backend)
    # plus numpy-float64 agreement at the tests/_numeric_gates.py epsilons
    # (rtol=1e-5, atol=1e-8: 5x headroom over the measured noise).
    eeg32 = np.asarray(eeg_proxy_transform(jnp.asarray(f["src"]), jnp.asarray(f["lead"])))
    meg32 = np.asarray(meg_proxy_transform(jnp.asarray(f["src"]), jnp.asarray(f["lead"])))
    assert _h(eeg32) == _h(meg32)
    ref64 = np.asarray(f["src"], dtype=np.float64) @ np.asarray(f["lead"], dtype=np.float64).T
    np.testing.assert_allclose(eeg32, ref64, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(meg32, ref64, rtol=1e-5, atol=1e-8)
    # emm_t sums L1/L2 terms over units (jnp.sum reductions): tree-reduction
    # order varies by backend (local 3.12 falsified the pin). numpy-float64
    # agreement at gate eps instead of a hash.
    emm32 = np.asarray(
        emm_proxy_transform(
            jnp.asarray(f["spikes"].mean(axis=1, keepdims=True)),
            jnp.asarray(f["src"]),
            jnp.asarray(f["phi"]),
        )
    )
    spk64 = np.asarray(f["spikes"], dtype=np.float64).mean(axis=1, keepdims=True)
    src64 = np.asarray(f["src"], dtype=np.float64)
    phi64 = np.asarray(f["phi"], dtype=np.float64)
    ref_emm64 = (
        spk64 + np.sum(np.abs(src64), axis=1, keepdims=True)
        + np.sum(np.square(phi64), axis=1, keepdims=True)
    ) / 3.0
    np.testing.assert_allclose(emm32, ref_emm64, rtol=1e-5, atol=1e-8)
    assert (
        _h(
            sample_phi_at_probe_depths(
                jnp.asarray(f["phi"]), jnp.asarray(f["field_z"]), jnp.asarray(f["probe_z"])
            )
        )
        == ORACLE["sample"]
    )
    fo = project_laminar_sources(jnp.asarray(f["src"]), jnp.asarray(f["pos"]), n_contacts=C)
    # source/phi/lfp proxies alias one array (sources @ kernel.T, a float32
    # matmul with the same 1-ulp BLAS fragility as eeg_t/meg_t above; the
    # 3.14 leg flipped proj_source). Pinned by twin equality (aliasing
    # structure) plus numpy-float64 agreement at gate eps, not by hash.
    assert _h(fo.source_proxy) == _h(fo.phi_e_proxy) == _h(fo.lfp_proxy)
    # Reference shares the impl's own kernel (bit-pinned below), so the
    # agreement isolates matmul summation order: pure ~2e-6 noise, 5x headroom.
    k32 = np.asarray(fo.kernel, dtype=np.float64)
    ref_proj64 = np.asarray(f["src"], dtype=np.float64) @ k32.T
    np.testing.assert_allclose(np.asarray(fo.source_proxy), ref_proj64, rtol=1e-5, atol=1e-8)
    assert _h(fo.csd_proxy) == ORACLE["proj_csd"]
    assert _h(fo.kernel) == ORACLE["proj_kernel"]
    q1, _ = construct_source_tensor(
        mode="total_membrane_current_proxy",
        total_membrane_current=jnp.asarray(f["src"]),
        scale=2.0,
    )
    assert _h(q1) == ORACLE["q_total"]
    q2, _ = construct_source_tensor(mode="spike_proxy", spike_proxy=jnp.asarray(f["spikes"]))
    assert _h(q2) == ORACLE["q_spike"]
    fs, _ = filtered_spike_source(
        jnp.asarray(f["spikes"]), {"cell_type": ["E"] * N}, tau_ms=5.0, dt_ms=0.1
    )
    # q_filt is a 24-step scan accumulation (alpha*carry + spike): FMA fusion
    # and rounding vary by backend. Reference is the same recurrence in
    # float64 with a plain loop (fixed time order); signs are all +1 here
    # (cell_type ["E"] * N).
    alpha64 = float(np.exp(-0.1 / 5.0))
    s64 = np.asarray(f["spikes"], dtype=np.float64)
    carry = np.zeros(s64.shape[1])
    ref_qfilt64 = np.empty_like(s64)
    for t in range(s64.shape[0]):
        carry = alpha64 * carry + s64[t]
        ref_qfilt64[t] = carry
    np.testing.assert_allclose(np.asarray(fs), ref_qfilt64, rtol=1e-5, atol=1e-8)
    assert (
        _h(csd_tensor(jnp.asarray(f["phi"]), jnp.asarray(0.1, dtype=jnp.float32)))
        == ORACLE["csd_t"]
    )
    # LinearReadout.apply is src @ W.T with W = lead: the same matmul as
    # eeg_t/meg_t, so the same agreement treatment (ref64 shares W=lead).
    lr = LinearReadout(name="t", W=jnp.asarray(f["lead"]))
    np.testing.assert_allclose(np.asarray(lr.apply(jnp.asarray(f["src"]))), ref64, rtol=1e-5, atol=1e-8)


def test_q_path_bit_identical_to_array_path(frozen):
    f = frozen
    cases = [
        (spk_probe, (f["spikes"],)),
        (vm_probe, (f["vm"],)),
        (source_probe, (f["src"],)),
        (csd_proxy_probe, (f["csd"],)),
        (eeg_proxy_probe, (f["eeg"],)),
        (meg_proxy_probe, (f["meg"],)),
        (emm_proxy_probe, (f["emm"],)),
    ]
    for fn, (arr,) in cases:
        a = fn(jnp.asarray(arr))
        q = fn(canonical_source(jnp.asarray(arr), source_mode="oracle"))
        np.testing.assert_array_equal(np.asarray(q.data), np.asarray(a.data))
        assert _h(q.data) == _h(a.data)
        assert q.report["source_identity"] == "canonical_source_Q"
        assert q.report["source_mode"] == "oracle"
        assert "source_identity" not in a.report
    a = lfp_proxy_probe(jnp.asarray(f["phi"]))
    q = lfp_proxy_probe(canonical_source(jnp.asarray(f["phi"]), source_mode="oracle"))
    np.testing.assert_array_equal(np.asarray(q.data), np.asarray(a.data))
    a = eeg_proxy_transform(jnp.asarray(f["src"]), jnp.asarray(f["lead"]))
    q = eeg_proxy_transform(
        canonical_source(jnp.asarray(f["src"]), source_mode="oracle"), jnp.asarray(f["lead"])
    )
    np.testing.assert_array_equal(np.asarray(q), np.asarray(a))
    fo_a = project_laminar_sources(jnp.asarray(f["src"]), jnp.asarray(f["pos"]), n_contacts=C)
    fo_q = project_laminar_sources(
        canonical_source(jnp.asarray(f["src"]), source_mode="oracle"),
        jnp.asarray(f["pos"]),
        n_contacts=C,
    )
    for attr in ("source_proxy", "phi_e_proxy", "csd_proxy", "lfp_proxy", "kernel"):
        np.testing.assert_array_equal(
            np.asarray(getattr(fo_q, attr)), np.asarray(getattr(fo_a, attr))
        )


def test_q_refuses_non_relative_representation(frozen):
    with pytest.raises(ValueError, match="relative-only"):
        CanonicalSource(data=jnp.asarray(frozen["src"]), representation="physical_V")
    with pytest.raises(ValueError, match="relative-only"):
        CanonicalSource(data=jnp.asarray(frozen["src"]), representation="relative_physical")


def test_q_requires_2d_finite(frozen):
    with pytest.raises(ValueError, match="2D"):
        canonical_source(jnp.asarray(frozen["src"][0]))
    with pytest.raises(ValueError, match="finite"):
        bad = np.asarray(frozen["src"])
        bad[0, 0] = np.inf
        canonical_source(jnp.asarray(bad))


def test_q_representation_constant():
    assert CANONICAL_SOURCE_REPRESENTATION == "relative"
    q = canonical_source(jnp.zeros((4, 3), dtype=np.float32))
    assert q.representation == "relative"
