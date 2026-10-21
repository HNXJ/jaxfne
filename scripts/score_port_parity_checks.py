#!/usr/bin/env python3
"""Core-vs-oracle parity checks of jaxfne routes against independent references.

Each check recomputes one agreement pinned by a test, with that test's inputs, tolerance
and reference, and records the worst case beside the test's bound:

- adex_single_cell_oracle: tests/test_emitters_adex.py:165 (transcription)
- adex_network_oracle: tests/test_emitters_adex.py:196 (transcription; four cases)
- multicompartment_hh_jaxley: tests/test_hh_multicompartment.py:44 (external; skipped when jaxley is absent)
- gsdr_mcdp_oracle_zscored: tests/test_gsdr_mcdp.py:64 (transcription)

Where the test states rtol and atol, error is the worst ratio of |core - reference| to the
bound atol + rtol*|reference|: 1 is the test's own assert_allclose limit. Spike mismatches
count as whole units. Each row also records worst_abs_deviation in the test's units.

Writes artifacts/etudes/core_checks/ports/score.json (one {error, tolerance, pass, reference}
row per check) and README.md beside it:

    python scripts/score_port_parity_checks.py         # write both files
    python scripts/score_port_parity_checks.py --check # exit 1 past rtol 1e-9
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import contextlib
import importlib.util
import itertools
import json
import math

import jax
import numpy as np

from jaxfne.emitters_hh import make_hh_params, reference_4comp_morphology, simulate_hh
from jaxfne.optim.gsdr import mcdp_factors

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
OUT_DIR = ROOT / "artifacts" / "etudes" / "core_checks" / "ports"
SCORE_PATH = OUT_DIR / "score.json"
README_PATH = OUT_DIR / "README.md"
RTOL = 1e-9
NOISE_FRACTION = 0.1  # errors below this fraction of the tolerance are compared as rounding residuals

# (rtol, atol) of the pinned test's assert_allclose.
TOLERANCES = {
    "adex_single_cell_oracle": (1e-6, 1e-9),
    "adex_network_oracle": (1e-6, 1e-9),
    "multicompartment_hh_jaxley": (1e-12, 1e-9),
    "gsdr_mcdp_oracle_zscored": (1e-5, 0.0),
}


@contextlib.contextmanager
def _x64():
    """x64 on for one block, restored after (as the pinned tests' fixtures do)."""
    orig = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", orig)


def _load_test(stem):
    """The pinned test module, loaded by path, so its oracle helpers and inputs are the test's own."""
    path = TESTS / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"_pinned_{stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ratio(got, want, rtol, atol):
    """Worst |got - want| over the bound atol + rtol*|want|; above 1 fails assert_allclose."""
    got, want = np.asarray(got, float), np.asarray(want, float)
    bound = atol + rtol * np.abs(want)
    assert np.all(bound > 0), "fixture bound must be positive"
    return float(np.max(np.abs(got - want) / bound))


def _row(error, tolerance, reference, passed, **extra):
    row = {"error": float(error), "tolerance": float(tolerance),
           "pass": bool(passed), "reference": reference}
    row.update(extra)
    return row


def _skipped(reference, reason):
    return {"error": None, "tolerance": None, "pass": None, "reference": reference, "skipped": reason}


def check_adex_single_cell_oracle():
    """Single cells under a step current against the float64 transcription (rtol 1e-6, atol 1e-9)."""
    t = _load_test("test_emitters_adex")
    labels = ("E", "PV", "SOM")
    with _x64():
        I_pA = t._step_current(len(labels))
        V_mV, w_pA, _, spk = t._kernel_cells(t.make_adex_params(labels), I_pA)
        V_o, w_o, spk_o = t._oracle_cells(labels, I_pA * 1e-12, t.DT_S)
    for j, lab in enumerate(labels):
        assert spk[:, j].sum() > 0, f"{lab} must spike under 300 pA"
    mismatches = int(np.count_nonzero(spk != spk_o))
    rtol, atol = TOLERANCES["adex_single_cell_oracle"]
    error = max(_ratio(V_mV, V_o * 1e3, rtol, atol), _ratio(w_pA, w_o * 1e12, rtol, atol), float(mismatches))
    return _row(
        error, 1.0, "float64 NumPy transcription of caminos_jax.adex_step (SI units)", error <= 1.0,
        test="tests/test_emitters_adex.py:165",
        worst_abs_deviation={
            "V_mV": float(np.max(np.abs(V_mV - V_o * 1e3))),
            "w_pA": float(np.max(np.abs(w_pA - w_o * 1e12))),
            "spike_mismatches": mismatches,
        },
    )


def check_adex_network_oracle():
    """Two-cell E->PV and PV->E network against the hand-coded float64 loop, all four test cases."""
    t = _load_test("test_emitters_adex")
    labels = ("E", "PV")
    ne, nr = t.NETWORK_EDGES, t.NETWORK_RECEPTORS
    edges = (ne["edge_pre"], ne["edge_post"], ne["edge_weight_nS"], ne["edge_receptor"],
             ne["edge_delay_steps"], nr["receptor_reversal_mV"], nr["receptor_tau_ms"])
    rtol, atol = TOLERANCES["adex_network_oracle"]
    cases = {}
    with _x64():
        I_pA = t._step_current(len(labels))
        for decay, step_order in itertools.product(("exp", "euler"), ("default", "brian2")):
            out = t.simulate_adex(
                t.make_adex_params(labels), n_steps=t.N_STEPS_STEP, dt_ms=t.DT_MS,
                **ne, **nr, decay=decay, step_order=step_order, I_ext_pA=I_pA, record_v=True,
            )
            spk = np.asarray(out["spikes"])
            V_mV = np.asarray(out["V_mV"])
            assert spk.sum(axis=0).min() > 0, "both cells must spike in the network run"
            V_o, spk_o = t._oracle_network(labels, edges, t.N_STEPS_STEP, I_pA * 1e-12, decay, step_order)
            mismatches = int(np.count_nonzero(spk != spk_o))
            cases[f"{decay}/{step_order}"] = max(_ratio(V_mV, V_o * 1e3, rtol, atol), float(mismatches))
    error = max(cases.values())
    return _row(
        error, 1.0, "float64 NumPy hand-coded loop of the two-cell network (transcription)", error <= 1.0,
        test="tests/test_emitters_adex.py:196", cases=cases,
    )


def check_multicomp_hh_jaxley():
    """Four-compartment HH cell under a step current against jaxley 0.14.0 jx.integrate (rtol 1e-12, atol 1e-9)."""
    reference = "jaxley 0.14.0 jx.integrate, external"
    try:
        import jaxley
    except ImportError:
        return _skipped(reference, "jaxley not installed")
    t = _load_test("test_hh_multicompartment")
    rtol, atol = TOLERANCES["multicompartment_hh_jaxley"]
    n_steps = 200
    with _x64():
        stim_on = int(1.0 // t.DT_MS)
        cell = t._one_cell_oracle()
        assert cell.recordings.rec_index.to_numpy().tolist() == [0, 1, 2, 3], \
            "recordings must map row i to compartment i"
        stim = np.zeros(n_steps, dtype=np.float64)
        stim[stim_on:] = 0.02
        data_stim = cell.branch(1).loc(0.0).data_stimulate(stim)
        v_oracle = np.asarray(jaxley.integrate(
            cell, data_stimuli=data_stim, t_max=n_steps * t.DT_MS, delta_t=t.DT_MS,
            solver="bwd_euler", voltage_solver="jaxley.dhs",
        ))
        assert v_oracle.shape == (4, n_steps + 1)
        stim_nA = np.zeros((n_steps, 1), dtype=np.float64)
        stim_nA[stim_on:, 0] = 0.02
        out = simulate_hh(
            make_hh_params(["E"]), n_steps=n_steps, dt_ms=t.DT_MS, I_ext_nA=stim_nA,
            stim_comp=2, record_comp=0, morphology=reference_4comp_morphology(),
            gate_init=0.2, record_v=True,
        )
        v_ours = np.asarray(out["V_mV"])[:, 0, :]
    error = max(_ratio(v_ours[:, i], v_oracle[i, 1:], rtol, atol) for i in range(4))
    worst_abs = max(float(np.max(np.abs(v_ours[:, i] - v_oracle[i, 1:]))) for i in range(4))
    return _row(error, 1.0, reference, error <= 1.0,
                test="tests/test_hh_multicompartment.py:44", worst_abs_deviation_mV=worst_abs)


def check_gsdr_mcdp_oracle_zscored():
    """Two pre/post groups on a 2x60x8 trace, each against the lane-oracle transcription (rtol 1e-5); seed 23."""
    t = _load_test("test_gsdr_mcdp")
    rng = np.random.default_rng(23)
    v = rng.standard_normal((2, 60, 8))
    pre_a, post_a = np.asarray([0, 1, 2]), np.asarray([3, 4, 5])
    pre_g, post_g = np.asarray([1, 2, 6]), np.asarray([0, 5, 7])
    with _x64():
        got_a = np.asarray(mcdp_factors(v, pre_a, post_a, 10))
        got_g = np.asarray(mcdp_factors(v, pre_g, post_g, 10))
    want_a = t._oracle_factors(v, pre_a, post_a, 10)
    want_g = t._oracle_factors(v, pre_g, post_g, 10)
    rtol, atol = TOLERANCES["gsdr_mcdp_oracle_zscored"]
    error = max(_ratio(got_a, want_a, rtol, atol), _ratio(got_g, want_g, rtol, atol))
    return _row(error, 1.0, "float64 NumPy transcription of the lane oracle (PORT_SPEC section 10)",
                error <= 1.0, test="tests/test_gsdr_mcdp.py:64", seed=23,
                worst_abs_deviation=float(max(np.max(np.abs(got_a - want_a)), np.max(np.abs(got_g - want_g)))))


CHECKS = (
    ("adex_single_cell_oracle", check_adex_single_cell_oracle),
    ("adex_network_oracle", check_adex_network_oracle),
    ("multicompartment_hh_jaxley", check_multicomp_hh_jaxley),
    ("gsdr_mcdp_oracle_zscored", check_gsdr_mcdp_oracle_zscored),
)


def compute_all():
    """One score row per check, in CHECKS order."""
    checks = {name: fn() for name, fn in CHECKS}
    live = [row for row in checks.values() if "skipped" not in row]
    return {"checks": checks,
            "pass": bool(live) and all(row["pass"] for row in live),
            "skipped": sorted(name for name, row in checks.items() if "skipped" in row)}


def compare(old, new, rtol=RTOL):
    """Mismatch strings between a stored document and a fresh one; [] means agreement."""
    if set(old.get("checks", ())) != set(new.get("checks", ())):
        return [f"checks: {sorted(old.get('checks', ()))} != {sorted(new.get('checks', ()))}"]
    bad = []
    for name in old["checks"]:
        o, n = old["checks"][name], new["checks"][name]
        if "skipped" in n and "skipped" not in o:
            # This environment lacks the check's optional dependency; the stored row stands.
            continue
        if ("skipped" in o) != ("skipped" in n):
            bad.append(f"{name}: skipped {('skipped' in o)} != {('skipped' in n)}")
            continue
        if "skipped" in o:
            if o["skipped"] != n["skipped"]:
                bad.append(f"{name}.skipped: {o['skipped']!r} != {n['skipped']!r}")
            continue
        if o["pass"] != n["pass"]:
            bad.append(f"{name}.pass: {o['pass']!r} != {n['pass']!r}")
        for key in ("error", "tolerance"):
            if math.isclose(o[key], n[key], rel_tol=rtol, abs_tol=1e-15):
                continue
            # A rounding-level residual differs across platforms (BLAS, float32); below a tenth
            # of the tolerance in both copies it cannot change the verdict.
            floor = NOISE_FRACTION * min(o["tolerance"], n["tolerance"])
            if key == "error" and o[key] <= floor and n[key] <= floor:
                continue
            bad.append(f"{name}.{key}: {o[key]!r} != {n[key]!r} past rtol {rtol:g}")
    if old.get("pass") != new.get("pass"):
        bad.append(f"pass: {old.get('pass')!r} != {new.get('pass')!r}")
    return bad


README_TEXT = """# Port and oracle parity checks

Core routes of `jaxfne` against independent references, recomputed by
`scripts/score_port_parity_checks.py` with the inputs, tolerances and references of the
tests named below. `score.json` holds one `{error, tolerance, pass, reference}` row per check.
For a test that states `assert_allclose(rtol, atol)`, `error` is the worst ratio of
`|core - reference|` to `atol + rtol·|reference|`, so a passing test implies a passing row
(the limit is 1). `worst_abs_deviation` is the largest `|core - reference|` over all elements, in the
test's units; it need not be the element that sets the worst ratio.

Reference types: **external** (another library) and **transcription** (file-local float64
code of published equations or of a documented lane oracle).

| check | test | reference | type | bound |
|---|---|---|---|---|
| adex_single_cell_oracle | tests/test_emitters_adex.py:165 | float64 NumPy transcription of `caminos_jax.adex_step` (the CAMINOS project port, artifacts/etudes/papers/caminos/caminos_jax.py, which imports no jaxfne; not the authors' code) | transcription | rtol 1e-6, atol 1e-9; spikes exact |
| adex_network_oracle | tests/test_emitters_adex.py:196 | float64 hand-coded loop of the two-cell network, four cases | transcription | rtol 1e-6, atol 1e-9; spikes exact |
| multicompartment_hh_jaxley | tests/test_hh_multicompartment.py:44 | jaxley 0.14.0 jx.integrate, four compartments | external | rtol 1e-12, atol 1e-9 |
| gsdr_mcdp_oracle_zscored | tests/test_gsdr_mcdp.py:64 | float64 transcription of the lane oracle, seed 23 | transcription | rtol 1e-5 |

`multicompartment_hh_jaxley` needs the optional `jaxley` package; without it the row keeps
nulls with `"skipped": "jaxley not installed"`.

Runtime on the build machine, one run: adex single cell 1.4 s, adex network 2.6 s,
multicompartment 26.0 s, gsdr zscored 23.4 s.

Candidates not scored (budget: no check above 30 s; each needs a reference that is not the same code path):

| candidate | reason |
|---|---|
| tests/test_jaxley_emitter_bridge_e2e.py:60 | mechanistic only (finite V, spike above 0 mV); 115.5 s |
| tests/test_gsdr_mcdp.py:31 and :47 | transcription; 34.4 s and 33.1 s |
| tests/test_destexhe2009_port.py:76 | asserts a spike count only; no reference value |
| tests/test_intra_step_fixed_point_t22.py, `_oracle_first_step` | calls the core `adex_step`, so the same code path |
| tests/test_emitters_lif.py:288 and :419 | oracle side asserted inline on hand-set events |
| tests/test_istdp_vogels2011.py:202 | hand pairing values for the paper rule; open |

Rerun:

    python scripts/score_port_parity_checks.py         # rewrite score.json (this README is left as is)
    python scripts/score_port_parity_checks.py --check # recompute and exit 1 past rtol 1e-9
"""


def write(doc):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCORE_PATH, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    if not README_PATH.is_file():
        README_PATH.write_text(README_TEXT, encoding="utf-8", newline="\n")


def check(doc):
    try:
        stored = json.loads(SCORE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"error: cannot read {SCORE_PATH}: {exc}", file=sys.stderr)
        return 1
    bad = compare(stored, doc)
    if bad:
        for line in bad:
            print(f"mismatch: {line}", file=sys.stderr)
        return 1
    print(f"fresh: {SCORE_PATH} matches a recomputation within rtol {RTOL:g} "
          f"(errors under {NOISE_FRACTION:g} x tolerance in both copies count as rounding residuals)")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score the core-vs-oracle parity checks.")
    parser.add_argument("--check", action="store_true",
                        help="recompute and exit 1 past rtol 1e-9")
    args = parser.parse_args(argv)
    doc = compute_all()
    if args.check:
        return check(doc)
    write(doc)
    print(f"wrote {SCORE_PATH} ({len(doc['checks'])} checks, pass={doc['pass']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
