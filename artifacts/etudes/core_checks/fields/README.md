# Field-invariant core checks

Worst-case deviations of the field and probe operators in `jaxfne.fields` against independent
references, recomputed by `scripts/score_field_invariant_checks.py`. `score.json` holds one
`{error, tolerance, pass, reference}` row per check; `error` is the worst-case deviation and
`tolerance` is the bound named in the table. A passing row means the stated agreement holds at
the stated inputs; these are analytic and hand-formula agreements of a core method, not a
paper replication, and there is no paper value beside ours.

| check | pinned by | reference | tolerance |
|---|---|---|---|
| operator_superposition | tests/test_global_superposition.py:12 | hand formula Y = S W^T in float64 numpy; seed 100; float32 inputs as the test | worst element-wise ratio \|got - Y\| / (1e-5 + 1e-5 \|Y\|) against 1 (np.allclose as the test) |
| csd_flat_potential_zero | tests/test_csd_tensor.py:42 | zero second difference of a constant (3.7, 5 x 8) | 1e-5 absolute |
| csd_quadratic_potential | new (cf. tests/test_linear_readouts.py:44, which uses a test-local stencil) | -phi'' = -2a at interior contacts for phi = a z^2 + b z + c, float64 | 1e-6 · max\|-phi''\| |
| cable_filter_transfer | new (tests/test_cable_filter_tensor.py:60 is qualitative only) | exact discrete response of the cascade at 5, 20, 100 Hz, tau 20 ms, dt 0.5 ms, order 2 | 1e-6 relative |
| poisson_1d_flux_magnitude | tests/test_experimental_poisson_1d_convergence.py:71 | Gauss's law: interior flux = Q dx at N = 21, 41, 81, Q = 2 | 1e-3 relative |
| poisson_1d_flux_span | tests/test_experimental_poisson_1d_convergence.py:56 | Gauss's law: flux zero outside [src, sink), constant inside, N = 81 | 1e-4 absolute |

Conventions and choices, stated once:

- csd_tensor is -(second difference)/dz^2 with edge padding and no conductivity, so the reference is
  -phi'' with sigma = 1 (its docstring: "without the conductivity"). Edge contacts are not scored: the
  edge padding gives a one-sided stencil there, which departs from -phi'' by design (the fixture asserts
  the departure). The reference sign and convention come from the operator's own docstring.
- The cable filter is a fixed linear recursion per section, y[n] = alpha x[n] + (1 - alpha) y[n-1] with
  alpha = 1 - exp(-dt/tau), cascaded `order` times (cable_filter_sources, proxy.py). Its exact response is
  H1(z) = alpha / (1 - (1 - alpha) z^-1), so |H(f)| = (alpha / sqrt(1 - 2 (1 - alpha) cos(Omega) + (1 - alpha)^2))^order
  with Omega = 2 pi f dt. The reference is this closed form, not the continuous RC transfer. The gain is
  measured over 800 to 4000 samples, whole cycles of each frequency, after a 0.4 s transient.
- The tolerances of the new checks (csd_quadratic_potential, cable_filter_transfer) and of
  poisson_1d_flux_magnitude are this scorer's choices, recorded here; the other three are the pinned tests' own.

Rerun:

    python scripts/score_field_invariant_checks.py         # rewrite score.json (this README is left as is)
    python scripts/score_field_invariant_checks.py --check # recompute and exit 1 past rtol 1e-9

The suite runs under `JAX_PLATFORMS=cpu`.
