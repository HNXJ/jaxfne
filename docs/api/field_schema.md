# Field Schema API

Field outputs in jaxfne are computational-scaffold proxy readouts produced by
the field solver functions in `jaxfne/fields/`. This page documents the
metadata contract every solver output carries: claim level, solver status, and
amplitude-calibration status. It is a contract description, not a class
reference.

## Scope

- Every field output is a **Relative** proxy readout inside a computational
  scaffold: simulated, structural quantities with no physical amplitude
  calibration.
- Solver claim level, solver status, and amplitude-calibration status are
  explicit metadata fields on every output surface. Downstream code must read
  them from metadata; they must not be inferred from an array's shape or dtype.
- `physical_amplitude_calibrated` is `False` on every solver path documented
  on this page.

## Common metadata

Three metadata fields are required on every field solver output surface.

| Key | Meaning | Confirmed values |
|-----|---------|------------------|
| `field_claim_level` | Readout interpretation boundary | `proxy_readout` |
| `field_solver_status` | Solver implementation/status category | `linear_solver`; `experimental_pde_solver` |
| `physical_amplitude_calibrated` | Amplitude status | `false` |

`field_claim_level` describes how the output may be interpreted (`proxy_readout`
on every current path). `field_solver_status` describes the implementation
category of the solver that produced it — the laminar proxy projection is a
`linear_solver` operator, while the experimental 1D Poisson solve is an
`experimental_pde_solver`. These are independent dimensions: a solver with an
experimental numerical status still produces proxy-readout outputs, and its
amplitude status remains `physical_amplitude_calibrated = False`.

## Solver contract table

All four public solver entry points, with their verified contracts:

| Entry point | Input convention | Output surface | `field_claim_level` | `field_solver_status` | `physical_amplitude_calibrated` | Implementation note |
|---|---|---|---|---|---|---|
| `project_laminar_sources` | `sources` shape `[T, N]`; `positions` shape `[N, 3]` (third coordinate = relative laminar depth in `[0, 1]`); `n_contacts`, `width`, `mode`, `dtype` | `FieldOutput` (`source_proxy`, `phi_e_proxy`, `csd_proxy`, `lfp_proxy`, `kernel`, `contact_depths`, `diagnostics`), all proxy arrays `[T, n_contacts]` | `proxy_readout` | `linear_solver` | `False` | Gaussian-leadfield laminar projection; `mode="density_preserving"` (default) or `"row_normalize"` |
| `project_sources_to_laminar_field` | same as `project_laminar_sources` | `FieldOutput` (same convention) | `proxy_readout` | `linear_solver` | `False` | Thin wrapper that delegates to `project_laminar_sources` |
| `experimental_poisson_1d` | `sources` shape `[N]`; `conductivity` scalar or per-face array `(N-1,)`; `dx`; `boundary`; `gauge`; `precision` (`"float32"` default, `"float64"` requires x64 enabled) | `(phi, residual, manifest)`, arrays shape `[N]` | `proxy_readout` | `experimental_pde_solver` | `False` | Dense 1D Poisson solve, mean-zero-Neumann boundaries / mean-zero gauge; float32 convergence ceiling ~N 150-200 |
| `experimental_poisson_1d_from_neuron_table` | `neuron_table` (each row needs a numeric `"z"`); `sources` shape `[N]`; `conductivity`; `n_bins`; `z_min`/`z_max` | `(phi, residual, manifest)`, arrays shape `[n_bins]` | `proxy_readout` | `experimental_pde_solver` | `False` | Bins per-neuron source values into depth bins (summed), then delegates to `experimental_poisson_1d`; adds `bin_edges` and `neurons_per_bin` to the manifest |

Wrapper language is exact: `project_sources_to_laminar_field` delegates to
`project_laminar_sources`, and `experimental_poisson_1d_from_neuron_table`
bins neuron-table source values then delegates to `experimental_poisson_1d`.
Neither wrapper redefines the metadata contract.

## Metadata surfaces

The two return surfaces are intentionally different; there is no unified
output object and no new public API is proposed.

- **Proxy projection** (`project_laminar_sources` /
  `project_sources_to_laminar_field`) returns a `FieldOutput` and carries the
  metadata in `FieldOutput.diagnostics` (including the `field_admissibility`
  sub-dict).
- **Poisson functions** (`experimental_poisson_1d` /
  `experimental_poisson_1d_from_neuron_table`) return `(phi, residual,
  manifest)` where `manifest` carries the metadata. The neuron-table wrapper
  copies the base manifest and adds `bin_edges` / `neurons_per_bin`.

## Use guidance

- Use `project_laminar_sources` (or its alias `project_sources_to_laminar_field`)
  for the standard laminar proxy projection workflow — the default evidence
  path for field readouts.
- Use `experimental_poisson_1d` only for the explicit one-dimensional,
  mean-zero-Neumann / mean-zero-gauge experimental PDE workflow, and keep
  `n_bins` below roughly 150 for a reliably converged float32 solve.
- Use `experimental_poisson_1d_from_neuron_table` when the source values
  start on a `Model.neuron_table()` (per-neuron depths and sources).
- `field_solver_status` distinguishes implementation type; it does not change
  the proxy-readout truth boundary. Every path on this page remains
  `field_claim_level = "proxy_readout"` with
  `physical_amplitude_calibrated = False`.

## Tests

- `tests/test_phaseE_field_schema.py` — per-solver trio assertions
  (`field_claim_level`, `field_solver_status`,
  `physical_amplitude_calibrated = False`) plus finiteness, on each solver's
  actual output surface.
- `tests/test_field_admissibility_v020.py` — conductivity-tensor validation,
  field-array finiteness, `build_field_admissibility_report`, manifest
  integration.
- `tests/test_field_proxy_admissibility_v024.py` — proxy kernel
  normalization, conservation status, boundary/gauge declared-metadata-only,
  JSON-safe diagnostics.
