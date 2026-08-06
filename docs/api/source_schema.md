# Source Schema API

The source of a simulation is the raw proxy-current trace that each emitter kernel returns
as its third value. This page documents the shape, metadata surfaces, and truth status of
that trace across every emitter path. It is a contract description, not a class reference:
there is no `Source` or `SourceTensor` object in jaxfne.

## Scope

- The source is an array, not an object. Every emitter kernel returns
  `(spikes, ...intermediate_state..., source)` where `source` is the third return value.
- No `Source` class, `SourceTensor` class, per-kernel metadata field, or compatibility
  adapter exists or is planned. Downstream code consumes the plain array plus the metadata
  surfaces described below.
- The term "source schema" refers to the stable contract of (a) the trace array convention
  and (b) the metadata keys that describe it — both documented on this page.

## Array convention

- Shape `(T, N)`: `T` is the number of simulated time samples (the run dimension of the
  simulation), `N` is the number of neurons.
- Dtype `float32`; values are finite in short deterministic runs.
- Values are **Relative** native-unit proxy quantities: internal current-like values in the
  same unit frame as the model's other state (see "State Variables" on the
  [Emitters API](emitters.md) page), with no conversion to a physical unit.
- Do not infer the schema from the shape alone: `(T, N)` also describes intermediate state
  arrays. The metadata surfaces below are the authoritative descriptors; tests assert on
  metadata, never on shape.

## Metadata surfaces

Two surfaces exist; they are deliberately different and are not interchangeable.

1. **`signals.metadata["source_bookkeeping"]`** (standard paths). Assembled per run in
   `jaxfne/_model_simulate.py` (lines 561-571) from `_SOURCE_PROXY_METADATA`
   (`jaxfne/_model.py:221`). Present on the dense, edge-list, homeostasis-izhikevich, and
   HDP-izhikevich paths.
2. **Top-level metadata + params** (homeostatic E/I path). The homeostatic E/I emitter does
   not populate `source_bookkeeping`; instead it exposes
   `signals.metadata["source_calibration_status"]` at top level, and
   `HomeostaticEIParams.source_calibration_status` (`jaxfne/emitters_homeostatic_ei.py:337`)
   carries the default. This is an intentional design difference of that path.

Both surfaces carry the same truth gate: `physical_amplitude_calibrated` is `False` on
every path, and the status strings are the `uncalibrated_*_native_current` values listed
in the table below.

## Emitter-path table

| Path | Kernel (jaxfne/emitters.py) | Source trace | Calibration status | Metadata surface | `physical_amplitude_calibrated` |
|------|------------------------------|--------------|--------------------|------------------|--------------------------------|
| Dense | `simulate_eig_izhikevich` (line 389) | third return, `(T, N)` | `uncalibrated_izhikevich_native_current` | `source_bookkeeping` | `False` |
| Edge-list | `simulate_edge_recurrent_izhikevich` (line 561) | third return, `(T, N)` | `uncalibrated_izhikevich_native_current` | `source_bookkeeping` | `False` |
| Homeostasis-izhikevich | `simulate_edge_recurrent_izhikevich_homeostatic` (line 677) | third return, `(T, N)` | `uncalibrated_izhikevich_native_current` | `source_bookkeeping` | `False` |
| HDP-izhikevich | `simulate_edge_recurrent_izhikevich_hdp` (line 1040) | third return, `(T, N)` | `uncalibrated_izhikevich_native_current` | `source_bookkeeping` | `False` |
| Homeostatic E/I | `simulate_homeostatic_ei` (`emitters_homeostatic_ei.py`, line 462) | third return, `(T, N)` | `uncalibrated_homeostatic_ei_native_current` | top-level `source_calibration_status` + params | `False` |

All five paths return the trace in the same `(T, N)` convention; only the metadata surface
differs, and only on the homeostatic E/I path.

## Truth boundary

- Every value on this page is **Relative**: native-unit proxy quantities inside a
  computational scaffold. There is no physical amplitude calibration on any path, and
  `physical_amplitude_calibrated` is `False` by default and everywhere.
- The proxy characterization is the default evidence path for this repo: simulated,
  scaffold-level output. Readouts built from the source trace inherit the same status.
- Language in jaxfne's public docs describes these values as Relative, proxy, and
  scaffold-level; the metadata keys above are the machine-readable form of the same claim.

## Testing

- `tests/test_phaseD_source_schema.py` covers all five emitter paths: array convention,
  exact status strings, presence/absence of `source_bookkeeping` per path, and the truth
  gate (`physical_amplitude_calibrated` never `True`).
- `tests/test_source_bookkeeping_v020.py` covers the bookkeeping metadata contract for the
  standard paths.
- Both suites assert on metadata surfaces, never on trace shape alone.
