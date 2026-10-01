# Augment API — controlled model augmentation

`jaxfne.augment` applies typed, independently selectable transforms to one
`NeuronalTensor` **after** JDNA `develop` and **before** `construct`:

```text
base spec -> augmentation spec -> JDNA completion -> NeuronalTensor
    -> augment -> NeuronalTensor -> construct -> Model -> simulate
```

Transforms never mutate a `Model`, never add a simulation path, and never
extend the TFNE grammar. The input tensor is never touched: the result is a
deep copy with only the targeted values rewritten.

Import:

```python
import jaxfne as jtfne

jtfne.augment               # (tensor, spec) -> (new_tensor, record)
jtfne.clone_tensor          # exact clone: equal structure, zero changes
jtfne.AugmentationSpec      # typed spec: at most one record per axis + K_V
jtfne.AugmentationRecord    # what augment realized (order, changes, digests)
jtfne.ProvenanceEntry       # one changed value (address, axis, before/after)
jtfne.ScaleN                # N axis  (cardinality)
jtfne.GeometryTransform     # G axis  (poses + ranges; PoseEdit, RangeEdit)
jtfne.ThetaC                # Theta_C axis (delay, probability)
jtfne.ThetaX                # Theta_X axis (conductance, time constant)
jtfne.W0                    # W_0 axis (initial gain)
jtfne.H0                    # H_0 axis  (initial hidden state)
```

`jtfne.augment` is the function, not the module: after `import
jaxfne.augment` (with or without `as m`), the attribute `jaxfne.augment` is
the function, and so is a `monkeypatch` string target `"jaxfne.augment.X"`. Import names with `from jaxfne.augment import ...`
or use the root names above.

## Canonical order

Transforms apply in the fixed order `N -> G -> Theta_C -> Theta_X -> W_0 ->
H_0`, whatever order the caller lists them in. A transform that changes no
value is a no-op: it contributes no `realized_order` entry and no changes.

## Axes

| Axis | Record | Fields rewritten | Operation | Refusals |
|------|--------|------------------|-----------|----------|
| `N` | `ScaleN(factor)` | every `Layer.n_neurons` | multiplicative (`n * factor`, exact integer) | a factor mapping any layer to a non-integer or to zero; an allocation that zeroes a declared cell type |
| `G` | `GeometryTransform(pose_edits, range_edits)` | `Area.pose.translation` / `rotation_deg`; `Layer.geometry` `x/y/z_range` | pose: scale, then additive rotation/translation; ranges: absolute replacement | unknown area or layer; a range outside `[0, 1]` or with `lo > hi`; two edits to the same area or layer in one record |
| `Theta_C` | `ThetaC(delay_factor, delay_jitter, probability_factor, probability_jitter, targets)` | `delay_ms` (every connection kind), `probability` (`AreaConnection` only) | multiplicative factor with per-field uniform draw in `[1 - jitter, 1 + jitter]` | a probability draw pushed above 1 (never clipped); a stored probability outside `(0, 1]` or a rescaled one `<= 0`; a stored delay that is negative or not finite; a rescaled delay that is not finite and `>= 0` |
| `Theta_X` | `ThetaX(g_factor, g_jitter, tau_factor, tau_jitter, targets)` | `static.g_mech[conn.mechanism]`, `static.dT_ms` | multiplicative factor with per-field uniform draw in `[1 - jitter, 1 + jitter]` | scaling `g_mech` for a mechanism the connection does not declare; a stored `g_mech` or `dT_ms` that is negative or not finite; a rescaled value that is not positive-finite |
| `W_0` | `W0(factor, jitter, targets)` | `plastic.w_mech` per connection | multiplicative factor with per-connection uniform draw in `[1 - jitter, 1 + jitter]` | a stored gain that is negative or not finite; a rescaled gain that is not positive-finite |
| `H_0` | `H0(offset, jitter, targets)` | `plastic.H` per connection | additive shift with per-connection uniform draw in `[-jitter, +jitter]` | a stored or rescaled `H` that is not finite |

`targets` selects connection addresses (`"area/inter/i"`,
`"area_connection/i"`); `()` means all connections. Deterministic factors
are finite numbers `> 0` (an `H0` offset is finite, any sign); relative
jitters satisfy `0 <= jitter < 1`; the `H0` jitter is absolute (`>= 0`, no
upper bound).

## K_V — the augmentation seed

`K_V` is the augmentation PRNG seed, carried by `AugmentationSpec(k_v=...)`.
A stochastic transform without its own explicit `K_V` is refused. Each axis
draws from its own independent stream (seeded from `K_V` mixed with the
axis index), so two axes under one `K_V` draw uncorrelated streams.
`K_V` is independent of the JDNA development seed `K_D` and the simulation
seed `K_S`.

## Provenance

`augment` returns `(new_tensor, record)` where `record` is an
`AugmentationRecord(realized_order, changes, spec_digest, base_digest,
notes)`:

- `changes` — one `ProvenanceEntry(address, axis, before, after, origin)`
  per rewritten value, in sampling order.
- `origin` — `"augmented"` for a deterministic rewrite,
  `"augment-sampled"` for a value that drew under `K_V`.
- `spec_digest` — stable digest of the spec (canonical axis order).
- `base_digest` — identity digest of the input tensor.
- A no-op `ScaleN` writes no scaling note; scaling keeps the existing
  `w/sqrt(N)` edge-weight rule.

`clone_tensor` is `augment` with an empty spec: structurally equal,
distinct object, zero changed values.

## Limits

- A probability pushed above 1 is refused, never clipped, so a stochastic
  spec can fail for some `K_V`.
- `construct` quantises `delay_ms` to `round(delay_ms / dt_ms)` steps, so a
  small delay change may leave the model unchanged, and a positive delay
  rounding to 0 steps is refused by `construct`, not here.
- `H_0` reaches the model as the target-group mean of `H`
  (`model.params["hdp_initial_H"]`) and is inert unless HDP is on.
- `Theta_X` scales only `g_mech[conn.mechanism]` — the key `construct`
  consumes — and refuses an undeclared one rather than materialising the
  default.
- `N` scaling keeps the `w/sqrt(N)` rule and refuses an allocation that
  zeroes a declared cell type.

## Example

```python
import jaxfne as jtfne

base = jtfne.make_minimal_ei_tensor(n=8)

same, clone_record = jtfne.clone_tensor(base)
assert clone_record.realized_order == ()

spec = jtfne.AugmentationSpec(
    transforms=[jtfne.ScaleN(2.0), jtfne.W0(factor=1.5, jitter=0.1)],
    k_v=7,
)
grown, record = jtfne.augment(base, spec)
assert record.realized_order == ("N", "W_0")

model = jtfne.construct(
    grown,
    jtfne.RuntimeConfiguration(seed=0, duration_ms=20.0, dt_ms=0.5),
)
signals = jtfne.simulate(model)
```

## Related pages

- [NeuronalTensor](neuronal_tensor.md) — the phenotype schema augment rewrites.
- [JDNA](jdna.md) — development (`K_D`) runs before augment (`K_V`).
- [Core](core.md) — `construct` / `simulate` consume the augmented tensor.
