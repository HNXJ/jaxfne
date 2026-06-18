# Objectives API

Objectives define numerical targets for `Model.tune`.

## Group firing-rate targets

```python
objectives = jtfne.rate_targets(
    groups={"first_half": range(24), "second_half": range(24, 48)},
    targets_hz={"first_half": 5.0, "second_half": 10.0},
)
```

## AGSDR optimizer spec

```python
optimizer = jtfne.agsdr(
    parameters={"drive_scale_a": (0.35, 2.25), "drive_scale_b": (0.35, 2.25)},
    generations=8,
    population_size=6,
    seed=42,
)
```

## Tune

```python
result = model.tune(objectives=objectives, optimizer=optimizer)
print(result.best_score)
print(result.best_parameters)
print(result.summary)
```

## Single objective

```python
objective = jtfne.objective(name="rate", metric="spike_rate_hz", target=10.0)
result = model.tune(objective=objective, parameter="drive_gain", bounds=(0.5, 2.0))
```

## Rate + synchrony targets

`rate_synchrony_targets` builds an `Objective` with a population-rate term and a
synchrony (kappa) term. All four arguments are defaulted to the canonical
balanced operating point, so `jtfne.rate_synchrony_targets()` is the standard
starting objective for laminar tuning.

| Parameter | Default | Meaning |
|---|---|---|
| `target_rate_hz` | `10.0` | Target population firing rate (Hz). |
| `target_kappa_synchrony` | `0.0` | Target synchrony. `0.0` = asynchronous-irregular; required for an **unbiased spectrolaminar readout** (a global rhythm masks laminar band structure). |
| `rate_weight` | `1.0` | Weight of the rate term. |
| `synchrony_weight` | `0.25` | Weight of the synchrony term. |

**Returns:** an `Objective` usable with `Model.evaluate` / `Model.tune`.

```python
obj = jtfne.rate_synchrony_targets()                  # 10 Hz, kappa 0 (canonical)
obj = jtfne.rate_synchrony_targets(target_rate_hz=5.0, synchrony_weight=0.5)
result = model.tune(obj, optimizer="AGSDR", steps=50)
```

## Result object

`TuneResult` exposes:

- `best_score`
- `best_parameters`
- `history`
- `summary`
- `model`
- `to_dict()`
