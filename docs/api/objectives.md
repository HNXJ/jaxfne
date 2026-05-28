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

## Result object

`TuneResult` exposes:

- `best_score`
- `best_parameters`
- `history`
- `summary`
- `model`
- `to_dict()`
