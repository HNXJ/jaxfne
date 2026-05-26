# Objectives API

Objective functions for optimization workflows.

## Overview

Objectives specify targets for parameter optimization. Each objective defines:
1. A metric to minimize/maximize (e.g., `spike_rate_hz`)
2. A target value (e.g., 10.0 Hz)
3. A loss function (e.g., MSE, L1 distance)

Multiple objectives can be combined into a multi-objective optimization problem.

---

## Objective

```python
jaxfne.objective(name, metric, target, **kwargs)
```

Create a single optimization objective.

### Parameters

- `name` (str): Human-readable objective name
- `metric` (str): Target metric key (see [ReadoutSpec](core.md) for available metrics)
- `target` (float): Target value
- `weight` (float, optional): Loss weight in multi-objective problems (default: 1.0)
- `loss_type` (str, optional): Loss function (default: `"mse"`)

### Returns

`Objective` object with specification and evaluation methods

### Example: Single Objective

```python
import jaxfne as jtfne

obj_spike_rate = jtfne.objective(
    name="firing_rate",
    metric="spike_rate_hz",
    target=10.0,
    weight=1.0
)
```

### Example: Multi-Objective

```python
objectives = [
    jtfne.objective(name="firing_rate", metric="spike_rate_hz", target=10.0, weight=1.0),
    jtfne.objective(name="mean_voltage", metric="mean_V_m", target=-50.0, weight=0.5),
    jtfne.objective(name="burst_freq", metric="burst_frequency_hz", target=5.0, weight=0.8),
]
```

---

## Objective Class Attributes

### `name`
Human-readable label for the objective.

### `metric`
Target metric key, must match a valid ReadoutSpec metric.

### `target`
Target value for the metric.

### `weight`
Relative importance in multi-objective optimization. Higher weight → higher penalty for deviation.

### `loss_type`
Loss function:
- `"mse"` (default): Mean squared error
- `"mae"`: Mean absolute error (L1 distance)
- `"rmse"`: Root mean squared error

---

## Loss Computation

### Unweighted Loss

For a single objective with target value $t$ and observed value $v$:

**MSE:** $\ell = (v - t)^2$

**MAE:** $\ell = |v - t|$

**RMSE:** $\ell = \sqrt{(v - t)^2}$

### Weighted Multi-Objective Loss

For multiple objectives:

$$L = \sum_{i=1}^{n} w_i \ell_i(v_i, t_i)$$

where $w_i$ is the weight and $\ell_i$ is the loss for objective $i$.

---

## Objective Report

```python
jaxfne.ObjectiveReport
```

Container for objective evaluation results.

### Attributes

- `specs` (list[Objective]): Original objective specifications
- `values` (dict[str, float]): Observed metric values
- `targets` (dict[str, float]): Target values
- `losses` (dict[str, float]): Per-objective loss values
- `total_loss` (float): Weighted sum of losses

### Methods

#### `to_dict() -> dict`

Convert report to JSON-safe dictionary.

**Example:**
```python
report = model.evaluate_objectives(signals, objectives)
results = report.to_dict()
```

---

## Optimization Workflow

### Step 1: Define Objectives

```python
objectives = [
    jtfne.objective(name="rate", metric="spike_rate_hz", target=10.0),
    jtfne.objective(name="voltage", metric="mean_V_m", target=-55.0)
]
```

### Step 2: Create Model

```python
cfg = jtfne.Configuration()
cfg = cfg.runtime(seed=7, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
# ... configure network ...
model = jtfne.construct(cfg)
```

### Step 3: Run Simulation

```python
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=7)
```

### Step 4: Evaluate Objectives

```python
report = model.evaluate_objectives(signals, objectives)
print(f"Total loss: {report.total_loss:.4f}")
for name, loss in report.losses.items():
    print(f"  {name}: {loss:.4f}")
```

### Step 5: Optimize (Optional)

Use GSDR or AGSDR optimizers:

```python
# Define a parameter to optimize
optimizer = jtfne.gsdr(
    model=model,
    objectives=objectives,
    param_name="I_injected",  # Izhikevich input current
    param_range=(0.0, 20.0),
    n_steps=50
)

best_params = optimizer.optimize()
```

---

## Available Metrics

All ReadoutSpec metrics can be used as objective targets:

| Metric | Type | Range | Use Case |
|--------|------|-------|----------|
| `spike_rate_hz` | float | [0, ∞) | Control firing rate |
| `burst_frequency_hz` | float | [0, ∞) | Control burst pattern |
| `mean_V_m` | float | [-90, 30] | Control resting potential |
| `min_V_m` | float | [-90, 30] | Control hyperpolarization |
| `max_V_m` | float | [-90, 30] | Control depolarization |
| `mean_source` | float | (-∞, ∞) | Control source strength |
| `mean_LFP` | float | (-∞, ∞) | Control LFP amplitude |
| `mean_CSD` | float | (-∞, ∞) | Control CSD magnitude |
| `mean_EMM` | float | [0, ∞) | Control activity cost |

---

## Constraints (Future)

**Planned (v0.3.6+):** Constraint specifications for bounded optimization:

```python
constraint = jtfne.constraint(
    name="voltage_bounds",
    metric="max_V_m",
    lower_bound=-80.0,
    upper_bound=40.0
)
```

Currently constraints can be implemented via large weights on boundary objectives.

---

## Best Practices

1. **Normalize targets:** Use realistic ranges (e.g., spike_rate_hz = 5–50 Hz)
2. **Balance weights:** In multi-objective, use equal or similar weights unless prioritizing
3. **Validate convergence:** Check that losses decrease over optimization iterations
4. **Monitor individual losses:** Track each objective separately; total loss may hide poor solutions
5. **Use realistic objectives:** Combine objectives that are physically reasonable together

**Example: Balanced Multi-Objective**

```python
# Good: All metrics on similar scales
objectives = [
    jtfne.objective("rate", "spike_rate_hz", target=15.0, weight=1.0),
    jtfne.objective("voltage", "mean_V_m", target=-55.0, weight=1.0),
    jtfne.objective("activity", "mean_EMM", target=100.0, weight=1.0),
]

# Less ideal: Vastly different scales without normalization
objectives_bad = [
    jtfne.objective("rate", "spike_rate_hz", target=15.0, weight=1.0),
    jtfne.objective("lfp_power", "mean_LFP", target=1e-6, weight=1.0),  # Much smaller scale
]
```

---

## See also

- [Core API](core.md) — ReadoutSpec and ReadoutResult
- [API reference](index.md)
