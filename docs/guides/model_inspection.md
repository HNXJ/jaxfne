# Model inspection

Two figures, before any analysis: what you built, and what it immediately does.

A simulation that runs is not a simulation you understand. Scalar summaries can be
blind to structure the trajectories plainly show — a rate averaged over a window is
identical for two runs whose spike timing differs completely. Look at the circuit and
look at the activity first.

## 1. Construct, then see the circuit

```python
import jaxfne as jtfne

model = jtfne.construct(...)  # however you build it

jaxfne.vis.network_hspice(model, path="circuit.png")
```

A block schematic: input on the left, output on the right, areas as blocks, layers as
rows, cell classes as chips, and one annotated arrow per cross-area projection carrying
its edge count, mean weight and sign. Individual neurons are never drawn — this is a
construction receipt, not a connectivity plot.

![circuit](../_static/model_inspection/case2_three_area_hspice_light.png)

Everything comes from `model.neuron_table()` and `model.params["edge_list"]`, so any
naming scheme works. Stage order is inferred from the projection graph; pass `stages=`
to assert one the topology does not show, and `title=` to annotate with your own
architecture notation.

```python
jaxfne.vis.network_hspice(
    model,
    x="sound", y="Belt.supra",
    stages=[["Cochlea"], ["A1"], ["Belt"]],
    title="x : Cochlea -> A1 -> Belt : y",
    theme="dark",
    path="circuit_dark.png",
)
```

## 2. Simulate, then see the activity

```python
signals = jaxfne.simulate(model, duration_ms=1000)
jaxfne.vis.network_raster(model, signals, dt_ms=0.1, path="raster.png")
```

A spike raster with rows ordered area → layer → cell class, over a population-rate
trace. Call it again after whatever you changed, with the same arguments, and the two
figures are directly comparable because the row ordering is computed by one function.

![raster](../_static/model_inspection/case2_three_area_raster_light.png)

```python
jaxfne.vis.network_raster(
    model, signals,
    window_ms=(0.0, 1000.0),     # default: the whole recording
    areas=["Cochlea", "A1"],     # default: every area, naturally ordered
    dt_ms=0.1,
    theme="dark",
)
```

## 3. Inspect comprehensively

```python
jaxfne.vis.build_atlas(model, signals, out_dir="atlas/")
```

For the interactive seven-panel atlas this inspects toward, see the
[Canonical Atlas Suite](atlas_suite.md).

## Themes

`theme` takes `"light"`, `"dark"`, or a `Theme` instance. It changes presentation only,
never information or semantics: both renderers return a description of what they drew,
and that description is identical under every theme.

```python
from jaxfne.vis.network_inspect import Theme, THEMES

house = Theme(**{**THEMES["light"].__dict__, "name": "house", "background": "#fff8e7"})
jaxfne.vis.network_hspice(model, theme=house)
```

## Return values

Both renderers return a dict describing the figure, suitable for a run record:

```python
{"figure": "network_hspice", "path": "circuit.png", "theme": "light",
 "stages": [["Cochlea"], ["A1"], ["Belt"]], "areas": [...],
 "n_neurons": 99, "n_edges_total": 102, "n_edges_local": 18,
 "n_edges_long_range": 84, "n_projections": 3}

{"figure": "network_raster", "path": "raster.png", "theme": "light",
 "n_units": 99, "n_groups": 27, "window_ms": [0.0, 500.0], "dt_ms": 1.0,
 "n_spikes": 1079, "thinned": False, "mean_rate_hz": 21.798}
```

`thinned` is `True` when more spikes were found than `max_points` allowed to be drawn;
`n_spikes` still reports what was found, so the figure never quietly under-reports.

## Reference

### `network_hspice(model, *, x=None, y=None, theme="light", stages=None, title=None, path=None, inhibitory_receptors=(1,), class_order=("E","PV","SST","VIP"), figsize=None, dpi=150, show_legend=True)`

### `network_raster(model, signals, *, window_ms=None, theme="light", dt_ms=1.0, areas=None, title=None, subtitle="", path=None, class_order=("E","PV","SST","VIP"), figsize=(16,10), dpi=150, max_points=400_000, show_rate=True)`

`signals` may be a jaxfne `Signals` or any `(n_time, n_neurons)` spike array.

### `describe(model, *, inhibitory_receptors=(1,), class_order=..., stages=None)`

The structure both renderers draw from: populations per area/layer/class, block-level
projections with edge count, mean weight and sign, and the inferred stage of each area.
Useful on its own when you want the numbers without a figure.
