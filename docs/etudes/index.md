# Études

An **étude** is a frozen scientific proposition demonstrated with the jaxfne
grammar: a declared plant, protocol, controller or adaptation class, metrics,
and committed artifacts. Études answer *what the grammar can demonstrate*;
[tutorials](../tutorials/index.md) teach *how to use the grammar*.

## Available études

| Étude | Question | Document |
|-------|----------|----------|
| Homeostatic balance learning | Can a spiking network learn to balance an unstable object while a homeostatic loop holds its rates? | [Homeostatic balance learning](hdp_balance.md) |
| Predictive Coding Light | Does learned inhibition remove predictable spikes and keep stimulus information, standalone and as a registered HDP rule? | [Predictive Coding Light](pcl.md) |

Études from 0.4.x are in the [Archive](../archive/index.md).

Committed bundles live under `artifacts/etudes/<name>/` with `metrics.json` and
`manifest.json` provenance. Field and spectral readouts in études are **proxy**
quantities unless an explicit calibration receipt is supplied.
