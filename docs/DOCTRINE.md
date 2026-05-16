# jaxfne doctrine

## Identity

`jaxfne` means **JAX Field Neural Equations**.

It implements a JAX-native engine for TFNE-style source-to-field neurophysiology.

## Core chain

```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer
```

## Claims

Allowed early claim:

```text
jaxfne provides a JAX-native computational scaffold for declaring emitters, source projections, fields, probes, objectives, and manifests.
```

Forbidden early claim:

```text
jaxfne validates a biological mechanism or physical CSD/LFP amplitude without calibration.
```

## Reduced emitter rule

Izhikevich native current is not amperes unless calibrated.

Default label:

```text
source_calibration_status = uncalibrated_izhikevich_native_current
```

## Source bookkeeping

Use one source mode per run. Do not double-count synaptic current.

```text
q = chi * I_m_tot + q_ext
```

or

```text
q = q_cap_ion + q_syn + q_ext
```

but never both simultaneously.
