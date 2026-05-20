# Frequently Asked Questions

## Installation and setup

**Q: What Python version does jaxfne require?**

A: jaxfne requires Python 3.10 or later. Most examples and tests use Python 3.11 or 3.13.

**Q: Can I run jaxfne on GPU?**

A: Yes. jaxfne uses JAX natively, which supports GPU execution. Set JAX device configuration as needed. CPU-first examples validate correctness; GPU execution is optional.

**Q: How do I install JAX with GPU support?**

A: See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html). Install `jax` with `[cuda]` or `[tpu]` extras if using accelerators.

## Basic usage

**Q: What's the simplest way to use jaxfne?**

A: See [Quickstart](quickstart.md) for a minimal example. It takes ~5 lines of code to configure, build, and simulate a 100-neuron network.

**Q: Can I use jaxfne with Jaxley models?**

A: Yes. See [Jaxley interoperability](jaxley_interop.md) for how to mount Jaxley-style outputs as source tensors.

**Q: What readout operators are available?**

A: Eight operators: SPK (spikes), Vm (voltage), source, LFP-proxy, CSD-proxy, EEG-proxy, MEG-proxy, EMM-proxy. See [Probe operators](probe_operators.md).

## Advanced workflows

**Q: How do I calibrate outputs to physical units?**

A: Default readouts are computational proxies. Calibration requires empirical data, geometry specifications, and solver validation. See [Calibration](calibration.md) for the calibration-ready design and future roadmap.

**Q: What metadata does jaxfne attach to outputs?**

A: All outputs include JSON-safe manifests with operator status, units, assumptions, and validation metadata. See [Output bundles](output_bundles.md).

**Q: Can I use custom emitter models?**

A: Yes. Implement the emitter interface and pass outputs as JAX arrays. See [Emitters](api/emitters.md) for details.

## Troubleshooting

**Q: I get a JAX device error. What's wrong?**

A: Check JAX device availability with `jax.devices()`. jaxfne is CPU-safe by default. If using GPU, verify CUDA/GPU setup via JAX documentation.

**Q: How do I reproduce results?**

A: Use explicit PRNG seeds in simulations. jaxfne uses deterministic JAX operations; same seed → same trajectory.

**Q: Where can I ask more questions?**

A: Open an issue or discussion on [GitHub](https://github.com/HNXJ/jaxfne).
