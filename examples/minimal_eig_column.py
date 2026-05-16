import jax
import jax.numpy as jnp
import jaxfne as jtfne

cfg = jtfne.configuration()
cfg = cfg.network(
    name="V1",
    kind="cortical_column",
    n=64,
    layers=["L1", "L2/3", "L4", "L5", "L6"],
    cell_types={"E": 0.80, "PV": 0.10, "SST": 0.07, "VIP": 0.03},
)
cfg = cfg.emitter(family="izhikevich", preset="cortical_eig")
cfg = cfg.field(domain="laminar_column", conductivity="proxy", boundary="declared_proxy", gauge="mean_zero")
cfg = cfg.probe(name="laminar_probe", modes=["spikes", "V_m", "source", "phi_e", "J_e", "CSD", "LFP"])

model = jtfne.construct(cfg)
sim = jtfne.simulation(duration_ms=100.0, dt_ms=0.1, plasticity=0.0, seed=0)
signals = model.simulate(sim)
readout = model.record(signals, modes=["spikes", "V_m", "CSD", "LFP"])
report = model.evaluate(signals, objective="smoke")
print(report)
print(model.manifest(signals))
