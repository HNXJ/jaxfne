# Colab Smoke Test — jaxfne v0.1.1

**Purpose:** Validate `pip install jaxfne==0.1.1` works in a fresh Colab runtime and the canonical spectrolaminar proxy workflow executes reproducibly without local checkout.

**Workflow type:** Deterministic laminar proxy simulation with receipt-based computational audit. Results are JSON-safe and reproducible across runtimes. CPU execution is sufficient; GPU is optional.

---

## Cell 1 — Install from TestPyPI (pre-release gate)

Run this cell before the real PyPI release is published:

```python
%pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ jaxfne==0.1.1
```

## Cell 2 — Install from real PyPI (after release)

Run this cell after `pip install jaxfne==0.1.1` is live on real PyPI:

```python
%pip install jaxfne==0.1.1
```

---

## Cell 3 — Spectrolaminar proxy workflow execution

Copy-paste this entire cell into Colab after installing:

```python
import json
import jaxfne as jtfne

print("jaxfne", jtfne.__version__)

cfg = (
    jtfne.configuration()
    .network(
        name="V1_proxy",
        kind="cortical_column",
        n=20,
        layers=["L2/3", "L4", "L5", "L6"],
        cell_types={"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03},
    )
    .emitter(family="izhikevich", preset="cortical_eig")
    .field(
        domain="laminar_column",
        conductivity="proxy",
        boundary="mean_zero_neumann",
        gauge="mean_zero",
    )
    .probe(
        name="laminar_probe",
        modes=["spikes", "V_m", "CSD", "LFP"],
        n_contacts=8,
    )
)

model = jtfne.construct(cfg)
signals = model.simulate(jtfne.simulation(duration_ms=20.0, dt_ms=0.1, seed=0))

receipt = model.run_receipt(signals)
readouts = model.compute_readout(
    signals,
    [
        jtfne.readout_spec("rate", "spike_rate_hz"),
        jtfne.readout_spec("csd", "csd_abs_mean"),
    ],
)
manifest = model.manifest(signals, readouts)

json.dumps(manifest, allow_nan=False)

assert manifest["truth_mode"] == "truth_safe_unverified"
assert manifest["claim_level"] == "computational_scaffold"
assert manifest["field_solver_status"] == "laminar_proxy_no_pde"
assert manifest["physical_amplitude_claim_allowed"] is False

print("receipt_id:", receipt.receipt_id)
print("readout_status:", [r.status for r in readouts])
print("truth_mode:", manifest["truth_mode"])
print("field_solver_status:", manifest["field_solver_status"])
print("OK")
```

---

## Computational Model Scope

**What this smoke test demonstrates:**
- **Deterministic spectrolaminar proxy execution:** Izhikevich emitters parameterized by native cell-type presets, laminar geometry metadata, and deterministic seeding.
- **Receipt-based computational audit:** Every run produces a deterministic receipt with configuration hash, seed, and truth gates for reproducibility verification.
- **Laminar field proxy readouts:** CSD and LFP are computed from layered source contributions (proxy architecture) rather than solving the full resistive PDE. Metadata explicitly documents this scope boundary.
- **JSON-safe serialization:** All signals, receipts, manifests, and readouts serialize without NaN/Inf, supporting robust data transfer and archival.

**Metadata status fields:**
- `truth_mode = truth_safe_unverified` — computational scaffold, no empirical validation
- `field_solver_status = laminar_proxy_no_pde` — proxy readouts, not full PDE solution
- `physical_amplitude_claim_allowed = false` — no physical unit claims without external validation

**Runtime notes:**
- CPU execution is assumed and sufficient; GPU is optional.
- No distributed parallelization required for this demo.
