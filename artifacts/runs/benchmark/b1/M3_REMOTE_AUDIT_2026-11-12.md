# B1 M3: remote-source baseline and matched-model gate (2026-11-12)

Status: **SOURCE AUDIT, NOT A SCORED BENCHMARK**. Baseline `jaxfne@dev@342279db791a005e556cb52c6894970b0ed9c940`. Local `E:/` and `F:/` progress is treated as unavailable, not recovered or verified. Do not claim an executable receipt from this document.

## Authority and question

The signed `artifacts/runs/benchmark/prereg/PREREG_B1_0510.md` and signed amendment 1 define M3 as **N independent single-compartment classic-HH cells** at N = 1, 10, 100; dt = 0.1 ms; duration = 1000 ms; seeds 1–5, one warm-up and five timed runs. A second amendment or explicit owner ruling is needed for a consequential change in model, grid, stochastic-drive estimand or acceptance threshold. The benchmark's question is throughput for *matched* models; comparing unmatched dynamics is not a valid performance ratio.

## Source-level discrepancy inventory

| Field | JaxFNE `b1_harness.py` | Jaxley `b1_jaxley.py` | Finding |
|---|---|---|---|
| HH leak reversal | `emitters_hh.py` E default EL = -54.3 mV | `HH` dict E_L = -54.378 mV; `HHChannel()` called without parameter overrides | Declared values differ; actual channel defaults must be inspected at runtime |
| Drive | constant `I_ext=10` plus OU | OU series with `mu=10, sigma=5, theta=1`, converted to nA by area | Distinct stochastic processes and scaling |
| OU time constant | `ou_tau=20` ms | `lam=exp(-theta*dt_ms)`, theta=1 (per ms as implemented), implying 1 ms decay | Different correlation times |
| OU amplitude | `HH_OU_SIGMA_CLIP=0.05` passed directly | sigma=5 in OU stationary series | Not comparable without units, SDE and discrete-time variance mapping |
| Time grid | caller-supplied `dt_ms`, `n_steps` | hardcoded `GRID_DT_MS=0.025`, `GRID_DURATION_MS=500` | Violates signed 0.1 ms/1000 ms grid |
| Randomness | JAX PRNG split | NumPy PCG64 | Exact trajectories need not match; compare specified statistical estimands |
| Initial state | `make_hh_params` and kernel defaults | `cell.init_states()` | Initial voltages/gates must be checked and explicitly matched |
| Solver | `emitters_hh.py` kernel | Jaxley `integrate` | Compare convergence under grid refinement before attributing discrepancies to models |

Source paths: `artifacts/runs/benchmark/b1/b1_harness.py` (`_run_M3`), `artifacts/runs/benchmark/b1/b1_jaxley.py` (`HH`, `OU`, `ou_current`, `_run_single`), `jaxfne/emitters_hh.py` (defaults, `ou_step`). The HH module documents that its noise channel was originally expressed in mA/cm2 and that callers using uA/cm2 scale sigma by 1000. **Do not apply that factor blindly**: first establish the intended benchmark OU stationary standard deviation, area and physical-current convention. The baseline's `HH` dictionary does not establish which values `HHChannel()` actually consumes.

## Minimal repair and acceptance packet

1. **Freeze units**: write a single benchmark-only parameter table for `C_m` (uF/cm2), `g` (mS/cm2), `E` (mV), membrane area (um2), current density (uA/cm2), OU mean/variance/correlation time, dt (ms), duration (ms), initial gates and voltage, solver. Derive density-to-nA: `I_nA = I_(uA/cm2) * area_um2 * 1e-5`. Check dimensional cancellation.
2. **Choose a common OU process before coding**. For an exact stationary-discretized OU with mean `mu`, stationary SD `s`, time constant `tau`: `x[t+1] = mu + exp(-dt/tau)*(x[t]-mu) + s*sqrt(1-exp(-2*dt/tau))*xi[t]`. For Euler-Maruyama as in `ou_step`, declare the discretization error and use an appropriate matched step. Distinguish `sigma` as stationary SD from diffusion coefficient; they are not interchangeable.
3. **Minimal red probes on baseline**: check that the Jaxley constructor actually receives all intended HH constants, both backends use signed dt/duration, and OU empirical mean/variance/lag-1 autocorrelation match the declared process within prespecified seed-dependent tolerances. These checks must fail against the current mismatched baseline and pass after a single scoped fix.
4. **Independent deterministic oracle**: disable OU; drive both single cells with the same constant current density and explicit initial conditions. Compare voltage traces at dt=0.1 and dt=0.05 ms (and smaller if needed), spike counts/timing and convergence; explain solver-dependent deviations.
5. **Stochastic oracle**: provide the same externally generated current-density series to both backends if APIs allow; otherwise compare across independent seeds with a predeclared statistical tolerance, never assert spike-time identity for distinct PRNGs. Verify per-cell independence at N=10.
6. **Only then benchmark** N=1/10/100 under the existing signed protocol, record device/dtype, actual channel parameters, seed list, compile/run separation, RSS/VRAM, recording controls, and validity; failed validity excludes speed ratios. Keep signed prereg and original pilots immutable.

## Stop and evidence

No source edit or scored run until the shared model specification is established. If the intended OU process or parameter mapping requires changing signed scientific conditions, stop that change for a dated owner-approved amendment; continue independent deterministic oracle work. A successful test requires an observed failure on the old code, an observed pass on the repair, and a non-writer rerun. Current status: **source mismatch confirmed; causal runtime impact and matched performance UNKNOWN**.
