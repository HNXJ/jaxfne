# PCL full-network port: P2 spec (lane L1)

Target: N'dri et al. 2025, Figs. 3–6. Reference code: Zenodo 10.5281/zenodo.16836801
(CC-BY 4.0; local copy `F:\warehouse\pcl\code`, read-only). Data and trained networks:
Zenodo 10.5281/zenodo.16813454 (`F:\warehouse\pcl\zenodo`). Paths below are relative
to `PCL_C++/libs/network/src/` unless stated. Read 2026-10-11 by Claude.

## Network (default config, `config/DefaultConfig.cpp`)

| Item | Code value | Note |
|---|---|---|
| Input | DVS events (t µs, x, y, polarity), 346×260, 1 camera | HDF5 group `events`: t, x, y, p, c |
| Simple layer | 9×9 positions × 64 depths per patch; RF 10×10×2 (polarity); overlap 3 (stride 7) | patches x {150}, y {100}: one patch |
| Complex layer | 6×6 × 32; RF 4×4×64 simple cells; overlap 3 | |
| Weight sharing | `patch`: one simple weight tensor per (patch, depth); the local-inhibition vector is shared per depth | `SpikingNetwork.cpp:188` |
| Inhibition | local (other depths at the same x, y, plastic, shared), lateral dynamic (range 4 in x and y, all depths, plastic per pair), top-down (complex to simple, plastic per pair) | `connectLayer` `:347` |
| Propagation | event-driven and depth-first: a spiking simple cell updates its weights, then sends local inhibition, then lateral inhibition (only when `m_activation` is on), then excites complex cells, which can spike and send top-down inhibition, all at the same timestamp | `addEvent` `:41`, `addNeuronEvent` `:87` |

## Neuron (`neurons/Neuron.cpp`, `SimpleNeuron.cpp`, `ComplexNeuron.cpp`)

On each input of weight w at time t:
`V ← V·exp(−Δt/τ_m)`, `A ← A·exp(−Δt/τ_SRA)`, `V ← V + w − η_RP·exp(−(t−t_s)/τ_RP) − A`,
clamp at a negative limit, spike if `V > θ`, then `V ← V_reset` and `A ← A + Δ_SRA`.
Inhibitory inputs subtract w and add the refractory and adaptation terms rather than
subtract them (`SimpleNeuron.cpp:61`). Complex cells have no A term.

| Parameter | Simple | Complex |
|---|---|---|
| θ (VTHRESH) | 30 (default config); 10 (paper Table 2) | 3 |
| V_reset | −10 | −10 |
| τ_m, τ_RP, τ_SRA (ms) | 18, 5, 100 | 50, 5, — |
| τ_LTP, τ_LTD (ms) | 7, 7 | 40, 40 |
| η_RP | 10 | 10 |
| NORM_FACTOR (excitatory L1) | 50 | 1000 |
| ETA_INH (local-inhibition L1) | 1500 | 600 |
| LATERAL / TOPDOWN norm | 6500 / 2000 | — |
| η_LTP, η_LTD, η_ILTP, η_ILTD | 0.00077, −0.00021, 0.00046, −0.00046 | 0.002, 0.002, 0.008, −0.008 |

## Plasticity (`SimpleNeuron::weightUpdate`, `:186`)

This runs at each postsynaptic spike, over the events stored since the last update.
Excitatory: `ΔLTP = (3 − w)·0.33·decay·η_LTP·exp(−(t_s − t_i)/τ_LTP)`, and when the cell has
spiked before, `ΔLTD = w·0.33·η_LTD·exp(−(t_i − t_{s−1})/τ_LTD)`. Then clip at w ≥ 0 and
L1-normalise to NORM_FACTOR. The inhibitory rules have the same form with `w_max = 50` and
factor 0.02, normalised per type (local to ETA_INH, top-down to TOPDOWN_NORM_FACTOR,
lateral to LATERAL_NORM_FACTOR).

The soft-bound constants (3, 0.33, 50, 0.02 for simple cells; 4, 0.25 for complex cells)
are hard-coded and are not in the paper's equation. The code is authoritative for
replication; each difference from the Methods equation is recorded as a deviation.

## Port design

- **A: event-exact reference.** NumPy or JAX `lax.scan` over events, the C++ order
  reproduced, on CPU. Slow, and used as the oracle on short clips.
- **B: time-binned JAX on GPU (float32, WSL2 CUDA).** All events in a bin of `dt` are
  applied together, and spikes cascade within the bin in the fixed order simple, local,
  lateral, complex, top-down. `dt` is declared in each manifest.
- **Configuration.** Each trained network's own `configs/*.json` overrides the defaults
  above. Thresholds differ between the default config and Table 2, so the files decide.

## Gates

| Gate | Test | Pass |
|---|---|---|
| L1-G0 | A reproduces Fig. 2, as the existing `pcl_fig2.py` does | weight and suppression order 1 > 2 > 3 in every network |
| L1-G1 | A on a trained network from the archive (frozen weights, inference), on 20 test clips | spike counts per cell agree with the C++ output, if available, or are recorded as the oracle |
| L1-G2 | B against A on the same clips | per-cell count correlation ≥ 0.95 and total count within 5 %, at the declared `dt` (tolerances to be confirmed at P3) |
| L1-G3 | B trained from scratch on the natural-image event set | Fig. 3 observables, preregistered at P3 |

Fig. 4 (suppression curves) and Figs. 5–6 (classification, sparseness) follow on G3's
network. Each figure's preregistration is drafted with the gate and signed in a batch.
