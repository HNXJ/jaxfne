# Paper replication programme (2026-10-10)

Human (Hamm, 2026-10-10): replicate the plan's papers with jaxfne, as far as each
allows. PCL and oscillatory dynamics come first. Large runs use WSL2 CUDA JAX
in float32; the goal is insight, and float32 is accurate enough for it. This
file sequences the work. Method, gates and the HDP off/on design stay in
`literature_reproduction_plan.md`; P1 records stay in `literature_p1_verification.md`.

## Decisions (human, 2026-10-10)

- Preregistrations (P3): Claude drafts, second-reviewer checks, Hamm signs in
  batches. Builds and pilots go ahead meanwhile; scored runs wait for a signature.
- PCL Zenodo files (CC-BY 4.0) are downloaded to `F:\warehouse\pcl\zenodo`:
  code (10.5281/zenodo.16836801), input videos and datasets with trained
  networks (10.5281/zenodo.16813454).
- Blocked methods (Tahvili et al. 2025, van Vreeswijk and Farkhooi 2025) are
  read through Hamm's browser session.
- Compute: WSL2 Ubuntu, `~/venv_jaxfne` (jax[cuda12], RTX A4000), clone
  `~/jaxfne_bench`. Every manifest records the backend and dtype. GPU and CPU
  agree to about 1e-5 relative (jchat p-jaxfne #237), so tolerances never
  assume bit equality across backends.

## Lanes

| Lane | Paper | Observables (from the primary source) | Blocker | First step |
|---|---|---|---|---|
| L1 PCL | N'dri et al., Nat Commun 16:8880 (2025) | Fig. 3: simple and complex receptive fields, orientation and spatial-frequency tuning, flat phase tuning, frequency doubling. Fig. 4: surround, orientation-tuned and cross-orientation suppression. Fig. 5: spikes removed against downstream accuracy (N-MNIST, DVS128 Gesture). Fig. 6: population sparseness | none | port the published network (Table 1: retina 346×260×2, simple 9×9×64, complex 6×6×32) to JAX on GPU |
| L2 Oscillations | Tahvili, Vinck, di Volo, Cell Reports 2025, DOI 10.1016/j.celrep.2025.116131 | stochastic gamma with drive-dependent frequency; PV phase-locking and delayed SOM firing; PV and SOM perturbations; SOM/PV density against frequency | STAR Methods unread | AdEx emitter with conductance synapses and per-connection taus (GAP rows in P1) |
| L3 Spectrolaminar | Mendoza-Halliday et al. 2024; Mackey et al. 2025; Major et al. 2025 | superficial gamma and deep alpha-beta gradient, crossing at L4, under each source's band definition | none | run the canonical and balanced columns through the declared band rules |
| L4 Calcium homeostasis | van Vreeswijk and Farkhooi, bioRxiv 2025 | balanced E-I from calcium-regulated synaptic homeostasis | equations unread | read methods; map calcium to the HDP H state |
| L5 Predictive coding (comparison) | Lee, Pennartz, Mejias, PLOS Comput Biol 2025 | PV, SST and VIP silencing; deviant enhancement; about 6 Hz rhythm | needs a rate emitter | check whether a rate emitter exists |

## PCL change of criterion

The paper defines no orientation-selectivity index. It reads tuning from
tuning curves (Fig. 3c) and suppression curves (Fig. 4c, 4d). The A1 criterion
(OSI ≥ 0.3 on bars) used in C1, C1b and K1 was ours, not the paper's, so the
A1 failures do not count against replication. L1 replaces A1 with the
paper's qualitative observables, preregistered before any scored run.

## Order

1. L1 port and L2 emitter, in separate worktrees.
2. L3, which needs no new code.
3. L4 and L5 once their sources are read.
