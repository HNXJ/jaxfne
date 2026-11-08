# Lee, Pennartz and Mejias 2025: deep-read notes (O-87)

Target: Lee K, Pennartz CMA, Mejias JF (2025), Cortical networks with multiple interneuron types generate
oscillatory patterns during predictive coding, PLoS Comput Biol 21(9):e1013469, doi:10.1371/journal.pcbi.1013469
(Crossref-verified 2026-11-08; CC BY). Local copy `F:/warehouse/papers_o87/lee2025/paper.pdf`; `L<n>` is line n of
`paper_lines.txt` beside it (PyMuPDF text, page markers kept). Status tags: observed (the paper states it), inferred,
unknown (the paper does not give it). Code: github.com/Kwangjun-uva/CoCoPC (L27-28).

## Question and scope

Can a two-area laminar rate model with E, PV, SST and VIP microcircuits perform predictive coding (PC), and what
dynamics and cell-type roles follow (L46-58, L127-136). Model only, no data fit (L130-131). Signal: population
firing rates (arbitrary units); images: grayscale CIFAR-10 (L645-647), also MNIST and fashion-MNIST (L51).

## Argument map

| Step | Content | Locator |
|---|---|---|
| Premise | PE+ and PE- computed by separate circuits of the same E-PV-SST-VIP motif with different inputs | L53-55, L159-165 |
| Method | Rate model `tau dr/dt = -r + ReLU(sum W r)`, tau 20 ms for E and I; within-circuit weights fixed at 1 | L575-591, L702-705 |
| Method | Area 1: 1024 L4 E (one per pixel), 2 x 1024 L2/3 microcircuits; Area 2: 784 each of L4 E, L4 PV, L5 E; all-to-all inter-area W | L596-610 |
| Method | Motif by exhaustive search, 512 connectivity x 64 input patterns = 32,768; 173 pass; one picked at random | L620-643 |
| Method | Hebbian rule Eq. 2 on inter-area W; batch 64, 2 s per image, 0.2 s blank, 100 presentations | L663-687 |
| Result | PE falls over training; unseen images reconstructed | L221-226, Fig 2 |
| Result | Damped oscillations, about 6 Hz, stable focus; frequency depends largely on tau | L283-290, Fig 4C-D |
| Result | Oddball: deviant raises PE; earlier deviant gives a larger peak; ISI/tau < 1 strengthens the effect | L299-314, Fig 5 |
| Result | Silencing: PV off, PE rises and the ~6 Hz rhythm stops; SST off, undamped oscillation, no reconstruction; VIP off, PE silenced, no reconstruction | L344-373, Fig 6 |
| Conclusion | PC can emerge from an anatomically informed circuit with Hebbian learning (L561-572); the oscillations are proposed as temporal signatures of inference (L291-293) | L291-293, L561-572 |

## Claims table

| Claim | Evidence | Standing | Caveat |
|---|---|---|---|
| 173 of 32,768 motifs give opposite PEs for BU > TD and BU < TD and none for BU = TD | L636-640 | Provisional (model-internal, checkable without training) | the three test inputs' magnitudes are unknown |
| Oscillation about 6 Hz in Rep, PE+, PE- | L284-285 | Provisional | tau 20 ms; "approximately"; no spectral method given (unknown) |
| Frequency rises as tau falls | L288-290, Fig 4D | Provisional | values only in the figure |
| Final reconstruction MSE / SSIM: control 0.00538 / 0.923; PV off 0.0132 / 0.847; SST off 0.0711 / 0.521; VIP off 0.223 / 0.311 | L436-438 (Fig 6 legend) | Provisional | n images, seeds and SSIM window not stated (unknown) |
| Deviant peak larger when earlier in the sequence | L313-314, Fig 5C | Provisional | values only in the figure |
| PV silencing changes accuracy little; the authors attribute this to PE+ and PE- effects that tend to cancel | L358-361 | Provisional (author interpretation) | |

## Quantities

| Quantity | Value | Unit | Condition | Locator |
|---|---|---|---|---|
| tau (E and I) | 20 | ms | all neurons | L589-590 |
| Area 1 L4 E / L2/3 microcircuits | 1024 / 2 x 1024 | count | 32 x 32 px inferred from 1024 | L598-600 |
| Area 2 per type | 784 | count | L4 E, L4 PV, L5 E | L608 |
| Training set | 2560 images, 10 classes, 256 per class | count | stated as 5.1% of the total images in the dataset | L645-646 |
| Training fraction in Fig 2 legend | 4.3% | % | stated as of the total training images; differs from 5.1% (L646) | L236 |
| Presentation / blank | 2 / 0.2 | s | batch 64; 100 presentations | L667-671 |
| Oscillation frequency | about 6 | Hz | default tau | L284-285 |
| Propofol simulation | L5 to L2/3 weights -30% | % | S4 Fig | L776-779 |
| Learning rate gamma_w, integration step, initial W | unknown | | not in the main text (S1 Table may hold W) | L753 |

## Disputes and inconsistencies

- Training fraction 4.3% (L236) against 5.1% (L646): 2560/60,000 = 4.27% (CIFAR-10 train plus test) and
  2560/50,000 = 5.12% (train only) (inferred). L236 labels its 4.3% "of total training images" and L646 labels its
  5.1% "of the total images in the dataset": each value fits the other line's label, so the labels look swapped;
  the count 2560 is unaffected.
- MMN: the authors argue PC over stimulus-specific adaptation, citing [81], [32], [82] (L490-511); the model has no
  adaptation mechanism, so it cannot arbitrate (inferred).
- Oscillation band: the Introduction and Discussion allow alpha to gamma by hyper-parameters (L141-142, L471-473);
  the reported run is about 6 Hz (theta/alpha boundary).
- Library: Mejias 2016 (`etudes/papers/mejias2016/`) shares an author; Lee 2025 cites Wilson and Cowan 1972 [87] for
  its rate model (L575, L1073) with a ReLU transfer and unit within-circuit weights (L591, L704).

## Open questions

- Is the ~6 Hz rhythm a property of the trained W or of the motif alone? Settle: simulate the motif with random W of
  matched spectral radius against the trained W.
- Do the silencing classes (Fig 6) hold for the other 172 motifs (L641-643 asserts "qualitatively similar")? Settle:
  repeat Fig 6 on a sample of motifs.

## Candidate target figure

**Fig 6, interneuron silencing.** Estimand: for control and PV, SST, VIP silenced (r_i = 0, L316-321, in both PE circuits,
L347, L429-430), the final reconstruction MSE and SSIM (L436-438) and the class of E dynamics in PE+ and PE-: damped
oscillation to a fixed point (control), continuous increase without the rhythm (PV), sustained oscillation (SST),
quiescence (VIP) (L344-373). Criterion proposal: the four classes reproduced, and the MSE ranking control < PV < SST <
VIP with each MSE within a factor 2 of the printed value (proposal, for the prereg).

Fit to the core: a rate network on the jaxfne rate emitter (`jaxfne/emitters_rate.py`) with a ReLU transfer
(not among `RATE_TRANSFER_FUNCTIONS` today, inferred from `rate_transfer` docstring `phi(0) = 1`; a new transfer is a
core change) and a Hebbian rule on inter-area W (an HDP realization `dW/dt = F_W(H, ...)`). Cost drivers: training
on 2560 images x 100 presentations; the trained W, gamma_w and step size are unknown in the text, so the authors'
code is the oracle for those values. Cheap stage 0: the Fig 7 search (173 of 32,768) needs no training and checks
the motif before any training run.
