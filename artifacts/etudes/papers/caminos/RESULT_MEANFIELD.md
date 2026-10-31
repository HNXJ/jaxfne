# CAMINOS mean-field verdicts (PLoS CB 2026, Figs 1D and 6)

Code: `caminos_meanfield.py` (tests `test_caminos_meanfield.py`, 21 pass). Reproduce with
`python caminos_meanfield.py`. Paper text: PMC13286281, Results (Eqs. 1, 10-14) and Methods (Eqs. 19-20,
Tables 1-4). Tolerances, set after the data (no preregistration): frequency within 15%, and the sign and order of every
lead.

## Linear model (Fig 6), A = Q - I, tau = 20 ms

| Panel | Paper | Port | Verdict |
|---|---|---|---|
| 6A | about 40 Hz; PV leads SOM by 4 ms | 37.33 Hz; PV leads SOM by 4.06 ms, E by 5.80 ms | pass |
| 6B | on the red line Q_SOM->PV = 0.6 Q_SOM->E - 0.35: pure oscillation; above: damped; below: growing | the same three regimes at Q_SOM->E 2, 4, 6 (offset 0.5) | pass |
| 6C left | det(Y) = (59 Q_PV->SOM - 220)/5; damped below 220/59, a positive real eigenvalue above | damped at 3.719, positive real at 3.739 | pass |
| 6C right | damping grows with Q_SOM->SOM | Re -9.00, -17.70, -33.88, -58.29 1/s at 0.5, 1, 2, 4 | pass |

## Nonlinear AdEx mean field (Fig 1D)

The paper reports 35 Hz and a PV-SOM delay near 7 ms, with mean rates close to the network's (PV > SOM > E).
The port, Eq. 1 with the Methods transfer function and Tables 1-4 as printed, gives over 2-4 s of a 4 s run:
E 4.08, PV 7.28, SOM 29.21 Hz; E spectral peak 12.50 Hz. Fixed point (Eq. 20): E 1.694, PV 6.884, SOM 8.383 Hz.
Verdict: fail (frequency 64% low; SOM above PV).

Readings the text leaves open, each marked in the code:

- Adaptation: Eqs. 1 and 20 print a (mu_V - E_L) without 1/tau_w, which is dimensionally inconsistent; the
  port follows Eq. 16 and divides by tau_w.
- Table 4 carries no units; the port reads it in volts (P0 near -50 mV matches V_T).
- E's D mu_V0 is 0.05 against 0.01 for PV and SOM. Setting it to 0.01 gives 2.5 Hz and no SOM excess, so it
  does not explain the gap; the printed value stays.

Observed properties of the printed Table 3 polynomial (pinned by tests): the PV transfer rises with nu_PV at
low rates (72.75 to 74.99 Hz as nu_PV goes 0.5 to 2 Hz at nu_E = nu_SOM = 0.5 Hz), and the E transfer is not
monotone in nu_PV once nu_E >= 10 Hz.

Open: the authors' code (Zenodo 10.5281/zenodo.20391405) would settle whether the gap is a transcription
reading or the printed tables; not downloaded.
