# CAMINOS mean-field verdicts (PLoS CB 2026, Figs 1D and 6)

Code: `caminos_meanfield.py` (tests `test_caminos_meanfield.py`, 22 pass). Reproduce with
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

Oracle: the authors' notebook `E_PV_SOM_Meanfield.ipynb`, Zenodo 10.5281/zenodo.20391405 (CC-BY 4.0, zip md5
5717b20c10902a629bb7da999b4b4d53, local copy `F:\warehouse\caminos\zenodo_20391405`, read 2026-10-31). It was
read and transcribed, not executed.

Finding: Table 3 prints its quadratic columns as mu2, sigma2, tau2, mu_sigma, mu_tau, sigma_tau. The code
applies the same ten values in the order mu2, mu_sigma, mu_tau, sigma2, sigma_tau, tau2. The first port
followed the printed headers and gave 12.50 Hz with SOM 29.21 Hz above PV 7.28 Hz. The port now follows the
code's order.

| Model, mean over 2-4 s of a 4 s run | E | PV | SOM | E peak | PV leads SOM |
|---|---|---|---|---|---|
| Port, code term order | 1.56 Hz | 8.78 Hz | 7.98 Hz | 33.50 Hz | 7.76 ms |
| Port, Table 3 header order (before) | 4.08 Hz | 7.28 Hz | 29.21 Hz | 12.50 Hz | |
| Transcription of the authors' code | 1.25 Hz | 9.17 Hz | 6.73 Hz | 35.00 Hz | |

Fixed point (Eq. 20), code order: E 1.387, PV 8.872, SOM 6.363 Hz.
Verdict: pass (frequency 4% low; PV > SOM > E; delay 7.8 ms against about 7). `test_fig1d_*` holds it and
fails on the header order.

The code confirms three readings the text leaves open:

- The adaptation term divides a (mu_V - E_L) by tau_w (Eq. 16), where Eqs. 1 and 20 print it without.
- Table 4 is in volts.
- E's D mu_V0 is 0.05.

One remaining difference: the code weights inhibitory inputs in T_v by (Q_I U_I)^2 against the printed
(U_I tau_I)^2. The port keeps the printed form; the code's form gives 35.00 Hz in the transcription above.

Observed properties of the fitted polynomial (pinned by tests): the PV transfer rises with nu_PV at low rates
(72.73 to 74.15 Hz as nu_PV goes 0.5 to 2 Hz at nu_E = nu_SOM = 0.5 Hz). With weak inhibition the E transfer
rises with nu_PV (39.71 to 45.43 Hz over the same range); near the operating point it falls.
