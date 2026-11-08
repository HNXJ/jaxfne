# B1 amendment 1, erratum 1 (2026-11-08, before any scored run)

Corrects `PREREG_B1_0510_amendment1.md` (pinned; left as written) in two places.

1. Signer. Its status line reads "signed (Hamm, …)". Hamm did not sign the file himself: the jaxfne keeper agent
   signed it on Hamm's standing go, given in the session chat 2026-11-07 and quoted in that line.
2. "The two arms stay matched" holds for the model class only: the jaxfne harness (`b1/b1_harness.py:296-318`,
   `simulate_hh` without edges or morphology) and `b1/b1_jaxley.py:199-206` (N independent `jx.Cell`) both build N
   independent single-compartment classic HH cells. The drives differ: the harness uses I_ext 10 uA/cm2 plus an OU
   with sigma `HH_OU_SIGMA_CLIP` (0.05) and tau 20 ms; `b1_jaxley.py:110-119` uses an OU with mu 10, sigma 5, theta 1.
   `b1_jaxley.py:170-176` already requires one convention. The drive is unified before any scored M3 run.
