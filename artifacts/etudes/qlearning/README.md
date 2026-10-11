# Q-learning on a jaxfne E/I network (lane QL)

5-state chain (start 0; right from 3 pays 1, ends episode; gamma 0.9).
State code: 40 E + 10 I network, 200 ms drive to one of five 8-cell E
groups; `phi(s)` = E spike counts / max, cached per (seed, s).
Agent: linear `Q(s,a) = w_a . phi(s)`, epsilon-greedy (0.1), alpha 0.1.
Same loop with one-hot features is the tabular baseline.

Seeds 0-4 fix the network realization (per-seed input gain) and the
exploration stream. Analytic `Q*`: `Q*(s,right) = 0.9**(3-s)`.

Tests (`test_qlearn_jaxfne.py`, under 3 min):

1. Tabular Q within 1e-2 of `Q*` (2000 episodes, seed 0).
2. jaxfne features learn all-right in 0..3 for seeds 0-4 (<=1000
   episodes); per-seed episodes-to-policy and max `|Q-Q*|` printed.
3. Flat control (state-mean features) takes one action everywhere.
4. Seeds 0/1 give different `phi`, each peaking in its driven group.
5. `alpha = 0` leaves `w` at zero after 50 episodes.

Run: `pytest -q -p no:cacheprovider artifacts/etudes/qlearning/`.
