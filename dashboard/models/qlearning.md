## Question

Can a linear Q-learner read state from the spike counts of a jaxfne E/I network and learn the
optimal policy of a small task, measured against the analytic optimum $Q^*$? This page is a
demonstration: the learning rule is standard Q-learning (Watkins and Dayan 1992), and no paper
figure is the target.

## Formulation

Linear value function on network features, with the one-step update:

$$Q(s, a) = w_a \cdot \phi(s), \qquad w_a \leftarrow w_a + \alpha\left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]\phi(s)$$

$\epsilon$-greedy with $\epsilon$ = 0.1, $\alpha$ = 0.1, $\gamma$ = 0.9. $\phi(s)$ is the E spike-count
vector for state $s$, scaled to unit L2 norm, so $\alpha\lVert\phi\rVert^2 = \alpha$; the
max-normalized features gave $\alpha\lVert\phi\rVert^2$ of 2.07–2.12, above the stability limit 2.

Tasks (`jaxfne.tasks`):
- `chain5`: 5 states, start 0, right from 3 pays 1; $Q^*(s, \text{right}) = 0.9^{3-s}$.
- `corridor7`: states 0–6, terminals 0 and 6 pay 1 on entry, start uniform on 1–5; optimal
  action left in 1–2, right in 4–5, either in 3.

## Implementation

`artifacts/etudes/papers/qlearning/qlearn_jaxfne.py`, through the jaxfne library: a 40 E + 10 I
network on the edge-list backend, one `connections()` rule adding E→E edges (p 0.5, weight 1.0),
`connectivity(edge_seed=seed)`. State $s$ drives two neighbouring 8-cell E groups at amplitude
10 for 200 ms. Controls: tabular (one-hot features) and flat (one constant vector).

## Results against the analytic optimum

`chain5`, 1000 episodes: the greedy policy is correct from episode 1 in seeds 0–4, max
$|Q - Q^*|$ 0.017–0.046. `chain5` cannot discriminate: always-right is optimal in every state.

`corridor7`, 1000 episodes, seeds 0–4:

| Features | Seeds reaching and holding the policy | Episodes to policy | Max $\lvert Q - Q^*\rvert$, states 1–5 |
|---|---|---|---|
| network | 5 of 5 | 222–486 | 0.082–0.113 |
| tabular | 4 of 5 | 2–23 | 0.229–0.395 |
| flat | 0 of 5 | never | 0.215–0.364 |

The network code learns later than tabular, with about a third of its Q error.

## Deviations and limitations

- The flat control never learns `corridor7`, so the task needs state information; why the
  network code has lower Q error is untested (no ablation).
- Tabular seed 3 keeps a wrong action in state 2: $Q(2, \text{left})$ was visited 19 times in
  1000 episodes and reached 0.722 against $Q^* = 0.9$. This is data-limited under the fixed
  $\alpha$, $\epsilon$ and episode count.
- On the default dense backend the `connections()` rule is ignored (spike counts identical
  across seeds), so the étude runs on the edge-list backend.

## Reproduce

```
pytest -q -p no:cacheprovider artifacts/etudes/papers/qlearning/
```

Run from the repository root. Seeds 0–4 fix the network realization and the simulation noise;
`pytest -s` prints every row.
