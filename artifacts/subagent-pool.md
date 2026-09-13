# Subagent Pool (permanent, compact)

| Agent | Owns | Must not own |
|-------|------|--------------|
| `math` | equations, invariants, dimensional consistency, classifiers | repository mutations |
| `biophysics` | neural mechanisms, E/I regimes, RBS/RBD/HDP interpretation | API/release decisions |
| `numerics` | JAX, PRNG, dtype, memory, numerical stability, performance | biological interpretation alone |
| `falsification` | statistics, controls, NEGATIVE/UNRESOLVED/INVALID separation | implementation repair |
| `repo` | paths, packaging, Git, docs links, root structure | scientific reinterpretation |
| `evidence` | hashes, manifests, provenance, authority packets, frozen evidence | changing evidence |
| `manuscript` | claims, references, figures, prose boundaries | changing numerical results |
| `adversary` | counterexamples, stale assumptions, hidden coupling, scope drift | primary writing |

Selection: `A_T ⊆ {M,B,N,F,R,E,P,A}` minimal useful subset, at least two orthogonal where meaningful. Orchestration: `P→{W_i}∥→{R_j}∥→G_integrate→S`. Workers ≠ reviewers normally.
