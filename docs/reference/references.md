# References

Scholarly ancestry for the JDNA (PseudoGenome → NeuronalTensor development)
model and for the developmental/generative modeling concepts it uses.
These are background references; none is a dependency of the package.

## Developmental and generative neural modeling

- Waddington, C. H. (1957). *The Strategy of the Genes*. Allen & Unwin. —
  Epigenetic-landscape metaphor for developmental trajectories under
  genotype and environment; conceptual ancestry for `Z_D` trajectories.
- Kitano, H. (1990). Designing neural networks using genetic algorithms with
  graph generation system. *Complex Systems*, 4, 461–476. — Grammar-based
  network development; early "rules-not-phenotype" encoding.
- Gruau, F. (1995). Cellular encoding applied to neurocontrol. *Proc. 6th
  International Conference on Genetic Algorithms* (ICGA), 460–468. — Cellular
  encoding: a program (genome) whose execution develops a network. (Also
  circulated as an MIT AI Lab memo; the memo number could not be verified
  against a primary record on 2026-08-18 and is therefore omitted.)
- Elman, J. L. (1993). Learning and development in neural networks: the
  importance of starting small. *Cognition*, 48(1), 71–99. — Developmental
  staging as a computational device; the developmental operator is not the
  learning rule.
- Stanley, K. O., & Miikkulainen, R. (2002). Evolving neural networks through
  augmenting topologies. *Evolutionary Computation*, 10(2), 99–127. — NEAT;
  historical contrast: JDNA does not evolve genomes in 0.4.17.
- Stanley, K. O., D'Ambrosio, D. B., & Gauci, J. (2009). A hypercube-based
  encoding for evolving large-scale neural networks. *Artificial Life*,
  15(2), 185–212. — HyperNEAT; generative encoding of connectivity.
- Zador, A. M. (2019). A critique of pure learning and what artificial neural
  networks can learn from animal brains. *Nature Communications*, 10, 3770.
  — Genomes as compressed prior structure; motivates rule-compressed
  phenotype generation.
- Palm, R. B., Najarro, E., & Risi, S. (2021). Testing the genomic bottleneck
  hypothesis in Hebbian meta-learning. *JMLR*, 148 (NeurIPS pre-registration
  workshop 2020); arXiv:2011.06811. — The genomic-bottleneck hypothesis:
  limiting the number of learned Hebbian rules regularizes generalization;
  computational analogue of genome-constrained learning (verified against
  the arXiv primary record on 2026-08-18).

## Framework-adjacent

- Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of
  membrane current. *J. Physiol.* 117, 500–544. — Biophysical baseline for
  the runtime models; JDNA development never alters membrane physics.
- Izhikevich, E. M. (2003). Simple model of spiking neurons. *IEEE Trans.
  Neural Netw.* 14(6), 1569–1572. — Emitter family used by the canonical
  pipeline; JDNA controls structure, not dynamics constants.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
  *Neural Computation*, 9(8), 1735–1780. — Used by some recurrent emitters
  in the runtime stack.

## Doctrinal semantics

- RBS/RBD/HDP semantics follow `docs/doctrine/rbs_rbd_hdp.md` and
  `docs/doctrine/tfne_containment_architecture.md`; see
  [HDP guide](../guides/hdp.md).