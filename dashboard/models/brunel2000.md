## Question

Does the jaxfne LIF core, at the paper's parameters, reproduce the Fig. 8 rates, regularity
and global frequencies of the SR, SI fast, AI and SI slow states of Brunel (2000,
J Comput Neurosci 8:183-208)?

## Model summary

A sparse network of leaky integrate-and-fire neurons, 80% excitatory and 20% inhibitory,
each cell receiving a fixed number of recurrent excitatory (`CE`) and inhibitory (`CI = 0.25 CE`)
delta-current inputs with one fixed delay `D`, plus independent Poisson external drive at rate
`nuext`. The paper's analysis predicts four regimes in the inhibition/excitation balance `g`
and the drive ratio `nuext/nuthr`: synchronous regular firing when excitation dominates, and
asynchronous irregular, fast synchronous irregular (~180 Hz global with ~60 Hz irregular cells)
and slow synchronous irregular (~20 Hz global with ~5 Hz cells) states when inhibition dominates.

## Source record

`artifacts/etudes/papers/brunel2000/SOURCE_P1.md` (P1): equations, the Fig. 8 parameter set
(`J` 0.1 mV, `CE` 1000, `CI` 250, `D` 1.5 ms, 10,000 E + 2,500 I cells), the four target claims
with the paper's simulation-vs-theory numbers, and the gaps (plain-LIF check, fixed indegree,
uniform delay, unstated simulation numerics, authors' code unknown).

## Status

P1: source record only, no scored run. Next per rule B8: pilot, then signature, then fresh seeds.
