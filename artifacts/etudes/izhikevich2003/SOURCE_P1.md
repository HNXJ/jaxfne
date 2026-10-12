# Izhikevich 2003: P1 source record

Target: E. M. Izhikevich, "Simple model of spiking neurons", IEEE Trans Neural Netw 14(6):
1569–1572 (2003), doi:10.1109/TNN.2003.820440. Read 2026-10-12 from the author's PDF
(izhikevich.org/publications/spikes.pdf, text layer and page renders); the MATLAB program
`net.m` on the same site matches the listing printed on p. 1571. Hamm 2026-10-12: third 0.5.6
paper, on the existing Izhikevich emitter. Licence: the paper is IEEE copyright; we use its
equations and parameters, with no figure reproduced.

## Model (Eqs. 1–3)

$v' = 0.04v^2 + 5v + 140 - u + I$, $u' = a(bv - u)$; if $v \ge 30$ mV then $v \leftarrow c$,
$u \leftarrow u + d$. $v$ in mV, $t$ in ms.

## Fig. 2: single-cell firing patterns

Step of dc current $I = 10$, time resolution 0.1 ms. Parameters read from the Fig. 2 parameter
plots and the text (p. 1570–1571):

| Type | a | b | c | d | Claim in the text |
|---|---|---|---|---|---|
| RS | 0.02 | 0.2 | −65 | 8 | a few spikes at short period, then the period increases (adaptation) |
| IB | 0.02 | 0.2 | −55 | 4 | a burst, then repetitive single spikes |
| CH | 0.02 | 0.2 | −50 | 2 | bursts of closely spaced spikes; inter-burst frequency up to 40 Hz |
| FS | 0.1 | 0.2 | −65 | 2 | high-frequency periodic trains, practically without adaptation |
| LTS | 0.02 | 0.25 | −65 | 2 | high-frequency trains with noticeable adaptation |
| TC | 0.02 | 0.25 | −65 | 0.05 | tonic firing from rest near −60 mV; rebound burst after hyperpolarization near −90 mV |
| RZ | 0.1 | 0.26 | −65 | 2 | damped or sustained subthreshold oscillations; bistable rest and spiking |

The figure's step amplitudes for TC (negative step) and RZ (brief pulse) are drawn, without
values.

## Fig. 3: 1000-neuron pulse-coupled network (p. 1571)

- 800 excitatory, 200 inhibitory. Excitatory: $(a, b) = (0.02, 0.2)$,
  $(c, d) = (-65, 8) + (15, -6)r^2$; inhibitory: $(a, b) = (0.02, 0.25) + (0.08, -0.05)r$,
  $(c, d) = (-65, 2)$; $r \sim U[0, 1]$ per cell.
- All-to-all weights: from excitatory cells $0.5\,U[0,1]$, from inhibitory cells $-U[0,1]$.
  A presynaptic spike adds its weight to the postsynaptic input $I$ in the same 1 ms step.
- Thalamic input per ms: $5\,N(0,1)$ to excitatory, $2\,N(0,1)$ to inhibitory cells.
- Initial $v = -65$, $u = bv$. 1000 ms. Step 1 ms: $v$ is advanced twice by 0.5 ms (forward
  Euler), then $u$ once by 1 ms.
- Result as stated: asynchronous firing, Poisson-like trains, mean rates around 8 Hz;
  occasional synchronized episodes in the alpha and gamma ranges (10 and 40 Hz).

## Gaps

- The Fig. 3 claims are qualitative except "around 8 Hz" and "10 and 40 Hz"; criteria and
  bands will be ours (P3).
- No seed or random generator is given; replicates are seeds of our generator.
- The library presets in `jaxfne/presets.py` cite "Izhikevich 2003 Table 1". The paper has no
  table; RS, FS and LTS match Fig. 2, and `VIP_IS` (b −0.10, c −55, d 6) appears nowhere in
  this paper.
