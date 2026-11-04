# PCL-R2/R4 design study (read-only), 2026-10-27

READ-ONLY design: no runtime change, no scored run. NOT PCL REPRODUCED; the signed v5 T2 FAIL stands. Inputs: the R1
receipt (`r1_fidelity_audit_2026-10-24.md`, F1-F6), a state inventory of the C++ reference
(`F:/warehouse/pcl/code/Predictive-Coding-Light_code/PCL_C++/`; it has no git history, so its version is unpinned), the
port (`port/pcl_event_ref.py`, `port/run_tuning.py`) and the core event scan at jaxfne `0a6f8148`. C_t = (X, H, W,
B, K, A) is the continuation decomposition of `artifacts/programme/receipts/engine_053_054/continuation_053.md`.

## 1. Reference continuation rule

Within a file, and across its `nb_run` repeats, the C++ carries every state variable (`NetworkHandle.cpp:109-148`);
each repeat continues the clock by one file span (`NetworkHandle.cpp:509-512`). The training loop calls `save()` after every file
(`SurroundSuppression.cpp:64`). `save()` calls `resetAllNeurons()` before it writes (`NetworkHandle.cpp:200`).
`Neuron::resetNeuron` sets potential, adaptation, timers and counters to 0, the threshold to `VTHRESH`, the rate
average to 0 and the learning decay to 1.0, and keeps every weight (`neurons/Neuron.cpp:611-628`). Loading reads
`count_spike`, `learning_decay` and `spiking_rate` only (`Neuron.cpp:495-502`).

The reference therefore hands the next file W and the reset neuron state. R1's F4 arm that carries the trained
object across a file (S=11355, C=11543) keeps V, A, threshold and rate, which the C++ resets; the arm that copies
W into a fresh object (S=9496, C=10005) matches the C++ rule, because the snapshot's loaded rate and decay equal the
reset values (rate 0 and decay 1.0 in all 6336 cSTDP3 neuron files, 5184 simple and 1152 complex). F4 stands as recorded; for the design, C0 at a file boundary is
a declared reset map, and the handoff carries W and applies that map.

## 2. State map

| State | C++ | Port (update) | Core event scan | C_t (proposed) |
|---|---|---|---|---|
| potential V | `Neuron.hpp:51` | `pcl_event_ref.py:229-268` | carried | X |
| adaptation potential | `Neuron.cpp:392-394` | `:223,230,264` | carried | X (A is defensible) |
| last-event and spike times | `Neuron.hpp:45-54` | `:210-218,232-278` | carried (first spike only) | X |
| threshold | `Neuron.cpp:377-387` | `:757-761` | fixed at `VTHRESH`; adaptation refused (`run_tuning.py:301-302`) | H |
| rate average | `Neuron.cpp:357-364` | `:754` | absent | H |
| spike-rate counter | `Neuron.hpp:48` | `:219,629,755` | counted; a second spike refused (`run_tuning.py:325-328`) | H |
| measurement cursor | `NetworkHandle.cpp:137-148` | `:748-762` | refused past the interval | H |
| w_s, wi_s, w_c, wi_c, LAT, TD | `SimpleNeuron.cpp:197-290`, `ComplexNeuron.cpp:97-166` | frozen per file (`learning=False`, `run_tuning.py:350`) | tables, frozen (`enable_hdp` refused) | W |
| learning decay | `Neuron.cpp:499` | loaded 1.0 | absent | W (learning only) |
| STDP and inhibition buffers | `SimpleNeuron.hpp:18-22`, `ComplexNeuron.cpp:78-175` | appended, unread with learning off | absent | B (learning only) |
| delay queue | `SpikingNetwork.cpp:69-136` | refused | absent | B (inactive at `synapses=1`) |
| RNG | init only (`WeightMatrix.cpp:7`) | init only | none | K (no draws in inference) |

## 3. Two realizations

(a) Reference-faithful compatibility realization: a PCL-specific scan that reproduces the C++ state machine,
with multi-spike cells, interval adaptation of threshold and rate, STDP on all six weight families, local
inhibition wired only under `"local"` (F1) and the reset rule of section 1. It closes F1, F2, F4 and F5 in one
place, but it is a second kernel outside the core grammar, which the replications-core-only rule (Hamm
2026-10-13) excludes from scored runs.

(b) Minimal general extension of the core LIF event scan, four capabilities, each with its own test: several
spikes per cell with their times; H coordinates (threshold, rate trace) updated at a declared measurement
interval through a declared F_H; F_W (the registered `pcl_stdp` rule) inside the scan; and a declared boundary
map on `ContinuationState` that carries W and resets X and H. Local inhibition follows the config (the port's
`"static"` to `"local"` alias goes, F1); column selection (F6) is a protocol choice for the R4 prereg.

Recommendation: (b). The numba port stays the event-exact oracle, so (a)'s state machine is kept as the
comparator for every capability of (b).

## 4. Continuation-equivalence test

For files F1..Fn and the boundary rule of section 1, two runs must agree: one continuous run that applies the
reset at each boundary, and a chain of per-file runs through `ContinuationState`. Spike counts and spike times
per cell agree exactly; W agrees bitwise in float64. The test runs first oracle against oracle (numba continuous
against numba chained), then core chained against the oracle. Its negative control chains without the reset; on
a file pair with nonzero V or a moved threshold at the boundary it must differ (R1 F4 shows a difference of 1859
simple spikes on one such pair), so the test can fail. Within a file, chunked runs equal the continuous run
bitwise, as the 0.5.3 continuation contract already requires.

## 5. Beyond PCL

Each capability is general. Multi-spike event replay serves any event-driven LIF lane, including the DVS128 and
N-MNIST inputs of the data-statement row; interval-updated threshold and rate traces are H coordinates with RBD
dynamics (D26) for adaptive-threshold models; the boundary map applies to any trial or episode protocol that resets fast state and
keeps weights. The demonstration declared for R4 is a two-episode run of a small adaptive-threshold LIF network
outside PCL, where chained-with-reset equals continuous-with-reset and chained-without-reset differs.

## 6. Order and open points

R2 reproduces the reference on identical inputs with the oracle in the reference configuration (learning on,
reset at each file, config-faithful inhibition); whether the C++ builds on this machine, so that the oracle can be
compared with the authors' binary, is unknown. R3 attributes what remains; R4 lands (b) behind a new prereg that
Hamm signs before any confirmatory run; R5 scores exact and approximate reproduction separately. Open for Hamm:
(a) or (b); the F6 column the R4 prereg scores. Open for the design: adaptation as X or A, and spike times as X
or B.
