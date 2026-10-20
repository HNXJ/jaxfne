# Head-forward analytic checks

Worst-case deviations of the spherical-head forward code in `jaxfne.fields.head`
(Sarvas 1987 MEG; Frank 1952 homogeneous-sphere EEG) against independent references,
recomputed by `scripts/score_head_forward_checks.py` with the inputs and tolerances of
`tests/test_head_forward.py`. `score.json` holds one `{error, tolerance, pass, reference}`
row per check; `error` is the worst-case absolute deviation and `tolerance` is
`atol + rtol·max|ref|`, so a passing test implies a passing row. These are analytic
agreements of a core method, not a paper replication: there is no paper value beside ours.

| check | reference | tolerance |
|---|---|---|
| meg_radial | primary-current Biot-Savart field | 1e-6·scale + 2e-4·scale |
| meg_full_vector | scalar-potential quadrature (-mu0 grad U) | 1e-4·scale + 1e-3·scale |
| meg_radial_silence | radial-to-tangential amplitude ratio | ratio below 1e-6 |
| eeg_neumann | Neumann-series potential (400 terms) | 1e-4·scale + 1e-3·scale |
| eeg_mne | MNE single-shell forward (homogeneous) | 1e-4·scale + 1e-3·scale |
| eeg_scaling | conductivity reciprocity (0.33 against 0.66 S/m) | 1e-5·scale |
| eeg_reference | average-reference null | column means below 1e-6·scale |

Rerun:

    python scripts/score_head_forward_checks.py         # rewrite score.json (this README is left as is)
    python scripts/score_head_forward_checks.py --check # recompute and exit 1 past rtol 1e-9

`eeg_mne` needs the optional `mne` package; without it the row keeps nulls with
`"skipped": "mne not installed"` and its dashboard criterion shows pending. The suite runs
under `JAX_PLATFORMS=cpu`.
