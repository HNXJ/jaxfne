"""Author e2_ssa_spec.v2 from immutable v1: fills ONLY missing execution grammar.
Asserts every v1 field is byte-equal in v2 (no frozen semantics touched)."""
import sys, json, hashlib, pathlib, copy
sys.path.insert(0, '.')
ROOT = pathlib.Path('.')
PR = ROOT / 'artifacts/e2/preregistration'
v1_path = PR / 'e2_ssa_spec.json'
v1 = json.loads(v1_path.read_bytes())
import e2_exec_lib as _  # noqa: F401  # ensures repo import path works

def canon(j):
    return hashlib.sha256(json.dumps({k: v for k, v in j.items() if k != 'spec_hash'},
                                     sort_keys=True, separators=(',', ':')).encode()).hexdigest()

v1_hash = canon(v1)
assert v1_hash == v1['spec_hash'] == json.loads((PR / '36_AUDIT_INPUT_MANIFEST.json').read_bytes())['ssa_spec_hash']

v2 = copy.deepcopy(v1)
v2['schema'] = 'e2_ssa_spec.v2'
v2['parent_spec'] = {'schema': 'e2_ssa_spec.v1', 'spec_hash': v1['spec_hash'],
                     'sha256': hashlib.sha256(v1_path.read_bytes()).hexdigest().upper(),
                     'preserved_immutable': True}
# ---- execution grammar additions ONLY ----
v2['execution_grammar'] = {
    'time': {'dt_ms': 0.5, 'burn_in_discard_ms': 0,
             'note': 'block duration derived: events x ISI_ms; n_steps = duration/dt'},
    'paradigm': {
        'family': 'general_sequential_oddball',
        'events_per_block': 80,
        'p_deviant': 0.15,
        'p_standard': 0.85,
        'isi_ms_onset_to_onset': 200,
        'stim_duration_ms': 80,
        'stimulus_identity_mapping': {
            'A': 'drive pulse train to E-subpopulation [0,400) at relative amplitude 1.0',
            'B': 'drive pulse train to E-subpopulation [400,800) at relative amplitude 1.0',
            'identity_is_spatial_not_energy': True,
            'pulse_waveform': 'rectangular native-current pulse, duration stim_duration_ms, amplitude 1.0 added to drive_E during stimulus window'},
        'blocks_per_replicate': [
            {'name': 'oddball_A_std', 'standard': 'A', 'deviant': 'B', 'events': 80},
            {'name': 'oddball_B_std_flip', 'standard': 'B', 'deviant': 'A', 'events': 80},
            {'name': 'many_standards_control', 'p_each': 0.5, 'events': 80},
            {'name': 'recovery_isi500', 'standard_only': 'A', 'isi_ms': 500, 'events': 40}]},
    'replicates': {'n_outer_per_battery': 20,
                   'resolves_cardinality': 'v1 seeds.cardinalities.n_per_cell=20 applies here; V1-PING executed 5/arm under its 4/5 rule (documented UNRESOLVED_CARDINALITY there)'},
    'seed_derivation': {
        'scheme': 'master=PRNGKey(101); chained jax.random.fold_in through canonical_order with data=offset_domain+replicate_idx; child seeds via sha256(parent_bytes+tag)',
        'canonical_order': v1['seeds']['canonical_order'],
        'offsets': v1['seeds']['offsets'],
        'consumed_keys_must_lie_in_declared_domains': True},
    'response_extraction': {
        'R_event': 'population mean spikes over [onset+W_primary[0], onset+W_primary[1]) / window_seconds; W_primary inherited [30,110] ms',
        'SI_epsilon_theta': 'inherited unchanged from SSA.metrics'},
    'factor_staging': {
        'confirmatory_now': ['S0_no_decrement', 'S1_global', 'S2_gate', 'S2_stimulus_specific', 'S3_recovery'],
        'controls_executed': ['many_standards(p=0.5)', 'role_swap(flip block)', 'shuffled_history(2000 draws, K_analysis)'],
        'CONFIRMATORY_DEFERRED_v3': ['mechanism_matrix H_write x Gamma_H x HDP 8 cells'],
        'deferral_reason': 'S4 requires S2-in-D context and 8-cell matrix x 20 replicates exceeds authorized compute envelope; zero SSA trajectories exist so staging is prospective; thresholds/semantics untouched'},
    'executor_rules_from_corrigendum': [
        'gate conjuncts generated from this JSON only (no code literals)',
        'typed unit wrappers for deg/ms/dB/Hz',
        'stable seed derivation above; hash() forbidden',
        'independent frozen-only rescoring required before interpretation']}
# ---- spec_hash over new content ----
v2.pop('spec_hash', None)
v2['spec_hash'] = canon(v2)

# assert v1 fields preserved verbatim (schema is the intentional version bump)
for k, val in v1.items():
    if k in ('spec_hash', 'schema'):
        continue
    assert v2.get(k) == val, f'frozen field mutated: {k}'

out = PR / 'e2_ssa_spec.v2.json'
out.write_text(json.dumps(v2, indent=2) + '\n')
receipt = {
    'schema': 'e2_ssa_amendment_receipt.v1',
    'generated_at': '2026-08-25',
    'from': {'schema': 'e2_ssa_spec.v1', 'spec_hash': v1['spec_hash']},
    'to': {'schema': 'e2_ssa_spec.v2', 'spec_hash': v2['spec_hash'],
           'sha256': hashlib.sha256(out.read_bytes()).hexdigest().upper(), 'path': str(out).replace('\\', '/')},
    'natures': 'prospective specification: ZERO SSA trajectories generated before this amendment',
    'fields_missing_in_v1_now_filled': ['time block', 'paradigm event counts/probabilities',
                                        'stimulus identity mapping + waveform', 'block schedule incl. flip + many-standards + recovery',
                                        'explicit replicate count', 'exact seed derivation scheme',
                                        'response extraction definition', 'factor staging (mechanism matrix deferred to v3)'],
    'fields_untouched': 'all v1 fields byte-equal (asserted): thresholds theta_SI/delta_*/epsilon, SI form, adequacy gates, S0-S4 classifiers, controls definitions, topology, delays, masks',
    'write_once': True}
(PR / 'e2_ssa_spec_v2_amendment_receipt.json').write_text(json.dumps(receipt, indent=2))
print('v2 spec_hash:', v2['spec_hash'])
print('v2 sha256   :', receipt['to']['sha256'])
print('frozen-fields assertion: ALL PASS')
