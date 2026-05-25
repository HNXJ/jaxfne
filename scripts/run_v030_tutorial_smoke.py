#!/usr/bin/env python3
"""
v0.3 Tutorial Smoke Test Runner

Executes all v0.3 tutorial notebooks in smoke mode (reduced simulation duration
and network size for fast CI/CD validation). Validates manifest structure,
claim gates, and basic numerics without full runtime.

Usage:
    python scripts/run_v030_tutorial_smoke.py --help
    python scripts/run_v030_tutorial_smoke.py --out-dir outputs/smoke_test
    python scripts/run_v030_tutorial_smoke.py --list  # Show available tutorials

truth_mode: truth_safe_unverified
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import subprocess


# v0.3 Tutorial definitions (smoke test parameters)
V030_TUTORIALS = {
    'v030_01': {
        'name': 'Single Neuron I — Izhikevich',
        'script': 'examples/tutorials/v030_01_single_neuron_izhikevich.py',
        'smoke_duration_ms': 10.0,     # Reduced from 100ms
        'smoke_n_neurons': 1,           # Reduced from 10
    },
    'v030_02': {
        'name': 'Single Neuron II — Hodgkin-Huxley',
        'script': 'examples/tutorials/v030_02_single_neuron_hodgkin_huxley.py',
        'smoke_duration_ms': 10.0,
        'smoke_n_neurons': 1,
    },
    'v030_03': {
        'name': 'Synaptic Dynamics',
        'script': 'examples/tutorials/v030_03_synaptic_dynamics.py',
        'smoke_duration_ms': 10.0,
        'smoke_n_neurons': 1,
        'smoke_n_synapses': 10,
    },
    'v030_04': {
        'name': 'Two-Neuron E/I',
        'script': 'examples/tutorials/v030_04_two_neuron_ei.py',
        'smoke_duration_ms': 10.0,
        'smoke_n_neurons': 2,
    },
    'v030_05': {
        'name': 'Laminar Population',
        'script': 'examples/tutorials/v030_05_laminar_population.py',
        'smoke_duration_ms': 10.0,
        'smoke_n_neurons': 20,
        'smoke_n_layers': 5,
    },
}


def validate_tutorial_manifest(manifest: Dict[str, Any], scenario_id: str) -> Dict[str, Any]:
    """
    Validate tutorial manifest against hard acceptance gates.

    Args:
        manifest: Manifest dict from tutorial execution.
        scenario_id: Scenario identifier (e.g., 'v030_01').

    Returns:
        dict: Validation results with keys:
              - 'scenario': scenario_id
              - 'status': 'PASS' or 'FAIL'
              - 'checks': dict of individual gate results
              - 'errors': list of error messages if status='FAIL'
    """
    errors = []
    checks = {}

    # Check 1: Manifest structure
    required_blocks = ['basis', 'probe_report', 'validation_report']
    for block in required_blocks:
        if block not in manifest:
            errors.append(f"Manifest missing required block: {block}")
            checks[f'structure_{block}'] = False
        else:
            checks[f'structure_{block}'] = True

    # Check 2: Claim gates
    basis = manifest.get('basis', {})
    checks['claim_gate_physical_amplitude'] = (
        basis.get('physical_amplitude_claim_allowed') == False
    )
    if not checks['claim_gate_physical_amplitude']:
        errors.append("physical_amplitude_claim_allowed is not False")

    checks['claim_gate_biological_metabolism'] = (
        basis.get('biological_metabolism_claim_allowed') == False
    )
    if not checks['claim_gate_biological_metabolism']:
        errors.append("biological_metabolism_claim_allowed is not False")

    checks['claim_gate_level'] = (
        basis.get('claim_level') == 'computational_scaffold'
    )
    if not checks['claim_gate_level']:
        errors.append(f"claim_level is {basis.get('claim_level')}, not 'computational_scaffold'")

    # Check 3: Probe operators (8 required)
    probe_report = manifest.get('probe_report', {})
    required_probes = ['spikes', 'V_m', 'source', 'lfp_proxy', 'csd_proxy', 'eeg_proxy', 'meg_proxy', 'emm_proxy']
    for probe in required_probes:
        if probe in probe_report:
            checks[f'probe_{probe}'] = True
        else:
            checks[f'probe_{probe}'] = False
            errors.append(f"Probe operator {probe} missing")

    # Check 4: JSON safety (no NaN/Inf in manifest keys)
    import json
    try:
        json_str = json.dumps(manifest, allow_nan=False)
        checks['json_safe'] = True
    except ValueError as e:
        checks['json_safe'] = False
        errors.append(f"Manifest contains NaN/Inf: {e}")

    status = 'PASS' if not errors else 'FAIL'

    return {
        'scenario': scenario_id,
        'status': status,
        'checks': checks,
        'errors': errors,
    }


def run_tutorial_smoke(
    scenario_id: str,
    tutorial_info: Dict[str, Any],
    out_dir: str = 'outputs',
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Execute a single tutorial in smoke mode.

    Args:
        scenario_id: Scenario ID (e.g., 'v030_01').
        tutorial_info: Tutorial metadata (name, script, smoke parameters).
        out_dir: Output directory for results.
        verbose: If True, print detailed output.

    Returns:
        dict: Execution results with keys:
              - 'scenario': scenario_id
              - 'status': 'COMPLETE', 'FAILED', or 'SKIPPED'
              - 'manifest_validation': validation results (if executed)
              - 'error_message': error details (if failed)
    """
    result = {
        'scenario': scenario_id,
        'status': 'SKIPPED',
        'name': tutorial_info.get('name'),
    }

    script_path = tutorial_info.get('script')
    if not script_path or not os.path.exists(script_path):
        result['status'] = 'SKIPPED'
        result['error_message'] = f"Script not found: {script_path}"
        return result

    try:
        if verbose:
            print(f"\n[{scenario_id}] Running: {tutorial_info.get('name')}")

        # Set environment variables for smoke mode
        env = os.environ.copy()
        env['V030_SMOKE_MODE'] = '1'
        env['V030_SCENARIO_ID'] = scenario_id
        env['V030_OUT_DIR'] = out_dir

        # Add smoke parameters as environment variables
        for key, value in tutorial_info.items():
            if key.startswith('smoke_'):
                env_key = f'V030_{key.upper()}'
                env[env_key] = str(value)

        # Run tutorial script
        result_json_path = os.path.join(out_dir, f'{scenario_id}_manifest.json')

        cmd = [sys.executable, script_path]
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            timeout=300,  # 5 minute timeout per tutorial
            text=True,
        )

        if proc.returncode != 0:
            result['status'] = 'FAILED'
            result['error_message'] = f"Script exited with code {proc.returncode}\nStderr: {proc.stderr}"
            return result

        # Load and validate manifest
        if os.path.exists(result_json_path):
            with open(result_json_path) as f:
                manifest = json.load(f)

            validation = validate_tutorial_manifest(manifest, scenario_id)
            result['manifest_validation'] = validation

            if validation['status'] == 'PASS':
                result['status'] = 'COMPLETE'
            else:
                result['status'] = 'FAILED'
                result['error_message'] = '; '.join(validation['errors'])
        else:
            result['status'] = 'FAILED'
            result['error_message'] = f"Manifest not found: {result_json_path}"

    except subprocess.TimeoutExpired:
        result['status'] = 'FAILED'
        result['error_message'] = "Tutorial timed out (> 300 seconds)"
    except Exception as e:
        result['status'] = 'FAILED'
        result['error_message'] = f"Unexpected error: {e}"

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run v0.3 tutorials in smoke test mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python scripts/run_v030_tutorial_smoke.py --list
  python scripts/run_v030_tutorial_smoke.py --out-dir outputs/smoke
  python scripts/run_v030_tutorial_smoke.py v030_01 v030_02
        '''
    )

    parser.add_argument(
        'scenarios',
        nargs='*',
        help='Scenario IDs to run (default: all)',
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available scenarios and exit',
    )
    parser.add_argument(
        '--out-dir',
        default='outputs/v030_smoke',
        help='Output directory for results',
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Print detailed progress',
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        print("Available v0.3 tutorial scenarios:")
        for sid, info in V030_TUTORIALS.items():
            print(f"  {sid}: {info['name']}")
        sys.exit(0)

    # Determine which scenarios to run
    if args.scenarios:
        scenarios_to_run = {
            sid: V030_TUTORIALS[sid]
            for sid in args.scenarios
            if sid in V030_TUTORIALS
        }
        if not scenarios_to_run:
            print(f"ERROR: No valid scenarios specified. Use --list to see available.")
            sys.exit(1)
    else:
        scenarios_to_run = V030_TUTORIALS

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Run tutorials
    all_results = {
        'timestamp': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
        'smoke_mode': True,
        'scenarios': {},
    }

    for scenario_id in sorted(scenarios_to_run.keys()):
        result = run_tutorial_smoke(
            scenario_id,
            scenarios_to_run[scenario_id],
            out_dir=args.out_dir,
            verbose=args.verbose,
        )
        all_results['scenarios'][scenario_id] = result

        status_symbol = {
            'COMPLETE': '✓',
            'FAILED': '✗',
            'SKIPPED': '⊘',
        }.get(result['status'], '?')

        print(f"{status_symbol} {scenario_id}: {result['status']}")
        if result.get('error_message'):
            print(f"  {result['error_message']}")

    # Save summary
    summary_path = os.path.join(args.out_dir, 'smoke_test_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to {summary_path}")

    # Exit code
    failed_count = sum(
        1 for r in all_results['scenarios'].values()
        if r['status'] == 'FAILED'
    )
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == '__main__':
    main()
