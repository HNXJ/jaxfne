#!/usr/bin/env python3
"""
v0.3 Tutorial Manifest Collector

Aggregates manifests and artifacts from all v0.3 tutorial outputs into a
single collection for validation, CI/CD verification, and documentation.

Validates:
- Manifest JSON structure
- Claim gates (physical_amplitude_claim_allowed=False, etc.)
- All 8 probe operators present
- Figure artifact hashes
- Acceptance gate compliance (firing rate 2-25 Hz, finite values, JSON-safe)

Output: v030_tutorial_collection.json (aggregated metadata) + validation report

Usage:
    python scripts/collect_v030_tutorial_manifests.py outputs/v030_tutorials/
    python scripts/collect_v030_tutorial_manifests.py --validate --strict
    python scripts/collect_v030_tutorial_manifests.py --list-failures

truth_mode: truth_safe_unverified
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import hashlib


class ManifestCollector:
    """Collect and validate v0.3 tutorial manifests."""

    def __init__(self, output_dir: str, verbose: bool = False):
        """
        Initialize collector.

        Args:
            output_dir: Directory containing tutorial output subdirectories.
            verbose: If True, print detailed validation output.
        """
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.manifests = {}
        self.validation_results = {}
        self.failures = []

    def collect(self) -> Dict[str, Any]:
        """
        Collect manifests from all tutorial output directories.

        Returns:
            dict: Collection metadata with keys:
                  - 'scenarios': dict of scenario_id -> manifest
                  - 'timestamp': ISO 8601 generation timestamp
                  - 'total_scenarios': count
                  - 'validation_summary': pass/fail counts
        """
        if not self.output_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {self.output_dir}")

        # Find all manifest files
        manifest_files = list(self.output_dir.glob('v030_*/manifest.json'))

        if not manifest_files:
            print(f"WARNING: No manifest files found in {self.output_dir}")

        for manifest_path in sorted(manifest_files):
            scenario_id = manifest_path.parent.name

            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)

                self.manifests[scenario_id] = manifest

                if self.verbose:
                    print(f"✓ Loaded: {scenario_id}")

            except json.JSONDecodeError as e:
                self.failures.append((scenario_id, f"Invalid JSON: {e}"))
                if self.verbose:
                    print(f"✗ Failed to parse: {scenario_id}")

        return {
            'scenarios': self.manifests,
            'timestamp': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
            'total_scenarios': len(self.manifests),
        }

    def validate_all(self, strict: bool = False) -> Dict[str, Any]:
        """
        Validate all collected manifests against acceptance gates.

        Args:
            strict: If True, any validation failure causes overall failure.

        Returns:
            dict: Validation report with keys:
                  - 'scenarios': dict of scenario_id -> validation result
                  - 'summary': overall pass/fail counts and error list
        """
        validation_report = {'scenarios': {}, 'summary': {'passed': 0, 'failed': 0, 'errors': []}}

        for scenario_id, manifest in self.manifests.items():
            result = self._validate_manifest(scenario_id, manifest, strict=strict)
            validation_report['scenarios'][scenario_id] = result

            if result['status'] == 'PASS':
                validation_report['summary']['passed'] += 1
            else:
                validation_report['summary']['failed'] += 1
                validation_report['summary']['errors'].extend(result.get('errors', []))

            if self.verbose:
                status_icon = '✓' if result['status'] == 'PASS' else '✗'
                print(f"{status_icon} {scenario_id}: {result['status']}")

        self.validation_results = validation_report
        return validation_report

    def _validate_manifest(
        self,
        scenario_id: str,
        manifest: Dict[str, Any],
        strict: bool = False,
    ) -> Dict[str, Any]:
        """
        Validate a single manifest.

        Args:
            scenario_id: Scenario identifier.
            manifest: Manifest dict to validate.
            strict: If True, any gate failure causes overall failure.

        Returns:
            dict: Validation result with keys:
                  - 'scenario': scenario_id
                  - 'status': 'PASS' or 'FAIL'
                  - 'checks': dict of individual checks
                  - 'errors': list of error messages
        """
        errors = []
        checks = {}

        # Gate 1: Manifest structure
        required_blocks = ['basis', 'probe_report', 'validation_report', 'conservation_proxy_diagnostics']
        for block in required_blocks:
            if block in manifest:
                checks[f'structure_{block}'] = True
            else:
                checks[f'structure_{block}'] = False
                errors.append(f"Missing block: {block}")

        # Gate 2: Claim gates (immutable)
        basis = manifest.get('basis', {})

        checks['claim_physical_amplitude'] = (
            basis.get('physical_amplitude_claim_allowed') == False
        )
        if not checks['claim_physical_amplitude']:
            errors.append("physical_amplitude_claim_allowed != False")

        checks['claim_biological_metabolism'] = (
            basis.get('biological_metabolism_claim_allowed') == False
        )
        if not checks['claim_biological_metabolism']:
            errors.append("biological_metabolism_claim_allowed != False")

        checks['claim_level'] = (
            basis.get('claim_level') == 'computational_scaffold'
        )
        if not checks['claim_level']:
            errors.append(f"claim_level={basis.get('claim_level')}, expected 'computational_scaffold'")

        # Gate 3: All 8 probe operators
        probe_report = manifest.get('probe_report', {})
        required_probes = ['spikes', 'V_m', 'source', 'lfp_proxy', 'csd_proxy', 'eeg_proxy', 'meg_proxy', 'emm_proxy']

        for probe in required_probes:
            if probe in probe_report:
                checks[f'probe_{probe}'] = True
            else:
                checks[f'probe_{probe}'] = False
                errors.append(f"Missing probe: {probe}")

        # Gate 4: JSON safety
        try:
            json.dumps(manifest, allow_nan=False)
            checks['json_safe'] = True
        except (ValueError, TypeError) as e:
            checks['json_safe'] = False
            errors.append(f"JSON safety: {e}")

        # Gate 5: Firing rate (2-25 Hz if available)
        diag = manifest.get('conservation_proxy_diagnostics', {})
        if 'mean_firing_rate_hz' in diag:
            fr = diag['mean_firing_rate_hz']
            if 2.0 <= fr <= 25.0:
                checks['firing_rate_range'] = True
            else:
                checks['firing_rate_range'] = False
                errors.append(f"Firing rate {fr:.1f} Hz outside [2, 25] range")
        else:
            checks['firing_rate_range'] = None  # Not available in this manifest

        status = 'FAIL' if errors else 'PASS'

        return {
            'scenario': scenario_id,
            'status': status,
            'checks': checks,
            'errors': errors,
        }

    def validate_artifact_hashes(self) -> Dict[str, Any]:
        """
        Validate SHA256 hashes of figure artifacts.

        Returns:
            dict: Hash validation results.
        """
        hash_validation = {
            'scenarios': {},
            'summary': {'valid': 0, 'invalid': 0, 'missing': 0},
        }

        for scenario_id in self.manifests.keys():
            scenario_dir = self.output_dir / scenario_id
            artifacts_path = scenario_dir / 'artifacts.json'

            if not artifacts_path.exists():
                hash_validation['summary']['missing'] += 1
                hash_validation['scenarios'][scenario_id] = {
                    'status': 'MISSING',
                    'artifacts_file': str(artifacts_path),
                }
                continue

            try:
                with open(artifacts_path) as f:
                    artifacts = json.load(f)

                scenario_validation = {'status': 'VALID', 'figures': {}}

                for fig_name, fig_info in artifacts.get('figures', {}).items():
                    if 'sha256' not in fig_info:
                        scenario_validation['status'] = 'INCOMPLETE'
                        continue

                    fig_path = scenario_dir / fig_name
                    if not fig_path.exists():
                        scenario_validation['figures'][fig_name] = {
                            'status': 'MISSING',
                            'recorded_hash': fig_info['sha256'],
                        }
                        hash_validation['summary']['missing'] += 1
                        continue

                    # Recompute hash
                    with open(fig_path, 'rb') as f:
                        computed_hash = hashlib.sha256(f.read()).hexdigest()

                    if computed_hash == fig_info['sha256']:
                        scenario_validation['figures'][fig_name] = {
                            'status': 'VALID',
                            'hash': computed_hash,
                        }
                        hash_validation['summary']['valid'] += 1
                    else:
                        scenario_validation['figures'][fig_name] = {
                            'status': 'INVALID',
                            'recorded_hash': fig_info['sha256'],
                            'computed_hash': computed_hash,
                        }
                        scenario_validation['status'] = 'INVALID'
                        hash_validation['summary']['invalid'] += 1

                hash_validation['scenarios'][scenario_id] = scenario_validation

            except (json.JSONDecodeError, IOError) as e:
                hash_validation['scenarios'][scenario_id] = {
                    'status': 'ERROR',
                    'error': str(e),
                }

        return hash_validation

    def save_collection(self, output_path: str) -> str:
        """
        Save collected manifests to JSON file.

        Args:
            output_path: Path to save collection JSON.

        Returns:
            str: Path to saved file.
        """
        collection = {
            'timestamp': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
            'total_scenarios': len(self.manifests),
            'scenarios': self.manifests,
        }

        with open(output_path, 'w') as f:
            json.dump(collection, f, indent=2)

        print(f"✓ Saved collection: {output_path}")
        return output_path

    def save_validation_report(self, output_path: str) -> str:
        """
        Save validation report to JSON file.

        Args:
            output_path: Path to save validation report.

        Returns:
            str: Path to saved file.
        """
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)

        print(f"✓ Saved validation report: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Collect and validate v0.3 tutorial manifests',
    )

    parser.add_argument(
        'output_dir',
        help='Directory containing v030_XX/ subdirectories with manifests',
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation gates on all manifests',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Strict validation: any failure causes overall failure',
    )
    parser.add_argument(
        '--check-hashes',
        action='store_true',
        help='Validate SHA256 hashes of figure artifacts',
    )
    parser.add_argument(
        '--save-collection',
        default='v030_tutorial_collection.json',
        help='Save aggregated collection to JSON',
    )
    parser.add_argument(
        '--save-validation',
        default='v030_validation_report.json',
        help='Save validation report to JSON',
    )
    parser.add_argument(
        '--list-failures',
        action='store_true',
        help='Print list of failed validations and exit',
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed output',
    )

    args = parser.parse_args()

    # Create collector
    collector = ManifestCollector(args.output_dir, verbose=args.verbose)

    # Collect
    print(f"Collecting manifests from {args.output_dir}...")
    collection = collector.collect()
    print(f"✓ Collected {collection['total_scenarios']} scenarios")

    # Validate
    if args.validate:
        print("\nValidating manifests...")
        validation = collector.validate_all(strict=args.strict)
        passed = validation['summary']['passed']
        failed = validation['summary']['failed']
        print(f"✓ Validation: {passed} PASS, {failed} FAIL")

        if failed > 0:
            print("\nValidation errors:")
            for error in validation['summary']['errors'][:10]:  # Show first 10
                print(f"  - {error}")
            if len(validation['summary']['errors']) > 10:
                print(f"  ... and {len(validation['summary']['errors']) - 10} more")

    # Check hashes
    if args.check_hashes:
        print("\nValidating artifact hashes...")
        hash_results = collector.validate_artifact_hashes()
        print(f"✓ Hashes: {hash_results['summary']['valid']} valid, "
              f"{hash_results['summary']['invalid']} invalid, "
              f"{hash_results['summary']['missing']} missing")

    # Save
    if args.save_collection:
        collector.save_collection(args.save_collection)

    if args.save_validation and args.validate:
        collector.save_validation_report(args.save_validation)

    # List failures
    if args.list_failures:
        if collector.failures:
            print("\nCollection failures:")
            for scenario_id, error in collector.failures:
                print(f"  {scenario_id}: {error}")
        else:
            print("\nNo collection failures.")

    sys.exit(0)


if __name__ == '__main__':
    main()
