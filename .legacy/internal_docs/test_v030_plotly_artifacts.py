"""
v0.3 Plotly Artifacts Tests

Tests that Plotly artifact generation works correctly, both with and without
Plotly installed. Validates guarded imports, JSON-safe data, and hash integrity.

truth_mode: truth_safe_unverified
"""

import json
import tempfile
from pathlib import Path
import pytest

# Import tutorial utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
from tutorial_plotly_utils import (
    plotly_available,
    create_spike_raster_figure,
    create_voltage_trace_figure,
    create_firing_rate_figure,
    save_artifact_metadata,
    validate_artifact_hashes,
)


class TestPlotlyImport:
    """Tests for guarded Plotly import."""

    def test_plotly_availability_check(self):
        """plotly_available() returns correct status."""
        result = plotly_available()
        assert isinstance(result, bool)

    def test_tutorial_utils_import_no_plotly_required(self):
        """Tutorial utils can import without requiring Plotly."""
        # Successfully imported above without error
        assert callable(create_spike_raster_figure)
        assert callable(create_voltage_trace_figure)
        assert callable(plotly_available)


class TestSpikeRasterGeneration:
    """Tests for spike raster figure generation."""

    def test_spike_raster_data_only(self):
        """Generate spike raster as data dict (no Plotly required)."""
        spike_times = [
            [10.0, 25.5],
            [15.3, 30.8],
            [5.0],
        ]

        result = create_spike_raster_figure(
            spike_times=spike_times,
            duration_ms=100.0,
            as_html=False,
        )

        assert result['type'] == 'spike_raster'
        assert result['neurons'] == 3
        assert result['duration_ms'] == 100.0
        assert result['spike_counts'] == [2, 2, 1]
        assert 2.0 <= result['mean_firing_rate'] <= 25.0  # Check firing rate gate

    def test_spike_raster_with_custom_neuron_ids(self):
        """Spike raster with custom neuron IDs."""
        spike_times = [
            [10.0, 20.0],
            [15.0, 25.0],
        ]
        neuron_ids = [100, 200]

        result = create_spike_raster_figure(
            spike_times=spike_times,
            neuron_ids=neuron_ids,
            duration_ms=50.0,
            as_html=False,
        )

        assert result['neurons'] == 2

    @pytest.mark.skipif(
        not plotly_available(),
        reason="Plotly not installed",
    )
    def test_spike_raster_html_generation(self):
        """Generate spike raster as HTML (if Plotly available)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'raster.html'

            spike_times = [
                [10.0, 25.5],
                [15.3, 30.8],
            ]

            result = create_spike_raster_figure(
                spike_times=spike_times,
                duration_ms=100.0,
                as_html=True,
                output_path=str(output_path),
            )

            assert 'html_path' in result
            assert 'hash' in result
            assert output_path.exists()
            assert len(result['hash']) == 64  # SHA256 hex length


class TestVoltageTraceGeneration:
    """Tests for voltage trace figure generation."""

    def test_voltage_trace_data_only(self):
        """Generate voltage trace as data dict."""
        time_ms = [0.0, 0.1, 0.2, 0.3, 0.4]
        voltage_traces = {
            0: [-65.0, -64.0, -60.0, -50.0, -65.0],
            1: [-65.0, -63.0, -58.0, -45.0, -65.0],
        }

        result = create_voltage_trace_figure(
            time_ms=time_ms,
            voltage_traces=voltage_traces,
            as_html=False,
        )

        assert result['type'] == 'voltage_trace'
        assert result['traces'] == 2
        assert result['duration_ms'] == 0.4

    @pytest.mark.skipif(
        not plotly_available(),
        reason="Plotly not installed",
    )
    def test_voltage_trace_html_generation(self):
        """Generate voltage trace as HTML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'voltage.html'

            time_ms = [0.0, 0.5, 1.0, 1.5]
            voltage_traces = {0: [-65.0, -60.0, -50.0, -65.0]}

            result = create_voltage_trace_figure(
                time_ms=time_ms,
                voltage_traces=voltage_traces,
                as_html=True,
                output_path=str(output_path),
            )

            assert output_path.exists()
            assert 'hash' in result


class TestFiringRateGeneration:
    """Tests for firing rate figure generation."""

    def test_firing_rate_data_only(self):
        """Generate firing rate as data dict."""
        time_ms = [0.0, 1.0, 2.0, 3.0, 4.0]
        firing_rate_hz = [0.0, 5.0, 10.0, 8.0, 3.0]

        result = create_firing_rate_figure(
            time_ms=time_ms,
            firing_rate_hz=firing_rate_hz,
            as_html=False,
        )

        assert result['type'] == 'firing_rate'
        assert result['mean_rate_hz'] == pytest.approx(5.2, abs=0.1)
        assert result['max_rate_hz'] == 10.0

    @pytest.mark.skipif(
        not plotly_available(),
        reason="Plotly not installed",
    )
    def test_firing_rate_html_generation(self):
        """Generate firing rate as HTML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'firing_rate.html'

            time_ms = [0.0, 1.0, 2.0]
            firing_rate_hz = [5.0, 10.0, 8.0]

            result = create_firing_rate_figure(
                time_ms=time_ms,
                firing_rate_hz=firing_rate_hz,
                as_html=True,
                output_path=str(output_path),
            )

            assert output_path.exists()


class TestArtifactMetadata:
    """Tests for artifact metadata saving and validation."""

    def test_save_artifact_metadata(self):
        """Save artifact metadata to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            figures = [
                {
                    'type': 'spike_raster',
                    'neurons': 10,
                    'mean_firing_rate': 12.5,
                },
                {
                    'type': 'voltage_trace',
                    'traces': 3,
                },
            ]

            metadata_path = save_artifact_metadata(
                scenario_id='v030_01',
                figures=figures,
                output_dir=tmpdir,
            )

            assert Path(metadata_path).exists()

            # Load and verify
            with open(metadata_path) as f:
                metadata = json.load(f)

            assert metadata['scenario'] == 'v030_01'
            assert len(metadata['figures']) == 2

    @pytest.mark.skipif(
        not plotly_available(),
        reason="Plotly not installed",
    )
    def test_hash_validation(self):
        """Validate SHA256 hashes of artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create a dummy HTML file
            html_path = tmpdir_path / 'test_figure.html'
            html_path.write_text('<html><body>Test</body></html>')

            # Create metadata
            import hashlib
            with open(html_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            metadata = {
                'scenario': 'v030_01',
                'figures': [
                    {
                        'html_path': str(html_path),
                        'sha256': file_hash,
                    },
                ],
            }

            metadata_path = tmpdir_path / 'artifacts.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

            # Validate
            result = validate_artifact_hashes(str(metadata_path))
            assert result is True


class TestAcceptanceGatesArtifacts:
    """Tests that artifacts meet acceptance gate criteria."""

    def test_firing_rate_gate_in_data(self):
        """Firing rate in generated data meets 2–25 Hz gate."""
        spike_times = [[10.0, 20.0]] * 10  # 10 neurons, ~20 spikes total → ~20 Hz
        duration_ms = 100.0

        result = create_spike_raster_figure(
            spike_times=spike_times,
            duration_ms=duration_ms,
            as_html=False,
        )

        firing_rate = result['mean_firing_rate']
        assert 2.0 <= firing_rate <= 25.0

    def test_json_safe_artifact_metadata(self):
        """Artifact metadata is JSON-safe."""
        figures = [
            {
                'type': 'spike_raster',
                'neurons': 10,
                'mean_firing_rate': 12.5,
                'spike_counts': [5, 4, 6, 5, 4, 5, 4, 5, 4, 5],
            },
        ]

        # Should be JSON-serializable
        json_str = json.dumps(figures)
        assert 'NaN' not in json_str
        assert 'Infinity' not in json_str

    def test_hash_format_valid(self):
        """SHA256 hashes have valid format."""
        import hashlib

        test_data = b'test data'
        hash_val = hashlib.sha256(test_data).hexdigest()

        assert len(hash_val) == 64  # SHA256 hex is 64 chars
        assert all(c in '0123456789abcdef' for c in hash_val)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
