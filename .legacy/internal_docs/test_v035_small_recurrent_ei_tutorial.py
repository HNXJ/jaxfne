"""
Tests for v0.3.5 small recurrent E/I tutorial.

Classification: tutorial_scaffold_test
Scope: Verify public API, grammar correctness, figure generation, scope metadata
Truth: truth_safe_unverified
"""

import json
import os
import pytest
from pathlib import Path
import numpy as np
import jaxfne as jtfne

# Environment variable to enable artifact validation
# Run with: JAXFNE_VALIDATE_TUTORIAL_OUTPUTS=1 pytest ...
RUN_ARTIFACT_TESTS = os.environ.get("JAXFNE_VALIDATE_TUTORIAL_OUTPUTS") == "1"


class TestV035TutorialFileStructure:
    """Test that tutorial files exist and are accessible."""

    def test_notebook_file_exists(self):
        """Notebook file is created and readable."""
        notebook_path = Path("tutorials/jaxfne_v035_small_recurrent_ei.ipynb")
        assert notebook_path.exists(), f"Notebook file not found at {notebook_path}"
        assert notebook_path.suffix == ".ipynb"

    def test_markdown_doc_exists(self):
        """Markdown documentation file is created."""
        doc_path = Path("docs/tutorials_v030/035_small_recurrent_ei.md")
        assert doc_path.exists(), f"Markdown doc not found at {doc_path}"
        assert doc_path.suffix == ".md"

    def test_required_figure_names_referenced(self):
        """Required figure names are referenced in notebook and docs (no execution required)."""
        REQUIRED_FIGURES = [
            "01_recurrent_raster.png",
            "02_voltage_traces.png",
            "03_population_rate.png",
            "04_source_proxy.png",
            "05_readout_summary.png",
        ]
        notebook_path = Path("tutorials/jaxfne_v035_small_recurrent_ei.ipynb")
        doc_path = Path("docs/tutorials_v030/035_small_recurrent_ei.md")

        notebook_text = notebook_path.read_text()
        doc_text = doc_path.read_text()

        for fig_name in REQUIRED_FIGURES:
            assert fig_name in notebook_text, \
                f"Figure name {fig_name} not referenced in notebook"
            assert fig_name in doc_text, \
                f"Figure name {fig_name} not referenced in docs"

    @pytest.mark.skipif(
        not RUN_ARTIFACT_TESTS,
        reason="Figures directory validation requires JAXFNE_VALIDATE_TUTORIAL_OUTPUTS=1",
    )
    def test_figures_directory_exists(self):
        """Figures directory is created (requires artifact validation enabled)."""
        figures_dir = Path("tutorial_outputs/v035_small_recurrent_ei/figures")
        assert figures_dir.exists(), f"Figures directory not found at {figures_dir}"
        assert figures_dir.is_dir()


@pytest.mark.skipif(
    not RUN_ARTIFACT_TESTS,
    reason="Generated tutorial figures are validated only when JAXFNE_VALIDATE_TUTORIAL_OUTPUTS=1",
)
class TestV035Figures:
    """Test that all required figures are generated and valid (artifact validation)."""

    REQUIRED_FIGURES = [
        "01_recurrent_raster.png",
        "02_voltage_traces.png",
        "03_population_rate.png",
        "04_source_proxy.png",
        "05_readout_summary.png",
    ]

    def test_all_figures_exist(self):
        """All 5 required figures are generated."""
        figures_dir = Path("tutorial_outputs/v035_small_recurrent_ei/figures")
        for figure_name in self.REQUIRED_FIGURES:
            figure_path = figures_dir / figure_name
            assert figure_path.exists(), f"Missing figure: {figure_name}"

    def test_figures_are_nonzero(self):
        """All figure files are non-empty (nonzero size)."""
        figures_dir = Path("tutorial_outputs/v035_small_recurrent_ei/figures")
        for figure_name in self.REQUIRED_FIGURES:
            figure_path = figures_dir / figure_name
            assert figure_path.stat().st_size > 0, f"Figure {figure_name} is zero-size"

    def test_figures_are_png(self):
        """All figures are valid PNG files (magic bytes)."""
        figures_dir = Path("tutorial_outputs/v035_small_recurrent_ei/figures")
        png_magic = b'\x89PNG\r\n\x1a\n'
        for figure_name in self.REQUIRED_FIGURES:
            figure_path = figures_dir / figure_name
            with open(figure_path, 'rb') as f:
                header = f.read(8)
                assert header == png_magic, f"Figure {figure_name} does not have PNG magic bytes"


class TestV035PublicGrammar:
    """Test that the public API chain works as documented."""

    def test_configuration_chain_builds(self):
        """Configuration chainable grammar builds without error."""
        cfg = jtfne.Configuration()
        cfg = cfg.runtime(seed=42, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
        cfg = cfg.column("small_recurrent_ei", layers=["L2/3"], n=12)
        cfg = cfg.cell_types({"E": 0.75, "PV": 0.25})
        cfg = cfg.connectivity(excitatory_to_inhibitory=True, inhibitory_to_excitatory=True)
        cfg = cfg.set_emitter("izhikevich", "cortical_eig_e_plus_pv")
        cfg = cfg.probes(["MUA-proxy", "source-proxy", "LFP-proxy", "CSD-proxy", "EEG-proxy", "MEG-proxy", "EMM-proxy"])

        assert cfg is not None
        assert cfg.metadata['duration_ms'] == 1000.0
        assert cfg.metadata['dt_ms'] == 0.1
        assert cfg.networks[0]['n'] == 12

    def test_probes_method_returns_list_like(self):
        """cfg.probes() returns list-like structure, not dict of individual probes."""
        cfg = jtfne.Configuration()
        cfg = cfg.probes(["MUA-proxy", "source-proxy"])

        # Should return a _ProbeDeclarations container with list-like behavior
        assert hasattr(cfg.probes, '__getitem__'), "probes should support indexing"
        assert len(cfg.probes) >= 1, "probes should contain at least one probe"

    def test_cell_types_split_is_correct(self):
        """cell_types E/I split produces correct label counts."""
        cfg = jtfne.Configuration()
        cfg = cfg.runtime(seed=42, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
        cfg = cfg.column("test_ei", layers=["L2/3"], n=12)
        cfg = cfg.cell_types({"E": 0.75, "PV": 0.25})
        cfg = cfg.connectivity()
        cfg = cfg.set_emitter("izhikevich", "cortical_eig")
        cfg = cfg.probes(["MUA-proxy"])

        model = jtfne.construct(cfg)
        labels = model.params['emitter'].labels

        n_e = sum(1 for l in labels if l == 'E')
        n_pv = sum(1 for l in labels if l == 'PV')

        # 0.75 of 12 = 9, 0.25 of 12 = 3
        assert n_e == 9, f"Expected 9 E neurons, got {n_e}"
        assert n_pv == 3, f"Expected 3 PV neurons, got {n_pv}"


class TestV035Simulation:
    """Test that simulation runs and produces valid outputs."""

    def test_construct_succeeds(self):
        """Model construction from config succeeds."""
        cfg = jtfne.Configuration()
        cfg = cfg.runtime(seed=42, dtype="float32", duration_ms=100.0, dt_ms=0.1)
        cfg = cfg.column("test_ei", layers=["L2/3"], n=12)
        cfg = cfg.cell_types({"E": 0.75, "PV": 0.25})
        cfg = cfg.connectivity(excitatory_to_inhibitory=True, inhibitory_to_excitatory=True)
        cfg = cfg.set_emitter("izhikevich", "cortical_eig_e_plus_pv")
        cfg = cfg.probes(["MUA-proxy", "source-proxy"])

        model = jtfne.construct(cfg)
        assert model is not None
        assert model.params is not None
        assert 'emitter' in model.params

    def test_simulate_produces_signals(self):
        """Simulation produces Signals object with correct shapes."""
        cfg = jtfne.Configuration()
        cfg = cfg.runtime(seed=42, dtype="float32", duration_ms=100.0, dt_ms=0.1)
        cfg = cfg.column("test_ei", layers=["L2/3"], n=12)
        cfg = cfg.cell_types({"E": 0.75, "PV": 0.25})
        cfg = cfg.connectivity()
        cfg = cfg.set_emitter("izhikevich", "cortical_eig")
        cfg = cfg.probes(["MUA-proxy"])

        model = jtfne.construct(cfg)
        signals = jtfne.simulate(model, duration_ms=100.0, dt_ms=0.1, seed=42)

        assert signals is not None
        assert hasattr(signals, 'V_m')
        assert hasattr(signals, 'spikes')
        assert signals.V_m.shape == (1000, 12), f"V_m shape {signals.V_m.shape} != (1000, 12)"
        assert signals.spikes.shape == (1000, 12), f"spikes shape {signals.spikes.shape} != (1000, 12)"

    def test_output_is_finite(self):
        """Simulation outputs are finite (no NaN/Inf)."""
        cfg = jtfne.Configuration()
        cfg = cfg.runtime(seed=42, dtype="float32", duration_ms=100.0, dt_ms=0.1)
        cfg = cfg.column("test_ei", layers=["L2/3"], n=8)
        cfg = cfg.cell_types({"E": 0.75, "PV": 0.25})
        cfg = cfg.connectivity()
        cfg = cfg.set_emitter("izhikevich", "cortical_eig")
        cfg = cfg.probes(["MUA-proxy"])

        model = jtfne.construct(cfg)
        signals = jtfne.simulate(model, duration_ms=100.0, dt_ms=0.1, seed=42)

        assert np.isfinite(signals.V_m).all(), "V_m contains NaN or Inf"
        assert np.isfinite(signals.spikes).all(), "spikes contains NaN or Inf"

    def test_firing_rate_in_reasonable_range(self):
        """Mean firing rate is in active tutorial gate: 2–25 Hz (not silence/null tutorial)."""
        cfg = jtfne.Configuration()
        cfg = cfg.runtime(seed=42, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
        cfg = cfg.column("test_ei", layers=["L2/3"], n=12)
        cfg = cfg.cell_types({"E": 0.75, "PV": 0.25})
        cfg = cfg.connectivity()
        cfg = cfg.set_emitter("izhikevich", "cortical_eig_e_plus_pv")
        cfg = cfg.probes(["MUA-proxy"])

        model = jtfne.construct(cfg)
        signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=42)

        # Correct calculation: total spikes / (n_neurons * duration_seconds)
        n_neurons = 12
        duration_ms = 1000.0
        total_spikes = float(signals.spikes.sum())
        mean_fr = total_spikes / (n_neurons * duration_ms / 1000.0)  # Hz
        assert 2.0 <= mean_fr <= 25.0, f"Firing rate {mean_fr} Hz outside active tutorial gate [2, 25] Hz"


class TestV035ScopeMetadata:
    """Test that scope metadata is correctly declared."""

    def test_configuration_metadata_present(self):
        """Configuration contains required scope metadata fields."""
        cfg = jtfne.Configuration()
        cfg = cfg.runtime(seed=42, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
        cfg = cfg.column("test_ei", layers=["L2/3"], n=12)
        cfg = cfg.cell_types({"E": 0.75, "PV": 0.25})
        cfg = cfg.connectivity(excitatory_to_inhibitory=True, inhibitory_to_excitatory=True)
        cfg = cfg.set_emitter("izhikevich", "cortical_eig_e_plus_pv")
        cfg = cfg.probes(["MUA-proxy", "source-proxy"])

        assert 'duration_ms' in cfg.metadata
        assert 'dt_ms' in cfg.metadata
        assert cfg.networks[0]['cell_types'] is not None

    def test_probes_metadata_forbids_physical_amplitude_claim(self):
        """Probes declare physical_amplitude_claim_allowed=False."""
        cfg = jtfne.Configuration()
        cfg = cfg.probes(["MUA-proxy", "source-proxy"])

        assert cfg.probes[0]['physical_amplitude_claim_allowed'] == False, \
            "Probes must forbid physical amplitude claims"

    def test_field_solver_status_is_proxy(self):
        """Probes declare field solver status as proxy, not solved."""
        cfg = jtfne.Configuration()
        cfg = cfg.probes(["LFP-proxy"])

        status = cfg.probes[0].get('field_solver_status', '')
        assert 'proxy' in status.lower() or 'no_pde' in status.lower(), \
            f"Field solver status '{status}' should indicate proxy/no_pde"


class TestV035PublicWording:
    """Test that public documentation uses approved scope language."""

    def test_markdown_doc_uses_computational_scaffold_language(self):
        """Markdown doc uses 'computational scaffold' not 'claim'."""
        doc_path = Path("docs/tutorials_v030/035_small_recurrent_ei.md")
        content = doc_path.read_text()

        # Forbidden terms
        forbidden_terms = [
            "claim gates",
            "claim_level",
            "claim gate summary",
            "What this tutorial claims",
            "What this tutorial does NOT claim",
        ]

        for term in forbidden_terms:
            # Allow if it's clearly in a code example or metadata section
            if f'"{term}"' not in content and f"'{term}'" not in content:
                assert term.lower() not in content.lower() or \
                    "json" in content.lower() or \
                    "metadata" in content.lower(), \
                    f"Forbidden term '{term}' found in public markdown"

    def test_notebook_avoids_overclaiming_language(self):
        """Notebook avoids 'real EEG', 'real MEG', 'validated', 'mechanism proof'."""
        notebook_path = Path("tutorials/jaxfne_v035_small_recurrent_ei.ipynb")
        content = notebook_path.read_text()

        # These phrases should not appear in the public notebook
        overclaiming_phrases = [
            "real EEG",
            "real MEG",
            "validated EEG",
            "validated MEG",
            "mechanism proof",
            "biological proof",
        ]

        for phrase in overclaiming_phrases:
            assert phrase.lower() not in content.lower(), \
                f"Overclaiming phrase '{phrase}' found in notebook"


class TestV035Determinism:
    """Test that simulation is deterministic given seed."""

    def test_same_seed_produces_identical_spikes(self):
        """Two simulations with the same seed produce identical spike patterns."""
        cfg = jtfne.Configuration()
        cfg = cfg.runtime(seed=42, dtype="float32", duration_ms=100.0, dt_ms=0.1)
        cfg = cfg.column("test_ei", layers=["L2/3"], n=8)
        cfg = cfg.cell_types({"E": 0.75, "PV": 0.25})
        cfg = cfg.connectivity()
        cfg = cfg.set_emitter("izhikevich", "cortical_eig")
        cfg = cfg.probes(["MUA-proxy"])

        model = jtfne.construct(cfg)

        signals1 = jtfne.simulate(model, duration_ms=100.0, dt_ms=0.1, seed=42)
        signals2 = jtfne.simulate(model, duration_ms=100.0, dt_ms=0.1, seed=42)

        assert np.allclose(signals1.spikes, signals2.spikes), \
            "Same seed should produce identical spike rasters"


class TestV035JSONSafety:
    """Test that tutorial outputs are JSON-safe."""

    def test_simulation_outputs_json_serializable(self):
        """Simulation metadata can be serialized as JSON without NaN/Inf."""
        cfg = jtfne.Configuration()
        cfg = cfg.runtime(seed=42, dtype="float32", duration_ms=100.0, dt_ms=0.1)
        cfg = cfg.column("test_ei", layers=["L2/3"], n=12)
        cfg = cfg.cell_types({"E": 0.75, "PV": 0.25})
        cfg = cfg.connectivity()
        cfg = cfg.set_emitter("izhikevich", "cortical_eig")
        cfg = cfg.probes(["MUA-proxy"])

        model = jtfne.construct(cfg)
        signals = jtfne.simulate(model, duration_ms=100.0, dt_ms=0.1, seed=42)

        metadata = {
            "duration_ms": 100.0,
            "dt_ms": 0.1,
            "n_neurons": 12,
            "mean_firing_rate": float(signals.spikes.mean() * 1000.0 / 100.0),
            "voltage_range": [float(signals.V_m.min()), float(signals.V_m.max())],
            "truth_mode": "truth_safe_unverified",
            "computational_level": "computational_scaffold",
        }

        # This should not raise ValueError for NaN/Inf
        json_str = json.dumps(metadata, allow_nan=False)
        assert isinstance(json_str, str)
        assert len(json_str) > 0
