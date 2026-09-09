"""Atlas suite: 6 fixed panels for any model, incl. N=1, 0 edges, no spikes, field/no-field, multi-area."""

import copy
import json
import pathlib
import numpy as np
import pytest

import jaxfne as J
from jaxfne.vis.atlas_suite import (
    DT_SOURCE_FALLBACK,
    DT_SOURCE_INFERRED,
    OPTIONAL_FIELD_FILE,
    PANELS,
    build_atlas,
    classify_dt_ms,
)

FIXED = [p[0] for p in PANELS]


def _build(kind: str, tmp: pathlib.Path, **kwargs):
    if kind == "single":
        cfg = J.suite2_single_neuron_config(seed=7, duration_ms=200.0, dt_ms=0.1)
    elif kind == "zero_edge":
        cfg = (
            J.Configuration()
            .runtime(seed=7, duration_ms=100.0, dt_ms=0.1)
            .column("V1", ["L2/3"], 10)
            .set_emitter("izhikevich", "cortical_eig")
            .connectivity(kind="empty")
            .probes(["spikes", "V_m"])
        )
    elif kind == "multi_area":
        cfg = J.suite2_v1_v4_config(duration_ms=150.0, dt_ms=0.1, seed=7)
    elif kind == "with_field":
        cfg = (
            J.Configuration()
            .runtime(seed=7, duration_ms=150.0, dt_ms=0.1)
            .column("V1", ["L2/3", "L4"], 16)
            .cell_types({"E": 0.75, "PV": 0.25})
            .connectivity(kind="laminar_signed_metadata", recurrent=True)
            .set_emitter("izhikevich", "cortical_eig")
            .field(domain="laminar_column", conductivity="proxy")
            .probes(["spikes", "V_m", "source", "LFP-proxy", "CSD-proxy"], n_contacts=8)
        )
    else:
        cfg = J.suite2_net1_config(seed=7, n=10, duration_ms=200.0, dt_ms=0.1)
    model = J.construct(cfg)
    out = tmp / kind
    manifest = build_atlas(model, out_dir=str(out), duration_ms=kwargs.get("duration_ms", 200.0), dt_ms=0.1)
    return model, out, manifest


def test_atlas_single_neuron(tmp_path):
    """N=1 degradation test: all 6 fixed panels emitted, single neuron card."""
    _, out, manifest = _build("single", tmp_path)
    assert manifest["n_neurons"] == 1
    for f in FIXED:
        p = out / f
        assert p.exists(), f"missing fixed panel {f}"
        assert p.stat().st_size > 0
    assert (out / "index.html").exists()
    assert (out / "manifest.json").exists()
    disk = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert disk["n_neurons"] == 1 and len(disk["panels"]) >= 6


def test_atlas_small_network(tmp_path):
    """Ordinary small network test: all 6 fixed panels present."""
    _, out, manifest = _build("net10", tmp_path)
    assert manifest["n_neurons"] == 10
    for f in FIXED:
        p = out / f
        assert p.exists(), f"missing fixed panel {f}"
        assert p.stat().st_size > 0


def test_atlas_zero_edges(tmp_path):
    """Zero-edge network test: single neuron has exactly 0 edges, verifies empty-matrix connectivity card."""
    _, out, manifest = _build("single", tmp_path, duration_ms=100.0)
    assert manifest["n_edges"] == 0
    for f in FIXED:
        assert (out / f).exists()


def test_atlas_no_spikes(tmp_path):
    """Zero spikes test: signals with all-zero spikes array."""
    cfg = J.suite2_net1_config(seed=42, n=8, duration_ms=100.0, dt_ms=0.1)
    model = J.construct(cfg)
    sim = J.Simulation(duration_ms=100.0, dt_ms=0.1, seed=42)
    sig = J.simulate(model, sim)
    sig_quiet = J.Signals(
        time_ms=sig.time_ms,
        V_m=sig.V_m,
        spikes=np.zeros_like(sig.spikes),
        sources=sig.sources,
        field=sig.field,
        metadata=sig.metadata,
    )
    out = tmp_path / "quiet"
    manifest = build_atlas(model, sig_quiet, out_dir=str(out))
    for f in FIXED:
        assert (out / f).exists()


def test_atlas_short_run(tmp_path):
    """Short run (e.g. 5 ms): PSD handles gracefully without crashing."""
    cfg = J.suite2_single_neuron_config(duration_ms=5.0, dt_ms=0.1)
    model = J.construct(cfg)
    out = tmp_path / "short"
    manifest = build_atlas(model, out_dir=str(out), duration_ms=5.0, dt_ms=0.1)
    for f in FIXED:
        assert (out / f).exists()


def test_atlas_field_vs_no_field(tmp_path):
    """Field present emits optional field.html; no field skips it."""
    # With field:
    _, out_field, manifest_field = _build("with_field", tmp_path)
    assert (out_field / OPTIONAL_FIELD_FILE).exists()
    assert any(p["file"] == OPTIONAL_FIELD_FILE for p in manifest_field["panels"])

    # Without field (pass signals with field=None):
    cfg_nofield = J.suite2_net1_config(seed=7, n=10, duration_ms=100.0, dt_ms=0.1)
    model_nofield = J.construct(cfg_nofield)
    sim_nofield = J.Simulation(duration_ms=100.0, dt_ms=0.1, seed=7, record_fields=False)
    sig_nofield = J.simulate(model_nofield, sim_nofield)
    assert sig_nofield.field is None

    out_nofield = tmp_path / "nofield"
    manifest_nofield = build_atlas(model_nofield, sig_nofield, out_dir=str(out_nofield))
    assert not (out_nofield / OPTIONAL_FIELD_FILE).exists()
    assert all(p["file"] != OPTIONAL_FIELD_FILE for p in manifest_nofield["panels"])


def test_atlas_multi_area(tmp_path):
    """Multi-area hierarchical column configuration."""
    _, out, manifest = _build("multi_area", tmp_path, duration_ms=150.0)
    for f in FIXED:
        assert (out / f).exists()


def test_atlas_non_mutation_invariant(tmp_path):
    """Atlas generation must not mutate model, signals, or simulation state."""
    cfg = J.suite2_net1_config(seed=42, n=10, duration_ms=100.0, dt_ms=0.1)
    model = J.construct(cfg)
    sig = J.simulate(model, J.Simulation(duration_ms=100.0, dt_ms=0.1, seed=42))

    orig_spikes = np.array(sig.spikes, copy=True)
    orig_vm = np.array(sig.V_m, copy=True)
    orig_time = np.array(sig.time_ms, copy=True)
    orig_summary = dict(model.summary())

    out = tmp_path / "mutation_test"
    build_atlas(model, sig, out_dir=str(out))

    np.testing.assert_array_equal(sig.spikes, orig_spikes)
    np.testing.assert_array_equal(sig.V_m, orig_vm)
    np.testing.assert_array_equal(sig.time_ms, orig_time)
    assert dict(model.summary()) == orig_summary
# --- README / docs unity gates ------------------------------------------------
#
# The README's static stills and the docs' interactive panels must stay derived
# from the same canonical panel set. The README binaries had no generator, so
# nothing could detect drift between the two surfaces or a renamed panel.

ROOT = pathlib.Path(__file__).resolve().parents[1]
README_PNG_DIR = ROOT / "docs" / "assets" / "readme"


def test_readme_still_exists_for_every_fixed_panel():
    """Each canonical panel has a committed README still under its own name."""
    for filename in FIXED:
        stem = filename.removesuffix(".html")
        png = README_PNG_DIR / f"{stem}.png"
        assert png.exists(), (
            f"missing README still {png.relative_to(ROOT)} for panel {filename!r}; "
            "regenerate with python scripts/generate_readme_atlas.py"
        )
        assert png.stat().st_size > 0


def test_no_orphan_readme_stills():
    """No README still may survive a panel rename."""
    expected = {f.removesuffix(".html") + ".png" for f in FIXED}
    actual = {p.name for p in README_PNG_DIR.glob("*.png")}
    orphans = actual - expected
    assert not orphans, (
        f"README stills with no corresponding panel: {sorted(orphans)}; "
        "regenerate with python scripts/generate_readme_atlas.py"
    )


def test_public_pages_reference_current_panel_names():
    """README, docs landing page and atlas guide must link the current panels.

    Checks asset and panel *references*, not prose -- the atlas guide names the
    retired panel deliberately when explaining why it was renamed.
    """
    pages = [
        ROOT / "README.md",
        ROOT / "docs" / "index.md",
        ROOT / "docs" / "guides" / "atlas_suite.md",
    ]
    retired_assets = ("operating_point.html", "operating_point.png")
    for page in pages:
        text = page.read_text(encoding="utf-8")
        for ref in retired_assets:
            assert ref not in text, f"{page.name} still references retired asset {ref!r}"

    # The 6-panel grammar list must name the current panel.
    for page in (ROOT / "README.md", ROOT / "docs" / "index.md"):
        text = page.read_text(encoding="utf-8")
        assert "`spectral`, `state_summary`)" in text, (
            f"{page.name} does not list the current 6-panel grammar"
        )

    guide = (ROOT / "docs" / "guides" / "atlas_suite.md").read_text(encoding="utf-8")
    for filename in FIXED:
        assert filename in guide, f"atlas guide does not document panel {filename!r}"


def test_readme_atlas_generator_pins_the_published_configuration():
    """The generator's pinned config is the one the published atlas records."""
    from scripts.generate_readme_atlas import EXPECTED_CONFIG_HASH

    manifest = json.loads(
        (ROOT / "docs" / "_static" / "atlas" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["config_hash"] == EXPECTED_CONFIG_HASH
    published = {p["file"] for p in manifest["panels"]}
    assert set(FIXED) <= published, (
        f"published atlas is missing canonical panels: {sorted(set(FIXED) - published)}"
    )


# --- Provenance regression gates -------------------------------------------
#
# build_atlas used to record its dt_ms *parameter default* (0.1) even when the
# caller passed realized signals simulated at another timestep, so the
# canonical 0.5 ms atlas was published as 0.1 ms with no version recorded.
# These gates derive truth from the signals grid and fail on that behavior.


def _small_signals(dt_ms: float = 0.5, duration_ms: float = 20.0, seed: int = 3):
    cfg = J.suite2_net1_config(seed=seed, n=8, duration_ms=duration_ms, dt_ms=dt_ms)
    model = J.construct(cfg)
    sig = J.simulate(model, J.Simulation(duration_ms=duration_ms, dt_ms=dt_ms, seed=seed))
    return model, sig


def test_build_atlas_derives_dt_from_signals(tmp_path):
    """Manifest dt must come from the signals time grid, not the default."""
    model, sig = _small_signals(dt_ms=0.5)
    manifest = build_atlas(model, sig, out_dir=str(tmp_path / "dt"))
    assert manifest["dt_ms"] == 0.5, (
        f"manifest dt_ms {manifest['dt_ms']!r} != simulated 0.5 "
        "(parameter default leaked into provenance)"
    )
    assert manifest["dt_source"] == DT_SOURCE_INFERRED
    assert "dt_fallback_reason" not in manifest
    assert manifest["n_steps"] == len(np.asarray(sig.time_ms))


# --- W2: dt inference classification ----------------------------------------


@pytest.mark.parametrize(
    "time_ms,expected_source,expected_dt,expected_reason",
    [
        (np.arange(0.0, 2.0, 0.5), DT_SOURCE_INFERRED, 0.5, None),
        (np.array([0.0]), DT_SOURCE_FALLBACK, 0.25, "insufficient_samples"),
        (np.array([]), DT_SOURCE_FALLBACK, 0.25, "insufficient_samples"),
        (None, DT_SOURCE_FALLBACK, 0.25, "missing_time"),
    ],
)
def test_classify_dt_ms_valid_paths(time_ms, expected_source, expected_dt, expected_reason):
    result = classify_dt_ms(time_ms, 0.25)
    assert result.dt_source == expected_source
    assert result.dt_ms == expected_dt
    assert result.fallback_reason == expected_reason


@pytest.mark.parametrize(
    "time_ms,reason",
    [
        (np.array([0.0, np.nan, 1.0]), "non_finite"),
        (np.array([0.0, np.inf, 1.0]), "non_finite"),
        (np.array([0.0, 0.0, 1.0]), "non_monotonic_or_nonpositive"),
        (np.array([1.0, 0.0]), "non_monotonic_or_nonpositive"),
        (np.array([0.0, 0.5, 1.2]), "non_uniform"),
    ],
)
def test_classify_dt_ms_invalid_grid_raises(time_ms, reason):
    with pytest.raises(ValueError, match=f"INVALID_TIME_GRID \\({reason}\\)"):
        classify_dt_ms(time_ms, 0.25)


def test_classify_dt_ms_propagates_unexpected_errors(monkeypatch):
    def _boom(_t):
        raise RuntimeError("unexpected inference failure")

    monkeypatch.setattr("jaxfne.vis.atlas_suite.np.diff", _boom)
    with pytest.raises(RuntimeError, match="unexpected inference failure"):
        classify_dt_ms(np.array([0.0, 0.5, 1.0]), 0.1)


def test_build_atlas_rejects_invalid_grid_without_outputs(tmp_path):
    model, sig = _small_signals(dt_ms=0.5)
    bad_time = np.asarray(sig.time_ms, dtype=float).copy()
    bad_time[3] += 0.07
    sig_bad = J.Signals(
        time_ms=bad_time,
        V_m=sig.V_m,
        spikes=sig.spikes,
        sources=sig.sources,
        field=sig.field,
        metadata=sig.metadata,
    )
    out = tmp_path / "bad_dt"
    with pytest.raises(ValueError, match="INVALID_TIME_GRID \\(non_uniform\\)"):
        build_atlas(model, sig_bad, out_dir=str(out), dt_ms=0.1)
    assert not out.exists() or not any(out.iterdir())


def test_atlas_manifest_records_version_seed_duration(tmp_path):
    """Manifest carries version/seed/duration and steps*dt == duration."""
    model, sig = _small_signals(dt_ms=0.5, duration_ms=20.0, seed=3)
    manifest = build_atlas(
        model, sig, out_dir=str(tmp_path / "prov"),
        duration_ms=20.0, seed=3,
    )
    assert manifest["jaxfne_version"] == J.__version__
    assert manifest["seed"] == 3
    assert manifest["duration_ms"] == 20.0
    assert manifest["n_steps"] * manifest["dt_ms"] == manifest["duration_ms"]


def test_atlas_panel_cards_match_manifest(tmp_path):
    """Every panel's embedded provenance card repeats the manifest values."""
    model, sig = _small_signals(dt_ms=0.5, duration_ms=20.0, seed=3)
    out = tmp_path / "cards"
    manifest = build_atlas(
        model, sig, out_dir=str(out), duration_ms=20.0, seed=3,
    )
    for key in ("dt_ms", "jaxfne_version", "config_hash"):
        want = str(manifest[key])
        for panel in manifest["panels"]:
            text = (out / panel["file"]).read_text(encoding="utf-8")
            assert want in text, (
                f"{panel['file']} card is missing {key}={want!r}"
            )
    assert str(manifest["n_steps"]) in (out / "raster.html").read_text(encoding="utf-8")


def test_canonical_manifest_provenance():
    """The committed canonical atlas records the pinned run truthfully."""
    from scripts.generate_readme_atlas import (
        CANONICAL_DT_MS,
        CANONICAL_DURATION_MS,
        CANONICAL_SEED,
        EXPECTED_CONFIG_HASH,
    )

    manifest = json.loads(
        (ROOT / "docs" / "_static" / "atlas" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["jaxfne_version"] == J.__version__, (
        f"canonical atlas built by {manifest.get('jaxfne_version')!r}, "
        f"runtime is {J.__version__!r}; regenerate with "
        "python scripts/generate_readme_atlas.py"
    )
    assert manifest["dt_ms"] == CANONICAL_DT_MS == 0.5
    assert manifest["duration_ms"] == CANONICAL_DURATION_MS == 200.0
    assert manifest["seed"] == CANONICAL_SEED == 0
    assert manifest["n_steps"] == 400
    assert manifest["n_steps"] * manifest["dt_ms"] == manifest["duration_ms"]
    assert manifest["config_hash"] == EXPECTED_CONFIG_HASH
