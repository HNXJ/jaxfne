"""The multiscale and heterogeneous etudes run at the emitter default drive (T-1).

Their frozen figures were made while the declared E/PV drive of 8.0 was inert
(Configuration.cell_type_drives, P-014); a baseline drive in the config would
change the realized model and the figures.
"""

import scripts.run_heterogeneous_emitters_etude as hetero
import scripts.run_multiscale_observation_etude as multiscale


def _baseline_drive(cfg):
    drive = cfg.metadata.get("drive", {})
    return drive.get("baseline_drive_by_cell_type") if isinstance(drive, dict) else None


def test_multiscale_config_declares_no_baseline_drive():
    cfg = multiscale.build_config()
    assert set(cfg.metadata), "fixture: the config carries metadata"
    assert _baseline_drive(cfg) is None


def test_heterogeneous_izh_config_declares_no_baseline_drive():
    cfg = hetero.izh_config()
    assert set(cfg.metadata), "fixture: the config carries metadata"
    assert _baseline_drive(cfg) is None
