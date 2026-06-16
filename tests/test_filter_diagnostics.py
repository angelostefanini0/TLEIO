import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from filter_diagnostics import save_consistency_diagnostics


def test_consistency_diagnostics_runs_without_covariance_snapshots(tmp_path):
    path = save_consistency_diagnostics(
        tmp_path / "consistency_diagnostics.csv",
        np.array([0.0, 1.0]),
        np.zeros((2, 3)),
        np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (2, 1)),
        np.ones((2, 3)),
        np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (2, 1)),
    )

    assert path.exists()
    data = np.loadtxt(path, skiprows=1)
    assert data.shape == (2, 7)


def test_consistency_diagnostics_labels_approximate_nees(tmp_path):
    path = save_consistency_diagnostics(
        tmp_path / "consistency_diagnostics.csv",
        np.array([0.0]),
        np.zeros((1, 3)),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.zeros((1, 3)),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
    )

    text = path.read_text(encoding="utf-8")
    assert "approximate_nees_available" in text
    data = np.loadtxt(path, skiprows=1)
    assert data[-1] == 0.0
