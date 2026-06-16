from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from filter.measurement_triplet import build_triplet_update, make_default_joint_covariance
from filter.scekf import ImuMSCKF
from filter.utils.math_utils import repair_covariance


def _args(**overrides):
    base = dict(
        sigma_na=0.01,
        sigma_ng=0.001,
        sigma_nba=1e-4,
        sigma_nbg=1e-5,
        sigma_rel_t=0.05,
        meas_cov_scale=1.0,
        chi2_confidence=0.95,
        chi2_multiplier=1.0,
        enable_chi2_gating=True,
        use_fej=False,
        use_block_update=False,
        covariance_repair_mode="jitter",
        imu_noise_model="discrete",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _clone_rotations():
    return [
        Rotation.from_euler("xyz", [0.00, 0.00, 0.00]).as_matrix(),
        Rotation.from_euler("xyz", [0.05, -0.02, 0.01]).as_matrix(),
        Rotation.from_euler("xyz", [-0.03, 0.04, 0.02]).as_matrix(),
        Rotation.from_euler("xyz", [0.02, 0.03, -0.04]).as_matrix(),
        Rotation.from_euler("xyz", [-0.01, 0.02, 0.05]).as_matrix(),
    ]


def _clone_positions():
    return [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.2, 0.0, 0.0]),
        np.array([0.4, 0.1, 0.0]),
        np.array([0.7, 0.1, 0.2]),
        np.array([1.0, 0.2, 0.2]),
    ]


def _setup_filter(**overrides):
    ekf = ImuMSCKF(_args(**overrides))
    rotations = _clone_rotations()
    positions = _clone_positions()
    ekf.state.clone_Rs = [R.copy() for R in rotations]
    ekf.state.clone_ps = [p.copy() for p in positions]
    ekf.state.clone_Rs_fej = [R.copy() for R in rotations]
    ekf.state.clone_ps_fej = [p.copy() for p in positions]
    ekf.state.R = rotations[-1].copy()
    ekf.state.p = positions[-1].copy()
    ekf.state.P = np.eye(45) * 0.05
    return ekf


def _perfect_measurement(ekf):
    return np.vstack(
        [
            ekf.state.clone_Rs[i].T @ (ekf.state.clone_ps[i + 1] - ekf.state.clone_ps[i])
            for i in range(4)
        ]
    )


def test_chi2_threshold_matches_residual_dimension():
    ekf = _setup_filter()
    info = ekf.update(
        {
            "relative_pose": _perfect_measurement(ekf),
            "joint_covariance": make_default_joint_covariance(0.05),
        }
    )

    assert info["residual_dim"] == 12
    assert info["chi2_threshold"] == pytest.approx(21.0260698, rel=1e-6)


def test_gating_rejects_large_residual():
    ekf = _setup_filter()
    measurement = _perfect_measurement(ekf) + 10.0
    info = ekf.update(
        {
            "relative_pose": measurement,
            "joint_covariance": np.eye(12) * 1e-4,
        }
    )

    assert info["rejected"]
    assert info["delta_x"] is None
    assert info["mahalanobis_sq"] > info["chi2_threshold"]


def test_gating_can_be_disabled():
    ekf = _setup_filter(enable_chi2_gating=False)
    measurement = _perfect_measurement(ekf) + 10.0
    info = ekf.update(
        {
            "relative_pose": measurement,
            "joint_covariance": np.eye(12) * 1e-4,
        }
    )

    assert not info["rejected"]
    assert info["delta_x"] is not None
    assert info["mahalanobis_sq"] > info["chi2_threshold"]


def test_update_returns_gating_diagnostics():
    ekf = _setup_filter()
    info = ekf.update(
        {
            "relative_pose": _perfect_measurement(ekf),
            "joint_covariance": make_default_joint_covariance(0.05),
        }
    )

    for key in ("mahalanobis_sq", "chi2_threshold", "residual_dim", "rejected"):
        assert key in info


def test_clone_fej_values_created_on_augmentation():
    ekf = ImuMSCKF(_args())
    ekf.state.R = Rotation.from_euler("xyz", [0.1, 0.2, -0.1]).as_matrix()
    ekf.state.p = np.array([1.0, 2.0, 3.0])
    ekf.augment_clone()

    np.testing.assert_allclose(ekf.state.clone_Rs_fej[0], ekf.state.clone_Rs[0])
    np.testing.assert_allclose(ekf.state.clone_ps_fej[0], ekf.state.clone_ps[0])


def test_fej_values_do_not_change_after_update_injection():
    ekf = _setup_filter()
    R_fej_before = [R.copy() for R in ekf.state.clone_Rs_fej]
    p_fej_before = [p.copy() for p in ekf.state.clone_ps_fej]
    delta = np.zeros(45)
    delta[15:18] = np.array([0.01, -0.02, 0.03])
    delta[18:21] = np.array([0.1, 0.0, 0.0])

    ekf._inject_error_state(delta)

    for before, after in zip(R_fej_before, ekf.state.clone_Rs_fej):
        np.testing.assert_allclose(after, before)
    for before, after in zip(p_fej_before, ekf.state.clone_ps_fej):
        np.testing.assert_allclose(after, before)
    assert not np.allclose(ekf.state.clone_Rs[0], R_fej_before[0])


def test_marginalization_keeps_fej_and_nominal_clone_lists_aligned():
    ekf = _setup_filter()
    ekf.marginalize_oldest_clone()

    assert len(ekf.state.clone_Rs) == 4
    assert len(ekf.state.clone_Rs_fej) == 4
    np.testing.assert_allclose(ekf.state.clone_Rs_fej[0], _clone_rotations()[1])


def test_covariance_check_rejects_large_negative_diagonal_in_strict_mode():
    P = np.eye(3)
    P[0, 0] = -1e-2
    with pytest.raises(ValueError, match="strict mode"):
        repair_covariance(P, mode="strict", name="bad_covariance")


def test_jitter_mode_repairs_tiny_negative_eigenvalue():
    P = np.diag([-1e-12, 1.0, 2.0])
    repaired, diagnostics = repair_covariance(P, mode="jitter", epsilon=1e-9)

    assert diagnostics["repair_applied"]
    assert np.min(np.linalg.eigvalsh(repaired)) == pytest.approx(1e-9)


def test_update_does_not_require_nontrivial_covariance_repair_on_nominal_case():
    ekf = _setup_filter()
    info = ekf.update(
        {
            "relative_pose": _perfect_measurement(ekf),
            "joint_covariance": make_default_joint_covariance(0.05),
        }
    )

    assert not info["rejected"]
    update_diagnostics = [d for d in ekf.covariance_diagnostics if d["name"] == "update"]
    assert update_diagnostics
    assert not update_diagnostics[-1]["repair_applied"]


def test_block_update_matches_dense_update_for_random_psd_covariance():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(18, 18))
    P = A @ A.T + np.eye(18) * 0.1
    H = np.zeros((6, 18))
    H[:, 6:15] = rng.normal(size=(6, 9))
    R = np.eye(6) * 0.2

    K_dense, S_dense, _ = ImuMSCKF._compute_dense_kalman_gain(P, H, R)
    K_block, S_block, touched = ImuMSCKF._compute_block_kalman_gain(P, H, R)

    np.testing.assert_allclose(S_block, S_dense, atol=1e-10)
    np.testing.assert_allclose(K_block, K_dense, atol=1e-10)
    np.testing.assert_array_equal(touched, np.arange(6, 15))


def test_block_update_matches_dense_update_for_triplet_jacobian():
    ekf = _setup_filter()
    residual, H, R = build_triplet_update(
        ekf.state,
        {"relative_pose": _perfect_measurement(ekf)},
        make_default_joint_covariance(0.05),
    )

    K_dense, S_dense, _ = ImuMSCKF._compute_dense_kalman_gain(ekf.state.P, H, R)
    K_block, S_block, touched = ImuMSCKF._compute_block_kalman_gain(ekf.state.P, H, R)

    assert residual.shape == (12,)
    assert touched.min() >= 15
    np.testing.assert_allclose(S_block, S_dense, atol=1e-10)
    np.testing.assert_allclose(K_block, K_dense, atol=1e-10)


def test_whitened_update_matches_innovation_update_for_identity_R():
    ekf_innovation = _setup_filter(enable_chi2_gating=False, update_solve_method="innovation")
    ekf_whitened = _setup_filter(enable_chi2_gating=False, update_solve_method="whitened")
    measurement = _perfect_measurement(ekf_innovation) + 0.01
    covariance = np.eye(12)

    info_innovation = ekf_innovation.update(
        {"relative_pose": measurement, "joint_covariance": covariance}
    )
    info_whitened = ekf_whitened.update(
        {"relative_pose": measurement, "joint_covariance": covariance}
    )

    np.testing.assert_allclose(info_whitened["delta_x"], info_innovation["delta_x"], atol=1e-10)
    assert info_whitened["whitening_applied"]


def test_whitened_update_matches_innovation_update_for_diagonal_R():
    ekf_innovation = _setup_filter(enable_chi2_gating=False, update_solve_method="innovation")
    ekf_whitened = _setup_filter(enable_chi2_gating=False, update_solve_method="whitened")
    measurement = _perfect_measurement(ekf_innovation) + 0.01
    covariance = np.diag(np.linspace(0.01, 0.2, 12))

    info_innovation = ekf_innovation.update(
        {"relative_pose": measurement, "joint_covariance": covariance}
    )
    info_whitened = ekf_whitened.update(
        {"relative_pose": measurement, "joint_covariance": covariance}
    )

    np.testing.assert_allclose(info_whitened["delta_x"], info_innovation["delta_x"], atol=1e-10)
    np.testing.assert_allclose(
        info_whitened["mahalanobis_sq"],
        info_innovation["mahalanobis_sq"],
        atol=1e-10,
    )


def test_whitened_update_handles_extreme_regressed_sigmas():
    ekf = _setup_filter(enable_chi2_gating=False, update_solve_method="whitened")
    measurement = _perfect_measurement(ekf) + 1e-4
    sigmas = np.geomspace(1e-4, 1.0, 12)
    covariance = np.diag(sigmas**2)

    info = ekf.update({"relative_pose": measurement, "joint_covariance": covariance})

    assert not info["rejected"]
    assert np.isfinite(info["delta_x"]).all()
    assert np.isfinite(info["condition_number_R"])
    assert np.isfinite(info["condition_number_S"])


def test_condition_number_diagnostics_are_returned():
    ekf = _setup_filter(update_solve_method="whitened")
    info = ekf.update(
        {
            "relative_pose": _perfect_measurement(ekf),
            "joint_covariance": make_default_joint_covariance(0.05),
        }
    )

    assert "condition_number_R" in info
    assert "condition_number_S" in info
    assert "whitening_applied" in info
