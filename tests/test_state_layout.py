from types import SimpleNamespace

import numpy as np
import pytest

from filter.scekf import ImuMSCKF, State


def _args(**overrides):
    base = dict(
        sigma_na=0.01,
        sigma_ng=0.001,
        sigma_nba=1e-4,
        sigma_nbg=1e-5,
        sigma_rel_t=0.05,
        meas_cov_scale=1.0,
        covariance_repair_mode="jitter",
        imu_noise_model="discrete",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_clone_slice_matches_covariance_layout():
    assert State.clone_slice(0) == slice(15, 21)
    assert State.clone_slice(2) == slice(27, 33)


def test_state_layout_helpers_reject_invalid_clone_index():
    with pytest.raises(ValueError, match="non-negative"):
        State.clone_slice(-1)


def test_covariance_shape_matches_clone_count_after_augmentation():
    ekf = ImuMSCKF(_args())

    ekf.augment_clone()
    ekf.augment_clone()

    assert ekf.state.expected_covariance_dim() == 27
    assert ekf.state.clone_count_from_covariance() == 2
    ekf.state.assert_covariance_shape_matches_clones()


def test_covariance_shape_matches_clone_count_after_marginalization():
    ekf = ImuMSCKF(_args())
    for _ in range(3):
        ekf.augment_clone()

    ekf.marginalize_oldest_clone()

    assert ekf.state.expected_covariance_dim() == 27
    assert ekf.state.clone_count_from_covariance() == 2
    ekf.state.assert_clone_fej_consistency()
    ekf.state.assert_covariance_shape_matches_clones()


def test_marginalization_preserves_remaining_cross_correlations():
    ekf = ImuMSCKF(_args())
    for clone_idx in range(3):
        ekf.state.p = np.array([clone_idx, 0.0, 0.0], dtype=np.float64)
        ekf.augment_clone()
    ekf.state.P[State.clone_slice(1), State.clone_slice(2)] = np.eye(6) * 0.3
    ekf.state.P[State.clone_slice(2), State.clone_slice(1)] = np.eye(6) * 0.3

    ekf.marginalize_oldest_clone()

    np.testing.assert_allclose(
        ekf.state.P[State.clone_slice(0), State.clone_slice(1)],
        np.eye(6) * 0.3,
        atol=1e-12,
    )
