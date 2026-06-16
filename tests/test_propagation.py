from types import SimpleNamespace

import numpy as np

from filter.imu_buffer import ImuMeasurement
from filter.scekf import ImuMSCKF, propagate_covariance


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


def test_stationary_imu_nominal_state_stays_consistent():
    ekf = ImuMSCKF(_args())
    ekf.g = np.array([0.0, 0.0, -9.80665])
    ekf.initialize_with_state(
        t=0.0,
        R=np.eye(3),
        v=np.zeros(3),
        p=np.zeros(3),
        bg=np.zeros(3),
        ba=np.zeros(3),
    )
    measurements = [
        ImuMeasurement(
            timestamp=(idx + 1) * 0.01,
            dt=0.01,
            accel=np.array([0.0, 0.0, 9.80665]),
            gyro=np.zeros(3),
        )
        for idx in range(100)
    ]

    ekf.propagate(measurements)

    np.testing.assert_allclose(ekf.state.R, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(ekf.state.v, np.zeros(3), atol=1e-10)
    np.testing.assert_allclose(ekf.state.p, np.zeros(3), atol=1e-10)


def test_process_noise_covariance_growth_is_monotonic_for_continuous_model():
    P0 = np.zeros((15, 15))
    P1 = propagate_covariance(
        P0,
        np.eye(3),
        np.zeros(3),
        np.array([0.0, 0.0, 9.80665]),
        0.01,
        sg=0.001,
        sa=0.01,
        sbg=1e-5,
        sba=1e-4,
        imu_noise_model="continuous",
    )
    P2 = propagate_covariance(
        P1,
        np.eye(3),
        np.zeros(3),
        np.array([0.0, 0.0, 9.80665]),
        0.01,
        sg=0.001,
        sa=0.01,
        sbg=1e-5,
        sba=1e-4,
        imu_noise_model="continuous",
    )

    assert np.all(np.diag(P2) >= np.diag(P1) - 1e-15)
    assert np.all(np.diag(P2)[0:15] > 0.0)


def test_propagate_covariance_preserves_symmetry():
    P = np.eye(15) * 0.01
    P_next = propagate_covariance(
        P,
        np.eye(3),
        np.array([0.01, -0.02, 0.03]),
        np.array([0.1, 0.2, 9.7]),
        0.02,
        sg=0.001,
        sa=0.01,
        sbg=1e-5,
        sba=1e-4,
    )

    np.testing.assert_allclose(P_next, P_next.T, atol=1e-12)


def test_propagate_rejects_non_positive_dt():
    ekf = ImuMSCKF(_args())
    measurement = ImuMeasurement(
        timestamp=0.0,
        dt=0.0,
        accel=np.zeros(3),
        gyro=np.zeros(3),
    )

    try:
        ekf.propagate([measurement])
    except ValueError as exc:
        assert "non-positive dt" in str(exc)
    else:
        raise AssertionError("Expected non-positive dt to be rejected.")
