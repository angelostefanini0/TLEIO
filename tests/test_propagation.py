from types import SimpleNamespace

import numpy as np
import pytest

from filter.imu_buffer import ImuInterval, ImuMeasurement
from filter.scekf import (
    ImuMSCKF,
    apply_summed_imu_covariance,
    propagate_covariance,
    propagate_covariance_components,
)


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


def test_propagate_intervals_matches_sample_dt_for_constant_imu():
    measurements = [
        ImuMeasurement(
            timestamp=(idx + 1) * 0.01,
            dt=0.01,
            accel=np.array([0.0, 0.0, 9.80665]),
            gyro=np.zeros(3),
        )
        for idx in range(10)
    ]
    intervals = [
        ImuInterval(
            t0=idx * 0.01,
            t1=(idx + 1) * 0.01,
            accel0=np.array([0.0, 0.0, 9.80665]),
            gyro0=np.zeros(3),
            accel1=np.array([0.0, 0.0, 9.80665]),
            gyro1=np.zeros(3),
        )
        for idx in range(10)
    ]
    ekf_samples = ImuMSCKF(_args())
    ekf_intervals = ImuMSCKF(_args())
    for ekf in (ekf_samples, ekf_intervals):
        ekf.g = np.array([0.0, 0.0, -9.80665])
        ekf.initialize_with_state(0.0, np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))

    ekf_samples.propagate(measurements)
    ekf_intervals.propagate_intervals(intervals)

    np.testing.assert_allclose(ekf_intervals.state.R, ekf_samples.state.R, atol=1e-10)
    np.testing.assert_allclose(ekf_intervals.state.v, ekf_samples.state.v, atol=1e-10)
    np.testing.assert_allclose(ekf_intervals.state.p, ekf_samples.state.p, atol=1e-10)
    np.testing.assert_allclose(ekf_intervals.state.P, ekf_samples.state.P, atol=1e-12)


def test_midpoint_matches_euler_for_constant_imu():
    intervals = [
        ImuInterval(
            t0=0.0,
            t1=0.1,
            accel0=np.array([0.0, 0.0, 9.80665]),
            gyro0=np.zeros(3),
            accel1=np.array([0.0, 0.0, 9.80665]),
            gyro1=np.zeros(3),
        )
    ]
    ekf_euler = ImuMSCKF(_args(nominal_integration_method="euler"))
    ekf_midpoint = ImuMSCKF(_args(nominal_integration_method="midpoint"))
    for ekf in (ekf_euler, ekf_midpoint):
        ekf.g = np.array([0.0, 0.0, -9.80665])
        ekf.initialize_with_state(0.0, np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))

    ekf_euler.propagate_intervals(intervals)
    ekf_midpoint.propagate_intervals(intervals)

    np.testing.assert_allclose(ekf_midpoint.state.R, ekf_euler.state.R, atol=1e-10)
    np.testing.assert_allclose(ekf_midpoint.state.v, ekf_euler.state.v, atol=1e-10)
    np.testing.assert_allclose(ekf_midpoint.state.p, ekf_euler.state.p, atol=1e-10)


def test_summed_covariance_matches_per_sample_for_short_constant_sequence():
    intervals = [
        ImuInterval(
            t0=idx * 0.01,
            t1=(idx + 1) * 0.01,
            accel0=np.array([0.0, 0.0, 9.80665]),
            gyro0=np.zeros(3),
            accel1=np.array([0.0, 0.0, 9.80665]),
            gyro1=np.zeros(3),
        )
        for idx in range(5)
    ]
    ekf_per_sample = ImuMSCKF(_args(covariance_propagation_mode="per_sample"))
    ekf_summed = ImuMSCKF(_args(covariance_propagation_mode="summed"))
    for ekf in (ekf_per_sample, ekf_summed):
        ekf.g = np.array([0.0, 0.0, -9.80665])
        ekf.initialize_with_state(0.0, np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))

    ekf_per_sample.propagate_intervals(intervals)
    ekf_summed.propagate_intervals(intervals)

    np.testing.assert_allclose(ekf_summed.state.P, ekf_per_sample.state.P, atol=1e-12)


def test_summed_covariance_preserves_cross_covariance_shape():
    P = np.eye(21) * 0.1
    Phi, Q = propagate_covariance_components(
        np.eye(3),
        np.zeros(3),
        np.array([0.0, 0.0, 9.80665]),
        0.01,
        sg=0.001,
        sa=0.01,
        sbg=1e-5,
        sba=1e-4,
    )

    P_next = apply_summed_imu_covariance(P, Phi, Q)

    assert P_next.shape == (21, 21)
    np.testing.assert_allclose(P_next, P_next.T, atol=1e-12)


def test_summed_noise_is_symmetric_psd():
    _, Q = propagate_covariance_components(
        np.eye(3),
        np.zeros(3),
        np.array([0.1, 0.2, 9.7]),
        0.02,
        sg=0.001,
        sa=0.01,
        sbg=1e-5,
        sba=1e-4,
    )

    np.testing.assert_allclose(Q, Q.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(Q)) >= -1e-15


def test_propagate_intervals_rejects_empty_interval_dt():
    ekf = ImuMSCKF(_args())
    interval = ImuInterval(
        t0=0.0,
        t1=0.0,
        accel0=np.zeros(3),
        gyro0=np.zeros(3),
        accel1=np.zeros(3),
        gyro1=np.zeros(3),
    )

    with pytest.raises(ValueError, match="non-positive dt"):
        ekf.propagate_intervals([interval])
