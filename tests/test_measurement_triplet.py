from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from filter.measurement_triplet import (
    build_pair_residual_and_local_jacobian,
    build_triplet_update,
    make_default_joint_covariance,
)
from filter.utils.math_utils import mat_exp


def _make_state():
    rotations = [
        Rotation.from_euler("xyz", [0.10, -0.20, 0.05]).as_matrix(),
        Rotation.from_euler("xyz", [-0.05, 0.25, 0.20]).as_matrix(),
        Rotation.from_euler("xyz", [0.20, 0.10, -0.15]).as_matrix(),
        Rotation.from_euler("xyz", [-0.15, -0.05, 0.30]).as_matrix(),
        Rotation.from_euler("xyz", [0.05, 0.15, -0.25]).as_matrix(),
    ]
    positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.3, -0.2, 0.1]),
        np.array([0.8, 0.1, -0.1]),
        np.array([1.1, 0.5, 0.2]),
        np.array([1.5, 0.4, 0.3]),
    ]
    return SimpleNamespace(
        clone_Rs=[R.copy() for R in rotations],
        clone_ps=[p.copy() for p in positions],
        clone_Rs_fej=[R.copy() for R in rotations],
        clone_ps_fej=[p.copy() for p in positions],
        P=np.eye(45),
    )


def _perfect_measurement(state):
    return np.vstack(
        [
            state.clone_Rs[i].T @ (state.clone_ps[i + 1] - state.clone_ps[i])
            for i in range(4)
        ]
    )


def _copy_state_with_perturbation(state, clone_idx, delta):
    perturbed = SimpleNamespace(
        clone_Rs=[R.copy() for R in state.clone_Rs],
        clone_ps=[p.copy() for p in state.clone_ps],
        clone_Rs_fej=[R.copy() for R in state.clone_Rs_fej],
        clone_ps_fej=[p.copy() for p in state.clone_ps_fej],
        P=state.P.copy(),
    )
    perturbed.clone_Rs[clone_idx] = perturbed.clone_Rs[clone_idx] @ mat_exp(delta[:3])
    perturbed.clone_ps[clone_idx] = perturbed.clone_ps[clone_idx] + delta[3:6]
    return perturbed


def test_pair_translation_residual_zero_for_perfect_measurement():
    state = _make_state()
    measurement = _perfect_measurement(state)[0]

    residual, _ = build_pair_residual_and_local_jacobian(
        state.clone_Rs[0],
        state.clone_ps[0],
        state.clone_Rs[1],
        state.clone_ps[1],
        measurement,
    )

    np.testing.assert_allclose(residual, np.zeros(3), atol=1e-12)


def test_triplet_jacobian_matches_finite_difference():
    state = _make_state()
    measurement = _perfect_measurement(state)
    residual0, H, _ = build_triplet_update(
        state,
        {"relative_pose": measurement},
        make_default_joint_covariance(0.1),
    )

    eps = 1e-7
    for clone_idx in range(5):
        for local_col in range(6):
            delta = np.zeros(6)
            delta[local_col] = eps
            perturbed = _copy_state_with_perturbation(state, clone_idx, delta)
            residual_eps, _, _ = build_triplet_update(
                perturbed,
                {"relative_pose": measurement},
                make_default_joint_covariance(0.1),
            )
            numerical = (residual_eps - residual0) / eps
            analytic_col = H[:, 15 + 6 * clone_idx + local_col]
            np.testing.assert_allclose(numerical, analytic_col, atol=2e-6, rtol=2e-5)


def test_negative_kalman_step_sign_reduces_residual_in_linearized_case():
    state = _make_state()
    measurement = _perfect_measurement(state)
    measurement[0] += np.array([0.1, -0.05, 0.02])
    residual, H, R = build_triplet_update(
        state,
        {"relative_pose": measurement},
        make_default_joint_covariance(0.05),
    )
    P = np.eye(H.shape[1]) * 0.1
    S = H @ P @ H.T + R
    K = np.linalg.solve(S.T, (P @ H.T).T).T
    delta_x = -K @ residual
    residual_linearized = residual + H @ delta_x

    assert np.linalg.norm(residual_linearized) < np.linalg.norm(residual)


def test_fej_toggle_changes_jacobian_but_not_residual_after_clone_correction():
    state = _make_state()
    measurement = _perfect_measurement(state)
    correction = np.array([0.05, -0.02, 0.03, 0.1, -0.05, 0.02])
    state.clone_Rs[1] = state.clone_Rs[1] @ mat_exp(correction[:3])
    state.clone_ps[1] = state.clone_ps[1] + correction[3:6]

    residual_current, H_current, _ = build_triplet_update(
        state,
        {"relative_pose": measurement},
        make_default_joint_covariance(0.1),
        use_fej=False,
    )
    residual_fej, H_fej, _ = build_triplet_update(
        state,
        {"relative_pose": measurement},
        make_default_joint_covariance(0.1),
        use_fej=True,
    )

    np.testing.assert_allclose(residual_fej, residual_current)
    assert not np.allclose(H_fej, H_current)
