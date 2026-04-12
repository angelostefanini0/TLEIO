"""Build TLEIO triplet measurements for the clone-based EKF update.

This file converts the transformer's raw `2 x 7` output into the stacked EKF
objects that the filter needs: normalized relative poses, a minimal 12D
residual, sparse clone-only Jacobians, and a single joint measurement
covariance for the `(1 -> 2, 2 -> 3)` update.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from filter.utils.math_utils import Jl_SO3_inv, hat, mat_log


def extract_raw_triplet_measurement(network_output):
    """Extract the raw `2 x 7` relative-pose means from a flexible input format.

    The filter accepts either a bare NumPy array or a dictionary so that the
    runner can stay simple while the learned model interface is still evolving.
    """

    if isinstance(network_output, dict):
        for key in ("relative_pose", "rel_pose", "poses", "mean", "mean_2x7"):
            if key in network_output and network_output[key] is not None:
                raw = network_output[key]
                break
        else:
            raise KeyError(
                "network_output must provide a `2x7` relative-pose mean under "
                "one of: relative_pose, rel_pose, poses, mean, mean_2x7."
            )
    else:
        raw = network_output

    raw = np.asarray(raw, dtype=float)
    if raw.shape == (14,):
        raw = raw.reshape(2, 7)
    if raw.shape != (2, 7):
        raise ValueError(
            f"Expected raw relative-pose means with shape (2, 7), got {raw.shape}."
        )
    return raw.copy()


def normalize_triplet_measurement(raw_measurement):
    """Normalize the two output quaternions in-place and return a clean copy."""

    measurement = np.asarray(raw_measurement, dtype=float).copy()
    for idx in range(2):
        quat = measurement[idx, 3:7]
        quat_norm = np.linalg.norm(quat)
        if quat_norm < 1e-12:
            raise ValueError(f"Quaternion {idx} has near-zero norm and cannot be normalized.")
        measurement[idx, 3:7] = quat / quat_norm
    return measurement


def extract_joint_measurement_covariance(network_output):
    """Extract an optional joint 12x12 covariance from the network output.

    The EKF update operates in the stacked 12D residual space, so the provided
    covariance is expected to already live in that space.
    """

    if not isinstance(network_output, dict):
        return None

    covariance = None
    for key in ("joint_covariance", "joint_cov", "covariance", "cov12", "R"):
        if key in network_output and network_output[key] is not None:
            covariance = network_output[key]
            break

    if covariance is None:
        return None

    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape == (144,):
        covariance = covariance.reshape(12, 12)
    if covariance.shape != (12, 12):
        raise ValueError(
            f"Expected a joint measurement covariance with shape (12, 12), got {covariance.shape}."
        )
    return covariance


def make_default_joint_covariance(sigma_translation, sigma_rotation):
    """Build a conservative block-diagonal joint covariance for both pose edges."""

    pair_cov = np.diag(
        [sigma_translation**2] * 3 + [sigma_rotation**2] * 3
    )
    joint_cov = np.zeros((12, 12), dtype=float)
    joint_cov[0:6, 0:6] = pair_cov
    joint_cov[6:12, 6:12] = pair_cov
    return joint_cov


def predict_relative_pose(R_i, p_i, R_j, p_j):
    """Predict the clone-to-clone relative pose in clone `i`'s body frame."""

    delta_p = p_j - p_i
    t_hat = R_i.T @ delta_p
    R_hat = R_i.T @ R_j
    return t_hat, R_hat, delta_p


def build_pair_residual_and_local_jacobian(R_i, p_i, R_j, p_j, measurement_7d):
    """Build one 6D residual and its local 6x12 Jacobian for clones `(i, j)`.

    The local Jacobian is ordered as:
    `(delta_theta_i, delta_p_i, delta_theta_j, delta_p_j)`.

    The rotation residual is defined as `log(R_hat^T R_meas)` so that it follows
    the same correction sign convention as a standard `measurement - prediction`
    EKF innovation. With our left-multiplicative clone perturbations, this uses
    the inverse left Jacobian of SO(3).
    """

    t_meas = measurement_7d[:3]
    R_meas = Rotation.from_quat(measurement_7d[3:7]).as_matrix()

    t_hat, R_hat, delta_p = predict_relative_pose(R_i, p_i, R_j, p_j)

    residual_t = t_meas - t_hat
    residual_R = mat_log( R_hat.T @ R_meas )
    residual = np.concatenate([residual_t, residual_R], axis=0)

    local_jacobian = np.zeros((6, 12), dtype=float)

    local_jacobian[0:3, 0:3] = -hat(R_i.T @ delta_p)
    local_jacobian[0:3, 3:6] = R_i.T
    local_jacobian[0:3, 9:12] = -R_i.T

    left_jacobian_inv = Jl_SO3_inv(residual_R)
    local_jacobian[3:6, 0:3] = left_jacobian_inv @ R_j.T @ R_i
    local_jacobian[3:6, 6:9] = -left_jacobian_inv

    return residual, local_jacobian


def embed_pair_jacobian(local_jacobian, clone_i, clone_j, state_dim, imu_dim=15):
    """Embed one local 6x12 pairwise Jacobian into the global filter state."""

    global_jacobian = np.zeros((6, state_dim), dtype=float)

    i0 = imu_dim + 6 * clone_i
    j0 = imu_dim + 6 * clone_j

    global_jacobian[:, i0 : i0 + 3] = local_jacobian[:, 0:3]
    global_jacobian[:, i0 + 3 : i0 + 6] = local_jacobian[:, 3:6]
    global_jacobian[:, j0 : j0 + 3] = local_jacobian[:, 6:9]
    global_jacobian[:, j0 + 3 : j0 + 6] = local_jacobian[:, 9:12]

    return global_jacobian


def build_triplet_update(state, network_output, default_covariance, covariance_scale=1.0):
    """Build the stacked TLEIO measurement residual, Jacobian, and covariance.

    The filter expects exactly three clones because the learned triplet
    measurement corresponds to the consecutive clone pairs `(1 -> 2)` and
    `(2 -> 3)`.
    """

    if len(state.clone_Rs) != 3 or len(state.clone_ps) != 3:
        raise ValueError(
            "TLEIO update requires exactly three clones before building the triplet measurement."
        )

    raw_measurement = extract_raw_triplet_measurement(network_output)
    measurement = normalize_triplet_measurement(raw_measurement)
    covariance = extract_joint_measurement_covariance(network_output)

    residual_12, jacobians = [], []
    pair_specs = ((0, 1, measurement[0]), (1, 2, measurement[1]))

    state_dim = state.P.shape[0]
    for clone_i, clone_j, measurement_7d in pair_specs:
        residual_6, local_jacobian = build_pair_residual_and_local_jacobian(
            state.clone_Rs[clone_i],
            state.clone_ps[clone_i],
            state.clone_Rs[clone_j],
            state.clone_ps[clone_j],
            measurement_7d,
        )
        residual_12.append(residual_6)
        jacobians.append(embed_pair_jacobian(local_jacobian, clone_i, clone_j, state_dim))

    residual = np.concatenate(residual_12, axis=0)
    jacobian = np.vstack(jacobians)
    measurement_covariance = default_covariance if covariance is None else covariance

    measurement_covariance = covariance_scale * measurement_covariance
    measurement_covariance = 0.5 * (measurement_covariance + measurement_covariance.T)
    return residual, jacobian, measurement_covariance
