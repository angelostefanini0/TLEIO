"""Build TLEIO triplet measurements for the clone-based EKF update.

This file converts the transformer's raw `2 x 3` output into the stacked EKF
objects that the filter needs: normalized relative poses, a minimal 6D
residual, sparse clone-only Jacobians, and a single joint measurement
covariance for the `(1 -> 2, 2 -> 3)` update.
"""

import numpy as np
from filter.utils.math_utils import hat,enforce_symmetry_and_pos_def


def extract_raw_triplet_measurement(network_output):
    """Extract the raw `2 x 3` relative-pose means from a flexible input format.

    The filter accepts either a bare NumPy array or a dictionary so that the
    runner can stay simple while the learned model interface is still evolving.
    """
    # Handles a dictionary
    if isinstance(network_output, dict):
        for key in ("relative_pose", "rel_pose", "poses", "mean", "mean_2x3"):
            if key in network_output and network_output[key] is not None:
                raw = network_output[key]
                break
        else:
            raise KeyError(
                "network_output must provide a `2x3` relative-pose mean under "
                "one of: relative_pose, rel_pose, poses, mean, mean_2x3."
            )
    else:  # Handles a NumPy array
        raw = network_output
    # Flattened 6D arrays are reshaped into 2 rows (one for each edge), 3 columns (x, y, z)
    raw = np.asarray(raw, dtype=float)
    if raw.shape == (6,):
        raw = raw.reshape(2, 3)
    if raw.shape != (2, 3):
        raise ValueError(
            f"Expected raw relative-pose means with shape (2, 3), got {raw.shape}."
        )
    return raw.copy()


def normalize_triplet_measurement(raw_measurement):
    """Normalize the two output quaternions in-place and return a clean copy."""

    measurement = np.asarray(raw_measurement, dtype=float).copy()
    # Iterate over the two edges in the triplet
    for idx in range(2):
        # Extract the quaternion and normalize
        quat = measurement[idx, 3:7]
        quat_norm = np.linalg.norm(quat)
        # Prevent division by zero
        if quat_norm < 1e-12:
            raise ValueError(f"Quaternion {idx} has near-zero norm and cannot be normalized.")
        # Normalize to ensure it represents a valid SO(3) rotation
        measurement[idx, 3:7] = quat / quat_norm
    return measurement


def extract_joint_measurement_covariance(network_output):
    """Extract an optional joint 6x6 covariance from the network output.

    The EKF update operates in the stacked 6D residual space, so the provided
    covariance is expected to already live in that space.
    """

    if not isinstance(network_output, dict):
        return None

    covariance = None
    for key in ("joint_covariance", "joint_cov", "covariance", "cov6", "R"):
        if key in network_output and network_output[key] is not None:
            covariance = network_output[key]
            break

    if covariance is None:
        return None

    covariance = np.asarray(covariance, dtype=float)
    # Reshape 1D arrays of length 36 into 6x6 matrices
    if covariance.shape == (36,):
        covariance = covariance.reshape(6, 6)
    if covariance.shape != (6, 6):
        raise ValueError(
            f"Expected a joint measurement covariance with shape (6, 6), got {covariance.shape}."
        )
    return covariance


def make_default_joint_covariance(sigma_translation):
    """Build a conservative block-diagonal joint covariance for both pose edges."""
    # Create a 3x3 diagonal matrix for a single edge
    pair_cov = np.diag(
        [sigma_translation**2] * 3 )
    # Stacks two 3x3 matrices in a 6x6 joint covariance matrix
    joint_cov = np.zeros((6, 6), dtype=float)
    joint_cov[0:3, 0:3] = pair_cov
    joint_cov[3:6, 3:6] = pair_cov
    return joint_cov

def predict_relative_pose(R_i, p_i, R_j, p_j):
    """Predict the clone-to-clone relative pose in clone `i`'s body frame.
    
    Args:
        R_i, p_i: World-to-body rotation and world position of clone i.
        R_j, p_j: World-to-body rotation and world position of clone j.
    """
    # Delta position in world frame
    delta_p = p_j - p_i 
    # Translate world delta into the local frame of clone i
    t_hat = R_i.T @ delta_p 
    # Relative rotation from i to j
    R_hat = R_i.T @ R_j
    
    return t_hat, R_hat, delta_p


def build_pair_residual_and_local_jacobian(R_i, p_i, R_j, p_j, measurement):
    """Build one 3D residual and its local 3x12 Jacobian for clones `(i, j)`.

    The local Jacobian is ordered as:
    `(delta_theta_i, delta_p_i, delta_theta_j, delta_p_j)`.

    The rotation residual is defined as `log(R_hat^T R_meas)` so that it follows
    the same correction sign convention as a standard `measurement - prediction`
    EKF innovation. With our left-multiplicative clone perturbations, this uses
    the inverse left Jacobian of SO(3).
    """
    # Extract the 3D measurements and get the current predictions
    t_meas = measurement[:3]
    t_hat, R_hat, delta_p = predict_relative_pose(R_i, p_i, R_j, p_j)
    #Calculate residual
    residual = t_meas - t_hat
    # Initialize the 3x12 Jacobian block mapping state errors to measurement errors
    local_jacobian = np.zeros((3, 12), dtype=float)
    # d(residual) / d(theta_i): Cross product matrix of predicted local translation
    local_jacobian[0:3, 0:3] = -hat(R_i.T @ delta_p)  
    # d(residual) / d(p_i): Identity rotated into frame i
    local_jacobian[0:3, 3:6] = R_i.T   
    # d(residual) / d(p_j): Negative identity rotated into frame i               
    local_jacobian[0:3, 9:12] = -R_i.T

    return residual, local_jacobian


def embed_pair_jacobian(local_jacobian, clone_i, clone_j, state_dim, imu_dim=15):
    """Embed one local 3x12 pairwise Jacobian into the global filter state."""
    # Create an empty Jacobian for the full state dimension (3 x state_dim )
    global_jacobian = np.zeros((3, state_dim), dtype=float)
    # Calculate starting column indices for clone i and clone j
    i0 = imu_dim + 6 * clone_i
    j0 = imu_dim + 6 * clone_j
    # Copy the local 3x12 blocks into their correct global positions
    global_jacobian[:, i0 : i0 + 3] = local_jacobian[:, 0:3] # delta_theta_i
    global_jacobian[:, i0 + 3 : i0 + 6] = local_jacobian[:, 3:6] # delta_p_i
    global_jacobian[:, j0 : j0 + 3] = local_jacobian[:, 6:9] # delta_theta_j
    global_jacobian[:, j0 + 3 : j0 + 6] = local_jacobian[:, 9:12] # delta_p_j

    return global_jacobian


def build_triplet_update(state, network_output, default_covariance, covariance_scale=1.0):
    """Build the stacked TLEIO measurement residual, Jacobian, and covariance.

    The filter expects exactly three clones because the learned triplet
    measurement corresponds to the consecutive clone pairs `(1 -> 2)` and
    `(2 -> 3)`.
    """
    # Enforce triplet constraint
    if len(state.clone_Rs) != 3 or len(state.clone_ps) != 3:
        raise ValueError(
            "TLEIO update requires exactly three clones before building the triplet measurement."
        )
    # Extract 2x3 measurement and 6x6 covariance
    measurement = extract_raw_triplet_measurement(network_output)
    covariance = extract_joint_measurement_covariance(network_output)

    residual_list, jacobians = [], []
    # Define the two edges: 0->1 and 1->2
    pair_specs = ((0, 1, measurement[0]), (1, 2, measurement[1]))

    state_dim = state.P.shape[0]
    # Loop over both edges to build and stack the local matrices
    for clone_i, clone_j, measurement_7d in pair_specs:
        residual_, local_jacobian = build_pair_residual_and_local_jacobian(
            state.clone_Rs[clone_i],
            state.clone_ps[clone_i],
            state.clone_Rs[clone_j],
            state.clone_ps[clone_j],
            measurement_7d,
        )
        # Store the 3D residual block
        residual_list.append(residual_)
        # Expand the local 3x12 Jacobian to 3xN and store it
        jacobians.append(embed_pair_jacobian(local_jacobian, clone_i, clone_j, state_dim))
    # Stack the two 3D residuals into a single 6D vector
    residual = np.concatenate(residual_list, axis=0)
    # Stack the two 3xN Jacobians into a single 6xN matrix
    jacobian = np.vstack(jacobians)
    # Fallback to default block-diagonal covariance
    measurement_covariance = default_covariance if covariance is None else covariance
    # Scale the covariance
    measurement_covariance = covariance_scale * measurement_covariance
    # Enforce symmetry
    measurement_covariance = enforce_symmetry_and_pos_def(measurement_covariance, epsilon=0.0)
    return residual, jacobian, measurement_covariance
