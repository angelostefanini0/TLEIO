"""Build TLEIO four-edge translation measurements for the clone-based EKF.

This module converts the transformer's raw `4 x 3` relative-translation output
into the stacked EKF objects used by the filter: a 12D residual, sparse
clone-only Jacobians, and one joint `12 x 12` measurement covariance for the
consecutive clone edges `(0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4)`.
"""

import numpy as np
from filter.utils.math_utils import hat,enforce_symmetry_and_pos_def


def extract_raw_triplet_measurement(network_output):
    """Extract raw `4 x 3` relative-translation means from flexible input.

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
                "network_output must provide a `4x3` relative-translation mean under "
                "one of: relative_pose, rel_pose, poses, mean, mean_2x3."
            )
    else:  # Handles a NumPy array
        raw = network_output
    # Flattened 12D arrays are reshaped into 4 rows (one for each edge), 3 columns (x, y, z)
    raw = np.asarray(raw, dtype=float)
    if raw.shape == (12,):
        raw = raw.reshape(4, 3)
    if raw.shape != (4, 3):
        raise ValueError(
            f"Expected raw relative-pose means with shape (4, 3), got {raw.shape}."
        )
    return raw.copy()


def extract_joint_measurement_covariance(network_output):
    """Extract an optional joint 12x12 covariance from the network output.

    The EKF update operates in the stacked 12D residual space, so the provided
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
    # Reshape 1D arrays of length 144 into 12x12 matrices
    if covariance.shape == (144,):
        covariance = covariance.reshape(12, 12)
    if covariance.shape != (12, 12):
        raise ValueError(
            f"Expected a joint measurement covariance with shape (12, 12), got {covariance.shape}."
        )
    return covariance


def make_default_joint_covariance(sigma_translation):
    """Build a block-diagonal joint covariance for all four translation edges."""
    # Create a 3x3 diagonal matrix for a single edge
    pair_cov = np.diag(
        [sigma_translation**2] * 3 )
    # Stacks four 3x3 matrices in a 12x12 joint covariance matrix
    joint_cov = np.zeros((12, 12), dtype=float)
    joint_cov[0:3, 0:3] = pair_cov
    joint_cov[3:6, 3:6] = pair_cov
    joint_cov[6:9, 6:9] = pair_cov
    joint_cov[9:12, 9:12] = pair_cov
    return joint_cov

def predict_relative_pose(R_i, p_i, R_j, p_j):
    """Predict the clone-to-clone relative translation in clone `i`'s body frame.
    
    Args:
        R_i, p_i: World-to-body rotation and world position of clone i.
        R_j, p_j: World-to-body rotation and world position of clone j.
    """
    # Delta position in world frame
    delta_p = p_j - p_i 
    # Translate world delta into the local frame of clone i
    t_hat = R_i.T @ delta_p 
    R_hat = R_i.T @ R_j
    
    return t_hat, R_hat, delta_p


def build_pair_residual_and_local_jacobian(
    R_i,
    p_i,
    R_j,
    p_j,
    measurement,
    jacobian_R_i=None,
    jacobian_p_i=None,
    jacobian_p_j=None,
):
    """Build one 3D residual and its local 3x12 Jacobian for clones `(i, j)`.

    The local Jacobian is ordered as:
    `(delta_theta_i, delta_p_i, delta_theta_j, delta_p_j)`.

    The residual is always evaluated at the current clone poses. When FEJ is
    enabled, the optional Jacobian poses are first-estimate clone poses used
    only for the Jacobian blocks.
    """
    # Extract the 3D measurements and get the current predictions
    t_meas = measurement[:3]
    t_hat, R_hat, delta_p = predict_relative_pose(R_i, p_i, R_j, p_j)
    #Calculate residual
    residual = t_meas - t_hat

    if jacobian_R_i is None:
        jacobian_R_i = R_i
    if jacobian_p_i is None:
        jacobian_p_i = p_i
    if jacobian_p_j is None:
        jacobian_p_j = p_j
    jacobian_delta_p = jacobian_p_j - jacobian_p_i
    # Initialize the 3x12 Jacobian block mapping state errors to measurement errors
    local_jacobian = np.zeros((3, 12), dtype=float)
    # d(residual) / d(theta_i): Cross product matrix of predicted local translation
    local_jacobian[0:3, 0:3] = -hat(jacobian_R_i.T @ jacobian_delta_p)
    # d(residual) / d(p_i): Identity rotated into frame i
    local_jacobian[0:3, 3:6] = jacobian_R_i.T
    # d(residual) / d(p_j): Negative identity rotated into frame i               
    local_jacobian[0:3, 9:12] = -jacobian_R_i.T

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


def build_triplet_update(state, network_output, default_covariance, covariance_scale=1.0, use_fej=False):
    """Build the stacked TLEIO measurement residual, Jacobian, and covariance.

    The filter expects exactly five clones because the learned triplet
    measurement corresponds to the consecutive clone pairs `(0 -> 1)`,
    `(1 -> 2)`, `(2 -> 3)`, and `(3 -> 4)`.
    """
    # Enforce triplet constraint
    if len(state.clone_Rs) != 5 or len(state.clone_ps) != 5:
        raise ValueError(
            "TLEIO update requires exactly five clones before building the triplet measurement."
        )
    if use_fej:
        if not hasattr(state, "clone_Rs_fej") or not hasattr(state, "clone_ps_fej"):
            raise AttributeError("FEJ update requested, but the state does not store FEJ clones.")
        if len(state.clone_Rs_fej) != 5 or len(state.clone_ps_fej) != 5:
            raise ValueError("FEJ clone lists must contain exactly five clones.")
        jacobian_Rs = state.clone_Rs_fej
        jacobian_ps = state.clone_ps_fej
    else:
        jacobian_Rs = state.clone_Rs
        jacobian_ps = state.clone_ps
    # Extract 4x3 measurement and 12x12 covariance
    measurement = extract_raw_triplet_measurement(network_output)
    covariance = extract_joint_measurement_covariance(network_output)

    residual_list, jacobians = [], []
    # Define the four consecutive translation edges.
    pair_specs = ((0, 1, measurement[0]), (1, 2, measurement[1]),(2,3,measurement[2]),(3,4,measurement[3]))

    state_dim = state.P.shape[0]
    # Loop over both edges to build and stack the local matrices
    for clone_i, clone_j, measurement_7d in pair_specs:
        residual_, local_jacobian = build_pair_residual_and_local_jacobian(
            state.clone_Rs[clone_i],
            state.clone_ps[clone_i],
            state.clone_Rs[clone_j],
            state.clone_ps[clone_j],
            measurement_7d,
            jacobian_R_i=jacobian_Rs[clone_i],
            jacobian_p_i=jacobian_ps[clone_i],
            jacobian_p_j=jacobian_ps[clone_j],
        )
        # Store the 3D residual block
        residual_list.append(residual_)
        # Expand the local 3x12 Jacobian to 3xN and store it
        jacobians.append(embed_pair_jacobian(local_jacobian, clone_i, clone_j, state_dim))
    # Stack the four 3D residuals into a single 12D vector
    residual = np.concatenate(residual_list, axis=0)
    # Stack the four 3xN Jacobians into a single 12xN matrix
    jacobian = np.vstack(jacobians)
    # Fallback to default block-diagonal covariance
    measurement_covariance = default_covariance if covariance is None else covariance
    # Scale the covariance
    measurement_covariance = covariance_scale * measurement_covariance
    # Enforce symmetry
    measurement_covariance = enforce_symmetry_and_pos_def(measurement_covariance, epsilon=0.0)
    return residual, jacobian, measurement_covariance
