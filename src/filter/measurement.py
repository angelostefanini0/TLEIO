"""Build TLEIO measurements for the clone-based EKF update.

This file converts the transformer's raw `4 x 3` output into the stacked EKF
objects that the filter needs: normalized relative poses, a minimal 12D
residual, sparse clone-only Jacobians, and a single joint measurement
covariance for the `(1 -> 2, 2 -> 3, 3 -> 4, 4 -> 5)` update.
"""

import numpy as np
from filter.utils.math_utils import hat,enforce_symmetry_and_pos_def


def extract_raw_measurement(network_output):
    """
    Extract the raw `4 x 3` relative-pose means from a the network output.
    """
    # Handles a dictionary
    raw = network_output["relative_pose"]

    # Flattened 12D arrays are reshaped into 4 rows (one for each edge), 3 columns (x, y, z)
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (4, 3):
        raise ValueError(f"Expected raw relative-pose means with shape (4, 3), got {raw.shape}.")
    return raw.copy()

def extract_joint_measurement_covariance(network_output):
    """Extract a joint 12x12 covariance from the network output.

    The EKF update operates in the stacked 12D residual space, so the provided
    covariance is expected to already live in that space.
    """

    covariance = network_output.get("joint_covariance")
    if covariance is None:
        print("Notice: no regressed covariance provided; falling back to the default measurement covariance.")
        return None

    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (12, 12):
        raise ValueError(f"Expected a joint measurement covariance with shape (12, 12), got {covariance.shape}.")
    return covariance
    


def make_default_joint_covariance(sigma_translation):
    """Build a conservative block-diagonal joint covariance for all pose edges."""
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

def predict_relative_pose(R_i, p_i, p_j):
    """Predict the clone-to-clone relative pose in clone `i`'s body frame.
    
    Args:
        R_i, p_i: World-to-body rotation and world position of clone i.
        R_j, p_j: World-to-body rotation and world position of clone j.
    """
    # Delta position in world frame
    delta_p = p_j - p_i 
    # Translate world delta into the local frame of clone i
    t_hat = R_i.T @ delta_p 
    
    return t_hat, delta_p


def build_pair_residual_and_local_jacobian(R_i, p_i, p_j, measurement):
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
    t_hat, delta_p = predict_relative_pose(R_i, p_i, p_j)
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


def embed_pair_jacobian(local_jacobian, clone_i, clone_j, state_dim, current_dim=15):
    """Embed one local 3x12 pairwise Jacobian into the global filter state."""
    # Create an empty Jacobian for the full state dimension (3 x state_dim )
    global_jacobian = np.zeros((3, state_dim), dtype=float)
    # Calculate starting column indices for clone i and clone j
    i0 =  6 * clone_i
    j0 =  6 * clone_j
    # Copy the local 3x12 blocks into their correct global positions
    global_jacobian[:, i0 : i0 + 3] = local_jacobian[:, 0:3] # delta_theta_i
    global_jacobian[:, i0 + 3 : i0 + 6] = local_jacobian[:, 3:6] # delta_p_i
    global_jacobian[:, j0 : j0 + 3] = local_jacobian[:, 6:9] # delta_theta_j
    global_jacobian[:, j0 + 3 : j0 + 6] = local_jacobian[:, 9:12] # delta_p_j

    return global_jacobian


def build_update(state, network_output, default_covariance, covariance_scale=1.0, network_scale=1.0):
    """Build the stacked TLEIO measurement residual, Jacobian, and covariance.

    The filter expects exactly five clones because the learned triplet
    measurement corresponds to the consecutive clone pairs `(1 -> 2)`,
    `(2 -> 3)`,`(3 -> 4)` and `(4 -> 5)`.
    """
    # Enforce clip constraint
    if len(state.clone_Rs) != 5 or len(state.clone_ps) != 5:
        raise ValueError("TLEIO update requires exactly five clones before building the clip measurement.")
    # Extract 4x3 measurement and 12x12 covariance
    measurement = extract_raw_measurement(network_output)
    # Scale the raw network output before it enters the residual computation
    measurement = network_scale * measurement
    covariance = extract_joint_measurement_covariance(network_output)

    residual_list, jacobians = [], []
    # Define the four edges: 0->1, 1->2, 2->3, 3->4
    pair_specs = ((0, 1, measurement[0]), (1, 2, measurement[1]),(2,3,measurement[2]),(3,4,measurement[3]))

    state_dim = state.P.shape[0]
    # Loop over both edges to build and stack the local matrices
    for clone_i, clone_j, meas in pair_specs:
        residual_, local_jacobian = build_pair_residual_and_local_jacobian(
            state.clone_Rs[clone_i],
            state.clone_ps[clone_i],
            state.clone_ps[clone_j],
            meas,
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
