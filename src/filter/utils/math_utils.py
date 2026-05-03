# Copyright 2004-present Facebook. All Rights Reserved.

import contextlib
import warnings

import numpy as np

from .from_scipy import compute_q_from_matrix
from .quiet_numba import jit


def get_rotation_from_gravity(acc):
    """
    Computes a rotation matrix that aligns the world z-axis (gravity direction)
    with the measured acceleration vector.
    """
    # Take the first accel data to get gravity direction
    ig_w = np.array([0, 0, 1.0]).reshape((3, 1))
    return rot_2vec(acc, ig_w)


def inv_SE3(T):
    """
    Computes the exact inverse of a 4x4 SE(3) transformation matrix.
    Given T = [R, p; 0, 1], the inverse is [R.T, -R.T * p; 0, 1].
    """
    Tinv = np.eye(4)
    Tinv[:3,:3] = T[:3,:3].T
    Tinv[:3,3:4] = - T[:3,:3].T @ T[:3,3:4]
    return Tinv

def Jr_exp(v):
    """
    Right Jacobian of the SO(3) exponential map.
    Maps a small change in the tangent space (so(3)) to a change in the 
    rotation matrix space (SO(3)). Handles the singularity at theta = 0 
    using a Taylor series expansion.
    """
    theta = np.linalg.norm(v)
    K = hat(v)
    I = np.eye(3)
    # Taylor expansion for stability nearby 0
    if theta < 1e-6:
        return I - 0.5 * K + (1/6) * K @ K
    
    K2 = K @ K
    # Rodrigues formula
    return I - ((1 - np.cos(theta)) / (theta**2)) * K + ((theta - np.sin(theta)) / (theta**3)) * K2


def hat(v):
    """
    Maps a 3D vector to a 3x3 skew-symmetric matrix (so(3)).
    Used to represent cross products as matrix multiplications (a x b = hat(a) * b).
    """
    v = np.squeeze(v)
    R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return R

def vee(w_x):
    """
    The inverse of the hat operator.
    Maps a 3x3 skew-symmetric matrix back to a 3D vector.
    """
    return np.array([w_x[2,1], w_x[0,2], w_x[1,0]])

@jit(nopython=True, parallel=False, cache=True)
def rot_2vec(a, b):
    """
    Computes the shortest-path rotation matrix that rotates vector 'a' onto vector 'b'.
    """
    assert a.shape == (3, 1)
    assert b.shape == (3, 1)
    # Redefined inside to satisfy numba's nopython requirement
    def hat(v):
        v = v.flatten()
        R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return R
    # Normalize input vectors
    a_n = np.linalg.norm(a)
    b_n = np.linalg.norm(b)
    a_hat = a / a_n
    b_hat = b / b_n
    # Cross product gives the axis of rotation
    omega = np.cross(a_hat.T, b_hat.T).T
    # Dot product gives cos(theta)
    c = 1.0 / (1 + np.dot(a_hat.T, b_hat))
    # Rodrigues' rotation formula
    R_ba = np.eye(3) + hat(omega) + c * hat(omega) @ hat(omega)
    return R_ba


@jit(nopython=True, parallel=False, cache=True)
def mat_exp(omega):
    """
    Matrix exponential map from so(3) to SO(3).
    Converts a 3D rotation vector (axis-angle representation) into a 3x3 Rotation matrix.
    Uses Rodrigues' rotation formula.
    """
    if len(omega) != 3:
        raise ValueError("tangent vector must have length 3")

    def hat(v):
        v = v.flatten()
        R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return R

    angle = np.linalg.norm(omega)

    # Taylor expansion for stability nearby 0
    if angle < 1e-10:
        return np.identity(3) + hat(omega)

    axis = omega / angle
    s = np.sin(angle)
    c = np.cos(angle)
    # Rodrigues' rotation formula
    return c * np.identity(3) + (1 - c) * np.outer(axis, axis) + s * hat(axis)

# Create a vectorized version of mat_exp to handle batches of tangent vectors
mat_exp_vec = np.vectorize(mat_exp, signature="(3)->(3,3)")

def enforce_orthogonality(R):
    """
    Projects a nearly-orthogonal 3x3 matrix back onto the SO(3) manifold
    using Singular Value Decomposition (SVD). Ensures the matrix is a valid rotation.
    """
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    # Ensure determinant is +1
    if np.linalg.det(R_ortho) < 0:
        U[:, 2] *= -1
        R_ortho = U @ Vt
        
    return R_ortho


def mat_log(R):
    """
    Logarithmic map from SO(3) to so(3).
    Converts a 3x3 Rotation matrix into a 3D rotation vector (tangent space).
    Converts to a quaternion first for numerical stability.
    """
    q = np.array(compute_q_from_matrix(R))
    
    if q[3] < 0:
        q = -q
        
    w = q[3]
    vec = q[0:3]
    n = np.linalg.norm(vec)
    epsilon = 1e-7
    # Taylor expansion for stability nearby 0
    if n < epsilon:
        w2 = w * w
        n2 = n * n
        atn = 2.0 / w - (2.0 * n2) / (w * w2)
    else:
        if np.absolute(w) < epsilon:
            if w > 0:
                atn = np.pi / n
            else:
                atn = -np.pi / n
        else:
            # General case
            atn = 2.0 * np.arctan2(n, w) / n
    tangent = atn * vec
    return tangent


def mat_log_vec(R):
    """
    Vectorized version of mat_log.
    Args:
        R [n x 3 x 3]: Batch of rotation matrices.
    Returns:
        [n x 3]: Batch of rotation vectors.
    """

    q = compute_q_from_matrix(R)
    w = q[:, 3]
    vec = q[:, 0:3]
    n = np.linalg.norm(vec, axis=1)
    epsilon = 1e-7
    # Masks for numerical stability 
    mask = n < epsilon
    atn_small = 2.0 / w - (2.0 * n * n) / (w * w * w)

    mask2 = np.absolute(w) < epsilon
    atn_normal_small = np.sign(w) * np.pi / n
    atn_normal_normal = 2.0 * np.arctan2(n, w) / n
    # Combine limits using masks
    atn = mask2 * atn_normal_small + (1 - mask2) * atn_normal_normal
    atn = mask * atn_small + (1 - mask2) * atn

    tangent = atn[0, np.newaxis] * vec
    return tangent


def hat_SE3(v):
    """
    Aligns with the Sophus convention of the 6x1 v being
    in the block order: [log(translation) log(rotation)]
    """
    Ohat = np.zeros((4,4))
    Ohat[:3,:3] = hat(v[3:])
    Ohat[:3,3] = v[:3]
    return Ohat


def exp_SE3(v):
    """
    Aligns with the Sophus convention of the 6x1 v being
    in the block order: [log(translation) log(rotation)]
    """
    Exp = np.eye(4)
    Exp[:3,:3] = mat_exp(v[3:])
    Exp[:3,3:4] = Jl_SO3(v[3:]) @ v[:3,None]
    return Exp


def log_SE3(T):
    """
    Aligns with the Sophus convention of the returned 6x1 v being
    in the block order: [log(translation) log(rotation)]
    """
    w = mat_log(T[:3,:3])
    V_inv = Jl_SO3_inv(w)
    v = V_inv @ T[:3,3:4]
    return np.concatenate([v[:,0], w], 0)


""" right jacobian for exp operation on SO(3) """


@jit(nopython=True, parallel=False, cache=True)
def Jr_exp(phi):
    """
    Jitted version of the right Jacobian for the exponential map on SO(3).
    Relates small perturbations in the tangent space to the manifold.
    """
    def hat(v):
        v = v.flatten()
        R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return R

    theta = np.linalg.norm(phi)
    # Taylor expansion for stability nearby 0
    if theta < 1e-3:
        J = np.eye(3) - 0.5 * hat(phi) + 1.0 / 6.0 * (hat(phi) @ hat(phi))
    else:
        # Closed-form evaluation
        J = (
            np.eye(3)
            - (1 - np.cos(theta)) / np.power(theta, 2.0) * hat(phi)
            + (theta - np.sin(theta)) / np.power(theta, 3.0) * (hat(phi) @ hat(phi))
        )
    return J


def Jr_log(phi):
    """ 
    Right jacobian for the log operation on SO(3). 
    This is the exact inverse of Jr_exp. Maps errors from the manifold 
    back to the tangent space.
    """
    theta = np.linalg.norm(phi)
    # Taylor expansion for stability nearby 0
    if theta < 1e-3:
        J = np.eye(3) + 0.5 * hat(phi)
    else:
        # Closed form evaluation
        phi_hat = hat(phi)
        J = (
            np.eye(3)
            + 0.5 * phi_hat
            + (
                1 / np.power(theta, 2.0)
                - (1 + np.cos(theta)) / (2 * theta * np.sin(theta))
            )
            * (phi_hat @ phi_hat)
        )
    return J


def Jr_SO3(phi):
    """ 
    Right Jacobian of SO(3)
    """
    phi_norm = np.linalg.norm(phi)
    # Taylor expansion for stability nearby 0
    if phi_norm < 1e-5:
        return np.eye(3)
    else:
        a = phi / phi_norm
        a = a.reshape(3,1)
        sin_phi_div_phi = np.sin(phi_norm) / phi_norm
        return sin_phi_div_phi * np.eye(3) + (1-sin_phi_div_phi) * a @ a.T + (1-np.cos(phi_norm))/phi_norm * hat(a)


def Jl_SO3(phi):
    """ 
    Left Jacobian of SO(3) 
    """
    theta = np.linalg.norm(phi)
    Om = hat(phi)
    if theta < 1e-5:
        return np.eye(3) + 0.5 * Om
    else:
        theta2 = theta ** 2
        return np.eye(3) + (1-np.cos(theta))/theta2 * Om + (theta-np.sin(theta))/(theta2*theta) * Om @ Om


def Jl_SO3_inv(phi):
    """ 
    Inverse of left Jacobian of SO(3) 
    """

    theta = np.linalg.norm(phi)
    Om = hat(phi)
    if theta < 1e-5:
        return np.eye(3) - 0.5 * Om + 1.0/12 * Om @ Om
    else:
        theta2 = theta ** 2
        return np.eye(3) - 0.5 * Om + (1 - 0.5 * theta * np.cos(theta/2) / np.sin(theta/2)) / theta**2 * Om @ Om


def unwrap_rpy(rpys):
    """
    Unwraps a sequence of Roll-Pitch-Yaw angles to prevent 360-degree (or 2*pi)
    discontinuities when angles wrap around boundaries (e.g., passing from 179 to -179 degrees).
    Angles are in degrees (checks for jumps > 300 deg).
    """
    diff = rpys[1:, :] - rpys[0:-1, :]
    uw_rpys = np.zeros(rpys.shape)
    uw_rpys[0, :] = rpys[0, :]
    diff[diff > 300] = diff[diff > 300] - 360
    diff[diff < -300] = diff[diff < -300] + 360
    uw_rpys[1:, :] = uw_rpys[0, :] + np.cumsum(diff, axis=0)
    return uw_rpys


def wrap_rpy(uw_rpys, radians=False):
    """
    Wraps an unwrapped sequence of RPY angles back into the standard range:
    [-180, 180] degrees or [-pi, pi] radians.
    """
    bound = np.pi if radians else 180
    rpys = uw_rpys
    while rpys.min() < -bound:
        rpys[rpys < -bound] = rpys[rpys < -bound] + 2*bound
    while rpys.max() >= bound:
        rpys[rpys >= bound] = rpys[rpys >= bound] - 2*bound
    return rpys
