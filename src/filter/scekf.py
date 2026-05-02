"""Implement the clone-based IMU propagation and TLEIO triplet EKF update.

This file is the core of the filter branch. It keeps the current IMU state,
manages stochastic pose clones at the three frame times, propagates with IMU
measurements, and performs the stacked `(1 -> 2, 2 -> 3)` relative-pose update
using the transformer's `2 x 3` output after converting it to a minimal 12D
EKF residual.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from filter.measurement_triplet import build_triplet_update, make_default_joint_covariance
from filter.utils.math_utils import hat, mat_exp,Jr_exp

eps=1e-9


class State:
    """Store the nominal filter state and its covariance.

    The implementation keeps the current IMU state first and then appends the
    pose clones, which is the layout already assumed by the existing code.

    Error State Vector Layout:
    [0:3]   - Rotation error (delta theta)
    [3:6]   - Velocity error (delta v)
    [6:9]   - Position error (delta p)
    [9:12]  - Gyroscope bias error (delta bg)
    [12:15] - Accelerometer bias error (delta ba)
    [15:]   - Appended stochastic clones (6 DoF per clone: 3 rotation, 3 position)
    """

    def __init__(
        self,
        initial_attitude_sigma_rad=0.01,
        initial_velocity_sigma_mps=0.5,
        initial_position_sigma_m=0.01,
        initial_bg_sigma_rps=0.01,
        initial_ba_sigma_mps2=0.2,
    ):
        """Initialize the nominal state, empty clone list, and small covariance."""

        self.R = np.eye(3)       # rotation from body to world
        self.v = np.zeros(3)     # velocity in world coordinates
        self.p = np.zeros(3)     # position in world coordinates
        self.bg = np.zeros(3)    # gyroscope bias
        self.ba = np.zeros(3)    # accelerometer bias 

        self.oldomega4 = np.zeros((4, 4)) # state matrix for third-order quaternion integration

        # Clones for MSCKF 
        self.clone_Rs = []       # cloned body-to-world rotations, oldest first
        self.clone_ps = []       # cloned world positions, oldest first
        # Initialize the covariance matrix for the IMU state
        self.P = np.zeros((15, 15))
        self.P[0:3, 0:3] = np.eye(3) * initial_attitude_sigma_rad**2
        self.P[3:6, 3:6] = np.eye(3) * initial_velocity_sigma_mps**2
        self.P[6:9, 6:9] = np.eye(3) * initial_position_sigma_m**2
        self.P[9:12, 9:12] = np.eye(3) * initial_bg_sigma_rps**2
        self.P[12:15, 12:15] = np.eye(3) * initial_ba_sigma_mps2**2

    def get_clone_count(self):
        """Return how many stochastic clones are currently stored."""

        return len(self.clone_Rs)


class ImuMSCKF:
    """Run IMU propagation and the TLEIO relative-pose update on cloned poses."""

    def __init__(self, args):
        """Read filter hyperparameters and prepare the default measurement noise."""

        self.args = args
        # IMU noise parameters
        self.sigma_na = getattr(args, "sigma_na", 0.01)
        self.sigma_ng = getattr(args, "sigma_ng", 0.001)
        self.sigma_nba = getattr(args, "sigma_nba", 1e-4)
        self.sigma_nbg = getattr(args, "sigma_nbg", 1e-5)
        # Transformer measurement assumptions
        self.sigma_rel_t = getattr(args, "sigma_rel_t", 0.10)
        self.sigma_rel_r = getattr(args, "sigma_rel_r", 0.10)
        self.meas_cov_scale = getattr(args, "meas_cov_scale", 1.0)
        # Initialization of P
        self.initial_attitude_sigma_rad = getattr(args, "initial_attitude_sigma_rad", 0.01)
        self.initial_velocity_sigma_mps = getattr(args, "initial_velocity_sigma_mps", 0.5)
        self.initial_position_sigma_m = getattr(args, "initial_position_sigma_m", 0.01)
        self.initial_bg_sigma_rps = getattr(args, "initial_bg_sigma_rps", 0.004)
        self.initial_ba_sigma_mps2 = getattr(args, "initial_ba_sigma_mps2", 0.04)

        self.g = np.array([0.0, 0.0, -9.80665])
        self.state = State(
            initial_attitude_sigma_rad=self.initial_attitude_sigma_rad,
            initial_velocity_sigma_mps=self.initial_velocity_sigma_mps,
            initial_position_sigma_m=self.initial_position_sigma_m,
            initial_bg_sigma_rps=self.initial_bg_sigma_rps,
            initial_ba_sigma_mps2=self.initial_ba_sigma_mps2,
        )
        self.default_measurement_covariance = make_default_joint_covariance(
            self.sigma_rel_t
        )
        self.t = 0.0
        self.adaptive_cov = AdaptiveCovariance(M1=2, M2=1, gamma=1e-5)

    def initialize_with_state(self, t, R, v, p, bg, ba, P=None):
        """Reset the filter to a known nominal state and clear all clones."""

        self.t = t
        self.state.R = R.copy()
        self.state.v = v.copy()
        self.state.p = p.copy()
        self.state.bg = bg.copy()
        self.state.ba = ba.copy()
        self.state.oldomega4 = np.zeros((4, 4))
        self.state.clone_Rs = []
        self.state.clone_ps = []
        if P is None:
            self.state.P = np.zeros((15, 15))
            self.state.P[0:3, 0:3] = np.eye(3) * self.initial_attitude_sigma_rad**2
            self.state.P[3:6, 3:6] = np.eye(3) * self.initial_velocity_sigma_mps**2
            self.state.P[6:9, 6:9] = np.eye(3) * self.initial_position_sigma_m**2
            self.state.P[9:12, 9:12] = np.eye(3) * self.initial_bg_sigma_rps**2
            self.state.P[12:15, 12:15] = np.eye(3) * self.initial_ba_sigma_mps2**2
        else:
            self.state.P = P.copy()

    def propagate(self, imu_data):
        """Propagate the current IMU state and covariance through queued IMU samples."""

        if len(imu_data) == 0:
            return

        for meas in imu_data:
            dt = meas.dt
            # Remove estimated biases from measurements
            wm = meas.gyro - self.state.bg
            am = meas.accel - self.state.ba

            R_prev = self.state.R.copy()
            v_prev = self.state.v.copy()
            p_prev = self.state.p.copy()

            dR = mat_exp(wm * dt)
            specific_force_world = R_prev @ am + self.g
            # Integrate nominal state (through third-order approximation)
            self.state.R, self.state.oldomega4 = integrate_quaternion_3rd_order(
                R_prev, wm, dt, self.state.oldomega4
            )
            self.state.v = v_prev + specific_force_world * dt
            self.state.p = p_prev + v_prev * dt + 0.5 * specific_force_world * dt**2
            # Propagate covariance
            self.state.P = propagate_covariance(
                self.state.P,
                R_prev,
                wm,
                am,
                dt,
                self.sigma_ng,
                self.sigma_na,
                self.sigma_nbg,
                self.sigma_nba,
            )

        self.t = imu_data[-1].timestamp

    def augment_clone(self):
        """Append the current pose as a stochastic clone and expand covariance."""

        n_clones = self.state.get_clone_count()
        n_imu = 15
        n_old = n_imu + 6 * n_clones
        n_new = n_old + 6
        # Save clone
        self.state.clone_Rs.append(self.state.R.copy())
        self.state.clone_ps.append(self.state.p.copy())
        # Jacobian mapping current IMU state to the new clone state 
        # (Identity matrices mapping IMU rotation to clone rotation, IMU pos to clone pos)
        clone_jacobian = np.zeros((6, n_old))
        clone_jacobian[0:3, 0:3] = np.eye(3)
        clone_jacobian[3:6, 6:9] = np.eye(3)
        # Expand covariance
        P_old = self.state.P
        P_new = np.zeros((n_new, n_new))
        P_new[:n_old, :n_old] = P_old
        P_new[n_old:, :n_old] = clone_jacobian @ P_old
        P_new[:n_old, n_old:] = (clone_jacobian @ P_old).T
        P_new[n_old:, n_old:] = clone_jacobian @ P_old @ clone_jacobian.T
        # Enforce symmetry
        P_sym = 0.5 * (P_new + P_new.T)
        self.state.P = P_sym + eps * np.eye(P_sym.shape[0])

    def marginalize_oldest_clone(self):
        """Drop the oldest clone and remove its covariance rows and columns."""

        if self.state.get_clone_count() == 0:
            return

        self.state.clone_Rs.pop(0)
        self.state.clone_ps.pop(0)

        n_imu = 15
        n_clones_after = self.state.get_clone_count()
        # Indices to keep: IMU state (0-14) + remaining clones (skipping the dropped one at index 15-20)
        keep = list(range(n_imu)) + list(
            range(n_imu + 6, n_imu + 6 + 6 * n_clones_after)
        )

        P = self.state.P
        P_sym = 0.5 * (P[np.ix_(keep, keep)] + P[np.ix_(keep, keep)].T)
        self.state.P = P_sym + eps * np.eye(P_sym.shape[0])

    def update(self, network_output):
        """Run the stacked TLEIO relative-pose EKF update on the three clones.

        The input is expected to contain the transformer's raw `2 x 3` mean
        output and, optionally, one joint `6 x 6` covariance for the stacked
        residual space.

        `build_triplet_update()` returns the Jacobian of the residual itself,
        not the Jacobian of a predicted measurement map. Because of that, the
        EKF correction must apply the negative Kalman step so the residual is
        driven toward zero instead of away from it.
        """
        covariance = network_output.get("joint_covariance", self.default_measurement_covariance)
        # Get residual (z - h(x)), Jacobian (H), and base Measurement Noise (R)
        residual, H, R = build_triplet_update(
            self.state,
            network_output,
            covariance,
            covariance_scale=self.meas_cov_scale,
        )

        P = self.state.P
        # Inflate measurement noise according to Adaptive Covariance
        R_adaptive = self.adaptive_cov.get_adaptive_R(residual, H, P, R)
        innovation_covariance = H @ P @ H.T + R_adaptive
        # Mahalanobis distance check
        mahalanobis_sq = residual.T @ np.linalg.solve(innovation_covariance, residual)
        chi2_threshold=12.59 #95% confidence for 6 DoF
        if mahalanobis_sq>chi2_threshold:
            return {
                "residual": residual,
                "jacobian": H,
                "measurement_covariance": R_adaptive,
                "innovation_covariance": innovation_covariance,
                "kalman_gain": None,
                "delta_x": None,
                "rejected": True 
            }
        PHt = P @ H.T
        # Kalman gain
        K = np.linalg.solve(innovation_covariance.T, PHt.T).T
        # Error state
        delta_x = -K @ residual
        # Apply correction
        self._inject_error_state(delta_x)
        self._sync_current_pose_with_latest_clone()
        # Covariance update in Joseph form
        identity = np.eye(P.shape[0])
        joseph_left = identity - K @ H
        P_updated = joseph_left @ P @ joseph_left.T + K @ R_adaptive @ K.T
        self.state.P = 0.5 * (P_updated + P_updated.T)

        return {
            "residual": residual,
            "jacobian": H,
            "measurement_covariance": R_adaptive,
            "innovation_covariance": innovation_covariance,
            "kalman_gain": K,
            "delta_x": delta_x,
        }

    def _inject_error_state(self, delta_x):
        """Apply one EKF correction to the nominal state and all active clones."""

        delta_x = np.asarray(delta_x, dtype=float)
        expected_dim = self.state.P.shape[0]
        if delta_x.shape != (expected_dim,):
            raise ValueError(
                f"Expected an error-state correction with shape ({expected_dim},), got {delta_x.shape}."
            )
        # Update IMU base state
        self.state.R = self.state.R @ mat_exp(delta_x[0:3])
        self.state.v = self.state.v + delta_x[3:6]
        self.state.p = self.state.p + delta_x[6:9]
        self.state.bg = self.state.bg + delta_x[9:12]
        self.state.ba = self.state.ba + delta_x[12:15]
        # Update cloned states
        for clone_idx in range(self.state.get_clone_count()):
            offset = 15 + 6 * clone_idx
            self.state.clone_Rs[clone_idx] = (
                self.state.clone_Rs[clone_idx] @ mat_exp(delta_x[offset : offset + 3])
                )
            self.state.clone_ps[clone_idx] = (
                self.state.clone_ps[clone_idx] + delta_x[offset + 3 : offset + 6]
            )

    def _sync_current_pose_with_latest_clone(self):
        """Keep the live IMU pose consistent with the newest clone at the same timestamp."""

        if self.state.get_clone_count() == 0:
            return
        self.state.R = self.state.clone_Rs[-1].copy()
        self.state.p = self.state.clone_ps[-1].copy()


class AdaptiveCovariance:
    """
    Implement Adaptive Algorithm for Covariance Estimation
    """
    def __init__(self, M1=5, M2=2, gamma=1e-5):
        self.M1 = M1          # Window size of residuals (to calculate empirical covariance)
        self.M2 = M2          # Number of iterations below gamma needed before resetting to Mode 1
        self.gamma = gamma    # Threshold to decide if theoretical and empirical covariances align
        self.residual_history = []
        self.mode1_counter = 0

    def get_adaptive_R(self, residual, H, P, R_base):
        res = residual.reshape(-1, 1)
        self.residual_history.append(res)
        # Maintain sliding window M1
        if len(self.residual_history) > self.M1:
            self.residual_history.pop(0)
        # Not enough data
        if len(self.residual_history) < self.M1:
            return R_base

        dim = res.shape[0]
        # Empirical covariance
        U_k = np.zeros((dim, dim))
        for r in self.residual_history:
            U_k += r @ r.T
        U_k /= self.M1
        # Theoretical covariance
        S_k = H @ P @ H.T + R_base
        # Eigendecomposition to compare empirical vs theoretical spread
        lambdas, U_vecs = np.linalg.eigh(U_k)

        max_diff = -np.inf
        Q_hat = np.zeros((dim, dim))

        for i in range(dim):
            u_i = U_vecs[:, i:i+1]
            # Theoretical variance projected along eigenvector i
            mu_i = (u_i.T @ S_k @ u_i)[0, 0]
            
            diff = lambdas[i] - mu_i
            if diff > max_diff:
                max_diff = diff
            # If empirical variance is larger, accumulate the difference to inflate R
            if diff > 0:
                Q_hat += diff * (u_i @ u_i.T)
        # If the network covariance matches the empirical reality for M2 consecutive frames, turn off the inflation.
        if max_diff < self.gamma:
            self.mode1_counter += 1
        else:
            self.mode1_counter = 0

        if self.mode1_counter > self.M2:
            Q_hat = np.zeros((dim, dim))

        R_adaptive = R_base + Q_hat
        # Enforce simmetry
        R_adaptive = 0.5 * (R_adaptive + R_adaptive.T)
        
        return R_adaptive


def propagate_rvt_and_jac(R, v, p, bg, ba, wm_raw, am_raw, dt, g):
    """Propagate the nominal IMU state and its first-order transition matrix."""
    # Unbias measurements
    wm = wm_raw - bg
    am = am_raw - ba
    # Integrate rotation, position and velocity
    phi = wm * dt
    dR = mat_exp(phi)
    R_new = R @ dR
    v_new = v + (R @ am + g) * dt
    p_new = p + v * dt + 0.5 * (R @ am + g) * dt**2
    # Continuous-to-discrete state transition matrix
    Phi = np.eye(15)
    Phi[0:3, 9:12] = -Jr_exp(phi) * dt
    Phi[3:6, 0:3] = -R @ hat(am) * dt
    Phi[3:6, 12:15] = -R * dt
    Phi[6:9, 3:6] = np.eye(3) * dt
    Phi[6:9, 0:3] = -0.5 * R @ hat(am) * dt**2
    Phi[6:9, 12:15] = -0.5 * R * dt**2
    Phi[0:3, 0:3] = dR.T 
    Phi[3:6, 9:12] = 0.5 * R @ hat(am) * dt**2
    Phi[6:9, 9:12] = (1.0 / 6.0) * R @ hat(am) * dt**3

    return R_new, v_new, p_new, Phi


def propagate_covariance(P, R, wm, am, dt, sg, sa, sbg, sba):
    """Propagate the current-plus-clones covariance through one IMU interval."""

    n = P.shape[0]
    n_imu = 15
    # State transition matrix
    _, _, _, Phi_imu = propagate_rvt_and_jac(
        R,
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        wm,
        am,
        dt,
        np.zeros(3),
    )
    # Continuous-time noise covariance
    Q_c = np.zeros((12, 12))
    Q_c[0:3, 0:3] = np.eye(3) * sg**2
    Q_c[3:6, 3:6] = np.eye(3) * sa**2
    Q_c[6:9, 6:9] = np.eye(3) * sbg**2
    Q_c[9:12, 9:12] = np.eye(3) * sba**2
    # Noise mapping matrix maps 12D noise into the 15D state space
    G = np.zeros((15, 12))
    G[0:3, 0:3] = -np.eye(3)
    G[3:6, 3:6] = -R
    G[9:12, 6:9] = np.eye(3)
    G[12:15, 9:12] = np.eye(3)
    # Discrete-time noise covariance
    Q_d = G @ Q_c @ G.T * dt
    # Full transition matrix
    Phi = np.eye(n)
    Phi[:n_imu, :n_imu] = Phi_imu
    # Propagate covariance
    P_new = Phi @ P @ Phi.T
    P_new[:n_imu, :n_imu] += Q_d
    P_sym = 0.5 * (P_new + P_new.T)
    
    return P_sym + eps * np.eye(P_sym.shape[0])

def integrate_quaternion_3rd_order(R, wm, dt, oldomega4):
    """
    Perform 3rd-order quaternion integration.
    """
    q_xyzw = Rotation.from_matrix(R).as_quat()
    q = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]) 
    #Skew-symmetric operator needed for quaternion multiplication  
    omega4 = np.array([
        [  0.0,  -wm[0], -wm[1], -wm[2]],
        [ wm[0],   0.0,   wm[2], -wm[1]],
        [ wm[1], -wm[2],   0.0,   wm[0]],
        [ wm[2],  wm[1], -wm[0],   0.0 ]
    ])
    
    I = np.eye(4)
    w_sq = np.sum(wm**2)
    # Third-order Taylor series approximation of the quaternion matrix exponential
    transition_matrix = (
        I 
        + 0.75 * omega4 * dt 
        - 0.25 * oldomega4 * dt 
        - (1.0 / 6.0) * w_sq * (dt**2) * I 
        - (1.0 / 24.0) * (omega4 @ oldomega4) * (dt**2) 
        - (1.0 / 48.0) * w_sq * omega4 * (dt**3)
    )
    
    q_next = transition_matrix @ q
    q_next /= np.linalg.norm(q_next)
    
    q_next_xyzw = np.array([q_next[1], q_next[2], q_next[3], q_next[0]])
    R_next = Rotation.from_quat(q_next_xyzw).as_matrix()
    
    return R_next, omega4
