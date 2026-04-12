"""Implement the clone-based IMU propagation and TLEIO triplet EKF update.

This file is the core of the filter branch. It keeps the current IMU state,
manages stochastic pose clones at the three frame times, propagates with IMU
measurements, and performs the stacked `(1 -> 2, 2 -> 3)` relative-pose update
using the transformer's `2 x 7` output after converting it to a minimal 12D
EKF residual.
"""

import numpy as np

from filter.measurement_triplet import build_triplet_update, make_default_joint_covariance
from filter.utils.math_utils import hat, mat_exp

eps=1e-9


class State:
    """Store the nominal filter state and its covariance.

    The implementation keeps the current IMU state first and then appends the
    pose clones, which is the layout already assumed by the existing code.
    """

    def __init__(self):
        """Initialize the nominal state, empty clone list, and small covariance."""

        self.R = np.eye(3)       # rotation from body to world
        self.v = np.zeros(3)     # velocity in world coordinates
        self.p = np.zeros(3)     # position in world coordinates
        self.bg = np.zeros(3)    # gyroscope bias
        self.ba = np.zeros(3)    # accelerometer bias

        self.clone_Rs = []       # cloned body-to-world rotations, oldest first
        self.clone_ps = []       # cloned world positions, oldest first
        self.P = np.zeros((15, 15))
        # Posa: possiamo fidarci degli anchor iniziali
        self.P[0:3, 0:3] = np.eye(3) * (0.01)**2  
        self.P[6:9, 6:9] = np.eye(3) * (0.01)**2  
        # Velocità: è calcolata male, diamo grande incertezza!
        self.P[3:6, 3:6] = np.eye(3) * (0.5)**2   
        # Bias: diamo al filtro il permesso di cambiarli
        self.P[9:12, 9:12] = np.eye(3) * (0.05)**2 # Bias gyro
        self.P[12:15, 12:15] = np.eye(3) * (0.2)**2 # Bias accel

    def get_clone_count(self):
        """Return how many stochastic clones are currently stored."""

        return len(self.clone_Rs)


class ImuMSCKF:
    """Run IMU propagation and the TLEIO relative-pose update on cloned poses."""

    def __init__(self, args):
        """Read filter hyperparameters and prepare the default measurement noise."""

        self.args = args

        self.sigma_na = getattr(args, "sigma_na", 0.01)
        self.sigma_ng = getattr(args, "sigma_ng", 0.001)
        self.sigma_nba = getattr(args, "sigma_nba", 1e-4)
        self.sigma_nbg = getattr(args, "sigma_nbg", 1e-5)

        self.sigma_rel_t = getattr(args, "sigma_rel_t", 0.10)
        self.sigma_rel_r = getattr(args, "sigma_rel_r", 0.10)
        self.meas_cov_scale = getattr(args, "meas_cov_scale", 1.0)

        self.g = np.array([0.0, 0.0, -9.80665])
        self.state = State()
        self.default_measurement_covariance = make_default_joint_covariance(
            self.sigma_rel_t
        )
        self.t = 0.0

    def initialize_with_state(self, t, R, v, p, bg, ba, P=None):
        """Reset the filter to a known nominal state and clear all clones."""

        self.t = t
        self.state.R = R.copy()
        self.state.v = v.copy()
        self.state.p = p.copy()
        self.state.bg = bg.copy()
        self.state.ba = ba.copy()
        self.state.clone_Rs = []
        self.state.clone_ps = []
        if P is None:
            self.state.P = np.zeros((15, 15))
            self.state.P[0:3, 0:3] = np.eye(3) * (0.01)**2  
            self.state.P[6:9, 6:9] = np.eye(3) * (0.01)**2  
            self.state.P[3:6, 3:6] = np.eye(3) * (0.5)**2   
            self.state.P[9:12, 9:12] = np.eye(3) * (0.05)**2
            self.state.P[12:15, 12:15] = np.eye(3) * (0.2)**2
        else:
            self.state.P = P.copy()

    def propagate(self, imu_data):
        """Propagate the current IMU state and covariance through queued IMU samples."""

        if len(imu_data) == 0:
            return

        for meas in imu_data:
            dt = meas.dt
            wm = meas.gyro - self.state.bg
            am = meas.accel - self.state.ba

            R_prev = self.state.R.copy()
            v_prev = self.state.v.copy()
            p_prev = self.state.p.copy()

            dR = mat_exp(wm * dt)
            specific_force_world = R_prev @ am + self.g

            self.state.R = R_prev @ dR
            self.state.v = v_prev + specific_force_world * dt
            self.state.p = p_prev + v_prev * dt + 0.5 * specific_force_world * dt**2

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

        self.state.clone_Rs.append(self.state.R.copy())
        self.state.clone_ps.append(self.state.p.copy())

        clone_jacobian = np.zeros((6, n_old))
        clone_jacobian[0:3, 0:3] = np.eye(3)
        clone_jacobian[3:6, 6:9] = np.eye(3)

        P_old = self.state.P
        P_new = np.zeros((n_new, n_new))
        P_new[:n_old, :n_old] = P_old
        P_new[n_old:, :n_old] = clone_jacobian @ P_old
        P_new[:n_old, n_old:] = (clone_jacobian @ P_old).T
        P_new[n_old:, n_old:] = clone_jacobian @ P_old @ clone_jacobian.T

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
        keep = list(range(n_imu)) + list(
            range(n_imu + 6, n_imu + 6 + 6 * n_clones_after)
        )

        P = self.state.P
        P_sym = 0.5 * (P[np.ix_(keep, keep)] + P[np.ix_(keep, keep)].T)
        self.state.P = P_sym + eps * np.eye(P_sym.shape[0])

    def update(self, network_output):
        """Run the stacked TLEIO relative-pose EKF update on the three clones.

        The input is expected to contain the transformer's raw `2 x 7` mean
        output and, optionally, one joint `12 x 12` covariance for the stacked
        residual space.

        `build_triplet_update()` returns the Jacobian of the residual itself,
        not the Jacobian of a predicted measurement map. Because of that, the
        EKF correction must apply the negative Kalman step so the residual is
        driven toward zero instead of away from it.
        """

        residual, H, R = build_triplet_update(
            self.state,
            network_output,
            self.default_measurement_covariance,
            covariance_scale=self.meas_cov_scale,
        )

        P = self.state.P
        innovation_covariance = H @ P @ H.T + R
        PHt = P @ H.T
        K = np.linalg.solve(innovation_covariance.T, PHt.T).T
        delta_x = -K @ residual

        self._inject_error_state(delta_x)

        identity = np.eye(P.shape[0])
        joseph_left = identity - K @ H
        P_updated = joseph_left @ P @ joseph_left.T + K @ R @ K.T
        self.state.P = 0.5 * (P_updated + P_updated.T)

        return {
            "residual": residual,
            "jacobian": H,
            "measurement_covariance": R,
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

        self.state.R = self.state.R @ mat_exp(delta_x[0:3])
        self.state.v = self.state.v + delta_x[3:6]
        self.state.p = self.state.p + delta_x[6:9]
        self.state.bg = self.state.bg + delta_x[9:12]
        self.state.ba = self.state.ba + delta_x[12:15]

        for clone_idx in range(self.state.get_clone_count()):
            offset = 15 + 6 * clone_idx
            self.state.clone_Rs[clone_idx] = (
                self.state.clone_Rs[clone_idx] @ mat_exp(delta_x[offset : offset + 3])
                )
            self.state.clone_ps[clone_idx] = (
                self.state.clone_ps[clone_idx] + delta_x[offset + 3 : offset + 6]
            )


def propagate_rvt_and_jac(R, v, p, bg, ba, wm_raw, am_raw, dt, g):
    """Propagate the nominal IMU state and its first-order transition matrix."""

    wm = wm_raw - bg
    am = am_raw - ba
    dR = mat_exp(wm * dt)
    R_new = R @ dR
    v_new = v + (R @ am + g) * dt
    p_new = p + v * dt + 0.5 * (R @ am + g) * dt**2

    F = np.zeros((15, 15))
    F[0:3, 0:3] = -hat(wm)
    F[0:3, 9:12] = -np.eye(3)
    F[3:6, 0:3] = -R @ hat(am)
    F[3:6, 12:15] = -R
    F[6:9, 3:6] = np.eye(3)

    Phi = np.eye(15) + F * dt
    return R_new, v_new, p_new, Phi


def propagate_covariance(P, R, wm, am, dt, sg, sa, sbg, sba):
    """Propagate the current-plus-clones covariance through one IMU interval."""

    n = P.shape[0]
    n_imu = 15

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

    Q_c = np.zeros((12, 12))
    Q_c[0:3, 0:3] = np.eye(3) * sg**2
    Q_c[3:6, 3:6] = np.eye(3) * sa**2
    Q_c[6:9, 6:9] = np.eye(3) * sbg**2
    Q_c[9:12, 9:12] = np.eye(3) * sba**2

    G = np.zeros((15, 12))
    G[0:3, 0:3] = -np.eye(3)
    G[3:6, 3:6] = -R
    G[9:12, 6:9] = np.eye(3)
    G[12:15, 9:12] = np.eye(3)

    Q_d = G @ Q_c @ G.T * dt

    Phi = np.eye(n)
    Phi[:n_imu, :n_imu] = Phi_imu

    P_new = Phi @ P @ Phi.T
    P_new[:n_imu, :n_imu] += Q_d
    P_sym = 0.5 * (P_new + P_new.T)
    
    return P_sym + eps * np.eye(P_sym.shape[0])
