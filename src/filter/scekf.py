import numpy as np
from filter.utils.math_utils import mat_exp, mat_log, hat, Jr_exp

# ─── State layout ────────────────────────────────────────────────────────────
# error state (33D): dR(3) dv(3) dp(3) dbg(3) dba(3)  [current IMU]
#                  + dR1(3) dp1(3)                      [clone 1]
#                  + dR2(3) dp2(3)                      [clone 2]
#                  + dR3(3) dp3(3)                      [clone 3]
# -----------------------------------------------------------------------------

class State:
    def __init__(self):
        self.R   = np.eye(3)       # rotation (body to world)
        self.v   = np.zeros(3)     # velocity in world
        self.p   = np.zeros(3)     # position in world
        self.bg  = np.zeros(3)     # gyro bias
        self.ba  = np.zeros(3)     # accel bias

        # Stochastic clones: list of (R, p) tuples, oldest first
        self.clone_Rs = []
        self.clone_ps = []

        # Covariance (33x33 when 3 clones present)
        self.P = np.eye(15) * 1e-6

    def get_clone_count(self):
        return len(self.clone_Rs)


class ImuMSCKF:

    def __init__(self, args):
        self.args = args

        # IMU noise parameters (from args or defaults)
        self.sigma_na  = getattr(args, 'sigma_na',  0.01)   # accel noise
        self.sigma_ng  = getattr(args, 'sigma_ng',  0.001)  # gyro noise
        self.sigma_nba = getattr(args, 'sigma_nba', 1e-4)   # accel bias walk
        self.sigma_nbg = getattr(args, 'sigma_nbg', 1e-5)   # gyro bias walk

        self.g = np.array([0, 0, -9.80665])  # gravity in world frame
        self.state = State()

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize_with_state(self, t, R, v, p, bg, ba, P=None):
        self.t = t
        self.state.R  = R.copy()
        self.state.v  = v.copy()
        self.state.p  = p.copy()
        self.state.bg = bg.copy()
        self.state.ba = ba.copy()
        self.state.clone_Rs = []
        self.state.clone_ps = []
        if P is not None:
            self.state.P = P.copy()
        else:
            self.state.P = np.eye(15) * 1e-6

    # ── IMU propagation ───────────────────────────────────────────────────────

    def propagate(self, imu_data):
        """Propagate state and covariance through a list of IMU measurements."""
        for meas in imu_data:
            dt = meas.dt
            wm = meas.gyro - self.state.bg
            am = meas.accel - self.state.ba

            R_prev = self.state.R.copy()
            v_prev = self.state.v.copy()
            p_prev = self.state.p.copy()

            # Strapdown integration (RK1 / Euler)
            dR = mat_exp(hat(wm) * dt)
            self.state.R = R_prev @ dR
            self.state.v = v_prev + (R_prev @ am + self.g) * dt
            self.state.p = p_prev + v_prev * dt + 0.5 * (R_prev @ am + self.g) * dt**2

            # Covariance propagation
            self.state.P = propagate_covariance(
                self.state.P, R_prev, wm, am, dt,
                self.sigma_ng, self.sigma_na,
                self.sigma_nbg, self.sigma_nba
            )

        self.t = imu_data[-1].timestamp if hasattr(imu_data[-1], 'timestamp') else self.t

    # ── Clone augmentation ────────────────────────────────────────────────────

    def augment_clone(self):
        """Append current pose as a new stochastic clone and expand covariance."""
        n_clones = self.state.get_clone_count()
        n_imu = 15
        n_old = n_imu + 6 * n_clones
        n_new = n_old + 6

        # New state
        self.state.clone_Rs.append(self.state.R.copy())
        self.state.clone_ps.append(self.state.p.copy())

        # Jacobian of new clone error w.r.t. old error state
        # dR_new = dR_imu, dp_new = dp_imu  (no velocity cloned)
        J = np.zeros((6, n_old))
        J[0:3, 0:3] = np.eye(3)   # dR
        J[3:6, 6:9] = np.eye(3)   # dp

        P_old = self.state.P
        P_new = np.zeros((n_new, n_new))
        P_new[:n_old, :n_old] = P_old
        P_new[n_old:, :n_old] = J @ P_old
        P_new[:n_old, n_old:] = (J @ P_old).T
        P_new[n_old:, n_old:] = J @ P_old @ J.T

        self.state.P = P_new

    # ── Clone marginalization ─────────────────────────────────────────────────

    def marginalize_oldest_clone(self):
        """Remove the oldest clone from the state and covariance."""
        if self.state.get_clone_count() == 0:
            return

        self.state.clone_Rs.pop(0)
        self.state.clone_ps.pop(0)

        n_imu = 15
        n_clones_after = self.state.get_clone_count()
        n_after = n_imu + 6 * n_clones_after

        # Indices to keep: IMU block + all clones except the first
        keep = list(range(n_imu)) + list(range(n_imu + 6, n_imu + 6 + 6 * n_clones_after))
        P = self.state.P
        self.state.P = P[np.ix_(keep, keep)]

    # ── Measurement update (Phase 2) ──────────────────────────────────────────

    def update(self, network_output):
        """
        TODO (Phase 2): implement the stacked 12D relative-pose update.
        network_output: dict with keys 'dx12' (12,) residual and 'cov12' (12,12).
        """
        raise NotImplementedError(
            "update() will be implemented in Phase 2 (measurement_triplet.py)"
        )


# ── Standalone propagation helpers ───────────────────────────────────────────

def propagate_rvt_and_jac(R, v, p, bg, ba, wm_raw, am_raw, dt, g):
    wm = wm_raw - bg
    am = am_raw - ba
    dR = mat_exp(hat(wm) * dt)
    R_new = R @ dR
    v_new = v + (R @ am + g) * dt
    p_new = p + v * dt + 0.5 * (R @ am + g) * dt**2

    # Continuous-time Jacobian blocks (first-order)
    F = np.zeros((15, 15))
    F[0:3, 0:3] = -hat(wm)
    F[0:3, 9:12] = -np.eye(3)
    F[3:6, 0:3] = -R @ hat(am)
    F[3:6, 3:6] = np.zeros((3, 3))
    F[3:6, 12:15] = -R
    F[6:9, 3:6] = np.eye(3)

    Phi = np.eye(15) + F * dt  # first-order ZOH

    return R_new, v_new, p_new, Phi


def propagate_covariance(P, R, wm, am, dt, sg, sa, sbg, sba):
    n = P.shape[0]
    n_imu = 15

    _, _, _, Phi_imu = propagate_rvt_and_jac(
        R, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), wm, am, dt,
        np.zeros(3)
    )

    # Noise covariance
    Q_c = np.zeros((12, 12))
    Q_c[0:3,   0:3]   = np.eye(3) * sg**2
    Q_c[3:6,   3:6]   = np.eye(3) * sa**2
    Q_c[6:9,   6:9]   = np.eye(3) * sbg**2
    Q_c[9:12,  9:12]  = np.eye(3) * sba**2

    G = np.zeros((15, 12))
    G[0:3,  0:3]  = -np.eye(3)
    G[3:6,  3:6]  = -R
    G[9:12, 6:9]  = np.eye(3)
    G[12:15,9:12] = np.eye(3)

    Q_d = G @ Q_c @ G.T * dt

    # Full Phi (IMU block only; clones are constant)
    Phi = np.eye(n)
    Phi[:n_imu, :n_imu] = Phi_imu

    P_new = Phi @ P @ Phi.T
    P_new[:n_imu, :n_imu] += Q_d

    return P_new