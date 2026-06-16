"""Implement the clone-based IMU propagation and TLEIO triplet EKF update.

This file is the core of the filter branch. It keeps the current IMU state,
manages stochastic pose clones at the three frame times, propagates with IMU
measurements, and performs the stacked `(1 -> 2, 2 -> 3, 3 -> 4, 4 -> 5)` relative-pose update
using the transformer's `4 x 3` output after converting it to a minimal 12D
EKF residual.
"""

import numpy as np
from scipy.stats import chi2
from scipy.spatial.transform import Rotation

from filter.measurement_triplet import build_triplet_update, make_default_joint_covariance
from filter.utils.math_utils import hat, mat_exp,Jr_exp, repair_covariance


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

    def __init__(self):
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
        self.clone_Rs_fej = []   # first-estimate clone rotations, oldest first
        self.clone_ps_fej = []   # first-estimate clone positions, oldest first
        # Initialize the covariance matrix for the IMU state
        self.P = np.zeros((15, 15))

    def get_clone_count(self):
        """Return how many stochastic clones are currently stored."""

        return len(self.clone_Rs)

    @staticmethod
    def clone_slice(clone_index):
        """Return the covariance slice for one clone in the global error state."""

        if clone_index < 0:
            raise ValueError(f"Clone index must be non-negative, got {clone_index}.")
        start = ImuMSCKF.IMU_STATE_DIM + ImuMSCKF.CLONE_STATE_DIM * clone_index
        return slice(start, start + ImuMSCKF.CLONE_STATE_DIM)

    def expected_covariance_dim(self):
        """Return the covariance dimension implied by the current clone count."""

        return ImuMSCKF.IMU_STATE_DIM + ImuMSCKF.CLONE_STATE_DIM * self.get_clone_count()

    def clone_count_from_covariance(self):
        """Infer clone count from covariance shape."""

        extra_dim = self.P.shape[0] - ImuMSCKF.IMU_STATE_DIM
        if extra_dim < 0 or extra_dim % ImuMSCKF.CLONE_STATE_DIM != 0:
            raise ValueError(f"Covariance shape {self.P.shape} is incompatible with the state layout.")
        return extra_dim // ImuMSCKF.CLONE_STATE_DIM

    def assert_covariance_shape_matches_clones(self):
        """Validate covariance size against active clone lists."""

        expected_dim = self.expected_covariance_dim()
        if self.P.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Covariance shape {self.P.shape} does not match {self.get_clone_count()} clones "
                f"(expected {(expected_dim, expected_dim)})."
            )
        if self.clone_count_from_covariance() != self.get_clone_count():
            raise ValueError("Covariance-implied clone count does not match clone list length.")

    def assert_clone_fej_consistency(self):
        """Validate that current and first-estimate clone lists stay aligned."""

        if len(self.clone_Rs) != len(self.clone_Rs_fej):
            raise ValueError("Current clone rotations and FEJ clone rotations have different lengths.")
        if len(self.clone_ps) != len(self.clone_ps_fej):
            raise ValueError("Current clone positions and FEJ clone positions have different lengths.")
        if len(self.clone_Rs) != len(self.clone_ps):
            raise ValueError("Clone rotation and position lists have different lengths.")


class ImuMSCKF:
    """Run IMU propagation and the TLEIO relative-pose update on cloned poses."""

    # Constants
    IMU_STATE_DIM = 15
    CLONE_STATE_DIM = 6


    def __init__(self, args):
        """Read filter hyperparameters and prepare the default measurement noise."""

        self.args = args
        # IMU noise parameters
        self.sigma_na = getattr(args, "sigma_na", 0.01)
        self.sigma_ng = getattr(args, "sigma_ng", 0.001)
        self.sigma_nba = getattr(args, "sigma_nba", 1e-4)
        self.sigma_nbg = getattr(args, "sigma_nbg", 1e-5)
        self.imu_noise_model = getattr(args, "imu_noise_model", "discrete")
        # Transformer measurement assumptions
        self.sigma_rel_t = getattr(args, "sigma_rel_t", 0.10)
        self.sigma_rel_r = getattr(args, "sigma_rel_r", 0.10)
        self.meas_cov_scale = getattr(args, "meas_cov_scale", 1.0)
        self.chi2_confidence = getattr(args, "chi2_confidence", 0.95)
        self.chi2_multiplier = getattr(args, "chi2_multiplier", 1.0)
        self.enable_chi2_gating = getattr(args, "enable_chi2_gating", True)
        self.use_fej = getattr(args, "use_fej", False)
        self.use_block_update = getattr(args, "use_block_update", False)
        self.covariance_propagation_mode = getattr(args, "covariance_propagation_mode", "per_sample")
        self.nominal_integration_method = getattr(args, "nominal_integration_method", "euler")
        self.update_solve_method = getattr(args, "update_solve_method", "innovation")
        self.gating_mode = getattr(args, "gating_mode", "global")
        self.fej_scope = getattr(args, "fej_scope", "clone_update")
        self.edge_robust_mode = getattr(args, "edge_robust_mode", "off")
        self.edge_inflation_factor = getattr(args, "edge_inflation_factor", 100.0)
        self.edge_chi2_multiplier = getattr(args, "edge_chi2_multiplier", 1.0)
        self.covariance_repair_mode = getattr(args, "covariance_repair_mode", "jitter")
        self.covariance_repair_epsilon = getattr(args, "covariance_repair_epsilon", 1e-9)
        # Initialization of P
        self.initial_attitude_sigma_rad = getattr(args, "initial_attitude_sigma_rad", 0.01)
        self.initial_velocity_sigma_mps = getattr(args, "initial_velocity_sigma_mps", 0.5)
        self.initial_position_sigma_m = getattr(args, "initial_position_sigma_m", 0.01)
        self.initial_z_sigma_m = getattr(args, "initial_z_sigma_m", 0.01)
        self.initial_bg_sigma_rps = getattr(args, "initial_bg_sigma_rps", 0.004)
        self.initial_ba_sigma_mps2 = getattr(args, "initial_ba_sigma_mps2", 0.04)

        self.g = np.array([0.0, 0.0, -9.80665])
        self.state = State()
        self.covariance_diagnostics = []
        self.initialize_with_state(
            t=0.0, 
            R=np.eye(3), 
            v=np.zeros(3), 
            p=np.zeros(3), 
            bg=np.zeros(3), 
            ba=np.zeros(3)
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
        self.state.clone_Rs_fej = []
        self.state.clone_ps_fej = []
        if P is None:
            self.state.P = np.zeros((self.IMU_STATE_DIM, self.IMU_STATE_DIM))
            self.state.P[0:3, 0:3] = np.eye(3) * self.initial_attitude_sigma_rad**2
            self.state.P[3:6, 3:6] = np.eye(3) * self.initial_velocity_sigma_mps**2
            self.state.P[6:8, 6:8] = np.eye(2) * self.initial_position_sigma_m**2
            self.state.P[8,8]=self.initial_z_sigma_m**2
            self.state.P[9:12, 9:12] = np.eye(3) * self.initial_bg_sigma_rps**2
            self.state.P[12:15, 12:15] = np.eye(3) * self.initial_ba_sigma_mps2**2
        else:
            self.state.P = P.copy()
        self.state.P = self._repair_covariance(self.state.P, "initialize")
        self.state.assert_clone_fej_consistency()
        self.state.assert_covariance_shape_matches_clones()

    def _repair_covariance(self, P, checkpoint):
        """Apply the configured covariance repair policy and store diagnostics."""

        P_repaired, diagnostics = repair_covariance(
            P,
            mode=self.covariance_repair_mode,
            epsilon=self.covariance_repair_epsilon,
            name=checkpoint,
        )
        self.covariance_diagnostics.append(diagnostics)
        return P_repaired

    def propagate(self, imu_data):
        """Propagate the current IMU state and covariance through queued IMU samples."""

        if len(imu_data) == 0:
            return

        for meas in imu_data:
            dt = meas.dt
            if not np.isfinite(dt) or dt <= 0.0:
                raise ValueError(f"IMU propagation received a non-positive dt: {dt}.")
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
                imu_noise_model=self.imu_noise_model,
                repair_mode=None,
            )
            self.state.P = self._repair_covariance(self.state.P, "propagate")

        self.t = imu_data[-1].timestamp

    def propagate_intervals(self, imu_intervals):
        """Propagate over explicit IMU sample intervals.

        This is the OpenVINS-style interval path. With the default Euler
        nominal integration and per-sample covariance propagation it is designed
        to match the original `propagate()` path closely while making the
        interval boundaries explicit.
        """

        if len(imu_intervals) == 0:
            return
        if self.nominal_integration_method not in {"euler", "midpoint", "midpoint_half_R"}:
            raise ValueError(f"Unknown nominal integration method {self.nominal_integration_method!r}.")
        if self.covariance_propagation_mode not in {"per_sample", "summed"}:
            raise ValueError(f"Unknown covariance propagation mode {self.covariance_propagation_mode!r}.")

        if self.covariance_propagation_mode == "summed":
            phi_summed = np.eye(self.IMU_STATE_DIM)
            q_summed = np.zeros((self.IMU_STATE_DIM, self.IMU_STATE_DIM), dtype=np.float64)

        for interval in imu_intervals:
            dt = float(interval.dt)
            if not np.isfinite(dt) or dt <= 0.0:
                raise ValueError(f"IMU interval propagation received a non-positive dt: {dt}.")

            if self.nominal_integration_method in {"midpoint", "midpoint_half_R"}:
                gyro_raw = 0.5 * (np.asarray(interval.gyro0, dtype=np.float64) + np.asarray(interval.gyro1, dtype=np.float64))
                accel_raw = 0.5 * (np.asarray(interval.accel0, dtype=np.float64) + np.asarray(interval.accel1, dtype=np.float64))
            else:
                gyro_raw = np.asarray(interval.gyro1, dtype=np.float64)
                accel_raw = np.asarray(interval.accel1, dtype=np.float64)

            if not np.isfinite(gyro_raw).all() or not np.isfinite(accel_raw).all():
                raise ValueError("IMU interval contains non-finite accel or gyro values.")

            wm = gyro_raw - self.state.bg
            am = accel_raw - self.state.ba

            R_prev = self.state.R.copy()
            v_prev = self.state.v.copy()
            p_prev = self.state.p.copy()

            if self.nominal_integration_method == "midpoint":
                self.state.R = R_prev @ mat_exp(wm * dt)
                specific_force_world = R_prev @ am + self.g
            elif self.nominal_integration_method == "midpoint_half_R":
                R_half = R_prev @ mat_exp(wm * dt * 0.5)
                self.state.R = R_prev @ mat_exp(wm * dt)
                specific_force_world = R_half @ am + self.g
            else:
                self.state.R, self.state.oldomega4 = integrate_quaternion_3rd_order(
                    R_prev, wm, dt, self.state.oldomega4
                )
                specific_force_world = R_prev @ am + self.g
            self.state.v = v_prev + specific_force_world * dt
            self.state.p = p_prev + v_prev * dt + 0.5 * specific_force_world * dt**2

            if self.covariance_propagation_mode == "summed":
                phi_i, q_i = propagate_covariance_components(
                    R_prev,
                    wm,
                    am,
                    dt,
                    self.sigma_ng,
                    self.sigma_na,
                    self.sigma_nbg,
                    self.sigma_nba,
                    imu_noise_model=self.imu_noise_model,
                )
                q_summed = phi_i @ q_summed @ phi_i.T + q_i
                q_summed = 0.5 * (q_summed + q_summed.T)
                phi_summed = phi_i @ phi_summed
            else:
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
                    imu_noise_model=self.imu_noise_model,
                    repair_mode=None,
                )
                self.state.P = self._repair_covariance(self.state.P, "propagate_interval")

        if self.covariance_propagation_mode == "summed":
            self.state.P = apply_summed_imu_covariance(
                self.state.P,
                phi_summed,
                q_summed,
            )
            self.state.P = self._repair_covariance(self.state.P, "propagate_interval_summed")

        self.t = float(imu_intervals[-1].t1)

    def augment_clone(self):
        """Append the current pose as a stochastic clone and expand covariance."""

        n_clones = self.state.get_clone_count()
        n_old = self.IMU_STATE_DIM + self.CLONE_STATE_DIM * n_clones
        n_new = n_old + self.CLONE_STATE_DIM
        # Save clone
        self.state.clone_Rs.append(self.state.R.copy())
        self.state.clone_ps.append(self.state.p.copy())
        self.state.clone_Rs_fej.append(self.state.R.copy())
        self.state.clone_ps_fej.append(self.state.p.copy())
        # Jacobian mapping current IMU state to the new clone state 
        # (Identity matrices mapping IMU rotation to clone rotation, IMU pos to clone pos)
        clone_jacobian = np.zeros((self.CLONE_STATE_DIM, n_old))
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
        self.state.P = self._repair_covariance(P_new, "augment_clone")
        self.state.assert_clone_fej_consistency()
        self.state.assert_covariance_shape_matches_clones()


    def marginalize_oldest_clone(self):
        """Drop the oldest clone and remove its covariance rows and columns."""

        if self.state.get_clone_count() == 0:
            return

        self.state.clone_Rs.pop(0)
        self.state.clone_ps.pop(0)
        self.state.clone_Rs_fej.pop(0)
        self.state.clone_ps_fej.pop(0)

        n_clones_after = self.state.get_clone_count()
        keep = list(range(self.IMU_STATE_DIM))
        for clone_idx in range(1, n_clones_after + 1):
            keep.extend(range(self.state.clone_slice(clone_idx).start, self.state.clone_slice(clone_idx).stop))

        P = self.state.P
        P_kept = P[np.ix_(keep, keep)]
        self.state.P = self._repair_covariance(P_kept, "marginalize_oldest_clone")
        self.state.assert_clone_fej_consistency()
        self.state.assert_covariance_shape_matches_clones()

    def _chi2_threshold(self, residual_dim):
        """Return the configured chi-square threshold for a residual dimension."""

        if residual_dim <= 0:
            raise ValueError(f"Residual dimension must be positive, got {residual_dim}.")
        return float(self.chi2_multiplier * chi2.ppf(self.chi2_confidence, residual_dim))

    @staticmethod
    def _compute_dense_kalman_gain(P, H, R):
        """Compute the innovation covariance and dense Kalman gain."""

        innovation_covariance = H @ P @ H.T + R
        PHt = P @ H.T
        K = np.linalg.solve(innovation_covariance.T, PHt.T).T
        return K, innovation_covariance, None

    @staticmethod
    def _compute_block_kalman_gain(P, H, R):
        """Compute the Kalman gain using only state columns touched by H."""

        touched_columns = np.flatnonzero(np.any(np.abs(H) > 0.0, axis=0))
        if touched_columns.size == 0:
            raise ValueError("Measurement Jacobian has no non-zero state columns.")
        H_small = H[:, touched_columns]
        P_small = P[np.ix_(touched_columns, touched_columns)]
        innovation_covariance = H_small @ P_small @ H_small.T + R
        PHt = P[:, touched_columns] @ H_small.T
        K = np.linalg.solve(innovation_covariance.T, PHt.T).T
        return K, innovation_covariance, touched_columns

    def _prepare_update_system(self, P, H, R, residual):
        """Return the possibly conditioned residual/Jacobian/noise update system."""

        solve_method = self.update_solve_method
        if solve_method == "qr":
            # Current TLEIO updates are 12D and the state is larger, so QR
            # compression has no rows to remove. Keep the explicit mode for
            # ablations while preserving the innovation update math.
            solve_method = "innovation"
        if solve_method not in {"innovation", "whitened"}:
            raise ValueError(f"Unknown update solve method {self.update_solve_method!r}.")

        condition_number_R = float(np.linalg.cond(R))
        if solve_method == "innovation":
            return {
                "residual": residual,
                "jacobian": H,
                "measurement_covariance": R,
                "condition_number_R": condition_number_R,
                "whitening_applied": False,
                "whitening_repaired_R": False,
            }

        R_for_whitening = 0.5 * (R + R.T)
        whitening_repaired_R = False
        try:
            L = np.linalg.cholesky(R_for_whitening)
        except np.linalg.LinAlgError:
            R_for_whitening = self._repair_covariance(R_for_whitening, "measurement_covariance_whitening")
            whitening_repaired_R = True
            L = np.linalg.cholesky(R_for_whitening)

        residual_w = np.linalg.solve(L, residual)
        H_w = np.linalg.solve(L, H)
        R_w = np.eye(R.shape[0])
        return {
            "residual": residual_w,
            "jacobian": H_w,
            "measurement_covariance": R_w,
            "condition_number_R": condition_number_R,
            "whitening_applied": True,
            "whitening_repaired_R": whitening_repaired_R,
        }

    def _compute_edge_mahalanobis(self, residual, H, P, R):
        """Compute four independent 3D chi-square statistics for the stacked edges."""

        edge_results = []
        threshold = float(
            self.edge_chi2_multiplier
            * self.chi2_multiplier
            * chi2.ppf(self.chi2_confidence, 3)
        )
        for edge_idx in range(4):
            edge_slice = slice(edge_idx * 3, edge_idx * 3 + 3)
            residual_edge = residual[edge_slice]
            H_edge = H[edge_slice, :]
            R_edge = R[edge_slice, edge_slice]
            S_edge = H_edge @ P @ H_edge.T + R_edge
            mahalanobis_sq = float(residual_edge.T @ np.linalg.solve(S_edge, residual_edge))
            ratio = mahalanobis_sq / threshold if threshold > 0.0 else np.inf
            edge_results.append(
                {
                    "edge": edge_idx,
                    "mahalanobis_sq": mahalanobis_sq,
                    "chi2_threshold": threshold,
                    "chi2_ratio": float(ratio),
                    "failed": bool(mahalanobis_sq > threshold),
                }
            )
        return edge_results

    def _edge_diagnostics_dict(self, edge_results):
        """Pack per-edge diagnostics into update return fields."""

        return {
            "edge_mahalanobis_sq": [float(result["mahalanobis_sq"]) for result in edge_results],
            "edge_chi2_thresholds": [float(result["chi2_threshold"]) for result in edge_results],
            "edge_chi2_ratios": [float(result["chi2_ratio"]) for result in edge_results],
            "failed_edge_indices": [int(result["edge"]) for result in edge_results if result["failed"]],
        }

    def _inflate_bad_edge_covariances(self, R, failed_edge_indices):
        """Inflate selected 3D edge covariance blocks."""

        if self.edge_inflation_factor <= 0.0 or not np.isfinite(self.edge_inflation_factor):
            raise ValueError("edge_inflation_factor must be finite and positive.")
        R_inflated = np.asarray(R, dtype=np.float64).copy()
        for edge_idx in failed_edge_indices:
            edge_slice = slice(edge_idx * 3, edge_idx * 3 + 3)
            R_inflated[edge_slice, edge_slice] *= self.edge_inflation_factor
        return R_inflated

    def update(self, network_output):
        """Run the stacked TLEIO relative-pose EKF update on the three clones.

        The input is expected to contain the transformer's raw `4 x 3` mean
        output and, optionally, one joint `12 x 12` covariance for the stacked
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
            use_fej=self.use_fej,
        )

        P = self.state.P
        # Inflate measurement noise according to Adaptive Covariance
        R_adaptive = self.adaptive_cov.get_adaptive_R(residual, H, P, R)
        if self.edge_robust_mode not in {"off", "inflate", "reject"}:
            raise ValueError(f"Unknown edge robust mode {self.edge_robust_mode!r}.")
        edge_results = self._compute_edge_mahalanobis(residual, H, P, R_adaptive)
        edge_diagnostics = self._edge_diagnostics_dict(edge_results)
        failed_edge_indices = edge_diagnostics["failed_edge_indices"]
        edge_rejected = bool(self.edge_robust_mode == "reject" and failed_edge_indices)
        num_inflated_edges = 0
        R_for_update = R_adaptive
        if self.edge_robust_mode == "inflate" and failed_edge_indices:
            R_for_update = self._inflate_bad_edge_covariances(R_adaptive, failed_edge_indices)
            num_inflated_edges = len(failed_edge_indices)
            edge_results = self._compute_edge_mahalanobis(residual, H, P, R_for_update)
            edge_diagnostics = self._edge_diagnostics_dict(edge_results)

        update_system = self._prepare_update_system(P, H, R_for_update, residual)
        residual_update = update_system["residual"]
        H_update = update_system["jacobian"]
        R_update = update_system["measurement_covariance"]
        if self.use_block_update:
            K, innovation_covariance, touched_columns = self._compute_block_kalman_gain(P, H_update, R_update)
        else:
            K, innovation_covariance, touched_columns = self._compute_dense_kalman_gain(P, H_update, R_update)
        # Mahalanobis distance check
        residual_dim = int(residual_update.shape[0])
        mahalanobis_sq = float(residual_update.T @ np.linalg.solve(innovation_covariance, residual_update))
        chi2_threshold = self._chi2_threshold(residual_dim)
        rejected = bool(
            edge_rejected
            or (self.enable_chi2_gating and mahalanobis_sq > chi2_threshold)
        )
        condition_number_S = float(np.linalg.cond(innovation_covariance))
        if rejected:
            return {
                "residual": residual,
                "jacobian": H,
                "measurement_covariance": R_for_update,
                "update_residual": residual_update,
                "update_jacobian": H_update,
                "update_measurement_covariance": R_update,
                "innovation_covariance": innovation_covariance,
                "kalman_gain": None,
                "delta_x": None,
                "rejected": True,
                "mahalanobis_sq": mahalanobis_sq,
                "chi2_threshold": chi2_threshold,
                "residual_dim": residual_dim,
                "touched_columns": touched_columns,
                "update_solve_method": self.update_solve_method,
                "condition_number_R": update_system["condition_number_R"],
                "condition_number_S": condition_number_S,
                "whitening_applied": update_system["whitening_applied"],
                "whitening_repaired_R": update_system["whitening_repaired_R"],
                "edge_robust_mode": self.edge_robust_mode,
                "num_inflated_edges": int(num_inflated_edges),
                "inflated_edge_indices": list(failed_edge_indices) if num_inflated_edges else [],
                "edge_rejected": edge_rejected,
                **edge_diagnostics,
            }
        # Error state
        delta_x = -K @ residual_update
        # Apply correction
        self._inject_error_state(delta_x)
        self._sync_current_pose_with_latest_clone()
        self.state.assert_clone_fej_consistency()
        self.state.assert_covariance_shape_matches_clones()
        # Covariance update in Joseph form
        identity = np.eye(P.shape[0])
        joseph_left = identity - K @ H_update
        P_updated = joseph_left @ P @ joseph_left.T + K @ R_update @ K.T
        self.state.P = self._repair_covariance(P_updated, "update")
        self.state.assert_covariance_shape_matches_clones()

        return {
            "residual": residual,
            "jacobian": H,
            "measurement_covariance": R_for_update,
            "update_residual": residual_update,
            "update_jacobian": H_update,
            "update_measurement_covariance": R_update,
            "innovation_covariance": innovation_covariance,
            "kalman_gain": K,
            "delta_x": delta_x,
            "rejected": False,
            "mahalanobis_sq": mahalanobis_sq,
            "chi2_threshold": chi2_threshold,
            "residual_dim": residual_dim,
            "touched_columns": touched_columns,
            "update_solve_method": self.update_solve_method,
            "condition_number_R": update_system["condition_number_R"],
            "condition_number_S": condition_number_S,
            "whitening_applied": update_system["whitening_applied"],
            "whitening_repaired_R": update_system["whitening_repaired_R"],
            "edge_robust_mode": self.edge_robust_mode,
            "num_inflated_edges": int(num_inflated_edges),
            "inflated_edge_indices": list(failed_edge_indices) if num_inflated_edges else [],
            "edge_rejected": edge_rejected,
            **edge_diagnostics,
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
            clone_slice = self.state.clone_slice(clone_idx)
            offset = clone_slice.start
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
        # Enforce symmetry and a tiny positive diagonal margin.
        R_adaptive, _ = repair_covariance(R_adaptive, mode="jitter", epsilon=1e-12, name="adaptive_measurement_covariance")
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


def propagate_covariance(P, R, wm, am, dt, sg, sa, sbg, sba, imu_noise_model="discrete", repair_mode="jitter"):
    """Propagate the current-plus-clones covariance through one IMU interval."""

    Phi_imu, Q_d = propagate_covariance_components(
        R,
        wm,
        am,
        dt,
        sg,
        sa,
        sbg,
        sba,
        imu_noise_model=imu_noise_model,
    )
    P_new = apply_summed_imu_covariance(P, Phi_imu, Q_d)

    if repair_mode is None:
        return 0.5 * (P_new + P_new.T)
    P_repaired, _ = repair_covariance(P_new, mode=repair_mode, name="propagate_covariance")
    return P_repaired


def propagate_covariance_components(R, wm, am, dt, sg, sa, sbg, sba, imu_noise_model="discrete"):
    """Return the IMU-state transition and discrete process noise for one interval."""

    if imu_noise_model not in {"discrete", "continuous"}:
        raise ValueError(f"Unknown IMU noise model {imu_noise_model!r}.")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Covariance propagation requires positive dt, got {dt}.")

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
    # Noise covariance. The default "discrete" branch preserves the original
    # TLEIO convention, where sigmas are interpreted as per-sample values. The
    # "continuous" branch treats sigmas as continuous-time densities and adds
    # the same-step accelerometer contribution to position uncertainty.
    Q_c = np.zeros((12, 12))
    G = np.zeros((15, 12))
    if imu_noise_model == "discrete":
        Q_c[0:3, 0:3] = np.eye(3) * sg**2
        Q_c[3:6, 3:6] = np.eye(3) * sa**2
        Q_c[6:9, 6:9] = np.eye(3) * sbg**2
        Q_c[9:12, 9:12] = np.eye(3) * sba**2
        G[0:3, 0:3] = -np.eye(3)
        G[3:6, 3:6] = -R
        G[9:12, 6:9] = np.eye(3)
        G[12:15, 9:12] = np.eye(3)
        Q_d = G @ Q_c @ G.T * dt
    else:
        phi = wm * dt
        Q_c[0:3, 0:3] = np.eye(3) * sg**2 / dt
        Q_c[3:6, 3:6] = np.eye(3) * sa**2 / dt
        Q_c[6:9, 6:9] = np.eye(3) * sbg**2 * dt
        Q_c[9:12, 9:12] = np.eye(3) * sba**2 * dt
        G[0:3, 0:3] = -Jr_exp(phi) * dt
        G[3:6, 3:6] = -R * dt
        G[6:9, 3:6] = -0.5 * R * dt**2
        G[9:12, 6:9] = np.eye(3)
        G[12:15, 9:12] = np.eye(3)
        Q_d = G @ Q_c @ G.T

    Q_d = 0.5 * (Q_d + Q_d.T)
    return Phi_imu, Q_d


def apply_summed_imu_covariance(P, Phi_imu, Q_imu):
    """Apply an accumulated IMU transition/noise pair to the full covariance."""

    P = np.asarray(P, dtype=np.float64)
    Phi_imu = np.asarray(Phi_imu, dtype=np.float64)
    Q_imu = np.asarray(Q_imu, dtype=np.float64)
    n_imu = 15
    if P.ndim != 2 or P.shape[0] != P.shape[1] or P.shape[0] < n_imu:
        raise ValueError(f"Expected a square covariance with at least {n_imu} rows, got {P.shape}.")
    if Phi_imu.shape != (n_imu, n_imu):
        raise ValueError(f"Expected Phi_imu shape {(n_imu, n_imu)}, got {Phi_imu.shape}.")
    if Q_imu.shape != (n_imu, n_imu):
        raise ValueError(f"Expected Q_imu shape {(n_imu, n_imu)}, got {Q_imu.shape}.")

    n = P.shape[0]
    Phi = np.eye(n)
    Phi[:n_imu, :n_imu] = Phi_imu
    P_new = Phi @ P @ Phi.T
    P_new[:n_imu, :n_imu] += Q_imu
    return 0.5 * (P_new + P_new.T)



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
