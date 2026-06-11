"""Generate synthetic TartanAir IMU samples from processed ground-truth poses.

Edit the USER CONFIGURATION section below, then run:

    python scripts/imu_synthesizer.py

The script reads:
    data/tartanair/<TRAJECTORY_NAME>/stamped_groundtruth.txt

and writes:
    data/tartanair/<TRAJECTORY_NAME>/imu.csv

Output columns are:
    timestamp_us,gx,gy,gz,ax,ay,az

Conventions:
    - ground-truth quaternions are [qx, qy, qz, qw]
    - rotations are body/camera-to-world
    - gyro is body-frame angular velocity [rad/s]
    - accel is body-frame specific force [m/s^2]
    - gravity is expressed in world coordinates
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline, Slerp


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Name of the trajectory folder inside data/tartanair/processed_train.
TRAJECTORY_NAME = "competition_Test_ME000"

# Root containing the processed TartanAir trajectory folders.
REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_TRAIN_ROOT = REPO_ROOT / "data/test/processed/"

# Input and output filenames inside the selected trajectory folder.
GROUNDTRUTH_FILENAME = "stamped_groundtruth.txt"
OUTPUT_IMU_FILENAME = "imu.csv"

# Synthetic IMU sampling rate.
IMU_RATE_HZ = 200.0

# World gravity used by src/main_filter.py.
GRAVITY_WORLD_MPS2 = np.array([0.0, 0.0, 9.80665], dtype=np.float64)

# Add zero-mean Gaussian noise after ideal IMU generation. Keep at 0.0 for an
# ideal IMU stream.
GYRO_NOISE_STD_RAD_S = 1.5e-3
ACCEL_NOISE_STD_MPS2 = 0.006

# Optional IMU biases. The initial bias is added to every sample. Random walk
# values are continuous-time std devs, scaled internally by sqrt(dt).
INITIAL_GYRO_BIAS_RAD_S = np.array([0.001, -0.00075, 0.00115], dtype=np.float64)
INITIAL_ACCEL_BIAS_MPS2 = np.array([0.00045, -0.0003, 0.0007], dtype=np.float64)
GYRO_BIAS_RANDOM_WALK_STD_RAD_S_SQRT_S = 8.0e-5
ACCEL_BIAS_RANDOM_WALK_STD_MPS2_SQRT_S = 8.0e-4
RANDOM_SEED = 7

# Refuse to overwrite an existing imu.csv unless this is True.
OVERWRITE_OUTPUT = True

# =============================================================================


def normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    """Normalize and sign-continuize quaternions in [qx, qy, qz, qw] order."""

    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("Found a zero-norm quaternion in the ground-truth file.")

    q = quaternions / norms
    for idx in range(1, len(q)):
        if np.dot(q[idx - 1], q[idx]) < 0.0:
            q[idx] = -q[idx]
    return q


def load_stamped_groundtruth(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load timestamp_us, position, and xyzw quaternion arrays."""

    data = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if data.shape[1] != 8:
        raise ValueError(
            f"{path} has {data.shape[1]} columns, expected 8: "
            "timestamp_us px py pz qx qy qz qw."
        )

    order = np.argsort(data[:, 0])
    data = data[order]

    timestamps_us = np.rint(data[:, 0]).astype(np.int64)
    keep = np.concatenate([[True], np.diff(timestamps_us) > 0])
    timestamps_us = timestamps_us[keep]
    positions = data[keep, 1:4].astype(np.float64)
    quaternions = normalize_quaternions(data[keep, 4:8].astype(np.float64))

    if len(timestamps_us) < 4:
        raise ValueError("Need at least four ground-truth poses to build splines.")

    return timestamps_us, positions, quaternions


def make_time_grid(timestamps_us: np.ndarray, imu_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Create a uniform IMU time grid covering the GT interval."""

    if imu_rate_hz <= 0.0:
        raise ValueError("IMU_RATE_HZ must be positive.")

    start_s = float(timestamps_us[0]) * 1e-6
    end_s = float(timestamps_us[-1]) * 1e-6
    dt_s = 1.0 / float(imu_rate_hz)
    count = int(np.floor((end_s - start_s) / dt_s)) + 1
    if count < 2:
        raise ValueError("Ground-truth interval is too short for the requested IMU rate.")

    times_s = start_s + np.arange(count, dtype=np.float64) * dt_s
    times_us = np.rint(times_s * 1e6).astype(np.int64)
    return times_s, times_us


def sample_position_acceleration(
    gt_times_s: np.ndarray,
    positions: np.ndarray,
    query_times_s: np.ndarray,
) -> np.ndarray:
    """Fit cubic splines to position and evaluate world-frame acceleration."""

    accel_world = np.empty((len(query_times_s), 3), dtype=np.float64)
    for axis in range(3):
        spline = CubicSpline(gt_times_s, positions[:, axis], bc_type="not-a-knot")
        accel_world[:, axis] = spline(query_times_s, 2)
    return accel_world


def sample_rotations_and_gyro(
    gt_times_s: np.ndarray,
    quaternions: np.ndarray,
    query_times_s: np.ndarray,
) -> tuple[Rotation, np.ndarray]:
    """Interpolate orientations and compute body-frame angular velocity."""

    gt_rotations = Rotation.from_quat(quaternions)

    try:
        rotation_spline = RotationSpline(gt_times_s, gt_rotations)
        rotations = rotation_spline(query_times_s)
        gyro_body = rotation_spline(query_times_s, order=1)
        return rotations, gyro_body
    except Exception:
        # Fallback for older SciPy behavior: SLERP rotations and finite-difference
        # body rates from R_i.T @ R_{i+1}.
        slerp = Slerp(gt_times_s, gt_rotations)
        rotations = slerp(query_times_s)
        matrices = rotations.as_matrix()
        gyro_body = np.empty((len(query_times_s), 3), dtype=np.float64)

        for idx in range(len(query_times_s)):
            if idx == 0:
                left, right = 0, 1
            elif idx == len(query_times_s) - 1:
                left, right = len(query_times_s) - 2, len(query_times_s) - 1
            else:
                left, right = idx - 1, idx + 1

            dt = query_times_s[right] - query_times_s[left]
            relative = matrices[left].T @ matrices[right]
            gyro_body[idx] = Rotation.from_matrix(relative).as_rotvec() / dt

        return rotations, gyro_body


def generate_imu_from_groundtruth(
    stamped_groundtruth_path: Path,
    imu_rate_hz: float,
    gravity_world_mps2: np.ndarray,
    gyro_noise_std_rad_s: float = 0.0,
    accel_noise_std_mps2: float = 0.0,
    initial_gyro_bias_rad_s: np.ndarray | None = None,
    initial_accel_bias_mps2: np.ndarray | None = None,
    gyro_bias_random_walk_std_rad_s_sqrt_s: float = 0.0,
    accel_bias_random_walk_std_mps2_sqrt_s: float = 0.0,
    random_seed: int = 7,
) -> np.ndarray:
    """Generate an IMU table with columns timestamp_us gx gy gz ax ay az."""

    timestamps_us, positions, quaternions = load_stamped_groundtruth(stamped_groundtruth_path)
    gt_times_s = timestamps_us.astype(np.float64) * 1e-6
    query_times_s, query_times_us = make_time_grid(timestamps_us, imu_rate_hz)

    accel_world = sample_position_acceleration(gt_times_s, positions, query_times_s)
    rotations, gyro_body = sample_rotations_and_gyro(gt_times_s, quaternions, query_times_s)

    specific_force_world = accel_world - np.asarray(gravity_world_mps2, dtype=np.float64)
    accel_body = rotations.inv().apply(specific_force_world)

    if (
        gyro_noise_std_rad_s > 0.0
        or accel_noise_std_mps2 > 0.0
        or gyro_bias_random_walk_std_rad_s_sqrt_s > 0.0
        or accel_bias_random_walk_std_mps2_sqrt_s > 0.0
    ):
        rng = np.random.default_rng(random_seed)
    else:
        rng = None

    gyro_bias = np.zeros_like(gyro_body)
    accel_bias = np.zeros_like(accel_body)

    gyro_bias[0] = (
        np.zeros(3, dtype=np.float64)
        if initial_gyro_bias_rad_s is None
        else np.asarray(initial_gyro_bias_rad_s, dtype=np.float64)
    )
    accel_bias[0] = (
        np.zeros(3, dtype=np.float64)
        if initial_accel_bias_mps2 is None
        else np.asarray(initial_accel_bias_mps2, dtype=np.float64)
    )

    if rng is not None:
        for idx in range(1, len(query_times_s)):
            dt_s = query_times_s[idx] - query_times_s[idx - 1]
            gyro_bias[idx] = gyro_bias[idx - 1]
            accel_bias[idx] = accel_bias[idx - 1]

            if gyro_bias_random_walk_std_rad_s_sqrt_s > 0.0:
                gyro_bias[idx] += rng.normal(
                    scale=float(gyro_bias_random_walk_std_rad_s_sqrt_s) * np.sqrt(dt_s),
                    size=3,
                )
            if accel_bias_random_walk_std_mps2_sqrt_s > 0.0:
                accel_bias[idx] += rng.normal(
                    scale=float(accel_bias_random_walk_std_mps2_sqrt_s) * np.sqrt(dt_s),
                    size=3,
                )

    gyro_body = gyro_body + gyro_bias
    accel_body = accel_body + accel_bias

    if rng is not None:
        if gyro_noise_std_rad_s > 0.0:
            gyro_body = gyro_body + rng.normal(
                scale=float(gyro_noise_std_rad_s),
                size=gyro_body.shape,
            )
        if accel_noise_std_mps2 > 0.0:
            accel_body = accel_body + rng.normal(
                scale=float(accel_noise_std_mps2),
                size=accel_body.shape,
            )

    return np.column_stack([query_times_us, gyro_body, accel_body])


def main() -> None:
    trajectory_dir = PROCESSED_TRAIN_ROOT / TRAJECTORY_NAME
    gt_path = trajectory_dir / GROUNDTRUTH_FILENAME
    imu_path = trajectory_dir / OUTPUT_IMU_FILENAME

    if not trajectory_dir.is_dir():
        raise FileNotFoundError(f"Trajectory folder does not exist: {trajectory_dir}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground-truth file does not exist: {gt_path}")
    if imu_path.exists() and not OVERWRITE_OUTPUT:
        raise FileExistsError(f"Output already exists and OVERWRITE_OUTPUT is False: {imu_path}")

    imu = generate_imu_from_groundtruth(
        stamped_groundtruth_path=gt_path,
        imu_rate_hz=IMU_RATE_HZ,
        gravity_world_mps2=GRAVITY_WORLD_MPS2,
        gyro_noise_std_rad_s=GYRO_NOISE_STD_RAD_S,
        accel_noise_std_mps2=ACCEL_NOISE_STD_MPS2,
        initial_gyro_bias_rad_s=INITIAL_GYRO_BIAS_RAD_S,
        initial_accel_bias_mps2=INITIAL_ACCEL_BIAS_MPS2,
        gyro_bias_random_walk_std_rad_s_sqrt_s=GYRO_BIAS_RANDOM_WALK_STD_RAD_S_SQRT_S,
        accel_bias_random_walk_std_mps2_sqrt_s=ACCEL_BIAS_RANDOM_WALK_STD_MPS2_SQRT_S,
        random_seed=RANDOM_SEED,
    )

    fmt = ["%d"] + ["%.10f"] * 6
    np.savetxt(imu_path, imu, delimiter=",", fmt=fmt)
    print(f"Wrote {imu.shape[0]} IMU rows to {imu_path}")


if __name__ == "__main__":
    main()
