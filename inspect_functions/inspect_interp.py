import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def load_table(path: Path, expected_cols: int) -> np.ndarray:
    with open(path, "r") as f:
        first = f.readline().strip()
    skiprows = 1 if first and (first[0].isalpha() or first.startswith("#")) else 0
    data = np.loadtxt(path, skiprows=skiprows, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != expected_cols:
        raise ValueError(f"{path} has {data.shape[1]} cols, expected {expected_cols}.")
    return data


def normalize_quat(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:
        q = (1.0 - alpha) * q0 + alpha * q1
        return q / np.linalg.norm(q)

    theta_0 = np.arccos(dot)
    theta = alpha * theta_0

    s0 = np.sin(theta_0 - theta) / np.sin(theta_0)
    s1 = np.sin(theta) / np.sin(theta_0)
    q = s0 * q0 + s1 * q1
    return q / np.linalg.norm(q)


def interpolate_pose(ts_gt, pos_gt, quat_gt, ts_query):
    right = np.searchsorted(ts_gt, ts_query, side="left")
    right = np.clip(right, 1, len(ts_gt) - 1)
    left = right - 1

    t0 = ts_gt[left].astype(np.float64)
    t1 = ts_gt[right].astype(np.float64)
    alpha = (ts_query.astype(np.float64) - t0) / (t1 - t0)
    alpha = np.clip(alpha, 0.0, 1.0)

    p0 = pos_gt[left]
    p1 = pos_gt[right]
    pos = (1.0 - alpha[:, None]) * p0 + alpha[:, None] * p1

    q0 = quat_gt[left]
    q1 = quat_gt[right]
    quat = np.stack([slerp(a, b, w) for a, b, w in zip(q0, q1, alpha)], axis=0)

    return pos, quat


def quat_angle_error_deg(q_ref: np.ndarray, q_est: np.ndarray) -> np.ndarray:
    q_ref = normalize_quat(q_ref)
    q_est = normalize_quat(q_est)
    dots = np.abs(np.sum(q_ref * q_est, axis=1))
    dots = np.clip(dots, -1.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dots))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    args = parser.parse_args()

    gt = load_table(args.gt, 8)
    anchors = load_table(args.anchors, 8)

    ts_gt = gt[:, 0].astype(np.int64)
    pos_gt = gt[:, 1:4]
    quat_gt = normalize_quat(gt[:, 4:8])

    ts_anchor = anchors[:, 0].astype(np.int64)
    pos_anchor = anchors[:, 1:4]
    quat_anchor = normalize_quat(anchors[:, 4:8])

    pos_interp, quat_interp = interpolate_pose(ts_gt, pos_gt, quat_gt, ts_anchor)

    pos_err = np.linalg.norm(pos_anchor - pos_interp, axis=1)
    rot_err = quat_angle_error_deg(quat_interp, quat_anchor)

    print(f"Position RMSE [m]:   {np.sqrt(np.mean(pos_err ** 2)):.6e}")
    print(f"Position max  [m]:   {np.max(pos_err):.6e}")
    print(f"Rotation RMSE [deg]: {np.sqrt(np.mean(rot_err ** 2)):.6e}")
    print(f"Rotation max  [deg]: {np.max(rot_err):.6e}")

    t_gt = (ts_gt - ts_gt[0]) * 1e-6
    t_anchor = (ts_anchor - ts_gt[0]) * 1e-6

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    for i, lbl in enumerate(["px", "py", "pz"]):
        axes[i].plot(t_gt, pos_gt[:, i], label="raw GT")
        axes[i].scatter(t_anchor, pos_anchor[:, i], s=12, label="anchor poses")
        axes[i].set_ylabel(lbl)
        axes[i].grid(True)

    axes[3].plot(t_anchor, pos_err, label="position error")
    axes[3].set_ylabel("pos err [m]")
    axes[3].set_xlabel("time [s]")
    axes[3].grid(True)

    axes[0].legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()