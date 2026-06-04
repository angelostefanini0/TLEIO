from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_table(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
    skiprows = 1 if first and (first[0].isalpha() or first.startswith("#")) else 0
    data = np.loadtxt(path, skiprows=skiprows, dtype=np.float64, delimiter=None)
    if data.ndim == 1:
        data = data[None, :]
    return data


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quat(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def rotvec_to_rotmat(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / theta
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def pose_to_T(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_rotmat(quat_xyzw)
    T[:3, 3] = position
    return T


def interpolate_gt_pose(gt: np.ndarray, query_ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gt_ts = gt[:, 0]
    pos = np.column_stack(
        [np.interp(query_ts, gt_ts, gt[:, axis]) for axis in range(1, 4)]
    )

    # Normalized lerp is sufficient here for choosing the initial anchor pose.
    quat = np.empty((len(query_ts), 4), dtype=np.float64)
    for idx, t in enumerate(query_ts):
        right = int(np.searchsorted(gt_ts, t, side="right"))
        if right <= 0:
            quat[idx] = gt[0, 4:8]
        elif right >= len(gt_ts):
            quat[idx] = gt[-1, 4:8]
        else:
            left = right - 1
            alpha = (t - gt_ts[left]) / max(gt_ts[right] - gt_ts[left], 1e-12)
            q0 = gt[left, 4:8]
            q1 = gt[right, 4:8]
            if np.dot(q0, q1) < 0.0:
                q1 = -q1
            quat[idx] = (1.0 - alpha) * q0 + alpha * q1
    return pos, normalize_quat(quat)


def reconstruct_tleio(tleio_rel: np.ndarray, gt: np.ndarray, gt_rel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if tleio_rel.shape[1] not in {5, 8}:
        raise ValueError(f"TLEIO file has {tleio_rel.shape[1]} columns, expected 5 or 8")
    if gt_rel.shape[1] != 8:
        raise ValueError(f"GT relative file has {gt_rel.shape[1]} columns, expected 8")
    if len(tleio_rel) != len(gt_rel):
        n = min(len(tleio_rel), len(gt_rel))
        tleio_rel = tleio_rel[:n]
        gt_rel = gt_rel[:n]

    if not np.allclose(tleio_rel[:, :2], gt_rel[:, :2]):
        raise ValueError("TLEIO and GT-relative timestamps do not match")

    anchor_ts = np.concatenate([tleio_rel[:1, 0], tleio_rel[:, 1]])
    init_pos, init_quat = interpolate_gt_pose(gt, anchor_ts[:1])
    T = pose_to_T(init_pos[0], init_quat[0])

    positions = [init_pos[0]]
    for pred_row, gt_rel_row in zip(tleio_rel, gt_rel):
        T_rel = np.eye(4, dtype=np.float64)
        T_rel[:3, :3] = rotvec_to_rotmat(gt_rel_row[5:8])
        T_rel[:3, 3] = pred_row[2:5]
        T = T @ T_rel
        positions.append(T[:3, 3].copy())

    return anchor_ts, np.asarray(positions)


def axis_equal_xy(ax, *arrays: np.ndarray) -> None:
    points = np.vstack([arr[:, :2] for arr in arrays if len(arr)])
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * max(float(np.max(maxs - mins)), 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")


def plot_sequence(seq: str, tleio_dir: Path, tlio_dir: Path, gt_root: Path, gt_rel_root: Path, out_dir: Path) -> Path:
    tleio_path = tleio_dir / f"{seq}.txt"
    tlio_traj_path = tlio_dir / seq / "trajectory.txt"
    gt_path = gt_root / seq / "stamped_groundtruth.txt"
    gt_rel_path = gt_rel_root / seq / "relative_motions.txt"

    tleio_rel = load_table(tleio_path)
    tlio_traj = np.loadtxt(tlio_traj_path, delimiter=",", dtype=np.float64)
    gt = load_table(gt_path)
    gt_rel = load_table(gt_rel_path)

    _, tleio_pos = reconstruct_tleio(tleio_rel, gt, gt_rel)
    # TLIO trajectory.txt stores TLIO prediction in columns 1:4 and the GT used
    # by the TLIO evaluator in columns 4:7.
    tlio_pos = tlio_traj[:, 1:4]
    gt_pos = tlio_traj[:, 4:7]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], color="black", linewidth=2.0, label="GT")
    ax.plot(tleio_pos[:, 0], tleio_pos[:, 1], color="#1f77b4", linewidth=1.7, label="TLEIO")
    ax.plot(tlio_pos[:, 0], tlio_pos[:, 1], color="#d62728", linewidth=1.5, label="TLIO")
    ax.scatter(gt_pos[0, 0], gt_pos[0, 1], color="black", marker="o", s=28, label="start")
    ax.set_title(seq)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    axis_equal_xy(ax, gt_pos, tleio_pos, tlio_pos)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{seq}_xy_tleio_tlio_gt.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tleio-dir", type=Path, default=Path("vggt_massive_v6_1_tartan_test"))
    parser.add_argument("--tlio-dir", type=Path, default=Path("data/TLIO_net_output"))
    parser.add_argument("--gt-root", type=Path, default=Path("data/tartanair/processed_test"))
    parser.add_argument("--gt-rel-root", type=Path, default=Path("data/tartanair/precomputed_test"))
    parser.add_argument("--out-dir", type=Path, default=Path("plots/vggt_massive_v6_1_tartan_test_tleio_tlio_gt_xy"))
    args = parser.parse_args()

    seqs = sorted(path.stem for path in args.tleio_dir.glob("*.txt"))
    saved = []
    skipped = []
    for seq in seqs:
        try:
            saved.append(plot_sequence(seq, args.tleio_dir, args.tlio_dir, args.gt_root, args.gt_rel_root, args.out_dir))
        except Exception as exc:
            skipped.append((seq, str(exc)))

    print(f"Saved {len(saved)} plot(s) to {args.out_dir}")
    for path in saved:
        print(path)
    if skipped:
        print("Skipped:")
        for seq, reason in skipped:
            print(f"{seq}: {reason}")


if __name__ == "__main__":
    main()
