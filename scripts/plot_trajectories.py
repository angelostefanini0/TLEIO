import argparse
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.spatial_math import (
    T_to_pose,
    interpolate_gt_pose,
    normalize_quat,
    pose_to_T,
    rotation_error_deg,
    rotvec_to_rotmat,
)
from scripts.utils.plotting import plot_covariance_error_cones, plot_relative_motion_inspection
from scripts.utils.config import default_config_path, parse_args_with_config

def load_table(path: Path) -> np.ndarray:
    with open(path, "r") as f:
        first = f.readline().strip()

    skiprows = 1 if first and (first[0].isalpha() or first.startswith("#")) else 0
    data = np.loadtxt(path, skiprows=skiprows, dtype=np.float64)

    if data.ndim == 1:
        data = data[None, :]

    return data


def translation_rel_to_T(pred_row: np.ndarray, gt_row: np.ndarray | None = None) -> np.ndarray:
    """Build a transform from a translation-only prediction row.

    If a GT relative-motion row is provided, its rotation vector is used for
    the rotation component while the predicted translation is kept.
    """
    if pred_row.shape[0] not in {5, 8}:
        raise ValueError(
            f"Predicted relative motion must have 5 columns [t0 t1 px py pz] "
            f"or 8 columns [t0 t1 px py pz sigma_x sigma_y sigma_z], "
            f"got {pred_row.shape[0]}."
        )
    if gt_row is not None and gt_row.shape[0] != 8:
        raise ValueError(
            f"GT relative motion must have 8 columns [t0 t1 px py pz rx ry rz], "
            f"got {gt_row.shape[0]}."
        )

    T_rel = np.eye(4, dtype=np.float64)
    if gt_row is not None:
        T_rel[:3, :3] = rotvec_to_rotmat(gt_row[5:8])
    T_rel[:3, 3] = pred_row[2:5]
    return T_rel


def gt_rows_by_timestamp(gt_rel: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    keys = gt_rel[:, :2].astype(np.int64)
    return {(int(t0), int(t1)): gt_rel[i] for i, (t0, t1) in enumerate(keys)}


def match_prediction_to_gt(pred: np.ndarray, gt_rel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gt_lookup = gt_rows_by_timestamp(gt_rel)
    matched_pred = []
    matched_gt = []
    for row in pred:
        key = (int(row[0]), int(row[1]))
        gt_row = gt_lookup.get(key)
        if gt_row is None:
            continue
        matched_pred.append(row)
        matched_gt.append(gt_row)

    if not matched_pred:
        return np.empty((0, pred.shape[1])), np.empty((0, gt_rel.shape[1]))
    return np.stack(matched_pred, axis=0), np.stack(matched_gt, axis=0)


def inspect_covariance_dataset(args) -> None:
    rel_files = sorted(
        path for path in args.rel_dir.glob(args.pattern)
        if path.is_file()
        and not path.stem.endswith("_raw")
        and "raw_model_outputs" not in path.stem
    )
    all_errors = []
    all_sigmas = []
    used_sequences = []
    skipped = []

    for rel_file in rel_files:
        seq_name = rel_file.stem
        gt_rel_file = args.gt_rel_root / seq_name / "relative_motions.txt"
        if not gt_rel_file.is_file():
            skipped.append((seq_name, "missing gt relative_motions.txt"))
            continue

        pred = load_table(rel_file)
        gt_rel = load_table(gt_rel_file)
        if pred.shape[1] != 8:
            skipped.append((seq_name, f"prediction has {pred.shape[1]} columns, expected 8"))
            continue
        if gt_rel.shape[1] != 8:
            skipped.append((seq_name, f"gt_rel has {gt_rel.shape[1]} columns, expected 8"))
            continue

        matched_pred, matched_gt = match_prediction_to_gt(pred, gt_rel)
        if len(matched_pred) == 0:
            skipped.append((seq_name, "no matching timestamps"))
            continue

        all_errors.append(matched_pred[:, 2:5] - matched_gt[:, 2:5])
        all_sigmas.append(matched_pred[:, 5:8])
        used_sequences.append((seq_name, len(matched_pred), len(pred)))

    if not all_errors:
        details = "\n".join(f"{seq}: {reason}" for seq, reason in skipped[:20])
        raise SystemExit(f"No covariance predictions could be aggregated.\n{details}")

    save_dir = args.save_dir or Path("plots/covariance_dataset")
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_path = save_dir / "covariance_error_cones_dataset.png"
    stats = plot_covariance_error_cones(
        rel_err_xyz=np.vstack(all_errors),
        rel_sigma=np.vstack(all_sigmas),
        save_path=plot_path,
        title=f"Dataset Uncertainty vs Translation Error ({len(used_sequences)} sequences)",
        max_points=args.max_points,
        error_limit=args.error_limit,
        sigma_limit=args.sigma_limit,
        sigma_multiplier=args.sigma_multiplier,
    )

    summary_path = save_dir / "covariance_error_cones_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"plot: {plot_path}\n")
        f.write(f"num_sequences: {len(used_sequences)}\n")
        f.write(f"num_valid_rows: {stats['num_valid']}\n")
        f.write(f"num_plotted_rows: {stats['num_plotted']}\n")
        f.write(
            f"norm: outside_{args.sigma_multiplier:g}sigma_percent={stats['outside_percent']:.6f} "
            f"rms_error_m={stats['rms_error']:.10f} "
            f"mean_sigma_m={stats['mean_sigma']:.10f}\n"
        )
        if skipped:
            f.write("\nskipped:\n")
            for seq_name, reason in skipped:
                f.write(f"{seq_name}: {reason}\n")

    print(f"Saved aggregate covariance plot: {plot_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Sequences used: {len(used_sequences)}")
    print(f"Rows used: {stats['num_valid']}")
    print(
        f"Outside {args.sigma_multiplier:g} sigma [%]: "
        f"norm={stats['outside_percent']:.2f}"
    )


def main():
    # PARSE ARGUMENTS 
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, default=None,
                        help="stamped_groundtruth.txt with columns: timestamp_us px py pz qx qy qz qw")
    parser.add_argument("--rel", type=Path, default=None,
                        help="relative motions: [t0_us t1_us px py pz] with optional sigma_x sigma_y sigma_z")
    parser.add_argument("--save_dir", type=Path, default=None,
                        help="Optional directory to save figures instead of showing them")
    
    parser.add_argument("--gt_rel", type=Path, default=None,
                        help="Optional GT relative motions used for rotations and relative-motion error.")
    parser.add_argument("--rel_dir", type=Path, default=None,
                        help="Dataset mode: directory of predicted relative-motion files.")
    parser.add_argument("--gt_rel_root", type=Path, default=None,
                        help="Dataset mode: root with <sequence>/relative_motions.txt files.")
    parser.add_argument("--pattern", type=str, default="*.txt",
                        help="Dataset mode: prediction filename glob.")
    parser.add_argument("--max_points", type=int, default=200_000,
                        help="Dataset covariance plot downsampling limit.")
    parser.add_argument("--error_limit", type=float, default=None,
                        help="Optional symmetric x-axis limit for covariance error-cone plots.")
    parser.add_argument("--sigma_limit", type=float, default=None,
                        help="Optional y-axis limit for covariance error-cone plots.")
    parser.add_argument("--sigma_multiplier", type=float, default=3.0,
                        help="Sigma bound multiplier for covariance error-cone plots.")
    
    args = parse_args_with_config(
        parser,
        default_config_path("plot_trajectories"),
        required=(),
    )

    if args.rel_dir is not None:
        if args.gt_rel_root is None:
            raise SystemExit("--gt_rel_root is required when using --rel_dir.")
        inspect_covariance_dataset(args)
        return

    if args.gt is None or args.rel is None:
        raise SystemExit("--gt and --rel are required unless using dataset mode with --rel_dir.")

    # LOAD GT AND RELATIVE MOTIONS AND CHECK DIMENSIONS
    gt = load_table(args.gt)
    rel = load_table(args.rel)
    gt_rel = load_table(args.gt_rel) if args.gt_rel is not None else None

    if gt.shape[1] != 8:
        raise ValueError(f"{args.gt} has {gt.shape[1]} columns, expected 8.")
    if rel.shape[1] not in {5, 8}:
        raise ValueError(f"{args.rel} has {rel.shape[1]} columns, expected 5 or 8")
    if gt_rel is not None and gt_rel.shape[1] != 8:
        raise ValueError(f"{args.gt_rel} has {gt_rel.shape[1]} columns, expected 8")

    gt_ts = gt[:, 0].astype(np.int64)
    gt_pos = gt[:, 1:4]
    gt_quat = normalize_quat(gt[:, 4:8])

    if len(rel) == 0:
        raise ValueError("Relative motions file is empty.")

    rel_t0 = rel[:, 0].astype(np.int64)
    rel_t1 = rel[:, 1].astype(np.int64)
    rel_sigma = rel[:, 5:8] if rel.shape[1] == 8 else None

    # CHECK FOR CONSISTENCY ACROSS COMPARED MOTIONS
    if not np.all(rel_t1 > rel_t0):
        raise ValueError("Each relative motion must satisfy t1_us > t0_us.")
    if len(rel) > 1 and not np.array_equal(rel_t0[1:], rel_t1[:-1]):
        raise ValueError("Relative motions do not form a continuous timestamp chain.")
    if gt_rel is not None:
        gt_rel_t0 = gt_rel[:, 0].astype(np.int64)
        gt_rel_t1 = gt_rel[:, 1].astype(np.int64)
        if len(gt_rel) != len(rel):
            raise ValueError("--gt_rel must have the same number of rows as --rel.")
        if not np.array_equal(rel_t0, gt_rel_t0) or not np.array_equal(rel_t1, gt_rel_t1):
            raise ValueError("--gt_rel timestamps do not match --rel timestamps.")

    # Anchor timestamps implied by relative motions
    anchor_ts = np.concatenate([rel_t0[:1], rel_t1])

    # Initial pose from source GT at first anchor
    init_pos, init_quat = interpolate_gt_pose(
        gt_ts, gt_pos, gt_quat, np.array([anchor_ts[0]], dtype=np.int64)
    )
    T_chain = pose_to_T(init_pos[0], init_quat[0])

    # Reconstruct trajectory only from relative motions
    recon_pos = [init_pos[0]]
    recon_quat = [init_quat[0]]

    for i in range(len(rel)):
        gt_rel_row = None if gt_rel is None else gt_rel[i]
        T_rel = translation_rel_to_T(rel[i], gt_rel_row)
        T_chain = T_chain @ T_rel
        p, q = T_to_pose(T_chain)
        recon_pos.append(p)
        recon_quat.append(q)

    recon_pos = np.stack(recon_pos, axis=0)
    recon_quat = normalize_quat(np.stack(recon_quat, axis=0))

    # GT reference at the same anchor timestamps
    ref_pos, ref_quat = interpolate_gt_pose(gt_ts, gt_pos, gt_quat, anchor_ts)
    ref_quat = normalize_quat(ref_quat)

    # CALCULATE ERROR STATS
    
    if gt_rel is not None:
        pos_err_rel = np.linalg.norm(rel[:, 2:5] - gt_rel[:, 2:5], axis=1)
        rel_err_xyz = rel[:, 2:5] - gt_rel[:, 2:5]
    else:
        rel_err_xyz = None

    pos_err = np.linalg.norm(recon_pos - ref_pos, axis=1)
    rot_err = rotation_error_deg(ref_quat, recon_quat)

    # LOG ERROR STATS 
    
    error_ref_label = "GT relative motions" if gt_rel is not None else "source GT"

    if gt_rel is not None:
        print(f"Relative motions vs {error_ref_label}")
        print(f"GT poses:                 {len(gt_ts)}")
        print(f"Relative motions:         {len(rel)}")
        print("Relative format:          translation + sigma" if rel_sigma is not None else "Relative format:          translation only")
        print("GT rel rotation:          used for trajectory integration")
        print(f"Reconstructed anchors:    {len(anchor_ts)}")
        print(f"Position RMSE [m]:        {np.sqrt(np.mean(pos_err_rel ** 2)):.6e}")

    print(f"Absolute error")
    print(f"GT poses:                 {len(gt_ts)}")
    print(f"Relative motions:         {len(rel)}")
    print("Relative format:          translation + sigma" if rel_sigma is not None else "Relative format:          translation only")
    print(f"Reconstructed anchors:    {len(anchor_ts)}")
    print(f"Position RMSE [m]:        {np.sqrt(np.mean(pos_err ** 2)):.6e}")
    print(f"Rotation RMSE [deg]:      {np.sqrt(np.mean(rot_err ** 2)):.6e}")

    plot_relative_motion_inspection(
        gt_ts=gt_ts,
        gt_pos=gt_pos,
        anchor_ts=anchor_ts,
        ref_pos=ref_pos,
        recon_pos=recon_pos,
        rel_t1=rel_t1,
        pos_err_rel=pos_err_rel if gt_rel is not None else None,
        rel_err_xyz=rel_err_xyz,
        rel_sigma=rel_sigma,
        error_ref_label=error_ref_label,
        save_dir=args.save_dir,
        sigma_multiplier=args.sigma_multiplier,
    )


if __name__ == "__main__":
    main()
