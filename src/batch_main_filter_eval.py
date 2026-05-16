"""Batch-run `main_filter.py` logic and evaluate each sequence with the RPG toolbox."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL_SRC = ROOT / "evaluation" / "rpg_trajectory_evaluation" / "src" / "rpg_trajectory_evaluation"
if str(EVAL_SRC) not in sys.path:
    sys.path.insert(0, str(EVAL_SRC))

try:
    from main_filter import CONFIG, run_filter
except ImportError:
    from .main_filter import CONFIG, run_filter

from trajectory import Trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run main_filter and evaluate each sequence with RPG trajectory metrics."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset folder name under data/ (for example: eds, tartanair).",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Optional comma-separated sequence list. If omitted, all processed sequences are used.",
    )
    parser.add_argument(
        "--gt",
        action="store_true",
        help="Use relative_motions.txt instead of transformer predictions.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame cap forwarded to the filter runner.",
    )
    return parser.parse_args()


def dataset_specific_overrides(dataset: str) -> dict:
    if dataset.lower() == "eds":
        return {
            "imu_axis_multipliers": (-1.0, -1.0, 1.0),
            "gravity_world_mps2": (0.0, 0.0, -9.80665),
        }
    return {}


def load_tuned_params(sequence: str) -> dict:
    json_path = ROOT / "outputs" / "tuning" / sequence / "best_filter_params.json"
    if not json_path.exists():
        return {}

    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("best", {}).get("params", {})


def parse_sequence_list(dataset: str, sequence_arg: str | None) -> list[str]:
    processed_dir = CONFIG.data_root / dataset / "processed"
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed dataset folder does not exist: {processed_dir}")

    if sequence_arg:
        sequences = [item.strip() for item in sequence_arg.split(",") if item.strip()]
    else:
        sequences = [
            path.name
            for path in processed_dir.iterdir()
            if path.is_dir() and (path / "stamped_groundtruth.txt").exists()
        ]

    if not sequences:
        raise ValueError("No sequences found to evaluate.")

    missing = [seq for seq in sequences if not (processed_dir / seq).is_dir()]
    if missing:
        raise FileNotFoundError(f"Unknown sequence(s) for dataset '{dataset}': {', '.join(missing)}")

    return sorted(sequences)


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def load_numeric_pose_table(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                rows.append([float(value) for value in parts])
            except ValueError:
                continue

    table = np.asarray(rows, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != 8:
        raise ValueError(f"{path} has shape {table.shape}, expected N x 8 pose rows.")
    return table


def infer_time_scale_to_seconds(timestamps: np.ndarray) -> float:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    positive_diffs = np.diff(timestamps)
    positive_diffs = positive_diffs[positive_diffs > 0]
    median_dt = float(np.median(positive_diffs)) if len(positive_diffs) > 0 else 0.0

    if median_dt > 1e7:
        return 1e-9
    if median_dt > 1e1:
        return 1e-6
    return 1.0


def write_seconds_pose_table(src_path: Path, dst_path: Path) -> Path:
    table = load_numeric_pose_table(src_path).copy()
    table[:, 0] *= infer_time_scale_to_seconds(table[:, 0])
    np.savetxt(dst_path, table, fmt="%.9f")
    return dst_path


def prepare_eval_directory(
    dataset: str,
    sequence: str,
    estimate_path: Path,
) -> tuple[Path, Path, Path]:
    gt_path = ROOT / "data" / dataset / "processed" / sequence / "stamped_groundtruth.txt"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth file does not exist: {gt_path}")
    if not estimate_path.exists():
        raise FileNotFoundError(f"Estimated trajectory file does not exist: {estimate_path}")

    seq_eval_dir = ROOT / "eval_outputs" / dataset / sequence
    seq_eval_dir.mkdir(parents=True, exist_ok=True)

    eval_gt_path = seq_eval_dir / "stamped_groundtruth.txt"
    eval_est_path = seq_eval_dir / "stamped_traj_estimate.txt"
    write_seconds_pose_table(gt_path, eval_gt_path)
    write_seconds_pose_table(estimate_path, eval_est_path)

    saved_results_dir = seq_eval_dir / "saved_results" / "traj_est"
    stale_matches = saved_results_dir / "stamped_est_gt_matches.txt"
    if stale_matches.exists():
        stale_matches.unlink()

    return seq_eval_dir, eval_gt_path, eval_est_path


def evaluate_with_rpg(results_dir: Path) -> dict:
    traj = Trajectory(str(results_dir), est_type="traj_est")
    if not traj.data_loaded:
        raise RuntimeError(f"RPG trajectory loader failed for {results_dir}")

    traj.compute_absolute_error()
    traj.compute_relative_errors()
    traj.cache_current_error()
    traj.write_errors_to_yaml()

    return {
        "ate_rmse_m": float(traj.abs_errors["abs_e_trans_stats"]["rmse"]),
        "abs_mean_m": float(traj.abs_errors["abs_e_trans_stats"]["mean"]),
        "abs_median_m": float(traj.abs_errors["abs_e_trans_stats"]["median"]),
        "abs_std_m": float(traj.abs_errors["abs_e_trans_stats"]["std"]),
        "abs_max_m": float(traj.abs_errors["abs_e_trans_stats"]["max"]),
        "rpg_rot_rmse_deg": float(traj.abs_errors["abs_e_rot_stats"]["rmse"]),
        "subtrajectory_lengths_m": [float(v) for v in traj.preset_boxplot_distances],
        "saved_results_dir": str(traj.saved_results_dir),
    }


def average_metric(rows: list[dict], key: str) -> float:
    return float(sum(float(row[key]) for row in rows) / len(rows))


def main() -> None:
    args = parse_args()
    sequences = parse_sequence_list(args.dataset, args.sequence)

    dataset_eval_dir = ROOT / "eval_outputs" / args.dataset
    dataset_eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    print(f"Sequences: {', '.join(sequences)}")

    rows: list[dict] = []
    failures: list[dict] = []

    for sequence in sequences:
        print("\n" + "=" * 72)
        print(f"Running sequence: {sequence}")
        print("=" * 72)

        try:
            config = replace(
                    CONFIG,
                    dataset=args.dataset,
                    sequence=sequence,
                    use_gt=args.gt,
                    max_frames=args.max_frames,
                    plot_transformer=True,
                    plot_projections=True,
                    interactive_plot=False,
                    plot_imu=False,
                    **dataset_specific_overrides(args.dataset),
                    **load_tuned_params(sequence),
                )

            filter_results = run_filter(config)
            estimate_path = Path(filter_results["saved_file"])
            seq_eval_dir, eval_gt_path, eval_est_path = prepare_eval_directory(
                args.dataset,
                sequence,
                estimate_path,
            )
            rpg_metrics = evaluate_with_rpg(seq_eval_dir)

            pos_rmse = float(filter_results["diagnostics"]["position_rmse_m"])
            rot_rmse = float(filter_results["diagnostics"]["rotation_rmse_deg"])
            ate_rmse = float(rpg_metrics["ate_rmse_m"])

            row = {
                "dataset": args.dataset,
                "sequence": sequence,
                "pos_rmse_m": pos_rmse,
                "rot_rmse_deg": rot_rmse,
                "ate_rmse_m": ate_rmse,
                "estimate_file": str(eval_est_path),
                "groundtruth_file": str(eval_gt_path),
                "filter_output_dir": str(estimate_path.parent),
                "eval_output_dir": str(seq_eval_dir),
                "rpg_metrics": rpg_metrics,
            }
            rows.append(row)

            (seq_eval_dir / "summary.json").write_text(
                json.dumps(to_jsonable(row), indent=2),
                encoding="utf-8",
            )

            print(
                f"{sequence}: "
                f"POS_RMSE={pos_rmse:.6f} m | "
                f"ROT_RMSE={rot_rmse:.6f} deg | "
                f"ATE={ate_rmse:.6f} m"
            )
        except Exception as exc:
            failures.append({"sequence": sequence, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{sequence}: FAILED -> {type(exc).__name__}: {exc}")

    if not rows:
        raise RuntimeError("No sequence completed successfully.")

    averages = {
        "pos_rmse_m": average_metric(rows, "pos_rmse_m"),
        "rot_rmse_deg": average_metric(rows, "rot_rmse_deg"),
        "ate_rmse_m": average_metric(rows, "ate_rmse_m"),
    }

    summary = {
        "dataset": args.dataset,
        "num_sequences_requested": len(sequences),
        "num_sequences_succeeded": len(rows),
        "num_sequences_failed": len(failures),
        "sequences": rows,
        "averages": averages,
        "failures": failures,
    }

    (dataset_eval_dir / "summary.json").write_text(
        json.dumps(to_jsonable(summary), indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("Per-sequence summary")
    print("=" * 72)
    for row in rows:
        print(
            f"{row['sequence']}: "
            f"POS_RMSE={row['pos_rmse_m']:.6f} m | "
            f"ROT_RMSE={row['rot_rmse_deg']:.6f} deg | "
            f"ATE={row['ate_rmse_m']:.6f} m"
        )

    print("\n" + "=" * 72)
    print("Averages")
    print("=" * 72)
    print(f"AVG POS_RMSE = {averages['pos_rmse_m']:.6f} m")
    print(f"AVG ROT_RMSE = {averages['rot_rmse_deg']:.6f} deg")
    print(f"AVG ATE      = {averages['ate_rmse_m']:.6f} m")

    if failures:
        print("\n" + "=" * 72)
        print("Failures")
        print("=" * 72)
        for failure in failures:
            print(f"{failure['sequence']}: {failure['error']}")

    print(f"\nSaved dataset summary to {dataset_eval_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
