"""Deterministic coarse-to-fine tuning for `main_filter.py` hyperparameters."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import random
from dataclasses import asdict, replace
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
EVAL_SRC = ROOT / "evaluation" / "rpg_trajectory_evaluation" / "src" / "rpg_trajectory_evaluation"
if str(EVAL_SRC) not in sys.path:
    sys.path.insert(0, str(EVAL_SRC))

try:
    from main_filter import CONFIG, RunnerConfig, _load_relative_motion_table, run_filter
except ImportError:
    from .main_filter import CONFIG, RunnerConfig, _load_relative_motion_table, run_filter

from trajectory import Trajectory

SEARCH_KEYS = (
    "sigma_na", "sigma_ng", "sigma_nba", "sigma_nbg",
    "assumed_sigma_rel_x_t", "assumed_sigma_rel_y_t", "assumed_sigma_rel_z_t", 
    "meas_cov_scale", "initial_attitude_sigma_deg",
    "initial_velocity_sigma_mps", "initial_position_sigma_m", "initial_z_sigma_m",
    "initial_bg_sigma_rps", "initial_ba_sigma_mps2",
    "network_scale_x", "network_scale_y", "network_scale_z",
)
SCALE_KEYS = ("network_scale_x", "network_scale_y", "network_scale_z")

COARSE_LOG10_RANGES = {
    "sigma_na": (-3.0, -0.7), "sigma_ng": (-3.0, -0.7),
    "sigma_nba": (-5.5, -2.0), "sigma_nbg": (-6.5, -3.5),
    "assumed_sigma_rel_x_t": (-2.3, -0.8), "assumed_sigma_rel_y_t": (-2.3, -0.8), "assumed_sigma_rel_z_t": (-2.3, -0.8), 
    "meas_cov_scale": (-0.7, 0.6),
    "initial_attitude_sigma_deg": (-1.0, 0.8), "initial_velocity_sigma_mps": (-1.2, 0.4),
    "initial_position_sigma_m": (-2.5, -0.3), "initial_z_sigma_m": (-2.5, -0.3),
    "initial_bg_sigma_rps": (-4.0, -1.5), "initial_ba_sigma_mps2": (-3.0, -0.3),
    "network_scale_x": (-1.5, 0.3), "network_scale_y": (-1.5, 0.3),
    "network_scale_z": (-1.5, 0.3),
}

REFINE_LOG10_HALF_WIDTH = {
    "sigma_na": 0.5, "sigma_ng": 0.5, "sigma_nba": 0.5, "sigma_nbg": 0.5,
    "assumed_sigma_rel_x_t": 0.5, "assumed_sigma_rel_y_t": 0.5, "assumed_sigma_rel_z_t": 0.5, 
    "meas_cov_scale": 0.35, "initial_attitude_sigma_deg": 0.5,
    "initial_velocity_sigma_mps": 0.5, "initial_position_sigma_m": 0.5, "initial_z_sigma_m": 0.5,
    "initial_bg_sigma_rps": 0.5, "initial_ba_sigma_mps2": 0.5,
    "network_scale_x": 0.2, "network_scale_y": 0.2, "network_scale_z": 0.2,
}


def score_run(
    results: dict,
    config: RunnerConfig,
    optimize_for_pos_rmse: bool = False,
    ate_only: bool = False,
) -> dict[str, float]:
    diagnostics = results["diagnostics"]
    pos_rmse = float(diagnostics["position_rmse_m"])
    rot_rmse_deg = float(diagnostics["rotation_rmse_deg"])
    rejected = int(results["num_updates_rejected"])

    try:
        ate_rmse = compute_ate_from_results(
            dataset=results["dataset"],
            sequence=results["sequence"],
            processed_root=config.processed_root,
            ground_truth_table=results["ground_truth"],
            estimate_table=results["trajectory"],
        )
    except Exception as exc:
        print(f"  -> Errore nel calcolo dell'ATE: {exc}")
        ate_rmse = 9999.0

    if ate_only:
        score = ate_rmse
    elif optimize_for_pos_rmse:
        score = pos_rmse + 0.05 * rot_rmse_deg + 0.001 * rejected
    else:
        score = ate_rmse + 0.05 * rot_rmse_deg + 0.001 * rejected

    return {
        "score": score,
        "ate_rmse": ate_rmse,
        "position_rmse_m": pos_rmse,
        "rotation_rmse_deg": rot_rmse_deg,
        "num_updates_rejected": rejected,
    }


def load_numeric_pose_table(path: Path) -> list[list[float]]:
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
    return rows


def infer_time_scale_to_seconds(timestamps: list[float]) -> float:
    if len(timestamps) < 2:
        return 1.0

    diffs = [curr - prev for prev, curr in zip(timestamps[:-1], timestamps[1:]) if curr > prev]
    median_dt = sorted(diffs)[len(diffs) // 2] if diffs else 0.0

    if median_dt > 1e7:
        return 1e-9
    if median_dt > 1e1:
        return 1e-6
    return 1.0


def write_seconds_pose_table(src_path: Path, dst_path: Path) -> Path:
    import numpy as np

    table = np.asarray(load_numeric_pose_table(src_path), dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != 8:
        raise ValueError(f"{src_path} has shape {table.shape}, expected N x 8 pose rows.")
    table[:, 0] *= infer_time_scale_to_seconds(table[:, 0].tolist())
    np.savetxt(dst_path, table, fmt="%.9f")
    return dst_path


def compute_ate_from_results(
    dataset: str,
    sequence: str,
    processed_root: Path,
    ground_truth_table,
    estimate_table,
) -> float:
    import numpy as np

    gt_path = processed_root / sequence / "stamped_groundtruth.txt"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth file does not exist: {gt_path}")

    with tempfile.TemporaryDirectory(prefix=f"optimum_search_{dataset}_{sequence}_", dir=ROOT) as temp_dir:
        eval_dir = Path(temp_dir)
        eval_gt_path = eval_dir / "stamped_groundtruth.txt"
        eval_est_path = eval_dir / "stamped_traj_estimate.txt"

        write_seconds_pose_table(gt_path, eval_gt_path)
        np.savetxt(eval_est_path, np.asarray(estimate_table, dtype=np.float64), fmt="%.9f")

        # Silence the RPG toolbox's verbose console output during random search.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            traj = Trajectory(str(eval_dir), est_type="traj_est")
            if not traj.data_loaded:
                raise RuntimeError(f"RPG trajectory loader failed for {eval_dir}")
            traj.compute_absolute_error()
        return float(traj.abs_errors["abs_e_trans_stats"]["rmse"])


def sample_log_uniform(rng: random.Random, low_log10: float, high_log10: float) -> float:
    return 10.0 ** rng.uniform(low_log10, high_log10)


def sample_coarse_params(rng: random.Random) -> dict[str, float]:
    return {key: sample_log_uniform(rng, *COARSE_LOG10_RANGES[key]) for key in SEARCH_KEYS}


def sample_refined_params(rng: random.Random, center: dict[str, float]) -> dict[str, float]:
    refined: dict[str, float] = {}
    for key in SEARCH_KEYS:
        half_width = REFINE_LOG10_HALF_WIDTH[key]
        center_log10 = math.log10(center[key])
        refined[key] = sample_log_uniform(rng, center_log10 - half_width, center_log10 + half_width)
    return refined


def sample_scale_warmup_params(
    rng: random.Random,
    base_params: dict[str, float],
    center: dict[str, float],
    half_width: float = 0.45,
) -> dict[str, float]:
    """Vary only XYZ scales around an analytic GT-derived starting point."""
    candidate = dict(base_params)
    for key in SCALE_KEYS:
        low_bound, high_bound = COARSE_LOG10_RANGES[key]
        center_log10 = math.log10(center[key])
        low = max(low_bound, center_log10 - half_width)
        high = min(high_bound, center_log10 + half_width)
        candidate[key] = sample_log_uniform(rng, low, high)
    return candidate


def estimate_axis_scale_seed(base_config: RunnerConfig) -> dict[str, float]:
    """Estimate an XYZ least-squares scale from this sequence's full GT."""
    import numpy as np

    sequence_path = base_config.processed_root / base_config.sequence
    pred = _load_relative_motion_table(
        sequence_path,
        use_gt=False,
        relative_motion_filename=base_config.relative_motion_filename,
    )
    gt = _load_relative_motion_table(sequence_path, use_gt=True)
    gt_lookup = {
        (int(row[0]), int(row[1])): row[2:5]
        for row in gt
    }

    pred_xyz = []
    gt_xyz = []
    max_transitions = (
        None
        if base_config.max_frames is None
        else max(0, base_config.max_frames - 1)
    )
    for row in pred[:max_transitions]:
        target = gt_lookup.get((int(row[0]), int(row[1])))
        if target is not None:
            pred_xyz.append(row[2:5])
            gt_xyz.append(target)
    if not pred_xyz:
        raise ValueError(
            f"No matching prediction/GT timestamps for {base_config.sequence}."
        )

    pred_xyz = np.asarray(pred_xyz, dtype=np.float64)
    gt_xyz = np.asarray(gt_xyz, dtype=np.float64)
    denominator = np.sum(pred_xyz * pred_xyz, axis=0)
    scale = np.divide(
        np.sum(pred_xyz * gt_xyz, axis=0),
        denominator,
        out=np.ones(3, dtype=np.float64),
        where=denominator > 1e-12,
    )
    scale = np.clip(scale, 10.0 ** -1.5, 10.0 ** 0.3)
    return {
        "network_scale_x": float(scale[0]),
        "network_scale_y": float(scale[1]),
        "network_scale_z": float(scale[2]),
    }


def evaluate_candidate(
    base_config: RunnerConfig,
    candidate: dict[str, float],
    optimize_for_pos_rmse: bool = False,
    ate_only: bool = False,
) -> dict | None:
    config = replace(
        base_config,
        **candidate,
        interactive_plot=False,
        plot_transformer=False,
        plot_projections=False,
        save_trajectory_file=False,
        save_diagnostic_plots=False,
    )
    try:
        results = run_filter(config)
    except Exception as exc:
        candidate_label = ", ".join(f"{key}={candidate[key]:.4g}" for key in SEARCH_KEYS[:4])
        print(f"  trial failed: {type(exc).__name__}: {exc} [{candidate_label}, ...]")
        return None

    metrics = score_run(
        results,
        config,
        optimize_for_pos_rmse=optimize_for_pos_rmse,
        ate_only=ate_only,
    )
    return {"params": candidate, "metrics": metrics}


def print_trial(label: str, trial_index: int, evaluated: dict) -> None:
    metrics = evaluated["metrics"]
    params = evaluated["params"]
    print(
        f"[{label} {trial_index:03d}] "
        f"score={metrics['score']:.6f} "
        f"ate={metrics['ate_rmse']:.6f} "
        f"pos={metrics['position_rmse_m']:.6f} "
        f"rot={metrics['rotation_rmse_deg']:.6f} "
        f"rej={metrics['num_updates_rejected']} "
        "scale_xyz=["
        f"{params['network_scale_x']:.5f},"
        f"{params['network_scale_y']:.5f},"
        f"{params['network_scale_z']:.5f}]"
    )


def maybe_update_best(best: dict | None, candidate: dict) -> tuple[dict | None, bool]:
    if best is None or candidate["metrics"]["score"] < best["metrics"]["score"]:
        return candidate, True
    return best, False


def run_search(
    base_config: RunnerConfig,
    scale_trials: int,
    coarse_trials: int,
    refine_trials: int,
    seed: int,
    optimize_for_pos_rmse: bool = False,
    ate_only: bool = False,
) -> dict:
    rng = random.Random(seed)
    completed: list[dict] = []
    best: dict | None = None

    analytic_scale_seed = estimate_axis_scale_seed(base_config)
    seeded_params = {key: float(getattr(base_config, key)) for key in SEARCH_KEYS}
    seeded_params.update(analytic_scale_seed)
    seeded = evaluate_candidate(
        base_config,
        seeded_params,
        optimize_for_pos_rmse=optimize_for_pos_rmse,
        ate_only=ate_only,
    )
    if seeded is not None:
        completed.append(seeded)
        print_trial("seed", 0, seeded)
        best, _ = maybe_update_best(best, seeded)

    for idx in range(scale_trials):
        evaluated = evaluate_candidate(
            base_config,
            sample_scale_warmup_params(rng, seeded_params, analytic_scale_seed),
            optimize_for_pos_rmse=optimize_for_pos_rmse,
            ate_only=ate_only,
        )
        if evaluated is None:
            continue
        completed.append(evaluated)
        print_trial("scale", idx, evaluated)
        best, improved = maybe_update_best(best, evaluated)
        if improved:
            print("  new best scale candidate")

    for idx in range(coarse_trials):
        evaluated = evaluate_candidate(
            base_config,
            sample_coarse_params(rng),
            optimize_for_pos_rmse=optimize_for_pos_rmse,
            ate_only=ate_only,
        )
        if evaluated is None:
            continue
        completed.append(evaluated)
        print_trial("coarse", idx, evaluated)
        best, improved = maybe_update_best(best, evaluated)
        if improved:
            print("  new best coarse candidate")

    if best is None:
        raise RuntimeError("No valid coarse trial completed.")

    refine_center = best["params"]
    for idx in range(refine_trials):
        evaluated = evaluate_candidate(
            base_config,
            sample_refined_params(rng, refine_center),
            optimize_for_pos_rmse=optimize_for_pos_rmse,
            ate_only=ate_only,
        )
        if evaluated is None:
            continue
        completed.append(evaluated)
        print_trial("refine", idx, evaluated)
        best, improved = maybe_update_best(best, evaluated)
        if improved:
            refine_center = best["params"]
            print("  new best refined candidate")

    completed_sorted = sorted(completed, key=lambda item: item["metrics"]["score"])
    return {
        "best": best,
        "top_k": completed_sorted[:10],
        "completed_trials": len(completed),
        "seed": seed,
        "scale_trials": scale_trials,
        "coarse_trials": coarse_trials,
        "refine_trials": refine_trials,
        "objective": (
            "ate_rmse_only"
            if ate_only
            else "position_rmse_m"
            if optimize_for_pos_rmse
            else "ate_rmse_with_stability_penalties"
        ),
        "protocol": (
            "per_sequence_supervised_prefix"
            if base_config.max_frames is not None
            else "per_sequence_supervised_oracle"
        ),
        "analytic_scale_seed": analytic_scale_seed,
        "base_config": asdict(base_config),
    }


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune main_filter hyperparameters with deterministic random search.")
    parser.add_argument("--dataset", type=str, default=CONFIG.dataset)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Override the default data/<dataset>/processed directory.",
    )
    parser.add_argument("--sequence", type=str, default=None, help="Sequence to tune. If omitted, tunes all sequences.")
    parser.add_argument("--gt", action="store_true", help="Use relative_motions.txt instead of regressed_relative_motions.txt.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--relative-motions-file",
        type=str,
        default=None,
        help="Raw prediction path; supports the {sequence} placeholder.",
    )
    parser.add_argument(
        "--scale-trials",
        type=int,
        default=150,
        help="Trials varying only XYZ scales before the full search.",
    )
    parser.add_argument("--coarse-trials", type=int, default=400)
    parser.add_argument("--refine-trials", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--optimize-for-pos-rmse",
        action="store_true",
        help="Optimize using position RMSE instead of ATE as the main score term.",
    )
    parser.add_argument(
        "--ate-only",
        action="store_true",
        help="Rank candidates by RPG ATE RMSE only, without rotation/rejection penalties.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "tuning",
        help="Directory to save the best result JSONs.",
    )
    return parser.parse_args()


def dataset_specific_overrides(dataset: str) -> dict:
    dataset_name = dataset.lower()
    if dataset_name == "eds":
        return {
            "imu_axis_multipliers": (-1.0, -1.0, 1.0),
            "gravity_world_mps2": (0.0, 0.0, -9.80665),
        }
    return {}


def main() -> None:
    args = parse_args()
    if args.ate_only and args.optimize_for_pos_rmse:
        raise ValueError("--ate-only and --optimize-for-pos-rmse are mutually exclusive.")

    processed_dir = (
        args.processed_dir
        if args.processed_dir is not None
        else CONFIG.data_root / args.dataset / "processed"
    )
    if not processed_dir.exists():
        print(f"Error: Dataset path {processed_dir} not found.")
        return

    if args.sequence:
        sequences = [args.sequence]
    else:
        sequences = [d.name for d in processed_dir.iterdir() if d.is_dir()]

    sequences.sort()

    for seq in sequences:
        print("\n" + "=" * 50)
        print(f"Optimizing: {seq}")
        print("=" * 50)

        sequence_dir = processed_dir / seq
        if not sequence_dir.exists():
            print(
                f"Error while optimizing {seq}: sequence folder not found at {sequence_dir}. "
                f"Check `--dataset` and `--sequence`."
            )
            continue

        base_config = replace(
            CONFIG,
            dataset=args.dataset,
            sequence=seq,
            processed_dir=processed_dir,
            use_gt=args.gt,
            max_frames=args.max_frames,
            relative_motion_filename=args.relative_motions_file,
            interactive_plot=False,
            plot_transformer=False,
            plot_projections=False,
            **dataset_specific_overrides(args.dataset),
        )

        try:
            summary = run_search(
                base_config=base_config,
                scale_trials=args.scale_trials,
                coarse_trials=args.coarse_trials,
                refine_trials=args.refine_trials,
                seed=args.seed,
                optimize_for_pos_rmse=args.optimize_for_pos_rmse,
                ate_only=args.ate_only,
            )
        except Exception as exc:
            print(f"Error while optimizing {seq}: {exc}")
            continue

        best = summary["best"]
        assert best is not None

        print("\nBest result")
        print(
            f"score={best['metrics']['score']:.6f} "
            f"ate={best['metrics']['ate_rmse']:.6f} "
            f"pos={best['metrics']['position_rmse_m']:.6f} "
            f"rot={best['metrics']['rotation_rmse_deg']:.6f} "
            f"rej={best['metrics']['num_updates_rejected']} "
            "scale_xyz=["
            f"{best['params']['network_scale_x']:.6f}, "
            f"{best['params']['network_scale_y']:.6f}, "
            f"{best['params']['network_scale_z']:.6f}]"
        )

        seq_output_file = args.output_dir / seq / "best_filter_params.json"
        seq_output_file.parent.mkdir(parents=True, exist_ok=True)
        seq_output_file.write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
        print(f"Saved summary to {seq_output_file}")


if __name__ == "__main__":
    main()
