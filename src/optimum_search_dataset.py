"""Deterministic coarse-to-fine tuning for `main_filter.py` hyperparameters across an entire dataset."""

from __future__ import annotations

import argparse
import concurrent.futures
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
    from main_filter import CONFIG, RunnerConfig, run_filter
except ImportError:
    from .main_filter import CONFIG, RunnerConfig, run_filter

from trajectory import Trajectory

SEARCH_KEYS = (
    "sigma_na", "sigma_ng", "sigma_nba", "sigma_nbg",
    "assumed_sigma_rel_x_t", "assumed_sigma_rel_y_t", "assumed_sigma_rel_z_t", 
    "meas_cov_scale", "initial_attitude_sigma_deg",
    "initial_velocity_sigma_mps", "initial_position_sigma_m", "initial_z_sigma_m",
    "initial_bg_sigma_rps", "initial_ba_sigma_mps2",
)

COARSE_LOG10_RANGES = {
    "sigma_na": (-3.0, -0.7), "sigma_ng": (-3.0, -0.7),
    "sigma_nba": (-5.5, -2.0), "sigma_nbg": (-6.5, -3.5),
    "assumed_sigma_rel_x_t": (-2.3, -0.8), "assumed_sigma_rel_y_t": (-2.3, -0.8), "assumed_sigma_rel_z_t": (-2.3, -0.8), 
    "meas_cov_scale": (-0.7, 0.6),
    "initial_attitude_sigma_deg": (-1.0, 0.8), "initial_velocity_sigma_mps": (-1.2, 0.4),
    "initial_position_sigma_m": (-2.5, -0.3), "initial_z_sigma_m": (-2.5, -0.3),
    "initial_bg_sigma_rps": (-4.0, -1.5), "initial_ba_sigma_mps2": (-3.0, -0.3),
}

REFINE_LOG10_HALF_WIDTH = {
    "sigma_na": 0.5, "sigma_ng": 0.5, "sigma_nba": 0.5, "sigma_nbg": 0.5,
    "assumed_sigma_rel_x_t": 0.5, "assumed_sigma_rel_y_t": 0.5, "assumed_sigma_rel_z_t": 0.5, 
    "meas_cov_scale": 0.35, "initial_attitude_sigma_deg": 0.5,
    "initial_velocity_sigma_mps": 0.5, "initial_position_sigma_m": 0.5, "initial_z_sigma_m": 0.5,
    "initial_bg_sigma_rps": 0.5, "initial_ba_sigma_mps2": 0.5,
}


def score_run(results: dict, config: RunnerConfig, optimize_for_pos_rmse: bool = False) -> dict[str, float]:
    del config

    diagnostics = results["diagnostics"]
    pos_rmse = float(diagnostics["position_rmse_m"])
    rot_rmse_deg = float(diagnostics["rotation_rmse_deg"])
    rejected = int(results["num_updates_rejected"])

    try:
        ate_rmse = compute_ate_from_results(
            dataset=results["dataset"],
            sequence=results["sequence"],
            ground_truth_table=results["ground_truth"],
            estimate_table=results["trajectory"],
        )
    except Exception as exc:
        print(f"  -> Errore nel calcolo dell'ATE: {exc}")
        ate_rmse = 9999.0

    if optimize_for_pos_rmse:
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
    ground_truth_table,
    estimate_table,
) -> float:
    import numpy as np

    gt_path = ROOT / "data" / dataset / "processed" / sequence / "stamped_groundtruth.txt"
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


def _eval_sequence_worker(args_tuple: tuple) -> dict | None:
    """Worker function per valutare una singola sequenza in un processo separato."""
    base_config, candidate, optimize_for_pos_rmse = args_tuple
    
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
        metrics = score_run(results, config, optimize_for_pos_rmse=optimize_for_pos_rmse)
        return metrics
    except Exception as exc:
        print(f"  [Worker Error] Sequence {config.sequence} failed: {exc}")
        return None


def evaluate_candidate_on_dataset(
    base_configs: list[RunnerConfig],
    candidate: dict[str, float],
    optimize_for_pos_rmse: bool = False,
    num_jobs: int = 4,
) -> dict | None:
    """Valuta lo stesso set di parametri su tutte le sequenze in parallelo e calcola la media."""
    num_sequences = len(base_configs)
    if num_sequences == 0:
        return None

    args_list = [(cfg, candidate, optimize_for_pos_rmse) for cfg in base_configs]
    results_list = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_jobs) as executor:
        results_list = list(executor.map(_eval_sequence_worker, args_list))

    # Se un qualsiasi worker ha restituito None (ha fallito), scartiamo il candidato
    if None in results_list:
        return None

    aggregate_metrics = {
        "score": 0.0,
        "ate_rmse": 0.0,
        "position_rmse_m": 0.0,
        "rotation_rmse_deg": 0.0,
        "num_updates_rejected": 0,
    }

    # Somma i risultati
    for metrics in results_list:
        aggregate_metrics["score"] += metrics["score"]
        aggregate_metrics["ate_rmse"] += metrics["ate_rmse"]
        aggregate_metrics["position_rmse_m"] += metrics["position_rmse_m"]
        aggregate_metrics["rotation_rmse_deg"] += metrics["rotation_rmse_deg"]
        aggregate_metrics["num_updates_rejected"] += metrics["num_updates_rejected"]

    # Calcola la media per tutte le metriche
    for key in aggregate_metrics:
        aggregate_metrics[key] /= num_sequences

    return {"params": candidate, "metrics": aggregate_metrics}


def print_trial(label: str, trial_index: int, evaluated: dict) -> None:
    metrics = evaluated["metrics"]
    print(
        f"[{label} {trial_index:03d}] "
        f"MEAN_score={metrics['score']:.6f} "
        f"MEAN_ate={metrics['ate_rmse']:.6f} "
        f"MEAN_pos={metrics['position_rmse_m']:.6f} "
        f"MEAN_rot={metrics['rotation_rmse_deg']:.6f} "
        f"MEAN_rej={metrics['num_updates_rejected']:.1f}"
    )


def maybe_update_best(best: dict | None, candidate: dict) -> tuple[dict | None, bool]:
    if best is None or candidate["metrics"]["score"] < best["metrics"]["score"]:
        return candidate, True
    return best, False


def run_search(
    base_configs: list[RunnerConfig],
    coarse_trials: int,
    refine_trials: int,
    seed: int,
    optimize_for_pos_rmse: bool = False,
    num_jobs: int = 4,
) -> dict:
    rng = random.Random(seed)
    completed: list[dict] = []
    best: dict | None = None

    for idx in range(coarse_trials):
        evaluated = evaluate_candidate_on_dataset(
            base_configs,
            sample_coarse_params(rng),
            optimize_for_pos_rmse=optimize_for_pos_rmse,
            num_jobs=num_jobs,
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
        evaluated = evaluate_candidate_on_dataset(
            base_configs,
            sample_refined_params(rng, refine_center),
            optimize_for_pos_rmse=optimize_for_pos_rmse,
            num_jobs=num_jobs,
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
    
    # Salviamo un base config generico di riferimento
    generic_base_config = asdict(base_configs[0])
    generic_base_config["sequence"] = "ALL_SEQUENCES"

    return {
        "best": best,
        "top_k": completed_sorted[:10],
        "completed_trials": len(completed),
        "seed": seed,
        "coarse_trials": coarse_trials,
        "refine_trials": refine_trials,
        "objective": "position_rmse_m" if optimize_for_pos_rmse else "ate_rmse",
        "base_config": generic_base_config,
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
    parser.add_argument("--sequence", type=str, default=None, help="Sequence to tune. If omitted, tunes all sequences.")
    parser.add_argument("--gt", action="store_true", help="Use relative_motions.txt instead of regressed_relative_motions.txt.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--coarse-trials", type=int, default=300)
    parser.add_argument("--refine-trials", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--optimize-for-pos-rmse",
        action="store_true",
        help="Optimize using position RMSE instead of ATE as the main score term.",
    )
    parser.add_argument(
        "--disable-adaptive-covariance",
        action="store_true",
        help="Disable residual-based adaptive covariance inflation during tuning.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "tuning",
        help="Directory to save the best result JSONs.",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=4,
        help="Numero di processi paralleli da lanciare per la valutazione delle sequenze.",
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

    processed_dir = CONFIG.data_root / args.dataset / "processed"
    if not processed_dir.exists():
        print(f"Error: Dataset path {processed_dir} not found.")
        return

    if args.sequence:
        sequences = [args.sequence]
    else:
        sequences = [d.name for d in processed_dir.iterdir() if d.is_dir()]

    sequences.sort()

    print("\n" + "=" * 50)
    print(f"Preparing configurations for Dataset: {args.dataset}")
    print("=" * 50)

    base_configs = []
    for seq in sequences:
        sequence_dir = processed_dir / seq
        if not sequence_dir.exists():
            print(f"Warning: sequence folder not found at {sequence_dir}. Skipping.")
            continue

        config = replace(
            CONFIG,
            dataset=args.dataset,
            sequence=seq,
            use_gt=args.gt,
            max_frames=args.max_frames,
            adaptive_covariance=not args.disable_adaptive_covariance,
            interactive_plot=False,
            plot_transformer=False,
            plot_projections=False,
            **dataset_specific_overrides(args.dataset),
        )
        base_configs.append(config)

    if not base_configs:
        print("Error: No valid sequences to optimize.")
        return

    print(f"Optimizing across {len(base_configs)} sequences: {', '.join([c.sequence for c in base_configs])}")
    print(f"Using {args.jobs} parallel jobs.")

    try:
        summary = run_search(
            base_configs=base_configs,
            coarse_trials=args.coarse_trials,
            refine_trials=args.refine_trials,
            seed=args.seed,
            optimize_for_pos_rmse=args.optimize_for_pos_rmse,
            num_jobs=args.jobs,
        )
    except Exception as exc:
        print(f"Error while optimizing the dataset: {exc}")
        return

    best = summary["best"]
    assert best is not None

    print("\nBest OVERALL result (Averages)")
    print(
        f"score={best['metrics']['score']:.6f} "
        f"ate={best['metrics']['ate_rmse']:.6f} "
        f"pos={best['metrics']['position_rmse_m']:.6f} "
        f"rot={best['metrics']['rotation_rmse_deg']:.6f} "
        f"rej={best['metrics']['num_updates_rejected']:.1f}"
    )

    dataset_output_file = args.output_dir / args.dataset / "best_dataset_filter_params.json"
    dataset_output_file.parent.mkdir(parents=True, exist_ok=True)
    dataset_output_file.write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
    print(f"Saved dataset summary to {dataset_output_file}")


if __name__ == "__main__":
    main()
