"""Deterministic coarse-to-fine tuning for `main_filter.py` hyperparameters."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

try:
    from compute_ate import compute_ate_from_tables
    from main_filter import CONFIG, RunnerConfig, run_filter
except ImportError:
    from .compute_ate import compute_ate_from_tables
    from .main_filter import CONFIG, RunnerConfig, run_filter

SEARCH_KEYS = (
    "sigma_na", "sigma_ng", "sigma_nba", "sigma_nbg",
    "assumed_sigma_rel_t", "meas_cov_scale", "initial_attitude_sigma_deg",
    "initial_velocity_sigma_mps", "initial_position_sigma_m", "initial_z_sigma_m",
    "initial_bg_sigma_rps", "initial_ba_sigma_mps2",
)

COARSE_LOG10_RANGES = {
    "sigma_na": (-3.0, -0.7), "sigma_ng": (-3.0, -0.7),
    "sigma_nba": (-5.5, -2.0), "sigma_nbg": (-6.5, -3.5),
    "assumed_sigma_rel_t": (-2.3, -0.8), "meas_cov_scale": (-0.7, 0.6),
    "initial_attitude_sigma_deg": (-1.0, 0.8), "initial_velocity_sigma_mps": (-1.2, 0.4),
    "initial_position_sigma_m": (-2.5, -0.3), "initial_z_sigma_m": (-2.5, -0.3),
    "initial_bg_sigma_rps": (-4.0, -1.5), "initial_ba_sigma_mps2": (-3.0, -0.3),
}

REFINE_LOG10_HALF_WIDTH = {
    "sigma_na": 0.5, "sigma_ng": 0.5, "sigma_nba": 0.5, "sigma_nbg": 0.5,
    "assumed_sigma_rel_t": 0.5, "meas_cov_scale": 0.35, "initial_attitude_sigma_deg": 0.5,
    "initial_velocity_sigma_mps": 0.5, "initial_position_sigma_m": 0.5, "initial_z_sigma_m": 0.5,
    "initial_bg_sigma_rps": 0.5, "initial_ba_sigma_mps2": 0.5,
}


def score_run(results: dict, config: RunnerConfig) -> dict[str, float]:
    del config

    diagnostics = results["diagnostics"]
    pos_rmse = float(diagnostics["position_rmse_m"])
    rot_rmse_deg = float(diagnostics["rotation_rmse_deg"])
    rejected = int(results["num_updates_rejected"])

    try:
        ate_rmse, _, _, _ = compute_ate_from_tables(
            groundtruth_table=results["ground_truth"],
            estimate_table=results["trajectory"],
            groundtruth_time_scale=1.0,
            estimate_time_scale=1.0,
            max_time_diff=0.02,
        )
    except Exception as exc:
        print(f"  -> Errore nel calcolo dell'ATE: {exc}")
        ate_rmse = 9999.0

    score = ate_rmse + 0.05 * rot_rmse_deg + 0.001 * rejected

    return {
        "score": score,
        "ate_rmse": ate_rmse,
        "position_rmse_m": pos_rmse,
        "rotation_rmse_deg": rot_rmse_deg,
        "num_updates_rejected": rejected,
    }


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


def evaluate_candidate(base_config: RunnerConfig, candidate: dict[str, float]) -> dict | None:
    config = replace(
        base_config,
        **candidate,
        interactive_plot=False,
        plot_transformer=False,
        plot_projections=False,
    )
    try:
        results = run_filter(config)
    except Exception as exc:
        candidate_label = ", ".join(f"{key}={candidate[key]:.4g}" for key in SEARCH_KEYS[:4])
        print(f"  trial failed: {type(exc).__name__}: {exc} [{candidate_label}, ...]")
        return None

    metrics = score_run(results, config)
    return {"params": candidate, "metrics": metrics}


def print_trial(label: str, trial_index: int, evaluated: dict) -> None:
    metrics = evaluated["metrics"]
    print(
        f"[{label} {trial_index:03d}] "
        f"score={metrics['score']:.6f} "
        f"ate={metrics['ate_rmse']:.6f} "
        f"pos={metrics['position_rmse_m']:.6f} "
        f"rot={metrics['rotation_rmse_deg']:.6f} "
        f"rej={metrics['num_updates_rejected']}"
    )


def maybe_update_best(best: dict | None, candidate: dict) -> tuple[dict | None, bool]:
    if best is None or candidate["metrics"]["score"] < best["metrics"]["score"]:
        return candidate, True
    return best, False


def run_search(base_config: RunnerConfig, coarse_trials: int, refine_trials: int, seed: int) -> dict:
    rng = random.Random(seed)
    completed: list[dict] = []
    best: dict | None = None

    for idx in range(coarse_trials):
        evaluated = evaluate_candidate(base_config, sample_coarse_params(rng))
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
        evaluated = evaluate_candidate(base_config, sample_refined_params(rng, refine_center))
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
        "coarse_trials": coarse_trials,
        "refine_trials": refine_trials,
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
    parser.add_argument("--sequence", type=str, default=None, help="Sequence to tune. If omitted, tunes all sequences.")
    parser.add_argument("--gt", action="store_true", help="Use relative_motions.txt instead of regressed_relative_motions.txt.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--coarse-trials", type=int, default=40)
    parser.add_argument("--refine-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
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

    processed_dir = CONFIG.data_root / args.dataset / "processed"
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

        base_config = replace(
            CONFIG,
            dataset=args.dataset,
            sequence=seq,
            use_gt=args.gt,
            max_frames=args.max_frames,
            interactive_plot=False,
            plot_transformer=False,
            plot_projections=False,
            **dataset_specific_overrides(args.dataset),
        )

        try:
            summary = run_search(
                base_config=base_config,
                coarse_trials=args.coarse_trials,
                refine_trials=args.refine_trials,
                seed=args.seed,
            )
        except Exception as exc:
            print(f"Errorwhile optimizing {seq}: {exc}")
            continue

        best = summary["best"]
        assert best is not None

        print("\nBest result")
        print(
            f"score={best['metrics']['score']:.6f} "
            f"ate={best['metrics']['ate_rmse']:.6f} "
            f"pos={best['metrics']['position_rmse_m']:.6f} "
            f"rot={best['metrics']['rotation_rmse_deg']:.6f} "
            f"rej={best['metrics']['num_updates_rejected']}"
        )

        seq_output_file = args.output_dir / seq / "best_filter_params.json"
        seq_output_file.parent.mkdir(parents=True, exist_ok=True)
        seq_output_file.write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
        print(f"Saved summary to {seq_output_file}")


if __name__ == "__main__":
    main()
