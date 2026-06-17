"""Run an EDS-wide V3 tuning grid and summarize cross-sequence metrics."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_eds_batch import run_eds_batch  # noqa: E402


DEFAULT_SEQUENCES = [
    "01_peanuts_light",
    "02_rocket_earth_light",
    "03_rocket_earth_dark",
    "06_ziggy_and_fuzz",
    "07_ziggy_and_fuzz_hdr",
    "08_peanuts_running",
    "09_ziggy_flying_pieces",
    "11_all_characters",
]


def scale_name(scale: float) -> str:
    return f"{scale:g}".replace(".", "p")


def make_grid(
    scales: list[float],
    include_secondary: bool,
    secondary_scales: list[float] | None = None,
) -> list[dict[str, object]]:
    """Build a focused V3 grid over covariance scale and extra V3 switches."""

    grid: list[dict[str, object]] = []
    for scale in scales:
        grid.append(
            {
                "run_name": f"scale_{scale_name(scale)}",
                "preset": "openvins_best",
                "extra": ["--meas_cov_scale", str(scale)],
                "family": "cov_scale",
                "scale": scale,
            }
        )

    if include_secondary:
        for scale in (secondary_scales or [1.2649054158337365, 2.5]):
            grid.append(
                {
                    "run_name": f"halfR_scale_{scale_name(scale)}",
                    "preset": "v3_midpoint_half_R",
                    "extra": ["--meas_cov_scale", str(scale)],
                    "family": "midpoint_half_R",
                    "scale": scale,
                }
            )
            for mode in ("inflate", "reject"):
                grid.append(
                    {
                        "run_name": f"edge_{mode}_scale_{scale_name(scale)}",
                        "preset": "openvins_best",
                        "extra": ["--meas_cov_scale", str(scale), "--edge_robust_mode", mode],
                        "family": f"edge_{mode}",
                        "scale": scale,
                    }
                )
    return grid


def _finite(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def summarize_run(run_name: str, family: str, scale: float, rows: list[dict[str, float | str]]) -> dict[str, float | str]:
    """Summarize one configuration across all successful EDS sequences."""

    successful = [row for row in rows if row.get("status") == "ok"]
    sim3 = _finite([float(row.get("sim3_aligned_scaled_ate_rmse_m", math.nan)) for row in successful])
    se3 = _finite([float(row.get("se3_aligned_ate_rmse_m", math.nan)) for row in successful])
    raw = _finite([float(row.get("raw_ate_rmse_m", math.nan)) for row in successful])
    first20 = _finite([float(row.get("first20_sim3_ate_rmse_m", math.nan)) for row in successful])
    rot = _finite([float(row.get("raw_rotation_rmse_deg", math.nan)) for row in successful])
    rejected = _finite([float(row.get("updates_rejected", math.nan)) for row in successful])

    def mean(arr: np.ndarray) -> float:
        return float(np.mean(arr)) if arr.size else math.nan

    def median(arr: np.ndarray) -> float:
        return float(np.median(arr)) if arr.size else math.nan

    def worst(arr: np.ndarray) -> float:
        return float(np.max(arr)) if arr.size else math.nan

    return {
        "run_name": run_name,
        "family": family,
        "scale": scale,
        "num_ok": len(successful),
        "num_failed": len(rows) - len(successful),
        "mean_sim3_ate_m": mean(sim3),
        "median_sim3_ate_m": median(sim3),
        "worst_sim3_ate_m": worst(sim3),
        "mean_se3_ate_m": mean(se3),
        "median_se3_ate_m": median(se3),
        "mean_raw_ate_m": mean(raw),
        "median_raw_ate_m": median(raw),
        "mean_first20_sim3_ate_m": mean(first20),
        "median_first20_sim3_ate_m": median(first20),
        "mean_rot_rmse_deg": mean(rot),
        "median_rot_rmse_deg": median(rot),
        "total_rejected_updates": float(np.sum(rejected)) if rejected.size else math.nan,
    }


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, float | str]]) -> None:
    ordered = sorted(rows, key=lambda row: (float(row["mean_sim3_ate_m"]), float(row["mean_se3_ate_m"])))
    lines = [
        "# EDS V3 Tuning Summary",
        "",
        "Sorted by mean Sim3 ATE across all successful EDS network-output sequences.",
        "",
        "| run | family | scale | ok | mean Sim3 | median Sim3 | worst Sim3 | mean SE3 | mean raw | mean first20 Sim3 | mean rot | rejected |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered:
        lines.append(
            "| {run_name} | {family} | {scale:.6g} | {num_ok} | {mean_sim3:.6f} | {median_sim3:.6f} | {worst_sim3:.6f} | {mean_se3:.6f} | {mean_raw:.6f} | {mean_first20:.6f} | {mean_rot:.6f} | {rejected:.0f} |".format(
                run_name=row["run_name"],
                family=row["family"],
                scale=float(row["scale"]),
                num_ok=int(row["num_ok"]),
                mean_sim3=float(row["mean_sim3_ate_m"]),
                median_sim3=float(row["median_sim3_ate_m"]),
                worst_sim3=float(row["worst_sim3_ate_m"]),
                mean_se3=float(row["mean_se3_ate_m"]),
                mean_raw=float(row["mean_raw_ate_m"]),
                mean_first20=float(row["mean_first20_sim3_ate_m"]),
                mean_rot=float(row["mean_rot_rmse_deg"]),
                rejected=float(row["total_rejected_updates"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an EDS-wide V3 filter tuning grid.")
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[0.5, 0.75, 1.0, 1.2649054158337365, 1.5, 2.0, 2.5, 3.0, 4.0],
        help="Global measurement covariance scales to evaluate.",
    )
    parser.add_argument("--output_root", type=Path, default=ROOT / "outputs" / "comparison_eds_v3_tuning")
    parser.add_argument("--skip_secondary", action="store_true", help="Only run the covariance-scale grid.")
    parser.add_argument(
        "--secondary_scales",
        nargs="+",
        type=float,
        default=None,
        help="Covariance scales used for midpoint_half_R and edge robust secondary tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, float | str]] = []
    for config in make_grid(
        args.scales,
        include_secondary=not args.skip_secondary,
        secondary_scales=args.secondary_scales,
    ):
        run_name = str(config["run_name"])
        print(f"=== Running {run_name} ===", flush=True)
        rows = run_eds_batch(
            sequences=list(args.sequences),
            preset=str(config["preset"]),
            run_name=run_name,
            extra_filter_args=list(config["extra"]),
            output_root=args.output_root / run_name,
        )
        summaries.append(
            summarize_run(
                run_name=run_name,
                family=str(config["family"]),
                scale=float(config["scale"]),
                rows=rows,
            )
        )
        write_csv(args.output_root / "v3_tuning_summary.csv", summaries)
        write_markdown(args.output_root / "v3_tuning_summary.md", summaries)

    print(f"Saved: {args.output_root / 'v3_tuning_summary.csv'}")
    print(f"Saved: {args.output_root / 'v3_tuning_summary.md'}")


if __name__ == "__main__":
    main()
