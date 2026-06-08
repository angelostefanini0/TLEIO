from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "gt": "tab:blue",
    "pred": "tab:red",
    "sigma_x": "tab:blue",
    "sigma_y": "tab:orange",
    "sigma_z": "tab:green",
    "total": "tab:purple",
}


def load_table(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline().strip()
    skiprows = 1 if first and not first[0].isdigit() and first[0] != "-" else 0
    data = np.loadtxt(path, skiprows=skiprows, dtype=np.float64)
    return np.atleast_2d(data)


def short_sequence_name(sequence: str) -> str:
    for token in reversed(sequence.split("_")):
        if len(token) >= 3 and token[:2] in {"ME", "MH"} and token[2:].isdigit():
            return token
    return sequence


def style_axis(ax) -> None:
    ax.grid(True, color="#9a9a9a", alpha=0.38, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=10)


def plot_sigma(path: Path, t_s: np.ndarray, sigmas: np.ndarray, sequence: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True)
    labels = [("x", COLORS["sigma_x"]), ("y", COLORS["sigma_y"]), ("z", COLORS["sigma_z"])]
    for idx, (label, color) in enumerate(labels):
        axes[idx].plot(t_s, sigmas[:, idx], color=color, linewidth=1.8, label=f"sigma {label}")
        axes[idx].set_title(f"Sigma {label.upper()}", fontsize=16, pad=7)
        axes[idx].set_ylabel("sigma [m]", fontsize=12)
        axes[idx].legend(loc="upper right", frameon=True, fontsize=13)
        style_axis(axes[idx])
    axes[-1].set_xlabel("time [s]", fontsize=12)
    fig.suptitle(f"Relative Uncertainty ({short_sequence_name(sequence)})", fontsize=18, y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_with_uncertainty(
    path: Path,
    t_s: np.ndarray,
    errors_xyz: np.ndarray,
    sigmas: np.ndarray,
    sequence: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    error_norm = np.linalg.norm(errors_xyz, axis=1)
    sigma_norm = np.linalg.norm(sigmas, axis=1)

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    labels = [("x", COLORS["sigma_x"]), ("y", COLORS["sigma_y"]), ("z", COLORS["sigma_z"])]
    for idx, (label, color) in enumerate(labels):
        err = errors_xyz[:, idx]
        sigma = sigmas[:, idx]
        axes[idx].plot(t_s, err, color=color, linewidth=1.6, label=f"error {label}")
        axes[idx].fill_between(t_s, -sigma, sigma, color=color, alpha=0.18, label=f"+/- sigma {label}")
        axes[idx].set_title(f"{label.upper()} Error with Uncertainty", fontsize=15, pad=7)
        axes[idx].set_ylabel("error [m]", fontsize=12)
        axes[idx].legend(loc="upper right", frameon=True, fontsize=12)
        style_axis(axes[idx])

    axes[3].plot(t_s, error_norm, color=COLORS["pred"], linewidth=1.8, label="translation error norm")
    axes[3].plot(t_s, sigma_norm, color=COLORS["total"], linewidth=1.8, linestyle="--", label="sigma norm")
    axes[3].set_title("Translation Error Norm", fontsize=15, pad=7)
    axes[3].set_ylabel("norm [m]", fontsize=12)
    axes[3].set_xlabel("time [s]", fontsize=12)
    axes[3].legend(loc="upper right", frameon=True, fontsize=12)
    style_axis(axes[3])

    fig.suptitle(f"Relative Error with Uncertainty ({short_sequence_name(sequence)})", fontsize=18, y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_cones(path: Path, errors_xyz: np.ndarray, sigmas: np.ndarray, sequence: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    error_norm = np.linalg.norm(errors_xyz, axis=1)
    sigma_norm = np.linalg.norm(sigmas, axis=1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(sigma_norm, error_norm, s=12, color=COLORS["pred"], alpha=0.75, label="relative edges")
    lim = max(float(np.max(sigma_norm)), float(np.max(error_norm)), 1e-9)
    xs = np.linspace(0.0, lim, 128)
    ax.plot(xs, xs, color="black", linewidth=1.5, label="1 sigma")
    ax.plot(xs, 2.0 * xs, color=COLORS["total"], linewidth=1.5, linestyle="--", label="2 sigma")
    ax.set_xlim(0.0, lim)
    ax.set_ylim(0.0, max(lim, float(np.max(error_norm))))
    ax.set_xlabel("predicted sigma norm [m]", fontsize=12)
    ax.set_ylabel("translation error norm [m]", fontsize=12)
    ax.set_title(f"Uncertainty Calibration ({short_sequence_name(sequence)})", fontsize=16, pad=10)
    style_axis(ax)
    ax.legend(loc="upper left", frameon=True, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sequence(pred_path: Path, gt_path: Path, out_dir: Path) -> None:
    pred = load_table(pred_path)
    gt = load_table(gt_path)
    if pred.shape[1] < 8:
        raise ValueError(f"{pred_path} must have at least 8 columns: t0 t1 px py pz sigma_x sigma_y sigma_z")
    if gt.shape[1] < 5:
        raise ValueError(f"{gt_path} must have at least 5 columns: t0 t1 px py pz")

    n = min(len(pred), len(gt))
    pred = pred[:n]
    gt = gt[:n]
    sequence = pred_path.stem
    t_s = (pred[:, 1] - pred[0, 0]) * 1e-6
    errors_xyz = pred[:, 2:5] - gt[:, 2:5]
    sigmas = pred[:, 5:8]

    seq_out = out_dir / sequence
    plot_sigma(seq_out / "relative_uncertainty_sigma.png", t_s, sigmas, sequence)
    plot_error_with_uncertainty(seq_out / "relative_error_with_uncertainty.png", t_s, errors_xyz, sigmas, sequence)
    plot_error_cones(seq_out / "relative_uncertainty_error_cones.png", errors_xyz, sigmas, sequence)
    print(f"saved covariance plots: {seq_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate covariance-only relative-motion plots.")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Folder with <sequence>.txt network outputs including sigma columns.")
    parser.add_argument("--gt-root", type=Path, required=True, help="Processed dataset root with <sequence>/relative_motions.txt.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output plot folder.")
    parser.add_argument("--sequence", type=str, default=None, help="Optional comma-separated sequence list.")
    args = parser.parse_args()

    pred_paths = sorted(args.pred_dir.glob("*.txt"))
    if args.sequence:
        keep = {item.strip() for item in args.sequence.split(",") if item.strip()}
        pred_paths = [path for path in pred_paths if path.stem in keep]
    if not pred_paths:
        raise FileNotFoundError(f"No prediction .txt files found in {args.pred_dir}")

    for pred_path in pred_paths:
        gt_path = args.gt_root / pred_path.stem / "relative_motions.txt"
        if not gt_path.exists():
            print(f"skip {pred_path.stem}: missing {gt_path}")
            continue
        plot_sequence(pred_path, gt_path, args.out_dir)


if __name__ == "__main__":
    main()
