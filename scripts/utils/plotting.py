from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


POSTER_COLORS = {
    "blue": "#2563A9",
    "light_blue": "#7EA6E0",
    "panel": "#FFFFFF",
    "red": "#C93F4A",
    "purple": "#6B7280",
    "grid": "#D0D5DD",
    "text": "#111827",
}


def plot_covariance_error_cones(
    rel_err_xyz: np.ndarray,
    rel_sigma: np.ndarray,
    save_path: Path | None,
    title: str = "Predicted Uncertainty vs Translation Error",
    max_points: int = 200_000,
    error_limit: float | None = None,
    sigma_limit: float | None = None,
    sigma_multiplier: float = 3.0,
    random_seed: int = 0,
) -> dict[str, np.ndarray | int]:
    """Plot predicted translation sigma norm against relative translation error norm.

    The dashed red boundary is the n-sigma condition: points below it have
    ``||error|| > n * ||sigma||`` and are counted as outside the bound.
    """
    rel_err_xyz = np.asarray(rel_err_xyz, dtype=np.float64)
    rel_sigma = np.asarray(rel_sigma, dtype=np.float64)
    if rel_err_xyz.shape != rel_sigma.shape or rel_err_xyz.ndim != 2 or rel_err_xyz.shape[1] != 3:
        raise ValueError(
            "Expected rel_err_xyz and rel_sigma with matching shape [N, 3], "
            f"got {rel_err_xyz.shape} and {rel_sigma.shape}."
        )

    finite = np.isfinite(rel_err_xyz).all(axis=1) & np.isfinite(rel_sigma).all(axis=1)
    nonnegative_sigma = (rel_sigma >= 0).all(axis=1)
    valid = finite & nonnegative_sigma
    rel_err_xyz = rel_err_xyz[valid]
    rel_sigma = rel_sigma[valid]
    if len(rel_err_xyz) == 0:
        raise ValueError("No valid covariance/error rows to plot.")

    if sigma_multiplier <= 0:
        raise ValueError(f"sigma_multiplier must be positive, got {sigma_multiplier}.")

    error_norm = np.linalg.norm(rel_err_xyz, axis=1)
    sigma_norm = np.linalg.norm(rel_sigma, axis=1)
    outside_mask = error_norm > sigma_multiplier * sigma_norm
    outside_percent = float(outside_mask.mean() * 100.0)
    rms_error = float(np.sqrt(np.mean(error_norm**2)))
    mean_sigma = float(np.mean(sigma_norm))

    plot_err = error_norm
    plot_sigma = sigma_norm
    if max_points > 0 and len(plot_err) > max_points:
        rng = np.random.default_rng(random_seed)
        keep = rng.choice(len(plot_err), size=max_points, replace=False)
        plot_err = plot_err[keep]
        plot_sigma = plot_sigma[keep]

    if error_limit is None:
        q_error = np.nanpercentile(plot_err, 99.5)
        error_limit = max(float(q_error), 1e-6)
    if sigma_limit is None:
        q_sigma = np.nanpercentile(plot_sigma, 99.5)
        sigma_limit = max(float(q_sigma), error_limit / 3.0, 1e-6)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    x_line = np.linspace(0.0, error_limit, 300)
    y_line = x_line / sigma_multiplier
    ax.scatter(plot_err, plot_sigma, s=3, alpha=0.28, linewidths=0, color=POSTER_COLORS["blue"])
    ax.plot(x_line, y_line, color=POSTER_COLORS["red"], linestyle="--", linewidth=1.2)
    ax.set_xlim(0.0, error_limit)
    ax.set_ylim(0.0, sigma_limit)
    ax.set_xlabel("||Error|| [m]")
    ax.set_ylabel("||Sigma|| [m]")
    ax.grid(True, color=POSTER_COLORS["grid"], alpha=0.65, linewidth=0.7)
    ax.set_title(f"{outside_percent:.2f}% outside {sigma_multiplier:g} sigma", color=POSTER_COLORS["text"], fontsize=12)
    fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return {
        "num_valid": len(rel_err_xyz),
        "num_plotted": len(plot_err),
        "outside_percent": outside_percent,
        "rms_error": rms_error,
        "mean_sigma": mean_sigma,
    }


def plot_relative_motion_inspection(
    gt_ts: np.ndarray,
    gt_pos: np.ndarray,
    anchor_ts: np.ndarray,
    ref_pos: np.ndarray,
    recon_pos: np.ndarray,
    rel_t1: np.ndarray,
    pos_err_rel: np.ndarray | None,
    rel_err_xyz: np.ndarray | None,
    rel_sigma: np.ndarray | None,
    error_ref_label: str,
    save_dir: Path | None,
    sigma_multiplier: float = 3.0,
) -> None:
    """Plot reconstructed relative-motion trajectories against GT."""
    t_gt = (gt_ts - gt_ts[0]) * 1e-6
    t_anchor = (anchor_ts - gt_ts[0]) * 1e-6
    t_err = (rel_t1 - gt_ts[0]) * 1e-6
    sequence_title = f" ({save_dir.name})" if save_dir is not None else ""

    fig1, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = ["x", "y", "z"]
    sigma_colors = [POSTER_COLORS["blue"], POSTER_COLORS["blue"], POSTER_COLORS["blue"]]
    for i in range(3):
        axes[i].plot(t_gt, gt_pos[:, i], label="raw stamped GT")
        axes[i].plot(t_anchor, ref_pos[:, i], "--", label="GT at anchor times")
        axes[i].plot(t_anchor, recon_pos[:, i], label="trajectory from relative motions")
        axes[i].set_ylabel(f"p{labels[i]} [m]")
        axes[i].grid(True)
    axes[0].legend()
    axes[-1].set_xlabel("time [s]")
    fig1.suptitle("Trajectory from Relative Motions vs Source GT")

    fig2, ax = plt.subplots(figsize=(8, 8))
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], label="raw stamped GT")
    ax.plot(ref_pos[:, 0], ref_pos[:, 1], "--", label="GT at anchor times")
    ax.plot(recon_pos[:, 0], recon_pos[:, 1], label="trajectory from relative motions")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True)
    ax.legend()
    ax.set_title("XY trajectory comparison")

    fig3 = plt.figure(figsize=(10, 8))
    ax3d = fig3.add_subplot(111, projection="3d")
    ax3d.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], label="raw stamped GT")
    ax3d.plot(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2], "--", label="GT at anchor times")
    ax3d.plot(
        recon_pos[:, 0],
        recon_pos[:, 1],
        recon_pos[:, 2],
        label="trajectory from relative motions",
    )
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.set_title("3D trajectory comparison")
    all_points = np.vstack([gt_pos, ref_pos, recon_pos])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    if radius < 1e-12:
        radius = 1.0
    ax3d.set_xlim(centers[0] - radius, centers[0] + radius)
    ax3d.set_ylim(centers[1] - radius, centers[1] + radius)
    ax3d.set_zlim(centers[2] - radius, centers[2] + radius)
    ax3d.legend()

    if pos_err_rel is not None:
        fig4, ax = plt.subplots(figsize=(12, 4), sharex=True)
        ax.plot(t_err, pos_err_rel)
        ax.set_ylabel("pos err [m]")
        ax.set_xlabel("time [s]")
        ax.grid(True)
        fig4.suptitle(f"Translation error vs {error_ref_label}")
    else:
        fig4 = None

    if rel_sigma is not None:
        fig5, axes = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True)
        for i, label in enumerate(labels):
            axes[i].plot(
                t_err,
                rel_sigma[:, i],
                label=f"sigma {label}",
                color=sigma_colors[i],
                linewidth=1.4,
            )
            axes[i].set_ylabel(f"sigma_{label} [m]")
            axes[i].grid(True, color=POSTER_COLORS["grid"], alpha=0.65, linewidth=0.7)
            axes[i].legend(loc="upper right", frameon=False, fontsize=9)
        axes[-1].set_xlabel("time [s]")
        fig5.suptitle("Predicted Translation Uncertainty", color=POSTER_COLORS["text"], fontsize=13)
    else:
        fig5 = None

    if rel_err_xyz is not None and rel_sigma is not None:
        fig6, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for i, label in enumerate(labels):
            sigma_i = rel_sigma[:, i]
            axes[i].plot(
                t_err,
                rel_err_xyz[:, i],
                label=f"error_{label}",
                color="tab:blue",
                linewidth=1.0,
            )
            axes[i].fill_between(
                t_err,
                -sigma_i,
                sigma_i,
                color="tab:blue",
                alpha=0.22,
                linewidth=0.0,
                edgecolor="none",
                label=f"+/- sigma_{label}",
            )
            axes[i].set_ylabel(f"e{label} [m]")
            axes[i].grid(True)
            axes[i].legend(loc="upper left", fontsize=8)
        axes[-1].set_xlabel("time [s]")
        fig6.suptitle(
            f"Translation Error with Predicted Uncertainty vs {error_ref_label}{sequence_title}",
            fontsize=10,
        )
    else:
        fig6 = None

    cone_stats = None

    plt.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig1.savefig(save_dir / "relative_vs_gt_xyz.png", dpi=150, bbox_inches="tight")
        fig2.savefig(save_dir / "relative_vs_gt_xy.png", dpi=150, bbox_inches="tight")
        fig3.savefig(save_dir / "relative_vs_gt_xyz_3d.png", dpi=150, bbox_inches="tight")
        if fig4 is not None:
            fig4.savefig(save_dir / "relative_vs_gt_error.png", dpi=150, bbox_inches="tight")
        if fig5 is not None:
            fig5.savefig(save_dir / "relative_uncertainty_sigma.png", dpi=150, bbox_inches="tight")
        if fig6 is not None:
            fig6.savefig(save_dir / "relative_error_with_uncertainty.png", dpi=150, bbox_inches="tight")
            cone_stats = plot_covariance_error_cones(
                rel_err_xyz=rel_err_xyz,
                rel_sigma=rel_sigma,
                save_path=save_dir / "relative_uncertainty_error_cones.png",
                title=f"Uncertainty vs Translation Error ({error_ref_label})",
                sigma_multiplier=sigma_multiplier,
            )
        print(f"Saved figures to {save_dir}")
        if cone_stats is not None:
            outside = cone_stats["outside_percent"]
            print(
                f"Outside {sigma_multiplier:g} sigma [%]: "
                f"norm={outside:.2f}"
            )
    else:
        plt.show()
