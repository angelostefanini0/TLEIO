from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
) -> None:
    """Plot reconstructed relative-motion trajectories against GT."""
    t_gt = (gt_ts - gt_ts[0]) * 1e-6
    t_anchor = (anchor_ts - gt_ts[0]) * 1e-6
    t_err = (rel_t1 - gt_ts[0]) * 1e-6

    fig1, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = ["x", "y", "z"]
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
        fig5, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for i, label in enumerate(labels):
            axes[i].plot(t_err, rel_sigma[:, i], label=f"sigma_{label}")
            axes[i].set_ylabel(f"sigma_{label} [m]")
            axes[i].grid(True)
            axes[i].legend()
        axes[-1].set_xlabel("time [s]")
        fig5.suptitle("Predicted Translation Uncertainty")
    else:
        fig5 = None

    if rel_err_xyz is not None and rel_sigma is not None:
        fig6, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for i, label in enumerate(labels):
            sigma_i = rel_sigma[:, i]
            axes[i].plot(t_err, rel_err_xyz[:, i], label=f"error_{label}")
            axes[i].fill_between(t_err, -sigma_i, sigma_i, alpha=0.25, label=f"+/- sigma_{label}")
            axes[i].plot(t_err, sigma_i, color="tab:orange", linewidth=0.8)
            axes[i].plot(t_err, -sigma_i, color="tab:orange", linewidth=0.8)
            axes[i].set_ylabel(f"e{label} [m]")
            axes[i].grid(True)
            axes[i].legend()
        axes[-1].set_xlabel("time [s]")
        fig6.suptitle(f"Translation Error with Predicted Uncertainty vs {error_ref_label}")
    else:
        fig6 = None

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
        print(f"Saved figures to {save_dir}")
    else:
        plt.show()
