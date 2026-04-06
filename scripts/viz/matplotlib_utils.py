from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class LiveTrajectoryPlot:
    fig: plt.Figure
    ax: plt.Axes
    pred_line: any
    gt_line: any
    pred_point: any
    gt_point: any
    status_text: any


@dataclass
class LiveTrajectoryViewer:
    fig: plt.Figure
    ax_image: plt.Axes
    ax_traj: plt.Axes
    image_artist: any
    pred_line: any
    gt_line: any
    pred_point: any
    gt_point: any
    status_text: any
    closed: bool = False
    quit_requested: bool = False


def create_live_trajectory_plot(title: str = "Live Trajectory") -> LiveTrajectoryPlot:
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))

    pred_line, = ax.plot([], [], color="blue", linewidth=2.0, label="predicted")
    gt_line, = ax.plot([], [], color="red", linewidth=2.0, label="ground truth")
    pred_point, = ax.plot([], [], "o", color="blue", markersize=6)
    gt_point, = ax.plot([], [], "o", color="red", markersize=6)
    status_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.show()

    return LiveTrajectoryPlot(
        fig=fig,
        ax=ax,
        pred_line=pred_line,
        gt_line=gt_line,
        pred_point=pred_point,
        gt_point=gt_point,
        status_text=status_text,
    )


def _set_equal_xy_limits(ax: plt.Axes, pred_xy: np.ndarray, gt_xy: np.ndarray) -> None:
    all_xy = np.vstack([pred_xy, gt_xy])
    mins = all_xy.min(axis=0)
    maxs = all_xy.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius = max(radius, 1e-3)
    pad = 0.05 * radius

    ax.set_xlim(center[0] - radius - pad, center[0] + radius + pad)
    ax.set_ylim(center[1] - radius - pad, center[1] + radius + pad)


def update_live_trajectory_plot(
    plotter: LiveTrajectoryPlot,
    pred_pos: np.ndarray,
    gt_pos: np.ndarray,
    frame_idx: int | None = None,
    timestamp_s: float | None = None,
) -> None:
    pred_xy = pred_pos[:, :2]
    gt_xy = gt_pos[:, :2]

    plotter.pred_line.set_data(pred_xy[:, 0], pred_xy[:, 1])
    plotter.gt_line.set_data(gt_xy[:, 0], gt_xy[:, 1])
    plotter.pred_point.set_data([pred_xy[-1, 0]], [pred_xy[-1, 1]])
    plotter.gt_point.set_data([gt_xy[-1, 0]], [gt_xy[-1, 1]])

    status_parts = []
    if frame_idx is not None:
        status_parts.append(f"frame: {frame_idx}")
    if timestamp_s is not None:
        status_parts.append(f"time: {timestamp_s:.3f} s")
    plotter.status_text.set_text(" | ".join(status_parts))

    _set_equal_xy_limits(plotter.ax, pred_xy, gt_xy)
    plotter.fig.canvas.draw_idle()
    plotter.fig.canvas.flush_events()
    plt.pause(0.001)


def create_live_trajectory_viewer(
    image_shape: tuple[int, int, int],
    title: str = "Live Trajectory Viewer",
) -> LiveTrajectoryViewer:
    plt.ion()
    fig, (ax_image, ax_traj) = plt.subplots(
        1,
        2,
        figsize=(14, 7),
        gridspec_kw={"width_ratios": [1.1, 1.0]},
    )

    empty_frame = np.full(image_shape, 255, dtype=np.uint8)
    image_artist = ax_image.imshow(empty_frame)
    ax_image.set_title("Events on RGB")
    ax_image.axis("off")

    pred_line, = ax_traj.plot([], [], color="blue", linewidth=2.0, label="predicted")
    gt_line, = ax_traj.plot([], [], color="red", linewidth=2.0, label="ground truth")
    pred_point, = ax_traj.plot([], [], "o", color="blue", markersize=6)
    gt_point, = ax_traj.plot([], [], "o", color="red", markersize=6)
    status_text = ax_traj.text(0.02, 0.98, "", transform=ax_traj.transAxes, va="top")

    ax_traj.set_title("XY trajectory")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.grid(True)
    ax_traj.legend()
    ax_traj.set_aspect("equal", adjustable="box")

    fig.suptitle(title)
    fig.tight_layout()

    viewer = LiveTrajectoryViewer(
        fig=fig,
        ax_image=ax_image,
        ax_traj=ax_traj,
        image_artist=image_artist,
        pred_line=pred_line,
        gt_line=gt_line,
        pred_point=pred_point,
        gt_point=gt_point,
        status_text=status_text,
    )

    def _on_close(_event) -> None:
        viewer.closed = True

    def _on_key(event) -> None:
        if event.key in ("q", "escape"):
            viewer.quit_requested = True

    fig.canvas.mpl_connect("close_event", _on_close)
    fig.canvas.mpl_connect("key_press_event", _on_key)
    fig.show()
    return viewer


def update_live_trajectory_viewer(
    viewer: LiveTrajectoryViewer,
    frame: np.ndarray,
    pred_pos: np.ndarray,
    gt_pos: np.ndarray,
    frame_idx: int | None = None,
    timestamp_s: float | None = None,
    pause_s: float = 0.001,
) -> None:
    viewer.image_artist.set_data(frame)

    pred_xy = pred_pos[:, :2]
    gt_xy = gt_pos[:, :2]

    viewer.pred_line.set_data(pred_xy[:, 0], pred_xy[:, 1])
    viewer.gt_line.set_data(gt_xy[:, 0], gt_xy[:, 1])
    viewer.pred_point.set_data([pred_xy[-1, 0]], [pred_xy[-1, 1]])
    viewer.gt_point.set_data([gt_xy[-1, 0]], [gt_xy[-1, 1]])

    status_parts = []
    if frame_idx is not None:
        status_parts.append(f"frame: {frame_idx}")
    if timestamp_s is not None:
        status_parts.append(f"time: {timestamp_s:.3f} s")
    viewer.status_text.set_text(" | ".join(status_parts))
    viewer.ax_image.set_title("Events on RGB" if not status_parts else "Events on RGB | " + " | ".join(status_parts))

    _set_equal_xy_limits(viewer.ax_traj, pred_xy, gt_xy)
    viewer.fig.canvas.draw_idle()
    viewer.fig.canvas.flush_events()
    plt.pause(max(pause_s, 0.001))
