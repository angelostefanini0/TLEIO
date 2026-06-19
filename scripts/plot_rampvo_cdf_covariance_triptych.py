from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import yaml
except ImportError:  # pragma: no cover - only used when YAML support is absent.
    yaml = None


SEQUENCES = [
    "ME000", "ME001", "ME002", "ME003", "ME004", "ME005", "ME006", "ME007",
    "MH000", "MH001", "MH002", "MH003", "MH004", "MH005", "MH006", "MH007",
]

METHODS = OrderedDict([
    ("TLIO", "3D RoNIN"),
    ("Eventsformer_no_vggt", "EventsFormer w/o RoPE"),
    ("Eventsformer_vggt", "EventsFormer w/ RoPE"),
    ("TLEIO", "TLEIO"),
    ("TLEIO_no_cov", "TLEIO w/o Adaptive Cov."),
])

STYLE = {
    "3D RoNIN": dict(color="#4C78A8", linestyle="-", linewidth=2.6),
    "EventsFormer w/o RoPE": dict(color="#54A24B", linestyle="-", linewidth=2.6),
    "EventsFormer w/ RoPE": dict(color="#E45756", linestyle="-", linewidth=2.6),
    "TLEIO": dict(color="#F2A900", linestyle="-", linewidth=3.0),
    "TLEIO w/o Adaptive Cov.": dict(color="#8A2BE2", linestyle="-", linewidth=2.7),
}


def configure_style(font_scale: float = 1.0) -> None:
    base_font = 15.0 * font_scale
    label_font = 17.2 * font_scale
    tick_font = 13.2 * font_scale
    legend_font = 13.0 * font_scale
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": base_font,
        "axes.labelsize": label_font,
        "xtick.labelsize": tick_font,
        "ytick.labelsize": tick_font,
        "legend.fontsize": legend_font,
        "axes.linewidth": 0.95,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "grid.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 350,
    })


def load_results(yaml_path: Path) -> OrderedDict[str, np.ndarray]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read resultsATE.yaml.")
    with yaml_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    metrics = raw["experiments_ate_metrics"]
    data: OrderedDict[str, np.ndarray] = OrderedDict()
    for key, label in METHODS.items():
        values = []
        results = metrics[key]["results"]
        for seq in SEQUENCES:
            values.append(float(results[seq]))
        data[label] = np.asarray(values, dtype=float)
    return data


def ecdf(values: np.ndarray, xmax: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(np.clip(values, 0.0, xmax))
    n = len(values)
    xs = [0.0]
    ys = [0.0]
    for idx, x in enumerate(values):
        xs.extend([x, x])
        ys.extend([idx / n, (idx + 1) / n])
    xs.append(xmax)
    ys.append(1.0)
    return np.asarray(xs), 100.0 * np.asarray(ys)


def normalized_auc(values: np.ndarray, xmax: float) -> float:
    xs, ys = ecdf(values, xmax)
    return float(np.trapz(ys / 100.0, xs) / xmax)


def sorted_methods(data: OrderedDict[str, np.ndarray], methods: list[str], auc_xmax: float) -> list[str]:
    return sorted(methods, key=lambda method: normalized_auc(data[method], auc_xmax), reverse=True)


def configure_cdf_axes(ax: plt.Axes, xmax: float) -> None:
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(0.0, 105.0)
    ax.set_xlabel("ATE [m]")
    ax.set_ylabel("Runs [%]")
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xticks([0, 1, 2, 3, 4] if xmax <= 4.0 else [0, 2, 4, 6, 8, 10])
    ax.grid(True, axis="y", color="0.72", alpha=0.95)
    ax.grid(False, axis="x")
    ax.tick_params(axis="both", pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.95)


def draw_cdf(ax: plt.Axes, data: OrderedDict[str, np.ndarray], methods: list[str], xmax: float, auc_xmax: float) -> None:
    for method in sorted_methods(data, methods, auc_xmax):
        xs, ys = ecdf(data[method], xmax)
        auc = normalized_auc(data[method], auc_xmax)
        ax.plot(xs, ys, label=f"{method}  AUC={auc:.2f}", **STYLE[method])

    configure_cdf_axes(ax, xmax)
    legend = ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        framealpha=0.94,
        facecolor="white",
        edgecolor="0.70",
        borderpad=0.34,
        handlelength=2.0,
        handletextpad=0.48,
        labelspacing=0.25,
    )
    legend.get_frame().set_linewidth(0.55)


def save_figure(fig: plt.Figure, out_base: Path, dpi: int) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def plot_full_cdf(data: OrderedDict[str, np.ndarray], out_base: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6.05, 4.45))
    draw_cdf(
        ax,
        data,
        ["3D RoNIN", "EventsFormer w/o RoPE", "EventsFormer w/ RoPE", "TLEIO"],
        xmax=11.0,
        auc_xmax=11.0,
    )
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.16, top=0.985)
    save_figure(fig, out_base, dpi)


def plot_cov_cdf(data: OrderedDict[str, np.ndarray], out_base: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6.05, 4.45))
    draw_cdf(
        ax,
        data,
        ["TLEIO", "TLEIO w/o Adaptive Cov."],
        xmax=4.0,
        auc_xmax=11.0,
    )
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.16, top=0.985)
    save_figure(fig, out_base, dpi)


def read_table(path: Path) -> np.ndarray:
    return np.loadtxt(path, skiprows=1)


def sequence_key(sequence: str) -> str:
    return sequence.replace("competition_Test_", "")


def load_prediction_error(pred_root: Path, gt_root: Path, sequence: str) -> tuple[np.ndarray, np.ndarray]:
    pred_path = pred_root / f"{sequence}.txt"
    gt_path = gt_root / sequence / "relative_motions.txt"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing GT relative motions: {gt_path}")

    pred = read_table(pred_path)
    gt = read_table(gt_path)
    n = min(len(pred), len(gt))
    if n == 0:
        raise ValueError(f"No samples for sequence {sequence}")

    error = pred[:n, 2:5] - gt[:n, 2:5]
    sigma = np.abs(pred[:n, 5:8])
    return error, sigma


def plot_error_vs_sigma(
    pred_root: Path,
    gt_root: Path,
    sequence: str,
    n_sigma: float,
    out_base: Path,
    dpi: int,
) -> None:
    error, sigma = load_prediction_error(pred_root, gt_root, sequence)
    time_s = np.arange(len(error), dtype=float) * 0.05

    fig, axes = plt.subplots(3, 1, figsize=(6.35, 4.55), sharex=True)
    axis_names = ["x", "y", "z"]
    y_labels = [r"$e_x$ [m]", r"$e_y$ [m]", r"$e_z$ [m]"]

    for idx, ax in enumerate(axes):
        bound = n_sigma * sigma[:, idx]
        ax.fill_between(
            time_s,
            -bound,
            bound,
            color="#7DB7E8",
            alpha=0.28,
            linewidth=0.0,
            label=rf"$\pm {n_sigma:g}\sigma_{axis_names[idx]}$",
        )
        ax.plot(
            time_s,
            error[:, idx],
            color="#2F73D0",
            linewidth=1.7,
            label=rf"$e_{axis_names[idx]}$",
        )
        ax.axhline(0.0, color="0.18", linewidth=0.7, alpha=0.75)
        ax.set_ylabel(y_labels[idx], labelpad=6)
        ax.grid(True, color="0.70", alpha=0.72)
        ax.legend(
            loc="upper left",
            frameon=True,
            fancybox=False,
            framealpha=0.90,
            facecolor="white",
            edgecolor="0.78",
            borderpad=0.25,
            handlelength=1.7,
            labelspacing=0.20,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)

    axes[-1].set_xlabel("Time [s]")
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.13, top=0.985, hspace=0.22)
    save_figure(fig, out_base, dpi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", default="resultsATE.yaml")
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--out-dir", default="figures/tartanair_rampvo_triptych")
    parser.add_argument("--sequence", default="competition_Test_MH001")
    parser.add_argument("--n-sigma", type=float, default=3.0)
    parser.add_argument("--dpi", type=int, default=350)
    parser.add_argument("--font-scale", type=float, default=1.0)
    args = parser.parse_args()

    configure_style(args.font_scale)
    out_dir = Path(args.out_dir)
    data = load_results(Path(args.yaml))

    plot_full_cdf(data, out_dir / "01_cdf_full_comparison", args.dpi)
    plot_cov_cdf(data, out_dir / "02_cdf_adaptive_covariance", args.dpi)
    plot_error_vs_sigma(
        Path(args.pred_root),
        Path(args.gt_root),
        args.sequence,
        args.n_sigma,
        out_dir / f"03_error_vs_3sigma_{args.sequence}",
        args.dpi,
    )

    print(f"Saved: {out_dir / '01_cdf_full_comparison.png'}")
    print(f"Saved: {out_dir / '02_cdf_adaptive_covariance.png'}")
    print(f"Saved: {out_dir / f'03_error_vs_3sigma_{args.sequence}.png'}")


if __name__ == "__main__":
    main()
